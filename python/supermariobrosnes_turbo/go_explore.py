"""Deterministic Go-Explore trajectory discovery without robustification."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import math
import sys
from typing import Any

import numpy as np

from .beam import CompletionEvent
from .jerk import ActionRun, JerkPolicy, canonicalize_runs, run_step_count


RECENT_CELL_VISIT_WINDOW = 10_000


def _deep_sizeof(value: Any, seen: set[int] | None = None) -> int:
    """Estimate retained Python and native bytes without double-counting objects."""
    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)

    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size if value.flags.owndata else size + int(value.nbytes)

    native_nbytes = getattr(value, "nbytes", None)
    if isinstance(native_nbytes, (int, np.integer)):
        return size + int(native_nbytes)

    if is_dataclass(value) and not isinstance(value, type):
        return size + sum(
            _deep_sizeof(getattr(value, item.name), visited) for item in fields(value)
        )
    if isinstance(value, Mapping):
        return size + sum(
            _deep_sizeof(key, visited) + _deep_sizeof(item, visited)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_sizeof(item, visited) for item in value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return size + _deep_sizeof(attributes, visited)
    return size


@dataclass(frozen=True)
class GoExploreCandidate:
    """One replayable action-run trajectory and its observed outcome."""

    runs: tuple[ActionRun, ...]
    episode_return: float
    progress: float
    completed: bool = False
    _canonical_step_count: int | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._canonical_step_count is None:
            runs = canonicalize_runs(self.runs)
            step_count = run_step_count(runs)
        else:
            runs = tuple(self.runs)
            step_count = int(self._canonical_step_count)
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "_canonical_step_count", step_count)

    @classmethod
    def _from_canonical(
        cls,
        *,
        runs: Sequence[ActionRun],
        step_count: int,
        episode_return: float,
        progress: float,
        completed: bool,
    ) -> GoExploreCandidate:
        """Freeze an internally maintained canonical trajectory without rebuilding it."""
        return cls(
            runs=tuple(runs),
            episode_return=episode_return,
            progress=progress,
            completed=completed,
            _canonical_step_count=step_count,
        )

    @property
    def step_count(self) -> int:
        assert self._canonical_step_count is not None
        return self._canonical_step_count

    @property
    def mean_return(self) -> float:
        return self.episode_return


@dataclass
class GoExploreCell:
    """One archived cell, exact restore point, and trajectory to reach it."""

    key: Hashable
    snapshot: Any
    runs: tuple[ActionRun, ...]
    episode_return: float
    progress: float
    program_steps: int = 0
    visits: int = 0
    selections: int = 0
    updates: int = 0

    @property
    def step_count(self) -> int:
        return self.program_steps


@dataclass(frozen=True)
class GoExploreObservation:
    """Work the environment adapter must perform after an observed step."""

    archive_mask: np.ndarray
    restart_mask: np.ndarray


@dataclass
class _PendingCell:
    key: Hashable
    lane: int
    runs: tuple[ActionRun, ...]
    episode_return: float
    progress: float
    step_count: int
    visits: int


@dataclass
class _GoExploreLaneState:
    runs: list[ActionRun] = field(default_factory=list)
    episode_return: float = 0.0
    progress: float = 0.0
    program_steps: int = 0
    steps_since_restart: int = 0
    exploration_action: int = 0
    exploration_remaining: int = 0


class GoExploreSearch:
    """Snapshot-based Go-Explore trajectory discovery.

    The search repeatedly restores archived cells and performs stochastic
    exploration from them. It intentionally implements only trajectory finding;
    no robustification, imitation learning, or stochastic-domain training is
    performed.
    """

    def __init__(
        self,
        *,
        n_envs: int,
        seed: int,
        action_names: Sequence[str],
        fallback_action: str,
        explore_steps: int,
        run_duration_mean: float,
        run_duration_max: int,
    ) -> None:
        if n_envs < 1:
            raise ValueError("Go-Explore requires at least one environment")
        if explore_steps < 1:
            raise ValueError("Go-Explore exploration steps must be positive")
        if run_duration_mean < 1.0:
            raise ValueError("Go-Explore run duration mean must be at least one")
        if run_duration_max < 1:
            raise ValueError("Go-Explore run duration maximum must be positive")
        self.n_envs = int(n_envs)
        self.seed = int(seed)
        self.action_names = tuple(str(name) for name in action_names)
        if not self.action_names:
            raise ValueError("Go-Explore requires at least one action name")
        try:
            self.fallback_action = self.action_names.index(str(fallback_action))
        except ValueError as exc:
            raise ValueError(
                "Go-Explore fallback action is absent from the task action set: "
                f"{fallback_action}"
            ) from exc
        self.explore_steps = int(explore_steps)
        self.run_duration_mean = float(run_duration_mean)
        self.run_duration_max = int(run_duration_max)
        self.global_step = 0
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.improvement_count = 0
        self.first_success_return: float | None = None
        self._archive: dict[Hashable, GoExploreCell] = {}
        self._archive_memory_bytes = _deep_sizeof(self._archive)
        self._archive_selection_count = 0
        self._archive_visit_count = 0
        self._archive_update_count = 0
        self._recent_cell_batches: deque[tuple[int, int]] = deque()
        self._recent_cell_visits = 0
        self._recent_new_cells = 0
        self._lanes = [_GoExploreLaneState() for _ in range(self.n_envs)]
        self._rngs = [
            np.random.default_rng(np.random.SeedSequence([self.seed, lane, 0x474F4558]))
            for lane in range(self.n_envs)
        ]
        self._pending: dict[Hashable, _PendingCell] = {}
        self._best_incomplete: GoExploreCandidate | None = None
        self._best_success: GoExploreCandidate | None = None
        self._completion_events: list[CompletionEvent] = []

    @property
    def archive_count(self) -> int:
        return len(self._archive)

    @property
    def archive_selection_count(self) -> int:
        return self._archive_selection_count

    @property
    def archive_visit_count(self) -> int:
        return self._archive_visit_count

    @property
    def archive_update_count(self) -> int:
        return self._archive_update_count

    @property
    def archive_memory_bytes(self) -> int:
        return self._archive_memory_bytes

    @property
    def archive_recent_new_cell_rate(self) -> float:
        if self._recent_cell_visits == 0:
            return 0.0
        return self._recent_new_cells / self._recent_cell_visits

    @property
    def archive_recent_visit_window(self) -> int:
        return self._recent_cell_visits

    @property
    def archive_visits_per_cell(self) -> float:
        if not self._archive:
            return 0.0
        return self.archive_visit_count / len(self._archive)

    @property
    def retained_count(self) -> int:
        return self.archive_count + self.successful_episodes

    @property
    def locked_count(self) -> int:
        return self.successful_episodes

    @property
    def incomplete_retained_count(self) -> int:
        return self.archive_count

    @property
    def best_success_return(self) -> float | None:
        if self._best_success is None:
            return None
        return self._best_success.episode_return

    @property
    def archive(self) -> Mapping[Hashable, GoExploreCell]:
        return self._archive

    def initialize(
        self,
        cell_keys: Sequence[Hashable],
        snapshots: Sequence[Any | None],
    ) -> None:
        """Seed the archive from the environment's already-reset lanes."""
        if len(cell_keys) != self.n_envs or len(snapshots) != self.n_envs:
            raise ValueError("Go-Explore initialization requires one value per lane")
        for lane, (key, snapshot) in enumerate(zip(cell_keys, snapshots)):
            if snapshot is None:
                raise ValueError("Go-Explore initialization snapshots cannot be empty")
            cell = self._archive.get(key)
            if cell is None:
                self._archive[key] = GoExploreCell(
                    key=key,
                    snapshot=snapshot,
                    runs=(),
                    episode_return=0.0,
                    progress=0.0,
                    visits=1,
                )
            else:
                cell.visits += 1
            self._archive_visit_count += 1
            self._lanes[lane] = _GoExploreLaneState()
        self._archive_memory_bytes = _deep_sizeof(self._archive)

    @staticmethod
    def _append_action(state: _GoExploreLaneState, action: int) -> None:
        if state.runs and state.runs[-1].action == action:
            previous = state.runs[-1]
            state.runs[-1] = ActionRun(action, previous.duration + 1)
        else:
            state.runs.append(ActionRun(action, 1))
        state.program_steps += 1
        state.steps_since_restart += 1

    def _sample_exploration_run(self, lane: int, state: _GoExploreLaneState) -> None:
        rng = self._rngs[lane]
        previous = state.runs[-1].action if state.runs else None
        action_count = len(self.action_names)
        if previous is None or action_count == 1:
            action = int(rng.integers(0, action_count))
        else:
            sampled = int(rng.integers(0, action_count - 1))
            action = sampled + int(sampled >= previous)
        duration = min(
            int(rng.geometric(1.0 / self.run_duration_mean)),
            self.run_duration_max,
        )
        state.exploration_action = action
        state.exploration_remaining = duration

    def next_actions(self) -> np.ndarray:
        if not self._archive:
            raise RuntimeError("Go-Explore must be initialized before stepping")
        actions = np.empty(self.n_envs, dtype=np.int64)
        for lane, state in enumerate(self._lanes):
            if state.exploration_remaining == 0:
                self._sample_exploration_run(lane, state)
            action = state.exploration_action
            state.exploration_remaining -= 1
            self._append_action(state, action)
            actions[lane] = action
        return actions

    @staticmethod
    def _record_facts(record: Any | None) -> tuple[bool, float]:
        if record is None:
            return False, 0.0
        if isinstance(record, Mapping):
            return (
                bool(record.get("completed", record.get("level_complete", False))),
                float(record.get("progress", record.get("max_x_pos", 0.0)) or 0.0),
            )
        metrics = getattr(record, "metrics", {}) or {}
        return (
            bool(
                getattr(record, "completed", False)
                or getattr(record, "outcome", None) == "success"
                or metrics.get("level_complete", False)
            ),
            float(
                getattr(
                    record,
                    "progress",
                    metrics.get("max_x_pos", metrics.get("global_max_x_pos", 0.0)),
                )
                or 0.0
            ),
        )

    @staticmethod
    def _cell_candidate_better(
        *,
        episode_return: float,
        step_count: int,
        cell: GoExploreCell | _PendingCell | None,
    ) -> bool:
        if cell is None:
            return True
        return episode_return > cell.episode_return or (
            episode_return == cell.episode_return and step_count < cell.step_count
        )

    def _consider_lane_candidate(
        self,
        state: _GoExploreLaneState,
        *,
        progress: float,
        completed: bool,
    ) -> None:
        """Freeze a lane trajectory only when it can affect retained search state."""
        previous = self._best_success if completed else self._best_incomplete
        if not completed and previous is not None:
            rank = (progress, state.episode_return, -state.program_steps)
            previous_rank = (
                previous.progress,
                previous.episode_return,
                -previous.step_count,
            )
            if rank < previous_rank:
                return
        candidate = GoExploreCandidate._from_canonical(
            runs=state.runs,
            step_count=state.program_steps,
            episode_return=state.episode_return,
            progress=progress,
            completed=completed,
        )
        self._consider_best(candidate)

    def _consider_best(self, candidate: GoExploreCandidate) -> None:
        if candidate.completed:
            previous = self._best_success
            improved = (
                previous is None or candidate.episode_return > previous.episode_return
            )
            if improved:
                self._best_success = candidate
                if self.first_success_return is None:
                    self.first_success_return = candidate.episode_return
                else:
                    self.improvement_count += 1
            self._completion_events.append(
                CompletionEvent(
                    runs=candidate.runs,
                    episode_return=candidate.episode_return,
                    progress=candidate.progress,
                    improved=improved,
                )
            )
            return
        previous = self._best_incomplete
        if previous is None or (
            candidate.progress,
            candidate.episode_return,
            -candidate.step_count,
            candidate.runs,
        ) > (
            previous.progress,
            previous.episode_return,
            -previous.step_count,
            previous.runs,
        ):
            self._best_incomplete = candidate

    def observe(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        cell_keys: Sequence[Hashable],
        records_by_lane: Mapping[int, Any] | None = None,
        *,
        progresses: Sequence[float] | None = None,
    ) -> GoExploreObservation:
        rewards_array = np.asarray(rewards, dtype=np.float64)
        dones_array = np.asarray(dones, dtype=np.bool_)
        if rewards_array.shape != (self.n_envs,) or dones_array.shape != (self.n_envs,):
            raise ValueError(
                "Go-Explore rewards and dones must contain one value per environment"
            )
        if len(cell_keys) != self.n_envs:
            raise ValueError(
                "Go-Explore cell keys must contain one value per environment"
            )
        progress_array = (
            np.zeros(self.n_envs, dtype=np.float64)
            if progresses is None
            else np.asarray(progresses, dtype=np.float64)
        )
        if progress_array.shape != (self.n_envs,):
            raise ValueError(
                "Go-Explore progresses must contain one value per environment"
            )
        records_by_lane = records_by_lane or {}
        self.global_step += self.n_envs
        self._pending = {}
        observed_counts: dict[Hashable, int] = {}
        restart_mask = np.zeros(self.n_envs, dtype=np.bool_)

        for lane, state in enumerate(self._lanes):
            state.episode_return += float(rewards_array[lane])
            state.progress = max(state.progress, float(progress_array[lane]))
            completed, record_progress = self._record_facts(records_by_lane.get(lane))
            progress = max(state.progress, record_progress)
            self._consider_lane_candidate(
                state,
                progress=progress,
                completed=completed,
            )
            if dones_array[lane]:
                self.completed_episodes += 1
                self.successful_episodes += int(completed)
                restart_mask[lane] = True
                continue

            key = cell_keys[lane]
            observed_counts[key] = observed_counts.get(key, 0) + 1
            cell = self._archive.get(key)
            current_pending = self._pending.get(key)
            comparison: GoExploreCell | _PendingCell | None = (
                current_pending if current_pending is not None else cell
            )
            if self._cell_candidate_better(
                episode_return=state.episode_return,
                step_count=state.program_steps,
                cell=comparison,
            ):
                self._pending[key] = _PendingCell(
                    key=key,
                    lane=lane,
                    runs=tuple(state.runs),
                    episode_return=state.episode_return,
                    progress=progress,
                    step_count=state.program_steps,
                    visits=0,
                )
            if state.steps_since_restart >= self.explore_steps:
                restart_mask[lane] = True

        for key, count in observed_counts.items():
            cell = self._archive.get(key)
            if cell is not None:
                cell.visits += count
                self._archive_visit_count += count
            pending = self._pending.get(key)
            if pending is not None:
                pending.visits = count

        recent_visits = sum(observed_counts.values())
        recent_new_cells = sum(
            pending.key not in self._archive for pending in self._pending.values()
        )
        if recent_visits:
            self._recent_cell_batches.append((recent_visits, recent_new_cells))
            self._recent_cell_visits += recent_visits
            self._recent_new_cells += recent_new_cells
            while (
                len(self._recent_cell_batches) > 1
                and self._recent_cell_visits - self._recent_cell_batches[0][0]
                >= RECENT_CELL_VISIT_WINDOW
            ):
                visits, new_cells = self._recent_cell_batches.popleft()
                self._recent_cell_visits -= visits
                self._recent_new_cells -= new_cells

        archive_mask = np.zeros(self.n_envs, dtype=np.bool_)
        for pending in self._pending.values():
            archive_mask[pending.lane] = True
        return GoExploreObservation(
            archive_mask=archive_mask,
            restart_mask=restart_mask,
        )

    def commit_archive(self, snapshots: Sequence[Any | None]) -> None:
        """Attach exact snapshots to the cell updates selected by ``observe``."""
        if len(snapshots) != self.n_envs:
            raise ValueError("Go-Explore archive snapshots require one value per lane")
        for pending in self._pending.values():
            snapshot = snapshots[pending.lane]
            if snapshot is None:
                raise ValueError(
                    "Go-Explore selected archive snapshots cannot be empty"
                )
            existing = self._archive.get(pending.key)
            if existing is None:
                dict_size_before = sys.getsizeof(self._archive)
                cell = GoExploreCell(
                    key=pending.key,
                    snapshot=snapshot,
                    runs=pending.runs,
                    episode_return=pending.episode_return,
                    progress=pending.progress,
                    program_steps=pending.step_count,
                    visits=pending.visits,
                )
                self._archive[pending.key] = cell
                self._archive_memory_bytes += (
                    sys.getsizeof(self._archive)
                    - dict_size_before
                    + _deep_sizeof(cell)
                )
                self._archive_visit_count += pending.visits
                continue
            previous_size = _deep_sizeof(existing)
            existing.snapshot = snapshot
            existing.runs = pending.runs
            existing.episode_return = pending.episode_return
            existing.progress = pending.progress
            existing.program_steps = pending.step_count
            existing.updates += 1
            self._archive_update_count += 1
            self._archive_memory_bytes += _deep_sizeof(existing) - previous_size
        self._pending = {}

    def _select_cell(self, lane: int) -> GoExploreCell:
        cells = tuple(self._archive.values())
        weights = np.asarray(
            [
                1.0 / math.sqrt(1.0 + cell.selections)
                + 1.0 / math.sqrt(1.0 + cell.visits)
                for cell in cells
            ],
            dtype=np.float64,
        )
        probabilities = weights / weights.sum()
        index = int(self._rngs[lane].choice(len(cells), p=probabilities))
        return cells[index]

    def restart(self, mask: Sequence[bool]) -> tuple[Any | None, ...]:
        """Choose archived cells and return lane-aligned restore snapshots."""
        restart_mask = np.asarray(mask, dtype=np.bool_)
        if restart_mask.shape != (self.n_envs,):
            raise ValueError("Go-Explore restart mask must contain one value per lane")
        snapshots: list[Any | None] = [None] * self.n_envs
        for lane in np.flatnonzero(restart_mask):
            index = int(lane)
            cell = self._select_cell(index)
            cell.selections += 1
            self._archive_selection_count += 1
            snapshots[index] = cell.snapshot
            self._lanes[index] = _GoExploreLaneState(
                runs=list(cell.runs),
                episode_return=cell.episode_return,
                progress=cell.progress,
                program_steps=cell.step_count,
            )
        return tuple(snapshots)

    def take_completion_events(self) -> tuple[CompletionEvent, ...]:
        events = tuple(self._completion_events)
        self._completion_events.clear()
        return events

    def best_candidate(self) -> GoExploreCandidate | None:
        return self._best_success or self._best_incomplete

    def policy(self) -> JerkPolicy:
        candidate = self.best_candidate()
        return JerkPolicy(
            action_names=self.action_names,
            action_runs=() if candidate is None else candidate.runs,
            fallback_action=self.fallback_action,
            timesteps=self.global_step,
            episodes=self.completed_episodes,
            best_reward=0.0 if candidate is None else candidate.episode_return,
            metadata={
                "search_algorithm": "go-explore",
                "go_explore_phase": "trajectory_finding",
                "robustification": False,
                "archive_count": self.archive_count,
                "explore_steps": self.explore_steps,
                "improvement_count": self.improvement_count,
                "terminate_on_life_loss": True,
                "terminate_on_level_change": False,
            },
        )
