from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .jerk import (
    ActionRun,
    JerkPolicy,
    canonicalize_runs,
    run_step_count,
    truncate_runs,
)


@dataclass
class BeamCandidate:
    """One canonical action-run program with outcome-specific scores."""

    runs: tuple[ActionRun, ...]
    incomplete_return: float = float("-inf")
    completed_return: float = float("-inf")
    progress: float = 0.0

    def __post_init__(self) -> None:
        self.runs = canonicalize_runs(self.runs)

    @property
    def step_count(self) -> int:
        return run_step_count(self.runs)

    @property
    def completed(self) -> bool:
        return math.isfinite(self.completed_return)

    @property
    def mean_return(self) -> float:
        return self.completed_return if self.completed else self.incomplete_return

    @property
    def rank(self) -> tuple[float, float]:
        return (float(self.completed), self.mean_return)

    def observe(self, value: float, *, completed: bool, progress: float) -> None:
        if completed:
            self.completed_return = max(self.completed_return, float(value))
        else:
            self.incomplete_return = max(self.incomplete_return, float(value))
        self.progress = max(self.progress, float(progress))


@dataclass(frozen=True)
class CompletionEvent:
    runs: tuple[ActionRun, ...]
    episode_return: float
    progress: float
    improved: bool


@dataclass(frozen=True)
class _ExpansionJob:
    parent: BeamCandidate
    replay_limit_runs: int
    branch: ActionRun
    generation: int
    parent_index: int
    branch_index: int
    sample_index: int = 0
    required: bool = True


@dataclass
class _BeamLaneState:
    mode: str = "explore"
    runs: list[ActionRun] = field(default_factory=list)
    step_count: int = 0
    episode_return: float = 0.0
    best_return: float = float("-inf")
    best_steps: int = 0
    best_progress: float = 0.0
    parent: BeamCandidate | None = None
    replay_limit_runs: int = 0
    replay_run_index: int = 0
    replay_run_remaining: int = 0
    branch: ActionRun | None = None
    branch_remaining: int = 0
    exploration_action: int = 0
    exploration_remaining: int = 0
    job_generation: int = -1
    job_required: bool = False
    rng: np.random.Generator | None = None


class BeamSearch:
    """Anytime beam search over open-loop action-run programs.

    Discovery retains the original steady-state behavior. An explicit
    continued-budget run switches to fair coverage generations after the first
    completion: every frozen parent receives its full branch neighborhood at a
    progressively deeper cut before the beam can be pruned again.
    """

    def __init__(
        self,
        *,
        n_envs: int,
        seed: int,
        action_names: Sequence[str],
        fallback_action: str,
        beam_width: int,
        refresh_episodes: int,
        protected_prefix_runs: int,
        mutation_runs: int,
        branch_durations: Sequence[int],
        run_duration_mean: float,
        run_duration_max: int,
        improve_after_completion: bool = False,
        improvement_protected_prefix_runs: int = 0,
    ) -> None:
        if n_envs < 1:
            raise ValueError("beam search requires at least one environment")
        self.n_envs = int(n_envs)
        self.seed = int(seed)
        self.action_names = tuple(str(name) for name in action_names)
        indices = {name: index for index, name in enumerate(self.action_names)}
        if not self.action_names:
            raise ValueError("beam search requires at least one action name")
        if fallback_action not in indices:
            raise ValueError(
                "beam fallback action is absent from the task action set: "
                f"{fallback_action}"
            )
        self.fallback_action = indices[fallback_action]
        self.beam_width = int(beam_width)
        self.refresh_episodes = int(refresh_episodes)
        self.protected_prefix_runs = int(protected_prefix_runs)
        self.mutation_runs = int(mutation_runs)
        self.branch_durations = tuple(int(value) for value in branch_durations)
        self.run_duration_mean = float(run_duration_mean)
        self.run_duration_max = int(run_duration_max)
        self.improve_after_completion = bool(improve_after_completion)
        self.improvement_protected_prefix_runs = int(
            improvement_protected_prefix_runs
        )
        if self.beam_width < 1:
            raise ValueError("beam width must be positive")
        if self.refresh_episodes < 1:
            raise ValueError("beam refresh episode count must be positive")
        if self.protected_prefix_runs < 0:
            raise ValueError("beam protected prefix must be non-negative")
        if self.improvement_protected_prefix_runs < 0:
            raise ValueError(
                "beam improvement protected prefix must be non-negative"
            )
        if self.mutation_runs < 1:
            raise ValueError("beam mutation runs must be positive")
        if not self.branch_durations or any(
            duration < 1 for duration in self.branch_durations
        ):
            raise ValueError("beam branch durations must be positive")
        if len(set(self.branch_durations)) != len(self.branch_durations):
            raise ValueError("beam branch durations must be unique")
        if self.run_duration_mean < 1.0:
            raise ValueError("beam run duration mean must be at least one")
        if self.run_duration_max < 1:
            raise ValueError("beam run duration maximum must be positive")

        self.global_step = 0
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.generation = 0
        self.improvement_count = 0
        self.first_success_return: float | None = None
        self._next_refresh_episode = self.refresh_episodes
        self._expansion_cursor = 0
        self._beam: dict[tuple[ActionRun, ...], BeamCandidate] = {}
        self._pending: dict[tuple[ActionRun, ...], BeamCandidate] = {}
        self._parents: tuple[BeamCandidate, ...] = ()
        self._best_success: BeamCandidate | None = None
        self._completion_events: list[CompletionEvent] = []
        self._lanes = [_BeamLaneState() for _ in range(self.n_envs)]
        self._rngs = [
            np.random.default_rng(
                np.random.SeedSequence([self.seed, lane, 0x4245414D])
            )
            for lane in range(self.n_envs)
        ]
        self._branches = tuple(
            ActionRun(action, duration)
            for duration in self.branch_durations
            for action in range(len(self.action_names))
        )
        self._improvement_mode = False
        self._cut_depth = 0
        self._coverage_queue: deque[_ExpansionJob] = deque()
        self._coverage_templates: tuple[_ExpansionJob, ...] = ()
        self._coverage_total = 0
        self._coverage_completed = 0
        self._overflow_cursor = 0

    @property
    def beam_count(self) -> int:
        return len(self._beam)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_candidate_count(self) -> int:
        return len(set(self._beam) | set(self._pending))

    @property
    def retained_count(self) -> int:
        return self.active_candidate_count + self.successful_episodes

    @property
    def locked_count(self) -> int:
        return self.successful_episodes

    @property
    def incomplete_retained_count(self) -> int:
        candidates = {**self._beam, **self._pending}
        return sum(not candidate.completed for candidate in candidates.values())

    @property
    def improvement_mode(self) -> bool:
        return self._improvement_mode

    @property
    def cut_depth(self) -> int:
        return self._cut_depth

    @property
    def coverage_total(self) -> int:
        return self._coverage_total

    @property
    def coverage_completed(self) -> int:
        return self._coverage_completed

    @property
    def best_success_return(self) -> float | None:
        if self._best_success is None:
            return None
        return self._best_success.completed_return

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
        completed = bool(
            getattr(record, "completed", False)
            or getattr(record, "outcome", None) == "success"
            or metrics.get("level_complete", False)
        )
        progress = float(
            getattr(
                record,
                "progress",
                metrics.get("max_x_pos", metrics.get("global_max_x_pos", 0.0)),
            )
            or 0.0
        )
        return completed, progress

    @staticmethod
    def _append_action(state: _BeamLaneState, action: int) -> None:
        if state.runs and state.runs[-1].action == action:
            previous = state.runs[-1]
            state.runs[-1] = ActionRun(action, previous.duration + 1)
        else:
            state.runs.append(ActionRun(action, 1))
        state.step_count += 1

    def _sample_exploration_run(
        self, lane: int, state: _BeamLaneState
    ) -> None:
        rng = state.rng if state.rng is not None else self._rngs[lane]
        action_count = len(self.action_names)
        previous = state.runs[-1].action if state.runs else None
        if previous is None or action_count == 1:
            action = int(rng.integers(0, action_count))
        else:
            sampled = int(rng.integers(0, action_count - 1))
            action = sampled + int(sampled >= previous)
        probability = 1.0 / self.run_duration_mean
        duration = min(int(rng.geometric(probability)), self.run_duration_max)
        state.exploration_action = action
        state.exploration_remaining = duration

    def _job_rng(self, job: _ExpansionJob) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.seed,
                    job.generation,
                    job.parent_index,
                    self._cut_depth,
                    job.branch_index,
                    job.sample_index,
                    0x4245414D,
                ]
            )
        )

    def _state_for_job(self, job: _ExpansionJob) -> _BeamLaneState:
        return _BeamLaneState(
            mode="replay" if job.replay_limit_runs else "branch",
            parent=job.parent,
            replay_limit_runs=job.replay_limit_runs,
            branch=job.branch,
            branch_remaining=job.branch.duration,
            job_generation=job.generation,
            job_required=job.required,
            rng=self._job_rng(job),
        )

    def _overflow_job(self) -> _ExpansionJob:
        template = self._coverage_templates[
            self._overflow_cursor % len(self._coverage_templates)
        ]
        sample_index = 1 + self._overflow_cursor // len(self._coverage_templates)
        self._overflow_cursor += 1
        return _ExpansionJob(
            parent=template.parent,
            replay_limit_runs=template.replay_limit_runs,
            branch=template.branch,
            generation=template.generation,
            parent_index=template.parent_index,
            branch_index=template.branch_index,
            sample_index=sample_index,
            required=False,
        )

    def _start_lane(self, lane: int) -> None:
        if self._improvement_mode and self._parents:
            job = (
                self._coverage_queue.popleft()
                if self._coverage_queue
                else self._overflow_job()
            )
            self._lanes[lane] = self._state_for_job(job)
            return

        state = _BeamLaneState()
        if self._parents:
            expansion = self._expansion_cursor
            parent = self._parents[expansion % len(self._parents)]
            branch_index = (expansion // len(self._parents)) % len(self._branches)
            self._expansion_cursor += 1
            mutable_runs = max(
                len(parent.runs) - self.protected_prefix_runs,
                0,
            )
            removed_runs = min(self.mutation_runs, mutable_runs)
            state.parent = parent
            state.replay_limit_runs = len(parent.runs) - removed_runs
            state.branch = self._branches[branch_index]
            state.branch_remaining = state.branch.duration
            state.mode = "replay" if state.replay_limit_runs else "branch"
        self._lanes[lane] = state

    @staticmethod
    def _next_replay_action(state: _BeamLaneState) -> int | None:
        parent = state.parent
        if parent is None or state.replay_run_index >= state.replay_limit_runs:
            return None
        if state.replay_run_remaining == 0:
            run = parent.runs[state.replay_run_index]
            state.replay_run_remaining = run.duration
        run = parent.runs[state.replay_run_index]
        state.replay_run_remaining -= 1
        if state.replay_run_remaining == 0:
            state.replay_run_index += 1
        return run.action

    def next_actions(self) -> np.ndarray:
        actions = np.empty(self.n_envs, dtype=np.int64)
        for lane, state in enumerate(self._lanes):
            action: int | None = None
            if state.mode == "replay":
                action = self._next_replay_action(state)
                if action is None:
                    state.mode = "branch"
            if state.mode == "branch":
                branch = state.branch
                if branch is None or state.branch_remaining == 0:
                    state.mode = "explore"
                else:
                    action = branch.action
                    state.branch_remaining -= 1
                    if state.branch_remaining == 0:
                        state.mode = "explore"
            if action is None and state.mode == "explore":
                if state.exploration_remaining == 0:
                    self._sample_exploration_run(lane, state)
                action = state.exploration_action
                state.exploration_remaining -= 1
            assert action is not None
            self._append_action(state, int(action))
            actions[lane] = action
        return actions

    def _candidate_for(self, canonical: tuple[ActionRun, ...]) -> BeamCandidate:
        candidate = self._pending.get(canonical) or self._beam.get(canonical)
        if candidate is None:
            candidate = BeamCandidate(runs=canonical)
            self._pending[canonical] = candidate
        return candidate

    def _upsert_candidate(
        self,
        runs: Sequence[ActionRun],
        *,
        score_return: float,
        completed: bool,
        progress: float,
    ) -> BeamCandidate:
        canonical = canonicalize_runs(runs)
        candidate = self._candidate_for(canonical)
        previous_best = self.best_success_return
        candidate.observe(score_return, completed=completed, progress=progress)
        if completed:
            improved = previous_best is None or score_return > previous_best
            if improved:
                self._best_success = candidate
                if self.first_success_return is None:
                    self.first_success_return = float(score_return)
                else:
                    self.improvement_count += 1
            self._completion_events.append(
                CompletionEvent(
                    runs=canonical,
                    episode_return=float(score_return),
                    progress=float(progress),
                    improved=improved,
                )
            )
        return candidate

    def _retain_lane(self, state: _BeamLaneState, record: Any | None) -> None:
        completed, progress = self._record_facts(record)
        if completed:
            runs = tuple(state.runs)
            score_return = state.episode_return
        else:
            runs = truncate_runs(state.runs, state.best_steps)
            score_return = state.best_return
            progress = state.best_progress
        if not runs or not math.isfinite(score_return):
            return
        self._upsert_candidate(
            runs,
            score_return=score_return,
            completed=completed,
            progress=progress,
        )

    @staticmethod
    def _candidate_sort_key(candidate: BeamCandidate) -> tuple[Any, ...]:
        return (candidate.mean_return, candidate.runs)

    def _select_retained(self) -> tuple[BeamCandidate, ...]:
        candidates = {**self._beam, **self._pending}
        if self._best_success is not None:
            candidates[self._best_success.runs] = self._best_success
        completed = sorted(
            (candidate for candidate in candidates.values() if candidate.completed),
            key=self._candidate_sort_key,
            reverse=True,
        )
        incomplete = sorted(
            (candidate for candidate in candidates.values() if not candidate.completed),
            key=self._candidate_sort_key,
            reverse=True,
        )
        if not completed:
            return tuple(incomplete[: self.beam_width])

        incomplete_target = min(len(incomplete), self.beam_width // 2)
        completed_target = min(len(completed), self.beam_width - incomplete_target)
        retained = completed[:completed_target] + incomplete[:incomplete_target]
        if len(retained) < self.beam_width:
            used = {candidate.runs for candidate in retained}
            remainder = sorted(
                (
                    candidate
                    for candidate in (*completed, *incomplete)
                    if candidate.runs not in used
                ),
                key=lambda candidate: (candidate.rank, candidate.runs),
                reverse=True,
            )
            retained.extend(remainder[: self.beam_width - len(retained)])
        return tuple(retained)

    def _refresh_beam(self) -> None:
        retained = self._select_retained()
        self._beam = {candidate.runs: candidate for candidate in retained}
        self._pending = {}
        self._parents = retained
        self.generation += 1
        if self._improvement_mode:
            self._prepare_coverage()

    def _branch_neighborhood(
        self, parent: BeamCandidate, replay_limit_runs: int
    ) -> tuple[ActionRun, ...]:
        durations = set(self.branch_durations)
        if replay_limit_runs < len(parent.runs):
            replaced_duration = parent.runs[replay_limit_runs].duration
            durations.update(
                min(max(replaced_duration + delta, 1), self.run_duration_max)
                for delta in (-2, -1, 0, 1, 2)
            )
        return tuple(
            ActionRun(action, duration)
            for duration in sorted(durations)
            for action in range(len(self.action_names))
        )

    def _prepare_coverage(self) -> None:
        jobs: list[_ExpansionJob] = []
        for parent_index, parent in enumerate(self._parents):
            mutable_runs = max(
                len(parent.runs) - self.improvement_protected_prefix_runs,
                0,
            )
            removed_runs = min(self._cut_depth, mutable_runs)
            replay_limit_runs = len(parent.runs) - removed_runs
            branches = self._branch_neighborhood(parent, replay_limit_runs)
            for branch_index, branch in enumerate(branches):
                jobs.append(
                    _ExpansionJob(
                        parent=parent,
                        replay_limit_runs=replay_limit_runs,
                        branch=branch,
                        generation=self.generation,
                        parent_index=parent_index,
                        branch_index=branch_index,
                    )
                )
        self._coverage_templates = tuple(jobs)
        self._coverage_queue = deque(jobs)
        self._coverage_total = len(jobs)
        self._coverage_completed = 0
        self._overflow_cursor = 0

    def _enable_improvement(self) -> None:
        self._improvement_mode = True
        self._cut_depth = 1
        self._refresh_beam()

    def _advance_improvement_generation(self) -> None:
        maximum_mutable = max(
            (
                max(
                    len(parent.runs) - self.improvement_protected_prefix_runs,
                    0,
                )
                for parent in self._parents
            ),
            default=1,
        )
        self._cut_depth = (
            1 if self._cut_depth >= max(maximum_mutable, 1) else self._cut_depth + 1
        )
        self._refresh_beam()

    def take_completion_events(self) -> tuple[CompletionEvent, ...]:
        events = tuple(self._completion_events)
        self._completion_events.clear()
        return events

    def observe(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        records_by_lane: Mapping[int, Any] | None = None,
        *,
        progresses: Sequence[float] | None = None,
    ) -> None:
        rewards_array = np.asarray(rewards, dtype=np.float64)
        dones_array = np.asarray(dones, dtype=bool)
        if rewards_array.shape != (self.n_envs,) or dones_array.shape != (
            self.n_envs,
        ):
            raise ValueError(
                "beam rewards and dones must contain one value per environment"
            )
        progress_array = (
            np.zeros(self.n_envs, dtype=np.float64)
            if progresses is None
            else np.asarray(progresses, dtype=np.float64)
        )
        if progress_array.shape != (self.n_envs,):
            raise ValueError("beam progresses must contain one value per environment")
        records_by_lane = records_by_lane or {}
        self.global_step += self.n_envs
        done_lanes: list[int] = []
        for lane, state in enumerate(self._lanes):
            reward = float(rewards_array[lane])
            state.episode_return += reward
            if state.episode_return > state.best_return:
                state.best_return = state.episode_return
                state.best_steps = state.step_count
                state.best_progress = float(progress_array[lane])
            if not dones_array[lane]:
                continue
            record = records_by_lane.get(lane)
            completed, _progress = self._record_facts(record)
            self._retain_lane(state, record)
            self.completed_episodes += 1
            self.successful_episodes += int(completed)
            if (
                self._improvement_mode
                and state.job_required
                and state.job_generation == self.generation
            ):
                self._coverage_completed += 1
            done_lanes.append(lane)

        if (
            self.improve_after_completion
            and self._best_success is not None
            and not self._improvement_mode
        ):
            self._enable_improvement()
        elif (
            self._improvement_mode
            and self._coverage_total > 0
            and self._coverage_completed >= self._coverage_total
        ):
            self._advance_improvement_generation()
        elif not self._improvement_mode:
            while self.completed_episodes >= self._next_refresh_episode:
                self._refresh_beam()
                self._next_refresh_episode += self.refresh_episodes

        for lane in done_lanes:
            self._start_lane(lane)

    def best_candidate(self) -> BeamCandidate | None:
        if self._best_success is not None:
            return self._best_success
        candidates = list(self._beam.values()) + list(self._pending.values())
        for state in self._lanes:
            if state.best_steps > 0:
                candidates.append(
                    BeamCandidate(
                        runs=truncate_runs(state.runs, state.best_steps),
                        incomplete_return=state.best_return,
                        progress=state.best_progress,
                    )
                )
        return max(
            candidates,
            key=lambda candidate: (candidate.incomplete_return, candidate.runs),
            default=None,
        )

    def policy(self) -> JerkPolicy:
        candidate = self.best_candidate()
        return JerkPolicy(
            action_names=self.action_names,
            action_runs=() if candidate is None else candidate.runs,
            fallback_action=self.fallback_action,
            timesteps=self.global_step,
            episodes=self.completed_episodes,
            best_reward=0.0 if candidate is None else candidate.mean_return,
            metadata={
                "search_algorithm": "beam",
                "beam_width": self.beam_width,
                "generation": self.generation,
                "improvement_mode": self._improvement_mode,
                "improvement_count": self.improvement_count,
                "cut_depth": self._cut_depth,
                "terminate_on_life_loss": True,
                "terminate_on_level_change": False,
            },
        )
