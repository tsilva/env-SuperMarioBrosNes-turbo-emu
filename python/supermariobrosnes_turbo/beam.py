from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .jerk import (
    ActionRun,
    JerkPolicy,
    RetainedProgram,
    canonicalize_runs,
    truncate_runs,
)


@dataclass
class _BeamLaneState:
    mode: str = "explore"
    runs: list[ActionRun] = field(default_factory=list)
    step_count: int = 0
    episode_return: float = 0.0
    best_return: float = float("-inf")
    best_steps: int = 0
    best_progress: float = 0.0
    parent: RetainedProgram | None = None
    replay_limit_runs: int = 0
    replay_run_index: int = 0
    replay_run_remaining: int = 0
    branch: ActionRun | None = None
    branch_remaining: int = 0
    exploration_action: int = 0
    exploration_remaining: int = 0


class BeamSearch:
    """Steady-state beam search over open-loop JERK action-run programs.

    The beam contains the best complete episode programs seen so far. Each
    expansion replays a parent without its final ``mutation_runs`` runs, tries
    one action/duration branch, and samples a suffix until the episode ends.
    Parent/branch pairs are visited round-robin so every beam member receives
    equal expansion pressure instead of return-weighted sampling.
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
    ) -> None:
        if n_envs < 1:
            raise ValueError("beam search requires at least one environment")
        self.n_envs = int(n_envs)
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
        if self.beam_width < 1:
            raise ValueError("beam width must be positive")
        if self.refresh_episodes < 1:
            raise ValueError("beam refresh episode count must be positive")
        if self.protected_prefix_runs < 0:
            raise ValueError("beam protected prefix must be non-negative")
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
        self._next_refresh_episode = self.refresh_episodes
        self._expansion_cursor = 0
        self._beam: dict[tuple[ActionRun, ...], RetainedProgram] = {}
        self._pending: dict[tuple[ActionRun, ...], RetainedProgram] = {}
        self._parents: tuple[RetainedProgram, ...] = ()
        self._lanes = [_BeamLaneState() for _ in range(self.n_envs)]
        self._rngs = [
            np.random.default_rng(np.random.SeedSequence([seed, lane, 0x4245414D]))
            for lane in range(self.n_envs)
        ]
        self._branches = tuple(
            ActionRun(action, duration)
            for duration in self.branch_durations
            for action in range(len(self.action_names))
        )

    @property
    def beam_count(self) -> int:
        return len(self._beam)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def retained_count(self) -> int:
        return len(set(self._beam) | set(self._pending))

    @property
    def locked_count(self) -> int:
        candidates = {**self._beam, **self._pending}
        return sum(candidate.completed for candidate in candidates.values())

    @property
    def incomplete_retained_count(self) -> int:
        return self.retained_count - self.locked_count

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
        rng = self._rngs[lane]
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

    def _start_lane(self, lane: int) -> None:
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

    def _upsert_candidate(
        self,
        runs: Sequence[ActionRun],
        *,
        score_return: float,
        completed: bool,
        progress: float,
    ) -> None:
        canonical = canonicalize_runs(runs)
        candidate = self._pending.get(canonical) or self._beam.get(canonical)
        if candidate is None:
            candidate = RetainedProgram(runs=canonical)
            self._pending[canonical] = candidate
        candidate.observe(score_return, completed=completed, progress=progress)

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

    def _refresh_beam(self) -> None:
        candidates = {**self._beam, **self._pending}
        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (candidate.rank, candidate.runs),
            reverse=True,
        )
        retained = ordered[: self.beam_width]
        self._beam = {candidate.runs: candidate for candidate in retained}
        self._pending = {}
        self._parents = tuple(retained)
        self.generation += 1

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
            while self.completed_episodes >= self._next_refresh_episode:
                self._refresh_beam()
                self._next_refresh_episode += self.refresh_episodes
            self._start_lane(lane)

    def best_candidate(self) -> RetainedProgram | None:
        candidates = list(self._beam.values()) + list(self._pending.values())
        for state in self._lanes:
            if state.best_steps > 0:
                candidates.append(
                    RetainedProgram(
                        runs=truncate_runs(state.runs, state.best_steps),
                        return_sum=state.best_return,
                        return_count=1,
                        progress=state.best_progress,
                    )
                )
        return max(candidates, key=lambda candidate: candidate.rank, default=None)

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
            },
        )
