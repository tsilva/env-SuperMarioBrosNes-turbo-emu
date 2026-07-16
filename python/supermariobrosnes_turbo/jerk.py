from __future__ import annotations

import json
import math
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .env import ACTION_SETS


JERK_POLICY_SCHEMA_VERSION = 2
JERK_POLICY_MEMBER = "jerk_policy.json"
LEVEL_NAME_PATTERN = re.compile(r"^Level[1-9][0-9]*-[1-9][0-9]*$")


def normalize_level_name(level: str) -> str:
    name = str(level).strip()
    if LEVEL_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"invalid level name {level!r}; expected a name such as 'Level1-1'"
        )
    return name


def run_directory_for_level(level: str, *, runs_root: str | Path = "runs") -> Path:
    name = normalize_level_name(level)
    return Path(runs_root) / f"{name}-jerk"


def policy_path_for_level(level: str, *, runs_root: str | Path = "runs") -> Path:
    name = normalize_level_name(level)
    return run_directory_for_level(name, runs_root=runs_root) / f"{name}.zip"


@dataclass(frozen=True, order=True)
class ActionRun:
    """One action held for a positive number of environment steps."""

    action: int
    duration: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", int(self.action))
        object.__setattr__(self, "duration", int(self.duration))
        if self.duration < 1:
            raise ValueError("JERK action-run durations must be positive")


def canonicalize_runs(runs: Sequence[ActionRun]) -> tuple[ActionRun, ...]:
    """Merge adjacent equal actions into the unique canonical run program."""
    canonical: list[ActionRun] = []
    for raw_run in runs:
        run = ActionRun(raw_run.action, raw_run.duration)
        if canonical and canonical[-1].action == run.action:
            previous = canonical[-1]
            canonical[-1] = ActionRun(previous.action, previous.duration + run.duration)
        else:
            canonical.append(run)
    return tuple(canonical)


def run_step_count(runs: Sequence[ActionRun]) -> int:
    return sum(run.duration for run in runs)


def truncate_runs(
    runs: Sequence[ActionRun], step_limit: int
) -> tuple[ActionRun, ...]:
    """Return the canonical prefix containing at most ``step_limit`` steps."""
    remaining = max(int(step_limit), 0)
    prefix: list[ActionRun] = []
    for run in runs:
        if remaining <= 0:
            break
        duration = min(run.duration, remaining)
        prefix.append(ActionRun(run.action, duration))
        remaining -= duration
    return canonicalize_runs(prefix)


@dataclass
class RetainedProgram:
    runs: tuple[ActionRun, ...]
    return_sum: float = 0.0
    return_count: int = 0
    completed: bool = False
    progress: float = 0.0

    def __post_init__(self) -> None:
        self.runs = canonicalize_runs(self.runs)

    @property
    def step_count(self) -> int:
        return run_step_count(self.runs)

    @property
    def mean_return(self) -> float:
        return (
            self.return_sum / self.return_count if self.return_count else float("-inf")
        )

    def observe(self, value: float, *, completed: bool, progress: float) -> None:
        self.return_sum += float(value)
        self.return_count += 1
        self.completed |= bool(completed)
        self.progress = max(self.progress, float(progress))

    @property
    def rank(self) -> tuple[float, ...]:
        if self.completed:
            return (
                1.0,
                -float(self.step_count),
                -float(len(self.runs)),
                self.mean_return,
                float(self.progress),
            )
        return (
            0.0,
            float(self.progress),
            self.mean_return,
            -float(self.step_count),
            -float(len(self.runs)),
        )


@dataclass
class _LaneState:
    mode: str = "explore"
    runs: list[ActionRun] = field(default_factory=list)
    step_count: int = 0
    episode_return: float = 0.0
    best_return: float = float("-inf")
    best_steps: int = 0
    archive_candidate: RetainedProgram | None = None
    replay_limit_runs: int = 0
    replay_run_index: int = 0
    replay_run_remaining: int = 0
    exploration_action: int = 0
    exploration_remaining: int = 0


class JerkSearch:
    """Vectorized Just Enough Retained Knowledge action-run search."""

    def __init__(
        self,
        *,
        n_envs: int,
        seed: int,
        total_timesteps: int,
        action_names: Sequence[str],
        fallback_action: str,
        archive_replay_probability_initial: float,
        archive_replay_probability_max: float,
        protected_prefix_runs: int,
        max_prefix_shorten_runs: int,
        deep_mutation_probability: float,
        run_duration_mean: float,
        run_duration_max: int,
        retained_limit: int,
    ) -> None:
        if n_envs < 1:
            raise ValueError("JERK requires at least one environment")
        self.n_envs = int(n_envs)
        self.total_timesteps = max(int(total_timesteps), 1)
        self.action_names = tuple(str(name) for name in action_names)
        indices = {name: index for index, name in enumerate(self.action_names)}
        if not self.action_names:
            raise ValueError("JERK requires at least one action name")
        if fallback_action not in indices:
            raise ValueError(
                "JERK fallback action is absent from the task action set: "
                f"{fallback_action}"
            )
        self.fallback_action = indices[fallback_action]
        self.archive_replay_probability_initial = float(
            archive_replay_probability_initial
        )
        self.archive_replay_probability_max = float(archive_replay_probability_max)
        if not (
            0.0
            <= self.archive_replay_probability_initial
            <= self.archive_replay_probability_max
            <= 1.0
        ):
            raise ValueError(
                "JERK probabilities must satisfy 0 <= archive_replay_probability_initial "
                "<= archive_replay_probability_max <= 1"
            )
        self.protected_prefix_runs = int(protected_prefix_runs)
        self.max_prefix_shorten_runs = int(max_prefix_shorten_runs)
        self.deep_mutation_probability = float(deep_mutation_probability)
        self.run_duration_mean = float(run_duration_mean)
        self.run_duration_max = int(run_duration_max)
        if self.protected_prefix_runs < 0:
            raise ValueError("JERK protected_prefix_runs must be non-negative")
        if self.max_prefix_shorten_runs < 1:
            raise ValueError("JERK max_prefix_shorten_runs must be positive")
        if not 0.0 <= self.deep_mutation_probability <= 1.0:
            raise ValueError("JERK deep_mutation_probability must be in [0, 1]")
        if self.run_duration_mean < 1.0:
            raise ValueError("JERK run_duration_mean must be at least one")
        if self.run_duration_max < 1:
            raise ValueError("JERK run_duration_max must be positive")
        self.retained_limit = int(retained_limit)
        if self.retained_limit < 1:
            raise ValueError("JERK retained_limit must be positive")
        self.global_step = 0
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.archive_replay_episodes = 0
        self.archive_selected_prefix_return_sum = 0.0
        self._retained: dict[tuple[ActionRun, ...], RetainedProgram] = {}
        self._lanes = [_LaneState() for _ in range(self.n_envs)]
        self._rngs = [
            np.random.default_rng(np.random.SeedSequence([seed, lane, 0x4A45524B]))
            for lane in range(self.n_envs)
        ]

    @property
    def archive_replay_probability(self) -> float:
        return min(
            self.archive_replay_probability_max,
            self.archive_replay_probability_initial
            + self.global_step / self.total_timesteps,
        )

    @property
    def archive_selected_prefix_return_mean(self) -> float:
        if not self.archive_replay_episodes:
            return 0.0
        return self.archive_selected_prefix_return_sum / self.archive_replay_episodes

    @property
    def retained_count(self) -> int:
        return len(self._retained)

    @property
    def locked_count(self) -> int:
        return sum(candidate.completed for candidate in self._retained.values())

    @property
    def incomplete_retained_count(self) -> int:
        return self.retained_count - self.locked_count

    def _retained_distribution(self) -> tuple[list[RetainedProgram], np.ndarray]:
        candidates = sorted(
            self._retained.values(), key=lambda candidate: candidate.runs
        )
        returns = np.asarray(
            [candidate.mean_return for candidate in candidates], dtype=np.float64
        )
        weights = returns - float(np.min(returns)) + 1e-12
        probabilities = weights / float(np.sum(weights))
        return candidates, probabilities

    def _sample_retained(self, lane: int) -> RetainedProgram:
        candidates, probabilities = self._retained_distribution()
        index = int(self._rngs[lane].choice(len(candidates), p=probabilities))
        return candidates[index]

    def _start_lane(self, lane: int) -> None:
        state = _LaneState()
        if (
            self._retained
            and self._rngs[lane].random() < self.archive_replay_probability
        ):
            candidate = self._sample_retained(lane)
            state.mode = "replay"
            state.archive_candidate = candidate
            length = len(candidate.runs)
            if length > self.protected_prefix_runs:
                if self._rngs[lane].random() < self.deep_mutation_probability:
                    state.replay_limit_runs = int(
                        self._rngs[lane].integers(self.protected_prefix_runs, length)
                    )
                else:
                    shorten_limit = min(
                        self.max_prefix_shorten_runs,
                        length - self.protected_prefix_runs,
                    )
                    shorten_runs = int(
                        self._rngs[lane].integers(1, shorten_limit + 1)
                    )
                    state.replay_limit_runs = length - shorten_runs
            else:
                state.replay_limit_runs = length
            self.archive_replay_episodes += 1
            self.archive_selected_prefix_return_sum += candidate.mean_return
        self._lanes[lane] = state

    @staticmethod
    def _append_action(state: _LaneState, action: int) -> None:
        if state.runs and state.runs[-1].action == action:
            previous = state.runs[-1]
            state.runs[-1] = ActionRun(action, previous.duration + 1)
        else:
            state.runs.append(ActionRun(action, 1))
        state.step_count += 1

    def _sample_exploration_run(self, lane: int, state: _LaneState) -> None:
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

    def _next_replay_action(self, state: _LaneState) -> int | None:
        candidate = state.archive_candidate
        if candidate is None or state.replay_run_index >= state.replay_limit_runs:
            return None
        if state.replay_run_remaining == 0:
            run = candidate.runs[state.replay_run_index]
            state.replay_run_remaining = run.duration
        run = candidate.runs[state.replay_run_index]
        action = run.action
        state.replay_run_remaining -= 1
        if state.replay_run_remaining == 0:
            state.replay_run_index += 1
        return action

    def next_actions(self) -> np.ndarray:
        actions = np.empty(self.n_envs, dtype=np.int64)
        for lane, state in enumerate(self._lanes):
            if state.mode == "replay":
                replay_action = self._next_replay_action(state)
                if replay_action is None:
                    state.mode = "explore"
                    state.archive_candidate = None
                else:
                    action = replay_action
            if state.mode == "explore":
                if state.exploration_remaining == 0:
                    self._sample_exploration_run(lane, state)
                action = state.exploration_action
                state.exploration_remaining -= 1
            self._append_action(state, int(action))
            actions[lane] = action
        return actions

    @staticmethod
    def _record_facts(record: Any | None) -> tuple[bool, float]:
        if record is None:
            return False, 0.0
        if isinstance(record, Mapping):
            completed = bool(
                record.get("completed", record.get("level_complete", False))
            )
            progress = float(
                record.get("progress", record.get("max_x_pos", 0.0)) or 0.0
            )
            return completed, progress
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

    def _retain_exploration(self, state: _LaneState, record: Any | None) -> None:
        completed, progress = self._record_facts(record)
        if completed:
            runs = tuple(state.runs)
            score_return = state.episode_return
        else:
            runs = truncate_runs(state.runs, state.best_steps)
            score_return = state.best_return
        if not runs or not math.isfinite(score_return):
            return
        self._upsert_retained(
            runs,
            score_return=score_return,
            completed=completed,
            progress=progress,
        )

    def _upsert_retained(
        self,
        runs: tuple[ActionRun, ...],
        *,
        score_return: float,
        completed: bool,
        progress: float,
    ) -> None:
        runs = canonicalize_runs(runs)
        candidate = self._retained.get(runs)
        if candidate is None:
            candidate = RetainedProgram(runs=runs)
            self._retained[runs] = candidate
        candidate.observe(score_return, completed=completed, progress=progress)
        incomplete = [
            item for item in self._retained.values() if not item.completed
        ]
        if len(incomplete) > self.retained_limit:
            locked = [item for item in self._retained.values() if item.completed]
            retained = locked + sorted(
                incomplete, key=lambda item: item.rank, reverse=True
            )[: self.retained_limit]
            self._retained = {
                item.runs: item for item in retained
            }

    def observe(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        records_by_lane: Mapping[int, Any] | None = None,
    ) -> None:
        rewards_array = np.asarray(rewards, dtype=np.float64)
        dones_array = np.asarray(dones, dtype=bool)
        if rewards_array.shape != (self.n_envs,) or dones_array.shape != (self.n_envs,):
            raise ValueError(
                "JERK rewards and dones must contain one value per environment"
            )
        records_by_lane = records_by_lane or {}
        self.global_step += self.n_envs
        for lane, state in enumerate(self._lanes):
            reward = float(rewards_array[lane])
            state.episode_return += reward
            if state.episode_return > state.best_return:
                state.best_return = state.episode_return
                state.best_steps = state.step_count
            if dones_array[lane]:
                record = records_by_lane.get(lane)
                completed, _progress = self._record_facts(record)
                self._retain_exploration(state, record)
                self.completed_episodes += 1
                self.successful_episodes += int(completed)
                self._start_lane(lane)

    def best_candidate(self) -> RetainedProgram | None:
        candidates = list(self._retained.values())
        for state in self._lanes:
            if state.mode != "replay" and state.best_steps > 0:
                candidates.append(
                    RetainedProgram(
                        runs=truncate_runs(state.runs, state.best_steps),
                        return_sum=state.best_return,
                        return_count=1,
                    )
                )
        return max(candidates, key=lambda candidate: candidate.rank, default=None)

    def policy(self) -> "JerkPolicy":
        candidate = self.best_candidate()
        return JerkPolicy(
            action_names=self.action_names,
            action_runs=() if candidate is None else candidate.runs,
            fallback_action=self.fallback_action,
        )


class JerkPolicy:
    """Portable open-loop policy produced by JERK search."""

    def __init__(
        self,
        *,
        action_names: Sequence[str],
        action_runs: Sequence[ActionRun],
        fallback_action: int,
        timesteps: int = 0,
        episodes: int = 0,
        best_reward: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.action_names = tuple(str(name) for name in action_names)
        self.action_runs = canonicalize_runs(action_runs)
        self.fallback_action = int(fallback_action)
        self.timesteps = int(timesteps)
        self.episodes = int(episodes)
        self.best_reward = float(best_reward)
        self.metadata = dict(metadata or {})
        self._run_indices = np.zeros(1, dtype=np.int64)
        self._run_remaining = np.zeros(1, dtype=np.int64)
        self._validate_actions()

    def _validate_actions(self) -> None:
        count = len(self.action_names)
        if count < 1:
            raise ValueError("JERK policy requires at least one action name")
        values = (*(run.action for run in self.action_runs), self.fallback_action)
        if any(action < 0 or action >= count for action in values):
            raise ValueError(
                "JERK policy contains an action outside its action-name table"
            )

    @property
    def action_set(self) -> str:
        for name, actions in ACTION_SETS.items():
            if tuple(actions) == self.action_names:
                return name
        raise ValueError("JERK action table does not match a native action set")

    @property
    def action_count(self) -> int:
        return len(self.action_names)

    @property
    def run_count(self) -> int:
        return len(self.action_runs)

    @property
    def step_count(self) -> int:
        return run_step_count(self.action_runs)

    @staticmethod
    def _batch_size(observation: Any) -> int:
        if isinstance(observation, Mapping):
            if not observation:
                return 1
            observation = next(iter(observation.values()))
        array = np.asarray(observation)
        return int(array.shape[0]) if array.ndim > 0 else 1

    def _ensure_lanes(self, count: int) -> None:
        if self._run_indices.shape != (count,):
            self._run_indices = np.zeros(count, dtype=np.int64)
            self._run_remaining = np.zeros(count, dtype=np.int64)

    def _next_action(self, lane: int) -> int:
        index = int(self._run_indices[lane])
        if index >= len(self.action_runs):
            return self.fallback_action
        run = self.action_runs[index]
        if self._run_remaining[lane] == 0:
            self._run_remaining[lane] = run.duration
        self._run_remaining[lane] -= 1
        if self._run_remaining[lane] == 0:
            self._run_indices[lane] += 1
        return run.action

    def reset(self) -> None:
        self._run_indices.fill(0)
        self._run_remaining.fill(0)

    reset_episode = reset

    def reset_lanes(self, dones: Sequence[bool]) -> None:
        mask = np.asarray(dones, dtype=bool)
        self._ensure_lanes(int(mask.size))
        self._run_indices[mask] = 0
        self._run_remaining[mask] = 0

    def predict(
        self, observation: Any, deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        del deterministic
        count = self._batch_size(observation)
        self._ensure_lanes(count)
        actions = np.asarray(
            [self._next_action(lane) for lane in range(count)], dtype=np.int64
        )
        return actions, None

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": JERK_POLICY_SCHEMA_VERSION,
            "algorithm_id": "jerk",
            "model_class": "rlab.jerk.JerkPolicy",
            "action_names": list(self.action_names),
            "action_runs": [
                [run.action, run.duration] for run in self.action_runs
            ],
            "fallback_action": self.fallback_action,
            "timesteps": self.timesteps,
            "episodes": self.episodes,
            "best_reward": self.best_reward,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            destination, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(
                JERK_POLICY_MEMBER,
                json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
                + "\n",
            )

    @classmethod
    def load(cls, path: str | Path) -> "JerkPolicy":
        with zipfile.ZipFile(Path(path)) as archive:
            payload = json.loads(archive.read(JERK_POLICY_MEMBER))
        if int(payload.get("schema_version") or 0) != JERK_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported JERK policy schema version")
        if payload.get("algorithm_id") != "jerk":
            raise ValueError("JERK policy payload has the wrong algorithm id")
        return cls(
            action_names=payload["action_names"],
            action_runs=tuple(ActionRun(*run) for run in payload["action_runs"]),
            fallback_action=payload["fallback_action"],
            timesteps=int(payload.get("timesteps", 0)),
            episodes=int(payload.get("episodes", 0)),
            best_reward=float(payload.get("best_reward", 0.0)),
            metadata=payload.get("metadata", {}),
        )


def save_jerk_checkpoint(
    path: str | Path,
    action_runs: Sequence[tuple[str, int]],
    *,
    timesteps: int,
    episodes: int,
    best_reward: float,
    action_set: str = "simple",
    metadata: dict[str, Any] | None = None,
) -> Path:
    if action_set not in ACTION_SETS:
        raise ValueError(f"unknown action set {action_set!r}")
    action_names = tuple(ACTION_SETS[action_set])
    indices = {name: index for index, name in enumerate(action_names)}
    try:
        runs = tuple(
            ActionRun(indices[str(action)], int(duration))
            for action, duration in action_runs
        )
    except KeyError as exc:
        raise ValueError(
            f"checkpoint action {exc.args[0]!r} is not in action_set={action_set!r}"
        ) from exc
    policy = JerkPolicy(
        action_names=action_names,
        action_runs=runs,
        fallback_action=indices["noop"],
        timesteps=timesteps,
        episodes=episodes,
        best_reward=best_reward,
        metadata=metadata,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix != ".zip":
        raise ValueError("JERK run checkpoints must use the .zip format")
    policy.save(target)
    return target


def load_jerk_checkpoint(path: str | Path) -> JerkPolicy:
    source = Path(path)
    if not zipfile.is_zipfile(source):
        raise ValueError(f"{source} is not a JERK run checkpoint")
    return JerkPolicy.load(source)
