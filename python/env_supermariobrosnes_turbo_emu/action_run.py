from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import ACTION_SETS, list_available_states


ACTION_RUN_POLICY_SCHEMA_VERSION = 1
ACTION_RUN_POLICY_MEMBER = "action_run_policy.json"
LEGACY_JERK_POLICY_MEMBER = "jerk_policy.json"


def validate_state_name(state: str) -> str:
    """Return an exact, path-safe state identifier without normalizing it."""
    if not isinstance(state, str) or not state:
        raise ValueError("state name must be a non-empty string")
    if state in {".", ".."} or "/" in state or "\\" in state:
        raise ValueError(
            f"invalid state name {state!r}; expected an exact state identifier"
        )
    return state


def resolve_state_name(
    state: str,
    *,
    state_dir: str | Path | None = None,
) -> str:
    """Resolve an exact state identifier from the configured state sources."""
    name = validate_state_name(state)
    available = list_available_states(state_dir)
    if name not in available:
        choices = ", ".join(available) or "<none>"
        raise ValueError(f"unknown state {name!r}; available states: {choices}")
    return name


def run_directory_for_state(
    state: str,
    *,
    runs_root: str | Path = "runs",
) -> Path:
    name = validate_state_name(state)
    return Path(runs_root) / name


def policy_path_for_state(
    state: str,
    *,
    runs_root: str | Path = "runs",
) -> Path:
    name = validate_state_name(state)
    return (
        run_directory_for_state(
            name,
            runs_root=runs_root,
        )
        / f"{name}.zip"
    )


def policy_paths_for_state(
    state: str,
    *,
    runs_root: str | Path = "runs",
) -> tuple[Path, ...]:
    """Return canonical and legacy policy paths in playback precedence order."""
    name = validate_state_name(state)
    root = Path(runs_root)
    return (
        policy_path_for_state(name, runs_root=root),
        root / f"{name}-beam" / f"{name}.zip",
        root / f"{name}-jerk" / f"{name}.zip",
    )


def find_policy_path_for_state(
    state: str,
    *,
    runs_root: str | Path = "runs",
) -> Path | None:
    """Find a canonical policy, preferring legacy beam over legacy JERK."""
    return next(
        (
            path
            for path in policy_paths_for_state(state, runs_root=runs_root)
            if path.is_file()
        ),
        None,
    )


@dataclass(frozen=True, order=True)
class ActionRun:
    """One action held for a positive number of environment steps."""

    action: int
    duration: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", int(self.action))
        object.__setattr__(self, "duration", int(self.duration))
        if self.duration < 1:
            raise ValueError("action-run durations must be positive")


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


def truncate_runs(runs: Sequence[ActionRun], step_limit: int) -> tuple[ActionRun, ...]:
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


class ActionRunPolicy:
    """Portable open-loop action-run policy."""

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
            raise ValueError("action-run policy requires at least one action name")
        values = (*(run.action for run in self.action_runs), self.fallback_action)
        if any(action < 0 or action >= count for action in values):
            raise ValueError(
                "action-run policy contains an action outside its action-name table"
            )

    @property
    def action_set(self) -> str:
        for name, actions in ACTION_SETS.items():
            if tuple(actions) == self.action_names:
                return name
        raise ValueError("action table does not match a native action set")

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
            "schema_version": ACTION_RUN_POLICY_SCHEMA_VERSION,
            "algorithm_id": "action-run",
            "model_class": "env_supermariobrosnes_turbo_emu.action_run.ActionRunPolicy",
            "action_names": list(self.action_names),
            "action_runs": [[run.action, run.duration] for run in self.action_runs],
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
                ACTION_RUN_POLICY_MEMBER,
                json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
                + "\n",
            )

    @classmethod
    def load(cls, path: str | Path) -> "ActionRunPolicy":
        with zipfile.ZipFile(Path(path)) as archive:
            member = (
                ACTION_RUN_POLICY_MEMBER
                if ACTION_RUN_POLICY_MEMBER in archive.namelist()
                else LEGACY_JERK_POLICY_MEMBER
            )
            payload = json.loads(archive.read(member))
        if member == ACTION_RUN_POLICY_MEMBER:
            if (
                int(payload.get("schema_version") or 0)
                != ACTION_RUN_POLICY_SCHEMA_VERSION
            ):
                raise ValueError("unsupported action-run policy schema version")
            if payload.get("algorithm_id") != "action-run":
                raise ValueError("action-run policy payload has the wrong algorithm id")
        elif (
            int(payload.get("schema_version") or 0) != 2
            or payload.get("algorithm_id") != "jerk"
        ):
            raise ValueError("unsupported legacy action-run policy schema")
        return cls(
            action_names=payload["action_names"],
            action_runs=tuple(ActionRun(*run) for run in payload["action_runs"]),
            fallback_action=payload["fallback_action"],
            timesteps=int(payload.get("timesteps", 0)),
            episodes=int(payload.get("episodes", 0)),
            best_reward=float(payload.get("best_reward", 0.0)),
            metadata=payload.get("metadata", {}),
        )


def save_action_run_checkpoint(
    path: str | Path,
    action_runs: Sequence[tuple[str, int]],
    *,
    timesteps: int,
    episodes: int,
    best_reward: float,
    action_set: str = "standard",
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
    policy = ActionRunPolicy(
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
        raise ValueError("action-run checkpoints must use the .zip format")
    policy.save(target)
    return target


def load_action_run_checkpoint(path: str | Path) -> ActionRunPolicy:
    source = Path(path)
    if not zipfile.is_zipfile(source):
        raise ValueError(f"{source} is not an action-run checkpoint")
    return ActionRunPolicy.load(source)
