"""Train an action-run policy on an exact Super Mario Bros NES state."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shlex
import struct
from typing import Any
import uuid

import numpy as np

from . import (
    ACTION_SETS,
    PlayerMotion,
    PlayerTask,
    SuperMarioBrosNesTurboVecEnv,
    list_available_states,
)
from .env import VISIBLE_HEIGHT, VISIBLE_WIDTH
from .action_run import (
    ActionRunPolicy,
    resolve_state_name,
)
from . import training_ui


LOGGER = logging.getLogger("training")
ACTION_SET = "standard"
DEFAULT_ALGORITHM = "go-explore"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 64
MAX_EPISODE_STEPS = 4_500
STALL_STEPS = 300
CHECKPOINT_FREQ = 0
LOG_INTERVAL_STEPS = 10_000
PROTECTED_PREFIX_RUNS = 8
RUN_DURATION_MEAN = 4.0
RUN_DURATION_MAX = 32
FALLBACK_ACTION = "noop"
STEP_COST = 0.1
REWARD_FUNCTION_PROGRESS_SCORE = "progress-score"
REWARD_FUNCTION_SCORE_FIRST = "score-first"
REWARD_FUNCTION_SPEEDRUN = "speedrun"
# Backwards-compatible names for callers that still describe these as modes.
REWARD_MODE_PROGRESS_SCORE = REWARD_FUNCTION_PROGRESS_SCORE
REWARD_MODE_SCORE_FIRST = REWARD_FUNCTION_SCORE_FIRST
INVALID_XSCROLL_MIN = 0xFF00
SCROLL_TRANSITION_MIN_DROP = 128
SCROLL_TRANSITION_BUCKET = 64
OBSERVATION_FREE_CROP = (0, VISIBLE_HEIGHT - 1, 0, VISIBLE_WIDTH - 1)
GO_EXPLORE_CELL_X_BUCKET_PIXELS = 8
GO_EXPLORE_CELL_Y_BUCKET_PIXELS = 16
GO_EXPLORE_ROUTE_COUNTER_MAX = 7
GO_EXPLORE_CELL_KEY_STRUCT = struct.Struct("<BBBBHHBBB")
GO_EXPLORE_CELL_KEY_BYTES = GO_EXPLORE_CELL_KEY_STRUCT.size
GO_EXPLORE_CELL_ENCODING = "packed-bytes"
GO_EXPLORE_CELL_REPRESENTATION = "level-sublevel-area-pointer-x8-y16-route-ground-power"
GO_EXPLORE_CELL_INFO_KEYS = (
    "area_id",
    "y_pos",
    "area_pointer",
    "loop_command_active",
    "loop_correct_count",
    "loop_pass_count",
    "player_motion",
    "player_power",
    "player_task",
)


@dataclass(frozen=True)
class RewardFunction:
    """Stable, user-selectable reward-function definition."""

    id: str
    progress_weight: float
    score_weight: float
    default_step_cost: float | None


REWARD_FUNCTIONS = {
    reward.id: reward
    for reward in (
        RewardFunction(REWARD_FUNCTION_PROGRESS_SCORE, 1.0, 0.01, STEP_COST),
        RewardFunction(REWARD_FUNCTION_SCORE_FIRST, 0.0, 1.0, None),
        RewardFunction(REWARD_FUNCTION_SPEEDRUN, 0.0, 0.0, 1.0),
    )
}
REWARD_FUNCTION_IDS = tuple(REWARD_FUNCTIONS)


def reward_function(reward_id: str) -> RewardFunction:
    """Resolve a stable reward-function ID."""
    try:
        return REWARD_FUNCTIONS[str(reward_id)]
    except KeyError as exc:
        raise ValueError(f"unknown reward function {reward_id!r}") from exc


class _TrainingArgumentParser(argparse.ArgumentParser):
    """Resolve the completion default from the selected training scope."""

    def parse_known_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        parsed, extras = super().parse_known_args(args, namespace)
        if parsed.continue_after_completion is None:
            parsed.continue_after_completion = parsed.state is not None
        return parsed, extras


def go_explore_route_phase(
    *,
    loop_command_active: bool,
    loop_correct_count: int,
    loop_pass_count: int,
    player_task: int,
) -> int:
    """Pack route-causal castle progress and pipe-transition state into one byte."""
    correct = min(max(int(loop_correct_count), 0), GO_EXPLORE_ROUTE_COUNTER_MAX)
    passed = min(max(int(loop_pass_count), 0), GO_EXPLORE_ROUTE_COUNTER_MAX)
    transition = int(player_task) != int(PlayerTask.PLAYER_CONTROL)
    return (
        int(transition)
        | (int(bool(loop_command_active)) << 1)
        | (correct << 2)
        | (passed << 5)
    )


@dataclass(frozen=True)
class EpisodeBoundary:
    done: bool
    life_loss: bool
    truncated: bool
    level_changed: bool
    stalled: bool = False


def episode_boundary(
    *,
    previous_lives: int,
    current_lives: int,
    previous_level: tuple[int, int],
    current_level: tuple[int, int],
    episode_steps: int,
    max_episode_steps: int,
    native_truncated: bool = False,
    stalled: bool = False,
) -> EpisodeBoundary:
    """End failed attempts on life loss, stall, or timeout, never level change."""
    life_loss = current_lives < previous_lives
    truncated = native_truncated or episode_steps >= max_episode_steps
    return EpisodeBoundary(
        done=life_loss or stalled or truncated,
        life_loss=life_loss,
        truncated=truncated,
        level_changed=current_level != previous_level,
        stalled=stalled,
    )


@dataclass(frozen=True)
class EpisodeRecord:
    lane: int
    completed: bool
    progress: float
    episode_return: float
    episode_length: int
    life_loss: bool
    stalled: bool
    truncated: bool


@dataclass(frozen=True)
class MarioTaskSnapshot:
    """One restorable emulator lane plus its reward-tracking state."""

    native: Any
    episode_steps: int
    last_progress_step: int
    episode_return: float
    previous_lives: int
    previous_level_hi: int
    previous_level_lo: int
    previous_score: int
    level_max_x: int
    completed_base: int
    max_global_x: int
    previous_x: int
    seen_scroll_transitions: frozenset[tuple[int, int]]


def sanitize_progress_x(current_x: np.ndarray, previous_x: np.ndarray) -> np.ndarray:
    return np.where(current_x >= INVALID_XSCROLL_MIN, previous_x, current_x)


def mark_new_scroll_transitions(
    previous_x: np.ndarray,
    current_x: np.ndarray,
    blocked: np.ndarray,
    seen: list[set[tuple[int, int]]],
) -> np.ndarray:
    """Mark each non-terminal scroll discontinuity once per episode."""
    transitions = (previous_x - current_x >= SCROLL_TRANSITION_MIN_DROP) & ~blocked
    novel = np.zeros_like(transitions, dtype=np.bool_)
    for lane in np.flatnonzero(transitions):
        index = int(lane)
        signature = (
            int(previous_x[index]) // SCROLL_TRANSITION_BUCKET,
            int(current_x[index]) // SCROLL_TRANSITION_BUCKET,
        )
        if signature not in seen[index]:
            seen[index].add(signature)
            novel[index] = True
    return novel


def shape_step_rewards(
    progress_delta: np.ndarray,
    score_delta: np.ndarray,
    life_loss: np.ndarray,
    *,
    step_cost: float,
    reward_mode: str = REWARD_MODE_PROGRESS_SCORE,
) -> np.ndarray:
    """Shape one step with the selected stable reward-function ID."""
    progress = np.asarray(progress_delta, dtype=np.float64)
    score = np.asarray(score_delta, dtype=np.float64)
    definition = reward_function(reward_mode)
    shaped = definition.progress_weight * progress + definition.score_weight * score
    return shaped - float(step_cost) - 25.0 * np.asarray(life_loss, dtype=np.float64)


def score_first_step_cost(max_episode_steps: int) -> float:
    """Keep the total episode time charge below one raw score point."""
    maximum = int(max_episode_steps)
    if maximum < 1:
        raise ValueError("max_episode_steps must be positive")
    return 1.0 / (maximum + 1)


def reward_function_step_cost(reward_id: str, max_episode_steps: int) -> float:
    """Return the default time cost for a reward-function ID."""
    definition = reward_function(reward_id)
    if definition.default_step_cost is None:
        return score_first_step_cost(max_episode_steps)
    return definition.default_step_cost


class MarioTask:
    """Vectorized state task shared by action-run search algorithms."""

    def __init__(
        self,
        *,
        state: str,
        state_dir: str | Path | None,
        rom_path: str | Path | None,
        seed: int,
        n_envs: int,
        max_episode_steps: int,
        stall_steps: int,
        step_cost: float | None,
        noop_reset_max: int = 0,
        action_set: str = ACTION_SET,
        reward_mode: str = REWARD_MODE_PROGRESS_SCORE,
        go_explore_cells: bool = False,
    ) -> None:
        self.n_envs = int(n_envs)
        self.state = state
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.stall_steps = int(stall_steps)
        self.reward_mode = str(reward_mode)
        self.go_explore_cells = bool(go_explore_cells)
        reward_function(self.reward_mode)
        default_step_cost = reward_function_step_cost(
            self.reward_mode,
            self.max_episode_steps,
        )
        self.step_cost = float(default_step_cost if step_cost is None else step_cost)
        self.noop_reset_max = int(noop_reset_max)
        if self.step_cost < 0.0:
            raise ValueError("step_cost must be non-negative")
        if self.noop_reset_max < 0:
            raise ValueError("noop_reset_max must be non-negative")
        observation_options: dict[str, Any] = {"obs_crop": OBSERVATION_FREE_CROP}
        self.native = SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            state=self.state,
            state_dir=state_dir,
            num_envs=self.n_envs,
            num_threads=self.n_envs,
            rom_path=rom_path,
            render_mode=None,
            use_restricted_actions=action_set,
            obs_copy="unsafe_view",
            obs_grayscale=True,
            obs_layout="chw",
            frame_skip=4,
            frame_stack=1,
            maxpool_last_two=False,
            noop_reset_max=self.noop_reset_max,
            sticky_action_prob=0.0,
            reward_clip=False,
            info_filter=(
                {"mode": "all", "keys": list(GO_EXPLORE_CELL_INFO_KEYS)}
                if self.go_explore_cells
                else "none"
            ),
            **observation_options,
        )
        self.action_names = tuple(self.native.action_meanings)
        self.episode_steps = np.zeros(self.n_envs, dtype=np.int64)
        self.last_progress_step = np.zeros(self.n_envs, dtype=np.int64)
        self.episode_returns = np.zeros(self.n_envs, dtype=np.float64)
        self.previous_lives = np.zeros(self.n_envs, dtype=np.int16)
        self.previous_level_hi = np.zeros(self.n_envs, dtype=np.int16)
        self.previous_level_lo = np.zeros(self.n_envs, dtype=np.int16)
        self.previous_score = np.zeros(self.n_envs, dtype=np.int64)
        self.level_max_x = np.zeros(self.n_envs, dtype=np.int64)
        self.completed_base = np.zeros(self.n_envs, dtype=np.int64)
        self.max_global_x = np.zeros(self.n_envs, dtype=np.int64)
        self.previous_x = np.zeros(self.n_envs, dtype=np.int64)
        self.cell_area_id = np.full(self.n_envs, -1, dtype=np.int16)
        self.cell_y_pos = np.zeros(self.n_envs, dtype=np.int32)
        self.cell_area_pointer = np.zeros(self.n_envs, dtype=np.int16)
        self.cell_loop_command_active = np.zeros(self.n_envs, dtype=np.bool_)
        self.cell_loop_correct_count = np.zeros(self.n_envs, dtype=np.int16)
        self.cell_loop_pass_count = np.zeros(self.n_envs, dtype=np.int16)
        self.cell_player_motion = np.full(self.n_envs, -1, dtype=np.int8)
        self.cell_player_power = np.full(self.n_envs, -1, dtype=np.int8)
        self.cell_player_task = np.full(self.n_envs, -1, dtype=np.int8)
        self.seen_scroll_transitions = [set() for _ in range(self.n_envs)]

    def _update_cell_state(self, infos: Mapping[str, Any], mask: np.ndarray) -> None:
        if not self.go_explore_cells:
            return
        selected = np.asarray(mask, dtype=np.bool_)
        values = {key: np.asarray(infos[key]) for key in GO_EXPLORE_CELL_INFO_KEYS}
        if any(value.shape != (self.n_envs,) for value in values.values()):
            raise ValueError("Go-Explore cell infos must contain one value per lane")
        self.cell_area_id[selected] = values["area_id"][selected]
        self.cell_y_pos[selected] = values["y_pos"][selected]
        self.cell_area_pointer[selected] = values["area_pointer"][selected]
        self.cell_loop_command_active[selected] = values["loop_command_active"][
            selected
        ]
        self.cell_loop_correct_count[selected] = values["loop_correct_count"][selected]
        self.cell_loop_pass_count[selected] = values["loop_pass_count"][selected]
        self.cell_player_motion[selected] = values["player_motion"][selected]
        self.cell_player_power[selected] = values["player_power"][selected]
        self.cell_player_task[selected] = values["player_task"][selected]

    def _initialize_lanes(self, mask: np.ndarray) -> None:
        current_x = self.native.xscroll_hi.astype(
            np.int64, copy=False
        ) * 256 + self.native.xscroll_lo.astype(np.int64, copy=False)
        current_x = sanitize_progress_x(current_x, np.zeros_like(current_x))
        self.episode_steps[mask] = 0
        self.last_progress_step[mask] = 0
        self.episode_returns[mask] = 0.0
        self.previous_lives[mask] = self.native.lives[mask]
        self.previous_level_hi[mask] = self.native.level_hi[mask]
        self.previous_level_lo[mask] = self.native.level_lo[mask]
        self.previous_score[mask] = self.native.score[mask]
        self.level_max_x[mask] = current_x[mask]
        self.completed_base[mask] = 0
        self.max_global_x[mask] = current_x[mask]
        self.previous_x[mask] = current_x[mask]
        for lane in np.flatnonzero(mask):
            self.seen_scroll_transitions[int(lane)].clear()

    def reset(self) -> np.ndarray:
        observations, infos = self.native.reset(seed=self.seed)
        all_lanes = np.ones(self.n_envs, dtype=np.bool_)
        self._update_cell_state(infos, all_lanes)
        self._initialize_lanes(all_lanes)
        return observations

    def reset_lanes(self, mask: np.ndarray) -> None:
        reset_mask = np.asarray(mask, dtype=np.bool_)
        if not np.any(reset_mask):
            return
        _observations, infos = self.native.reset(options={"reset_mask": reset_mask})
        self._update_cell_state(infos, reset_mask)
        self._initialize_lanes(reset_mask)

    def capture_snapshots(
        self, mask: np.ndarray
    ) -> tuple[MarioTaskSnapshot | None, ...]:
        """Capture selected live lanes together with task-local accounting."""
        capture_mask = np.asarray(mask, dtype=np.bool_)
        native_snapshots = self.native.capture_snapshots(capture_mask)
        snapshots: list[MarioTaskSnapshot | None] = []
        for lane, native_snapshot in enumerate(native_snapshots):
            if native_snapshot is None:
                snapshots.append(None)
                continue
            snapshots.append(
                MarioTaskSnapshot(
                    native=native_snapshot,
                    episode_steps=int(self.episode_steps[lane]),
                    last_progress_step=int(self.last_progress_step[lane]),
                    episode_return=float(self.episode_returns[lane]),
                    previous_lives=int(self.previous_lives[lane]),
                    previous_level_hi=int(self.previous_level_hi[lane]),
                    previous_level_lo=int(self.previous_level_lo[lane]),
                    previous_score=int(self.previous_score[lane]),
                    level_max_x=int(self.level_max_x[lane]),
                    completed_base=int(self.completed_base[lane]),
                    max_global_x=int(self.max_global_x[lane]),
                    previous_x=int(self.previous_x[lane]),
                    seen_scroll_transitions=frozenset(
                        self.seen_scroll_transitions[lane]
                    ),
                )
            )
        return tuple(snapshots)

    def restore_lanes(
        self,
        mask: np.ndarray,
        snapshots: Sequence[MarioTaskSnapshot | None],
    ) -> None:
        """Restore selected lanes without discarding archived task accounting."""
        restore_mask = np.asarray(mask, dtype=np.bool_)
        if not np.any(restore_mask):
            return
        if len(snapshots) != self.n_envs:
            raise ValueError("task snapshots must contain one value per lane")
        selected = [int(lane) for lane in np.flatnonzero(restore_mask)]
        if any(snapshots[lane] is None for lane in selected):
            raise ValueError("every restored lane requires a task snapshot")
        _observations, infos = self.native.reset(
            options={
                "reset_mask": restore_mask,
                "snapshots": [
                    None if snapshot is None else snapshot.native
                    for snapshot in snapshots
                ],
            }
        )
        self._update_cell_state(infos, restore_mask)
        for lane in selected:
            snapshot = snapshots[lane]
            assert snapshot is not None
            self.episode_steps[lane] = snapshot.episode_steps
            self.last_progress_step[lane] = snapshot.last_progress_step
            self.episode_returns[lane] = snapshot.episode_return
            self.previous_lives[lane] = snapshot.previous_lives
            self.previous_level_hi[lane] = snapshot.previous_level_hi
            self.previous_level_lo[lane] = snapshot.previous_level_lo
            self.previous_score[lane] = snapshot.previous_score
            self.level_max_x[lane] = snapshot.level_max_x
            self.completed_base[lane] = snapshot.completed_base
            self.max_global_x[lane] = snapshot.max_global_x
            self.previous_x[lane] = snapshot.previous_x
            self.seen_scroll_transitions[lane] = set(snapshot.seen_scroll_transitions)

    def go_explore_cell_keys(self, observations: np.ndarray) -> tuple[bytes, ...]:
        """Return compact route-aware semantic cell keys."""
        if len(observations) != self.n_envs:
            raise ValueError("Go-Explore requires one cell observation per lane")
        return tuple(
            GO_EXPLORE_CELL_KEY_STRUCT.pack(
                int(self.previous_level_hi[lane]) & 0xFF,
                int(self.previous_level_lo[lane]) & 0xFF,
                int(self.cell_area_id[lane]) & 0xFF,
                int(self.cell_area_pointer[lane]) & 0xFF,
                int(self.native.x_pos[lane]) // GO_EXPLORE_CELL_X_BUCKET_PIXELS,
                int(self.cell_y_pos[lane]) // GO_EXPLORE_CELL_Y_BUCKET_PIXELS,
                go_explore_route_phase(
                    loop_command_active=bool(self.cell_loop_command_active[lane]),
                    loop_correct_count=int(self.cell_loop_correct_count[lane]),
                    loop_pass_count=int(self.cell_loop_pass_count[lane]),
                    player_task=int(self.cell_player_task[lane]),
                ),
                int(self.cell_player_motion[lane] == int(PlayerMotion.GROUND)),
                int(self.cell_player_power[lane]) & 0xFF,
            )
            for lane in range(self.n_envs)
        )

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, dict[int, EpisodeRecord], np.ndarray
    ]:
        action_indices = np.asarray(actions, dtype=np.int64)
        observations, _native_rewards, native_terminated, native_truncated, infos = (
            self.native.step(action_indices)
        )
        self._update_cell_state(infos, np.ones(self.n_envs, dtype=np.bool_))
        current_lives = self.native.lives.astype(np.int64, copy=False)
        current_level_hi = self.native.level_hi.astype(np.int64, copy=False)
        current_level_lo = self.native.level_lo.astype(np.int64, copy=False)
        current_score = self.native.score.astype(np.int64, copy=False)
        current_x = self.native.xscroll_hi.astype(
            np.int64, copy=False
        ) * 256 + self.native.xscroll_lo.astype(np.int64, copy=False)
        current_x = sanitize_progress_x(current_x, self.level_max_x)

        life_loss = current_lives < self.previous_lives
        level_changed = (current_level_hi != self.previous_level_hi) | (
            current_level_lo != self.previous_level_lo
        )
        completed = level_changed & ~life_loss
        scroll_transition = mark_new_scroll_transitions(
            self.previous_x,
            current_x,
            life_loss | level_changed,
            self.seen_scroll_transitions,
        )

        segment_changed = completed | scroll_transition
        self.completed_base[segment_changed] += self.level_max_x[segment_changed]
        self.level_max_x[segment_changed] = 0
        effective_x = np.where(level_changed, 0, current_x)
        self.level_max_x = np.maximum(self.level_max_x, effective_x)
        global_max = self.completed_base + self.level_max_x
        progress_delta = np.maximum(global_max - self.max_global_x, 0)
        self.max_global_x = np.maximum(self.max_global_x, global_max)
        score_delta = np.maximum(current_score - self.previous_score, 0)

        progressed = progress_delta > 0
        self.last_progress_step[progressed] = self.episode_steps[progressed]
        self.episode_steps += 1
        stalled = (self.stall_steps > 0) & (
            self.episode_steps - self.last_progress_step >= self.stall_steps
        )
        timed_out = self.episode_steps >= self.max_episode_steps

        shaped_rewards = shape_step_rewards(
            progress_delta,
            score_delta,
            life_loss,
            step_cost=self.step_cost,
            reward_mode=self.reward_mode,
        )
        self.episode_returns += shaped_rewards

        failures = life_loss | stalled | timed_out | native_truncated
        unexpected_native_terminal = native_terminated & ~completed & ~life_loss
        failures |= unexpected_native_terminal
        records: dict[int, EpisodeRecord] = {}
        search_dones = failures | completed
        for lane in np.flatnonzero(search_dones):
            index = int(lane)
            records[index] = EpisodeRecord(
                lane=index,
                completed=bool(completed[index]),
                progress=float(self.max_global_x[index]),
                episode_return=float(self.episode_returns[index]),
                episode_length=int(self.episode_steps[index]),
                life_loss=bool(life_loss[index]),
                stalled=bool(stalled[index]),
                truncated=bool(timed_out[index] or native_truncated[index]),
            )

        self.previous_lives[:] = current_lives
        self.previous_level_hi[:] = current_level_hi
        self.previous_level_lo[:] = current_level_lo
        self.previous_score[:] = current_score
        self.previous_x[:] = current_x
        return observations, shaped_rewards, failures, records, completed

    def close(self) -> None:
        self.native.close()


def _force_policy_overwrite(args: argparse.Namespace) -> bool:
    """Allow replacement only for the canonical default run or explicit force."""
    reward_id = getattr(args, "reward_function", None)
    return bool(
        args.overwrite
        or (
            args.output is None
            and args.algorithm == DEFAULT_ALGORITHM
            and reward_id in {None, REWARD_FUNCTION_SPEEDRUN}
        )
    )


def _save_policy(policy: ActionRunPolicy, path: Path, *, force: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.zip"
    policy.save(temporary)
    try:
        if force:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite existing policy {path}; pass --overwrite to replace it"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _protect_existing_policies(run_dir: Path, *, force: bool) -> None:
    if force or not run_dir.exists():
        return
    existing = next(iter(sorted(run_dir.rglob("*.zip"))), None)
    if existing is not None:
        raise SystemExit(
            f"refusing to overwrite existing policy {existing}; "
            "pass --overwrite to replace policies in this run directory"
        )


def _format_box(title: str, rows: list[tuple[str, str]]) -> str:
    return training_ui.format_box(title, rows)


def _format_elapsed(seconds: float) -> str:
    return training_ui.format_elapsed(seconds)


def _format_progress(row: dict[str, Any], total_timesteps: int) -> str:
    return training_ui.format_progress(row, total_timesteps)


def _play_command(
    state: str,
    policy_path: Path,
    *,
    default_output: bool,
    rom_path: Path | None,
    action_set: str = ACTION_SET,
) -> str:
    argv = ["smb-turbo", "play", state]
    if not default_output:
        argv.extend(["--policy", str(policy_path)])
    if rom_path is not None:
        argv.extend(["--rom", str(rom_path)])
    if action_set != ACTION_SET:
        argv.extend(["--action-set", action_set])
    return shlex.join(argv)


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = _TrainingArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "state",
        nargs="?",
        help=(
            "exact state identifier, for example Level1-1; omit to train all "
            "32 canonical levels in order"
        ),
    )
    parser.add_argument(
        "--algorithm",
        choices=("beam", "go-explore"),
        default=DEFAULT_ALGORITHM,
        help=f"training search algorithm (default: {DEFAULT_ALGORITHM})",
    )
    parser.add_argument(
        "--rom",
        type=Path,
        help="ROM path; defaults to Stable Retro-compatible discovery",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="additional directory containing named .state files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="run directory; defaults to runs/<State>",
    )
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument(
        "--action-set",
        choices=tuple(ACTION_SETS),
        default=ACTION_SET,
        help=f"named search action set (default: {ACTION_SET})",
    )
    parser.add_argument("--transitions", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--lanes", type=int, default=N_ENVS)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--stall-steps", type=int, default=STALL_STEPS)
    parser.add_argument(
        "--noop-reset-max",
        type=int,
        default=0,
        metavar="FRAMES",
        help=(
            "maximum seeded random raw emulator frames applied after ordinary "
            "state resets (default: 0, disabled)"
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_FREQ)
    parser.add_argument("--log-every", type=int, default=LOG_INTERVAL_STEPS)
    shared = parser.add_argument_group("shared search tuning")
    shared.add_argument(
        "--protected-prefix-runs", type=int, default=PROTECTED_PREFIX_RUNS
    )
    shared.add_argument("--run-duration-mean", type=float, default=RUN_DURATION_MEAN)
    shared.add_argument("--run-duration-max", type=int, default=RUN_DURATION_MAX)
    shared.add_argument("--fallback-action", default=FALLBACK_ACTION)
    shared.add_argument(
        "--step-cost",
        type=float,
        default=None,
        help=(
            "per-step return charge; defaults to the selected reward function "
            "for Go-Explore and 1 / (max episode steps + 1) for beam"
        ),
    )

    beam = parser.add_argument_group("beam search tuning")
    beam.add_argument("--beam-width", type=int, default=None)
    beam.add_argument("--beam-refresh-episodes", type=int, default=None)
    beam.add_argument(
        "--beam-deepen-after-generations",
        type=int,
        default=None,
        help=(
            "switch unsolved beams from tail mutation to systematic geometric "
            "deepening after this many generations (default: 64)"
        ),
    )
    beam.add_argument("--mutation-runs", type=int, default=None)
    beam.add_argument("--initial-policy", type=Path, default=None)
    beam.add_argument(
        "--improvement-protected-prefix-runs",
        type=int,
        default=None,
        help=(
            "prefix runs protected only during post-completion improvement (default: 0)"
        ),
    )
    beam.add_argument(
        "--branch-durations", type=int, nargs="+", default=None, metavar="STEPS"
    )
    go_explore = parser.add_argument_group("Go-Explore tuning")
    go_explore.add_argument(
        "--go-explore-explore-steps",
        type=int,
        default=None,
        metavar="STEPS",
        help="random exploration horizon after each archived restore (default: 128)",
    )
    go_explore.add_argument(
        "--reward-function",
        choices=REWARD_FUNCTION_IDS,
        default=None,
        metavar="ID",
        help=(
            "trajectory reward function: speedrun (default), score-first, "
            "or progress-score"
        ),
    )
    completion = parser.add_mutually_exclusive_group()
    completion.add_argument(
        "--continue-after-completion",
        dest="continue_after_completion",
        action="store_true",
        help=(
            "continue to the transition budget and publish only higher-return "
            "completed paths (default when training an explicit state)"
        ),
    )
    completion.add_argument(
        "--stop-on-completion",
        dest="continue_after_completion",
        action="store_false",
        help="stop after the first completed path",
    )
    parser.set_defaults(continue_after_completion=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace policies in custom or non-default run directories",
    )
    parser.add_argument(
        "--ui",
        choices=("auto", "tui", "plain"),
        default="auto",
        help="training display (default: TUI in an interactive terminal)",
    )
    return parser


def _apply_algorithm_defaults(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    from .beam_training import (
        BEAM_REFRESH_EPISODES,
        BEAM_DEEPEN_AFTER_GENERATIONS,
        BEAM_WIDTH,
        BRANCH_DURATIONS,
        MUTATION_RUNS,
    )
    from .go_explore_training import GO_EXPLORE_EXPLORE_STEPS

    beam_defaults = {
        "beam_width": BEAM_WIDTH,
        "beam_refresh_episodes": BEAM_REFRESH_EPISODES,
        "beam_deepen_after_generations": BEAM_DEEPEN_AFTER_GENERATIONS,
        "mutation_runs": MUTATION_RUNS,
        "initial_policy": None,
        "improvement_protected_prefix_runs": 0,
        "branch_durations": list(BRANCH_DURATIONS),
    }
    go_explore_defaults = {
        "go_explore_explore_steps": GO_EXPLORE_EXPLORE_STEPS,
        "reward_function": REWARD_FUNCTION_SPEEDRUN,
    }
    defaults_by_algorithm = {
        "beam": beam_defaults,
        "go-explore": go_explore_defaults,
    }
    selected = defaults_by_algorithm[args.algorithm]
    rejected = {
        name: value
        for algorithm, defaults in defaults_by_algorithm.items()
        if algorithm != args.algorithm
        for name, value in defaults.items()
    }
    invalid = [name for name in rejected if getattr(args, name) is not None]
    if invalid:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in invalid)
        parser.error(f"{flags} cannot be used with --algorithm {args.algorithm}")
    for name, value in selected.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.step_cost is None:
        if args.algorithm == "go-explore":
            args.step_cost = reward_function_step_cost(
                args.reward_function,
                args.max_episode_steps,
            )
        else:
            args.step_cost = score_first_step_cost(args.max_episode_steps)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    _apply_algorithm_defaults(parser, args)
    if args.state is None:
        from .training_campaign import CANONICAL_LEVEL_STATES, run

        available = set(list_available_states(args.state_dir))
        missing = [state for state in CANONICAL_LEVEL_STATES if state not in available]
        if missing:
            raise SystemExit(
                "all-level training requires every canonical state; missing: "
                + ", ".join(missing)
            )
        return run(args, parser)
    try:
        args.state = resolve_state_name(args.state, state_dir=args.state_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.algorithm == "beam":
        from .beam_training import run

        return run(args, parser)
    if args.algorithm == "go-explore":
        from .go_explore_training import run

        return run(args, parser)
    raise AssertionError(f"unhandled training algorithm {args.algorithm!r}")


if __name__ == "__main__":
    raise SystemExit(main())
