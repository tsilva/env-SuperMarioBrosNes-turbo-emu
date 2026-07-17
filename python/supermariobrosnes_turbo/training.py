"""Train an action-run policy on an exact Super Mario Bros NES state."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shlex
import threading
import time
from typing import Any
import uuid

import numpy as np

from . import (
    ACTION_SETS,
    SuperMarioBrosNesTurboVecEnv,
)
from .env import VISIBLE_HEIGHT, VISIBLE_WIDTH
from .jerk import (
    JerkPolicy,
    JerkSearch,
    policy_path_for_state,
    resolve_state_name,
    run_directory_for_state,
)
from . import training_ui
from .training_ui import (
    PlainReporter,
    TrainingEvent,
    TrainingReporter,
    TrainingResult,
    TrainingSnapshot,
)


LOGGER = logging.getLogger("jerk_train")
ACTION_SET = "simple"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 64
MAX_EPISODE_STEPS = 4_500
STALL_STEPS = 300
CHECKPOINT_FREQ = 0
LOG_INTERVAL_STEPS = 10_000
ARCHIVE_REPLAY_PROBABILITY_INITIAL = 0.25
ARCHIVE_REPLAY_PROBABILITY_MAX = 0.9
PROTECTED_PREFIX_RUNS = 8
MAX_PREFIX_SHORTEN_RUNS = 16
DEEP_MUTATION_PROBABILITY = 0.25
RUN_DURATION_MEAN = 4.0
RUN_DURATION_MAX = 32
RETAINED_LIMIT = 256
FALLBACK_ACTION = "noop"
STEP_COST = 0.1
INVALID_XSCROLL_MIN = 0xFF00
OBSERVATION_FREE_CROP = (0, VISIBLE_HEIGHT - 1, 0, VISIBLE_WIDTH - 1)


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


def sanitize_progress_x(current_x: np.ndarray, previous_x: np.ndarray) -> np.ndarray:
    return np.where(current_x >= INVALID_XSCROLL_MIN, previous_x, current_x)


def shape_step_rewards(
    progress_delta: np.ndarray,
    score_delta: np.ndarray,
    life_loss: np.ndarray,
    *,
    step_cost: float,
) -> np.ndarray:
    """Reward new progress and score while charging every elapsed env step."""
    return (
        np.asarray(progress_delta, dtype=np.float64)
        + 0.01 * np.asarray(score_delta, dtype=np.float64)
        - float(step_cost)
        - 25.0 * np.asarray(life_loss, dtype=np.float64)
    )


class MarioJerkTask:
    """Vectorized state task matching rlab's JERK reward and failure contract."""

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
        step_cost: float,
    ) -> None:
        self.n_envs = int(n_envs)
        self.state = state
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.stall_steps = int(stall_steps)
        self.step_cost = float(step_cost)
        if self.step_cost < 0.0:
            raise ValueError("JERK step_cost must be non-negative")
        self.native = SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            state=self.state,
            state_dir=state_dir,
            num_envs=self.n_envs,
            num_threads=self.n_envs,
            rom_path=rom_path,
            render_mode=None,
            action_set=ACTION_SET,
            obs_copy="unsafe_view",
            obs_crop=OBSERVATION_FREE_CROP,
            obs_grayscale=True,
            obs_layout="chw",
            frame_skip=4,
            frame_stack=1,
            maxpool_last_two=False,
            sticky_action_prob=0.0,
            reward_clip=False,
            info_filter="none",
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

    def reset(self) -> np.ndarray:
        observations, _infos = self.native.reset(seed=self.seed)
        self._initialize_lanes(np.ones(self.n_envs, dtype=np.bool_))
        return observations

    def reset_lanes(self, mask: np.ndarray) -> None:
        reset_mask = np.asarray(mask, dtype=np.bool_)
        if not np.any(reset_mask):
            return
        self.native.reset(options={"reset_mask": reset_mask})
        self._initialize_lanes(reset_mask)

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, dict[int, EpisodeRecord], np.ndarray
    ]:
        action_indices = np.asarray(actions, dtype=np.int64)
        observations, _native_rewards, native_terminated, native_truncated, _infos = (
            self.native.step(action_indices)
        )
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

        self.completed_base[completed] += self.level_max_x[completed]
        self.level_max_x[completed] = 0
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
        return observations, shaped_rewards, failures, records, completed

    def close(self) -> None:
        self.native.close()


def exploit_probability(total_steps: int, total_timesteps: int) -> float:
    return min(
        ARCHIVE_REPLAY_PROBABILITY_MAX,
        ARCHIVE_REPLAY_PROBABILITY_INITIAL + total_steps / max(total_timesteps, 1),
    )


def _save_policy(policy: JerkPolicy, path: Path, *, force: bool = False) -> Path:
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


def _metric_row(
    search: JerkSearch, *, elapsed: float, accepted: bool
) -> dict[str, Any]:
    candidate = search.best_candidate()
    return {
        "timesteps": search.global_step,
        "episodes": search.completed_episodes,
        "retained_count": search.retained_count,
        "locked_count": search.locked_count,
        "incomplete_retained_count": search.incomplete_retained_count,
        "successful_episodes": search.successful_episodes,
        "archive_replay_probability": search.archive_replay_probability,
        "archive_selected_prefix_return_mean": (
            search.archive_selected_prefix_return_mean
        ),
        "best_program_steps": candidate.step_count if candidate else 0,
        "best_program_runs": len(candidate.runs) if candidate else 0,
        "best_mean_reward": candidate.mean_return if candidate else 0.0,
        "best_progress": candidate.progress if candidate else 0.0,
        "best_completed": candidate.completed if candidate else False,
        "accepted": accepted,
        "loop_fps": search.global_step / max(elapsed, 1e-9),
    }


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
) -> str:
    argv = ["smb-turbo", "play", state]
    if not default_output:
        argv.extend(["--policy", str(policy_path)])
    if rom_path is not None:
        argv.extend(["--rom", str(rom_path)])
    return shlex.join(argv)


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("state", help="Exact state identifier, for example Level1-1")
    parser.add_argument(
        "--algorithm",
        choices=("jerk", "beam"),
        default="beam",
        help="training search algorithm (default: beam)",
    )
    parser.add_argument(
        "--rom", type=Path, help="ROM path; defaults to Stable Retro-compatible discovery"
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
    parser.add_argument("--transitions", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--lanes", type=int, default=N_ENVS)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--stall-steps", type=int, default=STALL_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_FREQ)
    parser.add_argument("--log-every", type=int, default=LOG_INTERVAL_STEPS)
    shared = parser.add_argument_group("shared search tuning")
    shared.add_argument(
        "--protected-prefix-runs", type=int, default=PROTECTED_PREFIX_RUNS
    )
    shared.add_argument("--run-duration-mean", type=float, default=RUN_DURATION_MEAN)
    shared.add_argument("--run-duration-max", type=int, default=RUN_DURATION_MAX)
    shared.add_argument("--fallback-action", default=FALLBACK_ACTION)
    shared.add_argument("--step-cost", type=float, default=STEP_COST)

    jerk = parser.add_argument_group("JERK search tuning")
    jerk.add_argument(
        "--archive-replay-probability-initial",
        type=float,
        default=None,
    )
    jerk.add_argument(
        "--archive-replay-probability-max",
        type=float,
        default=None,
    )
    jerk.add_argument(
        "--max-prefix-shorten-runs", type=int, default=None
    )
    jerk.add_argument(
        "--deep-mutation-probability",
        type=float,
        default=None,
    )
    jerk.add_argument("--retained-limit", type=int, default=None)

    beam = parser.add_argument_group("beam search tuning")
    beam.add_argument("--beam-width", type=int, default=None)
    beam.add_argument("--beam-refresh-episodes", type=int, default=None)
    beam.add_argument("--mutation-runs", type=int, default=None)
    beam.add_argument(
        "--improvement-protected-prefix-runs",
        type=int,
        default=None,
        help=(
            "prefix runs protected only during post-completion improvement "
            "(default: 0)"
        ),
    )
    beam.add_argument(
        "--branch-durations", type=int, nargs="+", default=None, metavar="STEPS"
    )
    parser.add_argument(
        "--continue-after-completion",
        action="store_true",
        help=(
            "continue to the transition budget and publish only higher-return "
            "completed paths"
        ),
    )
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


def _initial_snapshot(args: argparse.Namespace) -> TrainingSnapshot:
    run_dir = args.output or run_directory_for_state(args.state)
    target_policy_path = (
        policy_path_for_state(args.state)
        if args.output is None
        else run_dir / f"{args.state}.zip"
    )
    return TrainingSnapshot(
        algorithm="JERK",
        state=args.state,
        seed=args.seed,
        lanes=args.lanes,
        stop_rule=(
            "first completion" if (not args.continue_after_completion) else "transition budget"
        ),
        output=target_policy_path,
        total_timesteps=args.transitions,
    )


def _run_training(
    args: argparse.Namespace,
    reporter: TrainingReporter,
    stop_event: threading.Event,
) -> TrainingResult:
    run_dir = args.output or run_directory_for_state(args.state)
    target_policy_path = (
        policy_path_for_state(args.state)
        if args.output is None
        else run_dir / f"{args.state}.zip"
    )
    metrics_path = run_dir / "episodes.jsonl"
    action_names = tuple(ACTION_SETS[ACTION_SET])
    search = JerkSearch(
        n_envs=args.lanes,
        seed=args.seed,
        total_timesteps=args.transitions,
        action_names=action_names,
        fallback_action=args.fallback_action,
        archive_replay_probability_initial=args.archive_replay_probability_initial,
        archive_replay_probability_max=args.archive_replay_probability_max,
        protected_prefix_runs=args.protected_prefix_runs,
        max_prefix_shorten_runs=args.max_prefix_shorten_runs,
        deep_mutation_probability=args.deep_mutation_probability,
        run_duration_mean=args.run_duration_mean,
        run_duration_max=args.run_duration_max,
        retained_limit=args.retained_limit,
    )
    task = MarioJerkTask(
        state=args.state,
        state_dir=args.state_dir,
        rom_path=args.rom,
        seed=args.seed,
        n_envs=args.lanes,
        max_episode_steps=args.max_episode_steps,
        stall_steps=args.stall_steps,
        step_cost=args.step_cost,
    )
    started_at = time.perf_counter()
    next_log = args.log_every
    next_checkpoint = args.checkpoint_every if args.checkpoint_every > 0 else None
    accepted = False
    accepted_lane: int | None = None
    first_success_step: int | None = None
    stopped_on_completion = False
    last_best: tuple[bool, float, float] | None = None
    initial_snapshot = _initial_snapshot(args)
    last_ui_publish = float("-inf")
    reporter.start(initial_snapshot)
    try:
        task.reset()
        while search.global_step < args.transitions and not stop_event.is_set():
            actions = search.next_actions()
            _observations, rewards, failure_dones, records, successes = task.step(
                actions
            )
            search_dones = failure_dones | successes
            search.observe(
                rewards,
                search_dones,
                records,
                progresses=getattr(task, "max_global_x", None),
            )
            step = search.global_step

            if np.any(successes):
                if not accepted:
                    accepted = True
                    first_success_step = step
                    accepted_lane = int(np.flatnonzero(successes)[0])
                    accepted_path = (
                        run_dir / "checkpoints" / f"{args.state}-{step}.zip"
                    )
                    _save_policy(search.policy(), accepted_path, force=args.overwrite)
                    elapsed = time.perf_counter() - started_at
                    success_row = _metric_row(
                        search, elapsed=elapsed, accepted=accepted
                    )
                    reporter.update(
                        training_ui.snapshot_from_row(
                            algorithm="JERK",
                            state=args.state,
                            seed=args.seed,
                            lanes=args.lanes,
                            stop_rule=initial_snapshot.stop_rule,
                            output=target_policy_path,
                            total_timesteps=args.transitions,
                            row={**success_row, "elapsed": elapsed},
                            status="Level completed",
                        ),
                        TrainingEvent(
                            "success",
                            "Level completed",
                            elapsed,
                            (
                                ("State", args.state),
                                ("Transition", f"{step:,}"),
                                ("Lane", f"{accepted_lane:,}"),
                                ("Checkpoint", str(accepted_path)),
                            ),
                        ),
                        force=True,
                    )
                if (not args.continue_after_completion):
                    stopped_on_completion = True
                    break

            task.reset_lanes(search_dones)

            elapsed = time.perf_counter() - started_at
            event = None
            if records:
                candidate = search.best_candidate()
                current_best = (
                    False if candidate is None else candidate.completed,
                    0.0 if candidate is None else candidate.progress,
                    0.0 if candidate is None else candidate.mean_return,
                )
                if current_best != last_best and candidate is not None:
                    event = TrainingEvent(
                        "new-best",
                        f"Best path updated: return {candidate.mean_return:,.1f} · x {candidate.progress:,.0f}",
                        elapsed,
                    )
                    last_best = current_best
            now = time.monotonic()
            routine_due = (
                now - last_ui_publish >= training_ui.UPDATE_INTERVAL_SECONDS
            )
            log_due = step >= next_log
            checkpoint_due = next_checkpoint is not None and step >= next_checkpoint
            ui_row = None
            snapshot = None
            if event is not None or routine_due or log_due or checkpoint_due:
                ui_row = _metric_row(search, elapsed=elapsed, accepted=accepted)
                snapshot = training_ui.snapshot_from_row(
                    algorithm="JERK",
                    state=args.state,
                    seed=args.seed,
                    lanes=args.lanes,
                    stop_rule=initial_snapshot.stop_rule,
                    output=target_policy_path,
                    total_timesteps=args.transitions,
                    row={**ui_row, "elapsed": elapsed},
                )
                reporter.update(snapshot, event)
                last_ui_publish = now

            if log_due:
                assert ui_row is not None and snapshot is not None
                row = ui_row
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                if isinstance(reporter, PlainReporter):
                    reporter.update(
                        snapshot,
                        force=True,
                    )
                while next_log <= step:
                    next_log += args.log_every

            while next_checkpoint is not None and step >= next_checkpoint:
                assert snapshot is not None
                checkpoint_path = _save_policy(
                    search.policy(),
                    run_dir / "checkpoints" / f"{args.state}-{step}.zip",
                    force=args.overwrite,
                )
                reporter.update(
                    snapshot,
                    TrainingEvent(
                        "checkpoint",
                        f"Checkpoint saved: {checkpoint_path}",
                        elapsed,
                    ),
                    force=True,
                )
                next_checkpoint += args.checkpoint_every

        final_candidate = search.best_candidate()
        final_policy = JerkPolicy(
            action_names=search.action_names,
            action_runs=() if final_candidate is None else final_candidate.runs,
            fallback_action=search.fallback_action,
        )
        user_stopped = stop_event.is_set()
        final_path = None
        if not user_stopped or final_candidate is not None:
            final_path = _save_policy(
                final_policy,
                target_policy_path,
                force=args.overwrite,
            )
        elapsed = time.perf_counter() - started_at
        final_row = _metric_row(search, elapsed=elapsed, accepted=accepted)
        final_row["accepted_lane"] = accepted_lane
        final_row["first_success_step"] = first_success_step
        final_row["budget_exhausted"] = search.global_step >= args.transitions
        final_row["stopped_on_completion"] = stopped_on_completion
        final_row["phase"] = "final"
        final_row["best_program_steps"] = final_policy.step_count
        final_row["best_program_runs"] = final_policy.run_count
        stop_reason = (
            "user"
            if user_stopped
            else "success"
            if accepted
            else "budget"
        )
        final_row["stop_reason"] = stop_reason
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final_row, sort_keys=True) + "\n")
    finally:
        task.close()

    play_command = (
        None
        if final_path is None
        else _play_command(
            args.state,
            final_path,
            default_output=args.output is None,
            rom_path=args.rom,
        )
    )
    result = TrainingResult(
        algorithm="JERK",
        stop_reason=stop_reason,
        exit_code=130 if stop_reason == "user" else 0 if accepted else 1,
        accepted=accepted,
        elapsed=elapsed,
        timesteps=search.global_step,
        episodes=search.completed_episodes,
        final_row=final_row,
        policy_path=final_path,
        play_command=play_command,
        error_message=(
            None
            if accepted or stop_reason == "user"
            else f"JERK exhausted {args.transitions} transitions without a {args.state} success event"
        ),
    )
    reporter.update(
        training_ui.snapshot_from_row(
            algorithm="JERK",
            state=args.state,
            seed=args.seed,
            lanes=args.lanes,
            stop_rule=initial_snapshot.stop_rule,
            output=target_policy_path,
            total_timesteps=args.transitions,
            row={**final_row, "elapsed": elapsed},
            status="Stopped" if stop_reason == "user" else "Complete",
        ),
        TrainingEvent(
            "stop" if stop_reason == "user" else "complete",
            "Training stopped safely" if stop_reason == "user" else "Training finished",
            elapsed,
        ),
        force=True,
    )
    return result


def _validate_args(args: argparse.Namespace) -> None:
    if (
        min(
            args.transitions,
            args.lanes,
            args.max_episode_steps,
            args.max_prefix_shorten_runs,
            args.run_duration_max,
            args.retained_limit,
            args.log_every,
        )
        <= 0
    ):
        raise SystemExit("JERK training sizes must be positive")
    if args.transitions % args.lanes:
        raise SystemExit("--transitions must be divisible by --lanes")
    if (
        args.stall_steps < 0
        or args.checkpoint_every < 0
        or args.protected_prefix_runs < 0
        or args.step_cost < 0.0
    ):
        raise SystemExit("JERK non-negative sizes must not be negative")
    if args.run_duration_mean < 1.0:
        raise SystemExit("--run-duration-mean must be at least one")
    if not 0.0 <= args.deep_mutation_probability <= 1.0:
        raise SystemExit("--deep-mutation-probability must be in [0, 1]")


def _apply_algorithm_defaults(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    jerk_defaults = {
        "archive_replay_probability_initial": ARCHIVE_REPLAY_PROBABILITY_INITIAL,
        "archive_replay_probability_max": ARCHIVE_REPLAY_PROBABILITY_MAX,
        "max_prefix_shorten_runs": MAX_PREFIX_SHORTEN_RUNS,
        "deep_mutation_probability": DEEP_MUTATION_PROBABILITY,
        "retained_limit": RETAINED_LIMIT,
    }
    from .beam_training import (
        BEAM_REFRESH_EPISODES,
        BEAM_WIDTH,
        BRANCH_DURATIONS,
        MUTATION_RUNS,
    )

    beam_defaults = {
        "beam_width": BEAM_WIDTH,
        "beam_refresh_episodes": BEAM_REFRESH_EPISODES,
        "mutation_runs": MUTATION_RUNS,
        "improvement_protected_prefix_runs": 0,
        "branch_durations": list(BRANCH_DURATIONS),
    }
    selected = jerk_defaults if args.algorithm == "jerk" else beam_defaults
    rejected = beam_defaults if args.algorithm == "jerk" else jerk_defaults
    invalid = [name for name in rejected if getattr(args, name) is not None]
    if invalid:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in invalid)
        parser.error(f"{flags} cannot be used with --algorithm {args.algorithm}")
    for name, value in selected.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        args.state = resolve_state_name(args.state, state_dir=args.state_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _apply_algorithm_defaults(parser, args)
    if args.algorithm == "beam":
        from .beam_training import run

        return run(args, parser)
    _validate_args(args)
    try:
        ui_mode = training_ui.resolve_ui_mode(args.ui)
    except ValueError as exc:
        parser.error(str(exc))

    run_dir = args.output or run_directory_for_state(args.state)
    _protect_existing_policies(run_dir, force=args.overwrite)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes.jsonl").write_text("", encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stop_event = threading.Event()
    if ui_mode == "tui":
        try:
            result = training_ui.run_training_app(
                _initial_snapshot(args),
                lambda reporter, shared_stop: _run_training(
                    args, reporter, shared_stop
                ),
                stop_event=stop_event,
            )
        except BaseException as error:
            training_ui.report_failure_traceback(error)
            return 1
    else:
        with training_ui.safe_sigint(stop_event):
            result = _run_training(args, PlainReporter(LOGGER), stop_event)

    training_ui.print_summary(result, LOGGER)
    if result.error_message is not None:
        raise RuntimeError(result.error_message)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
