#!/usr/bin/env python3
"""Train rlab-compatible JERK on a named Super Mario Bros NES level."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np

from supermariobrosnes_turbo import (
    ACTION_SETS,
    SuperMarioBrosNesTurboVecEnv,
    list_available_states,
)
from supermariobrosnes_turbo.jerk import (
    JerkPolicy,
    JerkSearch,
    normalize_level_name,
    policy_path_for_level,
    run_directory_for_level,
)


LOGGER = logging.getLogger("jerk_train")
ACTION_SET = "simple"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16
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
    """Vectorized level task matching rlab's JERK reward and failure contract."""

    def __init__(
        self,
        *,
        level: str,
        rom_path: str | Path | None,
        seed: int,
        n_envs: int,
        max_episode_steps: int,
        stall_steps: int,
        step_cost: float,
    ) -> None:
        self.n_envs = int(n_envs)
        self.level = normalize_level_name(level)
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.stall_steps = int(stall_steps)
        self.step_cost = float(step_cost)
        if self.step_cost < 0.0:
            raise ValueError("JERK step_cost must be non-negative")
        self.native = SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            state=self.level,
            num_envs=self.n_envs,
            num_threads=self.n_envs,
            rom_path=rom_path,
            render_mode="rgb_array",
            action_set=ACTION_SET,
            obs_copy="unsafe_view",
            obs_resize=(1, 1),
            obs_grayscale=True,
            obs_layout="chw",
            frame_skip=4,
            frame_stack=1,
            maxpool_last_two=False,
            sticky_action_prob=0.0,
            reward_clip=False,
            info_filter="all",
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


def _save_policy(policy: JerkPolicy, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.zip"
    policy.save(temporary)
    temporary.replace(path)
    return path


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", help="Packaged level state, for example Level1-1")
    parser.add_argument(
        "--rom", type=Path, help="ROM path; defaults to Stable Retro-compatible discovery"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Run directory; defaults to runs/<Level>-jerk",
    )
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--stall-steps", type=int, default=STALL_STEPS)
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    parser.add_argument("--log-interval-steps", type=int, default=LOG_INTERVAL_STEPS)
    parser.add_argument(
        "--archive-replay-probability-initial",
        type=float,
        default=ARCHIVE_REPLAY_PROBABILITY_INITIAL,
    )
    parser.add_argument(
        "--archive-replay-probability-max",
        type=float,
        default=ARCHIVE_REPLAY_PROBABILITY_MAX,
    )
    parser.add_argument(
        "--protected-prefix-runs", type=int, default=PROTECTED_PREFIX_RUNS
    )
    parser.add_argument(
        "--max-prefix-shorten-runs", type=int, default=MAX_PREFIX_SHORTEN_RUNS
    )
    parser.add_argument(
        "--deep-mutation-probability",
        type=float,
        default=DEEP_MUTATION_PROBABILITY,
    )
    parser.add_argument("--run-duration-mean", type=float, default=RUN_DURATION_MEAN)
    parser.add_argument("--run-duration-max", type=int, default=RUN_DURATION_MAX)
    parser.add_argument("--retained-limit", type=int, default=RETAINED_LIMIT)
    parser.add_argument("--fallback-action", default=FALLBACK_ACTION)
    parser.add_argument("--step-cost", type=float, default=STEP_COST)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        args.level = normalize_level_name(args.level)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    available_states = set(list_available_states())
    if args.level not in available_states:
        choices = ", ".join(sorted(available_states))
        raise SystemExit(
            f"unknown packaged level {args.level!r}; available states: {choices}"
        )
    if (
        min(
            args.timesteps,
            args.n_envs,
            args.max_episode_steps,
            args.max_prefix_shorten_runs,
            args.run_duration_max,
            args.retained_limit,
            args.log_interval_steps,
        )
        <= 0
    ):
        raise SystemExit("JERK training sizes must be positive")
    if args.timesteps % args.n_envs:
        raise SystemExit("--timesteps must be divisible by --n-envs")
    if (
        args.stall_steps < 0
        or args.checkpoint_freq < 0
        or args.protected_prefix_runs < 0
        or args.step_cost < 0.0
    ):
        raise SystemExit("JERK non-negative sizes must not be negative")
    if args.run_duration_mean < 1.0:
        raise SystemExit("--run-duration-mean must be at least one")
    if not 0.0 <= args.deep_mutation_probability <= 1.0:
        raise SystemExit("--deep-mutation-probability must be in [0, 1]")

    run_dir = args.output or run_directory_for_level(args.level)
    target_policy_path = (
        policy_path_for_level(args.level)
        if args.output is None
        else run_dir / f"{args.level}.zip"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "episodes.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    action_names = tuple(ACTION_SETS[ACTION_SET])
    search = JerkSearch(
        n_envs=args.n_envs,
        seed=args.seed,
        total_timesteps=args.timesteps,
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
        level=args.level,
        rom_path=args.rom,
        seed=args.seed,
        n_envs=args.n_envs,
        max_episode_steps=args.max_episode_steps,
        stall_steps=args.stall_steps,
        step_cost=args.step_cost,
    )
    task.reset()
    started_at = time.perf_counter()
    next_log = args.log_interval_steps
    next_checkpoint = args.checkpoint_freq if args.checkpoint_freq > 0 else None
    accepted = False
    accepted_lane: int | None = None
    first_success_step: int | None = None
    try:
        while search.global_step < args.timesteps:
            actions = search.next_actions()
            _observations, rewards, failure_dones, records, successes = task.step(
                actions
            )
            search_dones = failure_dones | successes
            search.observe(rewards, search_dones, records)
            step = search.global_step

            if np.any(successes):
                if not accepted:
                    accepted = True
                    first_success_step = step
                    accepted_lane = int(np.flatnonzero(successes)[0])
                    accepted_path = (
                        run_dir / "checkpoints" / f"{args.level}-{step}.zip"
                    )
                    _save_policy(search.policy(), accepted_path)
                    LOGGER.info(
                        "locked first %s success step=%d lane=%d path=%s",
                        args.level,
                        step,
                        accepted_lane,
                        accepted_path,
                    )

            task.reset_lanes(search_dones)

            if step >= next_log:
                row = _metric_row(
                    search, elapsed=time.perf_counter() - started_at, accepted=accepted
                )
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                LOGGER.info(
                    "timesteps=%d episodes=%d retained=%d locked=%d replay=%.3f "
                    "best=%.1f progress=%.0f program=%d/%d steps/runs fps=%.0f",
                    step,
                    search.completed_episodes,
                    search.retained_count,
                    search.locked_count,
                    search.archive_replay_probability,
                    row["best_mean_reward"],
                    row["best_progress"],
                    row["best_program_steps"],
                    row["best_program_runs"],
                    row["loop_fps"],
                )
                while next_log <= step:
                    next_log += args.log_interval_steps

            while next_checkpoint is not None and step >= next_checkpoint:
                _save_policy(
                    search.policy(),
                    run_dir / "checkpoints" / f"{args.level}-{step}.zip",
                )
                next_checkpoint += args.checkpoint_freq

        final_candidate = search.best_candidate()
        final_policy = JerkPolicy(
            action_names=search.action_names,
            action_runs=() if final_candidate is None else final_candidate.runs,
            fallback_action=search.fallback_action,
        )
        final_path = _save_policy(
            final_policy,
            target_policy_path,
        )
        final_row = _metric_row(
            search, elapsed=time.perf_counter() - started_at, accepted=accepted
        )
        final_row["accepted_lane"] = accepted_lane
        final_row["first_success_step"] = first_success_step
        final_row["budget_exhausted"] = search.global_step >= args.timesteps
        final_row["phase"] = "final"
        final_row["best_program_steps"] = final_policy.step_count
        final_row["best_program_runs"] = final_policy.run_count
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final_row, sort_keys=True) + "\n")
    finally:
        task.close()

    LOGGER.info(
        "saved=%s timesteps=%d episodes=%d retained=%d locked=%d accepted=%s",
        final_path,
        search.global_step,
        search.completed_episodes,
        search.retained_count,
        search.locked_count,
        accepted,
    )
    if not accepted and search.global_step >= args.timesteps:
        raise RuntimeError(
            f"JERK exhausted {args.timesteps} transitions without a {args.level} success event"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
