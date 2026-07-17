#!/usr/bin/env python3
"""Train an action-run policy with beam search on a named Mario level."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from supermariobrosnes_turbo import ACTION_SETS, list_available_states
from supermariobrosnes_turbo.beam import BeamSearch
from supermariobrosnes_turbo.jerk import JerkPolicy, normalize_level_name
from train import (
    ACTION_SET,
    FALLBACK_ACTION,
    LOG_INTERVAL_STEPS,
    MAX_EPISODE_STEPS,
    N_ENVS,
    PROTECTED_PREFIX_RUNS,
    RUN_DURATION_MAX,
    RUN_DURATION_MEAN,
    STALL_STEPS,
    STEP_COST,
    TOTAL_TIMESTEPS,
    MarioJerkTask,
    _format_box,
    _format_elapsed,
    _format_progress,
    _play_command,
    _protect_existing_policies,
    _save_policy,
)


LOGGER = logging.getLogger("beam_train")
BEAM_WIDTH = 16
BEAM_REFRESH_EPISODES = N_ENVS
MUTATION_RUNS = 8
BRANCH_DURATIONS = (1, 2, 4, 8, 16, 32)


def run_directory_for_level(level: str, *, runs_root: str | Path = "runs") -> Path:
    return Path(runs_root) / f"{normalize_level_name(level)}-beam"


def _metric_row(
    search: BeamSearch, *, elapsed: float, accepted: bool
) -> dict[str, Any]:
    candidate = search.best_candidate()
    return {
        "algorithm": "beam",
        "timesteps": search.global_step,
        "episodes": search.completed_episodes,
        "generation": search.generation,
        "beam_count": search.beam_count,
        "pending_count": search.pending_count,
        "retained_count": search.retained_count,
        "locked_count": search.locked_count,
        "incomplete_retained_count": search.incomplete_retained_count,
        "successful_episodes": search.successful_episodes,
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
        "--rom",
        type=Path,
        help="ROM path; defaults to Stable Retro-compatible discovery",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Run directory; defaults to runs/<Level>-beam",
    )
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--stall-steps", type=int, default=STALL_STEPS)
    parser.add_argument("--log-interval-steps", type=int, default=LOG_INTERVAL_STEPS)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument(
        "--beam-refresh-episodes", type=int, default=BEAM_REFRESH_EPISODES
    )
    parser.add_argument(
        "--protected-prefix-runs", type=int, default=PROTECTED_PREFIX_RUNS
    )
    parser.add_argument("--mutation-runs", type=int, default=MUTATION_RUNS)
    parser.add_argument(
        "--branch-durations",
        type=int,
        nargs="+",
        default=list(BRANCH_DURATIONS),
        metavar="STEPS",
    )
    parser.add_argument("--run-duration-mean", type=float, default=RUN_DURATION_MEAN)
    parser.add_argument("--run-duration-max", type=int, default=RUN_DURATION_MAX)
    parser.add_argument("--fallback-action", default=FALLBACK_ACTION)
    parser.add_argument("--step-cost", type=float, default=STEP_COST)
    parser.add_argument(
        "--stop-on-completion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop after the first level completion (default: enabled)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow existing policies in the run directory to be replaced",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
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
    positive_sizes = (
        args.timesteps,
        args.n_envs,
        args.max_episode_steps,
        args.log_interval_steps,
        args.beam_width,
        args.beam_refresh_episodes,
        args.mutation_runs,
        args.run_duration_max,
        *args.branch_durations,
    )
    if min(positive_sizes) <= 0:
        raise SystemExit("beam training sizes must be positive")
    if args.timesteps % args.n_envs:
        raise SystemExit("--timesteps must be divisible by --n-envs")
    if args.stall_steps < 0 or args.protected_prefix_runs < 0 or args.step_cost < 0:
        raise SystemExit("beam non-negative sizes must not be negative")
    if args.run_duration_mean < 1.0:
        raise SystemExit("--run-duration-mean must be at least one")

    run_dir = args.output or run_directory_for_level(args.level)
    policy_path = run_dir / f"{args.level}.zip"
    _protect_existing_policies(run_dir, force=args.force)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "episodes.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    action_names = tuple(ACTION_SETS[ACTION_SET])
    LOGGER.info(
        "\n%s",
        _format_box(
            "Beam training",
            [
                ("Level", args.level),
                ("Action set", f"{ACTION_SET} ({len(action_names)} actions)"),
                ("Parallel lanes", f"{args.n_envs:,}"),
                ("Beam width", f"{args.beam_width:,}"),
                ("Budget", f"{args.timesteps:,} transitions"),
                (
                    "Stop rule",
                    "first completion"
                    if args.stop_on_completion
                    else "transition budget",
                ),
                ("Policy", str(policy_path)),
            ],
        ),
    )
    search = BeamSearch(
        n_envs=args.n_envs,
        seed=args.seed,
        action_names=action_names,
        fallback_action=args.fallback_action,
        beam_width=args.beam_width,
        refresh_episodes=args.beam_refresh_episodes,
        protected_prefix_runs=args.protected_prefix_runs,
        mutation_runs=args.mutation_runs,
        branch_durations=args.branch_durations,
        run_duration_mean=args.run_duration_mean,
        run_duration_max=args.run_duration_max,
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
    accepted = False
    accepted_lane: int | None = None
    first_success_step: int | None = None
    stopped_on_completion = False
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
                    _save_policy(search.policy(), accepted_path, force=args.force)
                    LOGGER.info(
                        "\n%s",
                        _format_box(
                            "Level completed",
                            [
                                ("Level", args.level),
                                ("Transition", f"{step:,}"),
                                ("Lane", f"{accepted_lane:,}"),
                                ("Checkpoint", str(accepted_path)),
                            ],
                        ),
                    )
                if args.stop_on_completion:
                    stopped_on_completion = True
                    break

            task.reset_lanes(search_dones)
            if step >= next_log:
                row = _metric_row(
                    search, elapsed=time.perf_counter() - started_at, accepted=accepted
                )
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                LOGGER.info("%s", _format_progress(row, args.timesteps))
                while next_log <= step:
                    next_log += args.log_interval_steps

        candidate = search.best_candidate()
        final_policy = JerkPolicy(
            action_names=search.action_names,
            action_runs=() if candidate is None else candidate.runs,
            fallback_action=search.fallback_action,
            timesteps=search.global_step,
            episodes=search.completed_episodes,
            best_reward=0.0 if candidate is None else candidate.mean_return,
            metadata={
                "search_algorithm": "beam",
                "beam_width": args.beam_width,
                "generation": search.generation,
                "terminate_on_life_loss": True,
                "terminate_on_level_change": False,
            },
        )
        final_path = _save_policy(final_policy, policy_path, force=args.force)
        final_row = _metric_row(
            search, elapsed=time.perf_counter() - started_at, accepted=accepted
        )
        final_row.update(
            {
                "accepted_lane": accepted_lane,
                "first_success_step": first_success_step,
                "budget_exhausted": search.global_step >= args.timesteps,
                "stopped_on_completion": stopped_on_completion,
                "phase": "final",
                "best_program_steps": final_policy.step_count,
                "best_program_runs": final_policy.run_count,
            }
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final_row, sort_keys=True) + "\n")
    finally:
        task.close()

    elapsed = time.perf_counter() - started_at
    LOGGER.info(
        "\n%s",
        _format_box(
            "Beam training complete",
            [
                ("Result", "level completed" if accepted else "budget exhausted"),
                ("Elapsed", _format_elapsed(elapsed)),
                ("Transitions", f"{search.global_step:,}"),
                ("Episodes", f"{search.completed_episodes:,}"),
                ("Generations", f"{search.generation:,}"),
                ("Best reward", f"{final_row['best_mean_reward']:,.1f}"),
                ("Progress", f"{final_row['best_progress']:,.0f}"),
                (
                    "Policy size",
                    f"{final_policy.step_count:,} steps / "
                    f"{final_policy.run_count:,} runs",
                ),
                ("Saved", str(final_path)),
            ],
        ),
    )
    LOGGER.info(
        "\nPlay the policy:\n\n  %s\n",
        _play_command(
            args.level,
            final_path,
            default_output=False,
            rom_path=args.rom,
        ),
    )
    if not accepted and search.global_step >= args.timesteps:
        raise RuntimeError(
            f"beam search exhausted {args.timesteps} transitions without a "
            f"{args.level} success event"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
