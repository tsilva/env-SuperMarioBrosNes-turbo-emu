"""Train an action-run policy with Go-Explore trajectory discovery."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from typing import Any

import numpy as np

from . import ACTION_SETS
from .go_explore import GoExploreSearch, SUCCESS_GUIDED_RESTORE_PROBABILITY
from .jerk import JerkPolicy, run_directory_for_state
from . import training_ui
from .training_ui import (
    PlainReporter,
    TrainingEvent,
    TrainingReporter,
    TrainingResult,
    TrainingSnapshot,
)
from .training import (
    GO_EXPLORE_CELL_FRAME_SHAPE,
    GO_EXPLORE_CELL_HUD_MASK,
    GO_EXPLORE_CELL_KEY_BYTES,
    GO_EXPLORE_CELL_QUANTIZATION_BITS,
    GO_EXPLORE_CELL_REPRESENTATION,
    MarioJerkTask,
    REWARD_MODE_SCORE_FIRST,
    _play_command,
    _protect_existing_policies,
    _save_policy,
)


LOGGER = logging.getLogger("go_explore_train")
GO_EXPLORE_EXPLORE_STEPS = 128
ARCHIVE_GROWTH_EVENT_INTERVAL = 1_000


def _policy(search: GoExploreSearch) -> JerkPolicy:
    policy = search.policy()
    policy.metadata.update(
        {
            "cell_representation": GO_EXPLORE_CELL_REPRESENTATION,
            "cell_frame_shape": list(GO_EXPLORE_CELL_FRAME_SHAPE),
            "cell_quantization_bits": GO_EXPLORE_CELL_QUANTIZATION_BITS,
            "cell_encoding": "raw-bytes",
            "cell_key_bytes": GO_EXPLORE_CELL_KEY_BYTES,
            "success_guided_restore_probability": (
                SUCCESS_GUIDED_RESTORE_PROBABILITY
            ),
        }
    )
    return policy


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **vars(args),
        "go_explore_cell_representation": GO_EXPLORE_CELL_REPRESENTATION,
        "go_explore_cell_frame_shape": list(GO_EXPLORE_CELL_FRAME_SHAPE),
        "go_explore_cell_hud_mask": list(GO_EXPLORE_CELL_HUD_MASK),
        "go_explore_cell_quantization_bits": GO_EXPLORE_CELL_QUANTIZATION_BITS,
        "go_explore_cell_encoding": "raw-bytes",
        "go_explore_cell_key_bytes": GO_EXPLORE_CELL_KEY_BYTES,
        "go_explore_success_guided_restore_probability": (
            SUCCESS_GUIDED_RESTORE_PROBABILITY
        ),
    }


def _metric_row(
    search: GoExploreSearch, *, elapsed: float, accepted: bool
) -> dict[str, Any]:
    candidate = search.best_candidate()
    return {
        "algorithm": "go-explore",
        "cell_representation": GO_EXPLORE_CELL_REPRESENTATION,
        "timesteps": search.global_step,
        "episodes": search.completed_episodes,
        "archive_count": search.archive_count,
        "archive_selection_count": search.archive_selection_count,
        "archive_visit_count": search.archive_visit_count,
        "archive_update_count": search.archive_update_count,
        "archive_memory_bytes": search.archive_memory_bytes,
        "archive_recent_new_cell_rate": search.archive_recent_new_cell_rate,
        "archive_recent_visit_window": search.archive_recent_visit_window,
        "archive_visits_per_cell": search.archive_visits_per_cell,
        "success_guided_cell_count": search.success_guided_cell_count,
        "success_guided_selection_count": search.success_guided_selection_count,
        "retained_count": search.retained_count,
        "locked_count": search.locked_count,
        "incomplete_retained_count": search.incomplete_retained_count,
        "successful_episodes": search.successful_episodes,
        "best_program_steps": candidate.step_count if candidate else 0,
        "best_program_runs": len(candidate.runs) if candidate else 0,
        "best_mean_reward": candidate.mean_return if candidate else 0.0,
        "best_progress": candidate.progress if candidate else 0.0,
        "best_completed": candidate.completed if candidate else False,
        "first_success_reward": search.first_success_return,
        "best_success_reward": search.best_success_return,
        "improvement_count": search.improvement_count,
        "accepted": accepted,
        "loop_fps": search.global_step / max(elapsed, 1e-9),
    }


def _initial_snapshot(args: argparse.Namespace) -> TrainingSnapshot:
    run_dir = args.output or run_directory_for_state(args.state)
    return TrainingSnapshot(
        algorithm="Go-Explore",
        state=args.state,
        seed=args.seed,
        lanes=args.lanes,
        stop_rule=(
            "first completion"
            if not args.continue_after_completion
            else "transition budget"
        ),
        output=run_dir / f"{args.state}.zip",
        total_timesteps=args.transitions,
        action_set=f"{args.action_set} ({len(ACTION_SETS[args.action_set])} actions)",
    )


def _run_training(
    args: argparse.Namespace,
    reporter: TrainingReporter,
    stop_event: threading.Event,
) -> TrainingResult:
    run_dir = args.output or run_directory_for_state(args.state)
    policy_path = run_dir / f"{args.state}.zip"
    metrics_path = run_dir / "episodes.jsonl"
    successes_path = run_dir / "successes.jsonl"
    action_names = tuple(ACTION_SETS[args.action_set])
    search = GoExploreSearch(
        n_envs=args.lanes,
        seed=args.seed,
        action_names=action_names,
        fallback_action=args.fallback_action,
        explore_steps=args.go_explore_explore_steps,
        run_duration_mean=args.run_duration_mean,
        run_duration_max=args.run_duration_max,
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
        action_set=args.action_set,
        reward_mode=REWARD_MODE_SCORE_FIRST,
        visual_cell_observations=True,
    )
    started_at = time.perf_counter()
    next_log = args.log_every
    next_checkpoint = args.checkpoint_every if args.checkpoint_every > 0 else None
    accepted = False
    accepted_lane: int | None = None
    first_success_step: int | None = None
    stopped_on_completion = False
    last_archive_event_count = 0
    last_best: tuple[bool, float, float] | None = None
    initial_snapshot = _initial_snapshot(args)
    last_ui_publish = float("-inf")
    reporter.start(initial_snapshot)
    try:
        observations = task.reset()
        all_lanes = np.ones(args.lanes, dtype=np.bool_)
        search.initialize(
            task.go_explore_cell_keys(observations),
            task.capture_snapshots(all_lanes),
        )
        last_archive_event_count = search.archive_count
        while search.global_step < args.transitions and not stop_event.is_set():
            actions = search.next_actions()
            observations, rewards, failure_dones, records, successes = task.step(
                actions
            )
            search_dones = failure_dones | successes
            observation = search.observe(
                rewards,
                search_dones,
                task.go_explore_cell_keys(observations),
                records,
                progresses=getattr(task, "max_global_x", None),
            )
            if np.any(observation.archive_mask):
                search.commit_archive(task.capture_snapshots(observation.archive_mask))
            completion_events = search.take_completion_events()
            step = search.global_step

            for completion in completion_events:
                success_row = {
                    "timesteps": step,
                    "episode_return": completion.episode_return,
                    "progress": completion.progress,
                    "improved": completion.improved,
                    "action_runs": [
                        [run.action, run.duration] for run in completion.runs
                    ],
                }
                with successes_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(success_row, sort_keys=True) + "\n")
                if completion.improved:
                    _save_policy(_policy(search), policy_path, force=True)

            if completion_events:
                if not accepted:
                    accepted = True
                    first_success_step = step
                    accepted_lane = int(np.flatnonzero(successes)[0])
                    accepted_path = run_dir / "checkpoints" / f"{args.state}-{step}.zip"
                    _save_policy(_policy(search), accepted_path, force=args.overwrite)
                    elapsed = time.perf_counter() - started_at
                    success_row = _metric_row(
                        search, elapsed=elapsed, accepted=accepted
                    )
                    reporter.update(
                        training_ui.snapshot_from_row(
                            algorithm="Go-Explore",
                            state=args.state,
                            seed=args.seed,
                            lanes=args.lanes,
                            stop_rule=initial_snapshot.stop_rule,
                            output=policy_path,
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
                if not args.continue_after_completion:
                    stopped_on_completion = True
                    break

            if np.any(observation.restart_mask):
                task.restore_lanes(
                    observation.restart_mask,
                    search.restart(observation.restart_mask),
                )

            elapsed = time.perf_counter() - started_at
            events: list[TrainingEvent] = []
            if first_success_step != step:
                for completion in completion_events:
                    if completion.improved:
                        events.append(
                            TrainingEvent(
                                "new-best-success",
                                "Completed path improved: "
                                f"return {completion.episode_return:,.1f}",
                                elapsed,
                            )
                        )
            if (
                search.archive_count - last_archive_event_count
                >= ARCHIVE_GROWTH_EVENT_INTERVAL
            ):
                events.append(
                    TrainingEvent(
                        "archive-growth",
                        f"Archive expanded: {search.archive_count:,} cells",
                        elapsed,
                    )
                )
                last_archive_event_count = search.archive_count
            if records:
                candidate = search.best_candidate()
                current_best = (
                    False if candidate is None else candidate.completed,
                    0.0 if candidate is None else candidate.progress,
                    0.0 if candidate is None else candidate.mean_return,
                )
                if candidate is not None and current_best != last_best:
                    events.append(
                        TrainingEvent(
                            "new-best",
                            f"Best path updated: return {candidate.mean_return:,.1f} "
                            f"· x {candidate.progress:,.0f}",
                            elapsed,
                        )
                    )
                    last_best = current_best

            now = time.monotonic()
            routine_due = now - last_ui_publish >= training_ui.UPDATE_INTERVAL_SECONDS
            log_due = step >= next_log
            checkpoint_due = next_checkpoint is not None and step >= next_checkpoint
            ui_row = None
            snapshot = None
            if events or routine_due or log_due or checkpoint_due:
                ui_row = _metric_row(search, elapsed=elapsed, accepted=accepted)
                snapshot = training_ui.snapshot_from_row(
                    algorithm="Go-Explore",
                    state=args.state,
                    seed=args.seed,
                    lanes=args.lanes,
                    stop_rule=initial_snapshot.stop_rule,
                    output=policy_path,
                    total_timesteps=args.transitions,
                    row={**ui_row, "elapsed": elapsed},
                )
                if events:
                    for event in events:
                        reporter.update(snapshot, event, force=True)
                else:
                    reporter.update(snapshot)
                last_ui_publish = now
            if log_due:
                assert ui_row is not None and snapshot is not None
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(ui_row, sort_keys=True) + "\n")
                if isinstance(reporter, PlainReporter):
                    reporter.update(snapshot, force=True)
                while next_log <= step:
                    next_log += args.log_every

            while next_checkpoint is not None and step >= next_checkpoint:
                assert snapshot is not None
                checkpoint_path = _save_policy(
                    _policy(search),
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

        candidate = search.best_candidate()
        final_policy = _policy(search)
        user_stopped = stop_event.is_set()
        final_path = None
        if not user_stopped or candidate is not None:
            final_path = _save_policy(final_policy, policy_path, force=True)
        elapsed = time.perf_counter() - started_at
        final_row = _metric_row(search, elapsed=elapsed, accepted=accepted)
        stop_reason = "user" if user_stopped else "success" if accepted else "budget"
        final_row.update(
            {
                "accepted_lane": accepted_lane,
                "first_success_step": first_success_step,
                "first_success_reward": search.first_success_return,
                "best_success_reward": search.best_success_return,
                "improvement_count": search.improvement_count,
                "budget_exhausted": search.global_step >= args.transitions,
                "stopped_on_completion": stopped_on_completion,
                "phase": "final",
                "best_program_steps": final_policy.step_count,
                "best_program_runs": final_policy.run_count,
                "stop_reason": stop_reason,
            }
        )
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
            default_output=False,
            rom_path=args.rom,
            action_set=args.action_set,
        )
    )
    result = TrainingResult(
        algorithm="Go-Explore",
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
            else f"Go-Explore exhausted {args.transitions} transitions without "
            f"a {args.state} success event"
        ),
        extra_summary_rows=(
            ("Archive cells", f"{search.archive_count:,}"),
            (
                "Archive memory",
                training_ui.format_byte_size(search.archive_memory_bytes),
            ),
        ),
    )
    reporter.update(
        training_ui.snapshot_from_row(
            algorithm="Go-Explore",
            state=args.state,
            seed=args.seed,
            lanes=args.lanes,
            stop_rule=initial_snapshot.stop_rule,
            output=policy_path,
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
            args.log_every,
            args.run_duration_max,
            args.go_explore_explore_steps,
        )
        <= 0
    ):
        raise SystemExit("Go-Explore training sizes must be positive")
    if args.transitions % args.lanes:
        raise SystemExit("--transitions must be divisible by --lanes")
    if (
        args.stall_steps < 0
        or args.checkpoint_every < 0
        or args.protected_prefix_runs < 0
        or (args.step_cost is not None and args.step_cost < 0)
    ):
        raise SystemExit("Go-Explore non-negative sizes must not be negative")
    if args.run_duration_mean < 1.0:
        raise SystemExit("--run-duration-mean must be at least one")


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    _validate_args(args)
    try:
        ui_mode = training_ui.resolve_ui_mode(args.ui)
    except ValueError as exc:
        parser.error(str(exc))

    run_dir = args.output or run_directory_for_state(args.state)
    _protect_existing_policies(run_dir, force=args.overwrite)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes.jsonl").write_text("", encoding="utf-8")
    (run_dir / "successes.jsonl").write_text("", encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(_run_config(args), default=str, indent=2, sort_keys=True) + "\n",
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
    if result.error_message:
        LOGGER.error(result.error_message)
    return result.exit_code
