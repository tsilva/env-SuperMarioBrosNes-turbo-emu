"""Sequential training across every canonical Super Mario Bros level."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import logging
from pathlib import Path
import threading
from types import ModuleType

from . import training_ui
from .jerk import run_directory_for_state
from .training_ui import (
    PlainReporter,
    TrainingEvent,
    TrainingReporter,
    TrainingResult,
    TrainingSnapshot,
)


LOGGER = logging.getLogger("training_campaign")
CANONICAL_LEVEL_STATES = tuple(
    f"Level{world}-{level}" for world in range(1, 9) for level in range(1, 5)
)


class CampaignReporter:
    """Add campaign position to snapshots from an unchanged single-level trainer."""

    def __init__(
        self,
        reporter: TrainingReporter,
        *,
        index: int,
        total: int,
    ) -> None:
        self.reporter = reporter
        self.index = int(index)
        self.total = int(total)
        self.last_snapshot: TrainingSnapshot | None = None

    def _campaign_snapshot(self, snapshot: TrainingSnapshot) -> TrainingSnapshot:
        enriched = replace(
            snapshot,
            campaign_index=self.index,
            campaign_total=self.total,
            campaign_completed=self.index - 1,
        )
        self.last_snapshot = enriched
        return enriched

    def start(self, snapshot: TrainingSnapshot) -> None:
        self.reporter.start(self._campaign_snapshot(snapshot))

    def update(
        self,
        snapshot: TrainingSnapshot,
        event: TrainingEvent | None = None,
        *,
        force: bool = False,
    ) -> None:
        self.reporter.update(
            self._campaign_snapshot(snapshot),
            event,
            force=force,
        )

    def finish(self, result: TrainingResult) -> None:
        if self.last_snapshot is None:
            return
        finished = replace(
            self.last_snapshot,
            campaign_completed=self.index,
            status=(
                "Stopped"
                if result.stop_reason == "user"
                else "Level completed"
                if result.accepted
                else "Level budget exhausted"
            ),
        )
        self.last_snapshot = finished
        self.reporter.update(
            finished,
            TrainingEvent(
                "complete",
                (
                    f"Campaign level {self.index}/{self.total} completed: "
                    f"{finished.state}"
                    if result.accepted
                    else f"Campaign level {self.index}/{self.total} finished "
                    f"without success: {finished.state}"
                ),
                result.elapsed,
            ),
            force=True,
        )


def _algorithm_module(algorithm: str) -> ModuleType:
    if algorithm == "beam":
        from . import beam_training

        return beam_training
    if algorithm == "go-explore":
        from . import go_explore_training

        return go_explore_training
    from . import training

    return training


def _level_args(args: argparse.Namespace, state: str) -> argparse.Namespace:
    level_args = copy.copy(args)
    level_args.state = state
    if args.output is not None:
        level_args.output = Path(args.output) / state
    return level_args


def _force_existing(module: ModuleType, args: argparse.Namespace) -> bool:
    del module
    from .training import _force_policy_overwrite

    return _force_policy_overwrite(args)


def _preflight(
    args: argparse.Namespace,
    states: tuple[str, ...],
    module: ModuleType,
) -> None:
    if args.initial_policy is not None:
        raise SystemExit(
            "--initial-policy requires an explicit state and cannot seed an all-level campaign"
        )
    module._validate_args(_level_args(args, states[0]))
    from .training import _protect_existing_policies

    for state in states:
        level_args = _level_args(args, state)
        run_dir = level_args.output or run_directory_for_state(state)
        _protect_existing_policies(
            run_dir,
            force=_force_existing(module, level_args),
        )


def _prepare_level(
    args: argparse.Namespace,
    *,
    include_successes: bool,
) -> None:
    run_dir = args.output or run_directory_for_state(args.state)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes.jsonl").write_text("", encoding="utf-8")
    if include_successes:
        (run_dir / "successes.jsonl").write_text("", encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _campaign_result(
    results: list[TrainingResult],
    states: tuple[str, ...],
) -> TrainingResult:
    if not results:
        raise RuntimeError("all-level training campaign produced no results")
    last = results[-1]
    succeeded = sum(result.accepted for result in results)
    failed_states = [
        state
        for state, result in zip(states, results)
        if not result.accepted and result.stop_reason != "user"
    ]
    user_stopped = last.stop_reason == "user"
    all_succeeded = len(results) == len(states) and succeeded == len(states)
    stop_reason = "user" if user_stopped else "success" if all_succeeded else "budget"
    final_row = dict(last.final_row)
    final_row.update(
        {
            "campaign_total": len(states),
            "campaign_processed": len(results),
            "campaign_succeeded": succeeded,
            "campaign_failed_states": failed_states,
        }
    )
    return TrainingResult(
        algorithm=f"{last.algorithm} campaign",
        stop_reason=stop_reason,
        exit_code=130 if user_stopped else 0 if all_succeeded else 1,
        accepted=all_succeeded,
        elapsed=sum(result.elapsed for result in results),
        timesteps=sum(result.timesteps for result in results),
        episodes=sum(result.episodes for result in results),
        final_row=final_row,
        policy_path=None,
        play_command=None,
        error_message=(
            None
            if all_succeeded or user_stopped
            else "levels without a successful trajectory: " + ", ".join(failed_states)
        ),
        extra_summary_rows=(
            ("Levels processed", f"{len(results):,} / {len(states):,}"),
            ("Levels completed", f"{succeeded:,} / {len(states):,}"),
        ),
    )


def _execute(
    args: argparse.Namespace,
    states: tuple[str, ...],
    module: ModuleType,
    reporter: TrainingReporter,
    stop_event: threading.Event,
) -> TrainingResult:
    results: list[TrainingResult] = []
    total = len(states)
    for index, state in enumerate(states, start=1):
        level_args = _level_args(args, state)
        _prepare_level(
            level_args,
            include_successes=args.algorithm in {"beam", "go-explore"},
        )
        campaign_reporter = CampaignReporter(
            reporter,
            index=index,
            total=total,
        )
        result = module._run_training(level_args, campaign_reporter, stop_event)
        results.append(result)
        campaign_reporter.finish(result)
        if isinstance(reporter, PlainReporter):
            training_ui.print_summary(result, LOGGER)
        if result.stop_reason == "user":
            break
    return _campaign_result(results, states)


def _print_summary(result: TrainingResult, output: Path | None) -> None:
    row = result.final_row
    rows = [
        (
            "Levels processed",
            f"{int(row['campaign_processed']):,} / {int(row['campaign_total']):,}",
        ),
        (
            "Levels completed",
            f"{int(row['campaign_succeeded']):,} / {int(row['campaign_total']):,}",
        ),
        ("Transitions", f"{result.timesteps:,}"),
        ("Elapsed", training_ui.format_elapsed(result.elapsed)),
        ("Policies", str(output or Path("runs"))),
    ]
    LOGGER.info(
        "\n%s",
        training_ui.format_box(f"{result.algorithm} complete", rows),
    )
    if result.error_message:
        LOGGER.error(result.error_message)


def run(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    states: tuple[str, ...] = CANONICAL_LEVEL_STATES,
) -> int:
    module = _algorithm_module(args.algorithm)
    _preflight(args, states, module)
    try:
        ui_mode = training_ui.resolve_ui_mode(args.ui)
    except ValueError as exc:
        parser.error(str(exc))

    first_args = _level_args(args, states[0])
    initial = replace(
        module._initial_snapshot(first_args),
        campaign_index=1,
        campaign_total=len(states),
        campaign_completed=0,
    )
    stop_event = threading.Event()
    if ui_mode == "tui":
        try:
            result = training_ui.run_training_app(
                initial,
                lambda reporter, shared_stop: _execute(
                    args,
                    states,
                    module,
                    reporter,
                    shared_stop,
                ),
                stop_event=stop_event,
            )
        except BaseException as error:
            training_ui.report_failure_traceback(error)
            return 1
    else:
        with training_ui.safe_sigint(stop_event):
            result = _execute(
                args,
                states,
                module,
                PlainReporter(LOGGER),
                stop_event,
            )

    _print_summary(result, args.output)
    return result.exit_code
