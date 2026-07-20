from __future__ import annotations

import argparse
from pathlib import Path
import threading
from types import SimpleNamespace

from supermariobrosnes_turbo import training_campaign
from supermariobrosnes_turbo.training_ui import (
    TrainingEvent,
    TrainingResult,
    TrainingSnapshot,
)


class _CaptureReporter:
    def __init__(self) -> None:
        self.started: list[TrainingSnapshot] = []
        self.updated: list[tuple[TrainingSnapshot, TrainingEvent | None]] = []

    def start(self, snapshot: TrainingSnapshot) -> None:
        self.started.append(snapshot)

    def update(
        self,
        snapshot: TrainingSnapshot,
        event: TrainingEvent | None = None,
        *,
        force: bool = False,
    ) -> None:
        del force
        self.updated.append((snapshot, event))


def _snapshot(state: str, output: Path) -> TrainingSnapshot:
    return TrainingSnapshot(
        algorithm="Beam",
        state=state,
        seed=1,
        lanes=1,
        stop_rule="first completion",
        output=output,
        total_timesteps=10,
    )


def _result(state: str, output: Path, *, accepted: bool) -> TrainingResult:
    return TrainingResult(
        algorithm="Beam",
        stop_reason="success" if accepted else "budget",
        exit_code=0 if accepted else 1,
        accepted=accepted,
        elapsed=1.0,
        timesteps=2,
        episodes=1,
        final_row={
            "best_mean_reward": 1.0,
            "best_progress": 2.0,
            "best_program_steps": 1,
            "best_program_runs": 1,
        },
        policy_path=output / f"{state}.zip",
        play_command=None,
        error_message=None if accepted else "budget exhausted",
    )


def test_campaign_reporter_adds_and_completes_level_progress() -> None:
    captured = _CaptureReporter()
    reporter = training_campaign.CampaignReporter(
        captured,
        index=3,
        total=32,
    )
    snapshot = _snapshot("Level1-3", Path("runs/Level1-3/Level1-3.zip"))

    reporter.start(snapshot)
    reporter.finish(_result("Level1-3", Path("runs/Level1-3"), accepted=True))

    assert captured.started[0].campaign_index == 3
    assert captured.started[0].campaign_total == 32
    assert captured.started[0].campaign_completed == 2
    assert captured.updated[-1][0].campaign_completed == 3
    assert captured.updated[-1][1].kind == "complete"


def test_campaign_executes_every_level_in_isolated_output_directories(
    tmp_path: Path,
) -> None:
    states = ("Level1-1", "Level1-2")
    captured = _CaptureReporter()
    seen: list[tuple[str, Path]] = []

    def run_level(args, reporter, _stop_event):
        seen.append((args.state, args.output))
        reporter.start(_snapshot(args.state, args.output / f"{args.state}.zip"))
        return _result(
            args.state,
            args.output,
            accepted=args.state == "Level1-1",
        )

    module = SimpleNamespace(_run_training=run_level)
    args = argparse.Namespace(
        algorithm="beam",
        output=tmp_path,
    )

    result = training_campaign._execute(
        args,
        states,
        module,
        captured,
        threading.Event(),
    )

    assert seen == [
        ("Level1-1", tmp_path / "Level1-1"),
        ("Level1-2", tmp_path / "Level1-2"),
    ]
    assert all((tmp_path / state / "episodes.jsonl").exists() for state in states)
    assert result.exit_code == 1
    assert result.final_row["campaign_processed"] == 2
    assert result.final_row["campaign_succeeded"] == 1
    assert result.final_row["campaign_failed_states"] == ["Level1-2"]


def test_campaign_stops_before_starting_the_next_level_after_safe_stop(
    tmp_path: Path,
) -> None:
    captured = _CaptureReporter()
    seen: list[str] = []

    def stop_level(args, reporter, stop_event):
        seen.append(args.state)
        reporter.start(_snapshot(args.state, args.output / f"{args.state}.zip"))
        stop_event.set()
        result = _result(args.state, args.output, accepted=False)
        return TrainingResult(
            **{
                **result.__dict__,
                "stop_reason": "user",
                "exit_code": 130,
            }
        )

    result = training_campaign._execute(
        argparse.Namespace(algorithm="beam", output=tmp_path),
        ("Level1-1", "Level1-2"),
        SimpleNamespace(_run_training=stop_level),
        captured,
        threading.Event(),
    )

    assert seen == ["Level1-1"]
    assert result.exit_code == 130
    assert result.final_row["campaign_processed"] == 1
