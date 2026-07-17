from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from supermariobrosnes_turbo import training as train
from supermariobrosnes_turbo import beam_training as train_beam
from supermariobrosnes_turbo.jerk import JerkPolicy, policy_path_for_state


class _FakeFailureTask:
    def __init__(self, **_kwargs) -> None:
        self.steps = 0
        self.closed = False

    def reset(self) -> np.ndarray:
        return np.zeros((1, 1), dtype=np.uint8)

    def step(self, actions):
        self.steps += 1
        count = len(actions)
        records = {
            lane: SimpleNamespace(completed=False, progress=100.0)
            for lane in range(count)
        }
        return (
            np.zeros((count, 1), dtype=np.uint8),
            np.ones(count, dtype=np.float64),
            np.ones(count, dtype=np.bool_),
            records,
            np.zeros(count, dtype=np.bool_),
        )

    def reset_lanes(self, _mask) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeSuccessTask(_FakeFailureTask):
    def step(self, actions):
        self.steps += 1
        count = len(actions)
        records = {
            lane: SimpleNamespace(completed=True, progress=3154.0)
            for lane in range(count)
        }
        return (
            np.zeros((count, 1), dtype=np.uint8),
            np.ones(count, dtype=np.float64),
            np.zeros(count, dtype=np.bool_),
            records,
            np.ones(count, dtype=np.bool_),
        )


class _FakeImprovingSuccessTask(_FakeFailureTask):
    def step(self, actions):
        rewards = (1.0, 3.0, 2.0)
        reward = rewards[min(self.steps, len(rewards) - 1)]
        self.steps += 1
        count = len(actions)
        records = {
            lane: SimpleNamespace(completed=True, progress=3154.0)
            for lane in range(count)
        }
        return (
            np.zeros((count, 1), dtype=np.uint8),
            np.full(count, reward, dtype=np.float64),
            np.zeros(count, dtype=np.bool_),
            records,
            np.ones(count, dtype=np.bool_),
        )


class _StopAfterCandidateReporter:
    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event

    def start(self, _snapshot) -> None:
        return None

    def update(self, snapshot, event=None, *, force=False) -> None:
        del event, force
        if snapshot.timesteps:
            self.stop_event.set()


class _NullReporter:
    def start(self, _snapshot) -> None:
        return None

    def update(self, _snapshot, _event=None, *, force=False) -> None:
        del force


TRAINERS = [
    (train, ["--algorithm", "jerk"]),
    (
        train_beam,
        [
            "--algorithm",
            "beam",
            "--beam-width",
            "2",
            "--beam-refresh-episodes",
            "1",
            "--mutation-runs",
            "1",
            "--branch-durations",
            "1",
        ],
    ),
]


def _parse_args(values: list[str]):
    parser = train.build_parser()
    args = parser.parse_args(values)
    train._apply_algorithm_defaults(parser, args)
    return args


def test_default_beam_run_overwrites_existing_canonical_policy(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_beam, "MarioJerkTask", _FakeSuccessTask)
    policy_path = policy_path_for_state("Level1-1")
    policy_path.parent.mkdir(parents=True)
    policy_path.write_bytes(b"old policy")
    args = _parse_args(
        [
            "Level1-1",
            "--algorithm",
            "beam",
            "--transitions",
            "1",
            "--lanes",
            "1",
            "--log-every",
            "10",
        ]
    )

    result = train_beam._run_training(args, _NullReporter(), threading.Event())

    assert result.accepted
    assert JerkPolicy.load(policy_path).metadata["search_algorithm"] == "beam"


def test_continued_beam_publishes_later_better_completion_and_archives_all(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "continued-beam"
    output.mkdir()
    monkeypatch.setattr(train_beam, "MarioJerkTask", _FakeImprovingSuccessTask)
    args = _parse_args(
        [
            "Level1-1",
            "--algorithm",
            "beam",
            "--output",
            str(output),
            "--overwrite",
            "--continue-after-completion",
            "--transitions",
            "3",
            "--lanes",
            "1",
            "--beam-width",
            "1",
            "--beam-refresh-episodes",
            "1",
            "--mutation-runs",
            "1",
            "--branch-durations",
            "1",
            "--run-duration-max",
            "1",
            "--log-every",
            "10",
        ]
    )

    result = train_beam._run_training(
        args, _NullReporter(), threading.Event()
    )

    policy = JerkPolicy.load(output / "Level1-1.zip")
    successes = [
        json.loads(line)
        for line in output.joinpath("successes.jsonl").read_text().splitlines()
    ]
    assert result.accepted
    assert result.timesteps == 3
    assert policy.best_reward == 3.0
    assert result.final_row["first_success_reward"] == 1.0
    assert result.final_row["best_success_reward"] == 3.0
    assert result.final_row["improvement_count"] == 1
    assert [row["episode_return"] for row in successes] == [1.0, 3.0, 2.0]
    assert [row["improved"] for row in successes] == [True, True, False]


@pytest.mark.parametrize(("module", "extra_args"), TRAINERS)
@pytest.mark.parametrize(
    ("task_class", "expected_reason", "expected_code"),
    [
        (_FakeSuccessTask, "success", 0),
        (_FakeFailureTask, "budget", 1),
    ],
)
def test_both_trainers_finish_success_and_budget_paths(
    module,
    extra_args: list[str],
    task_class,
    expected_reason: str,
    expected_code: int,
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / f"{module.__name__}-{expected_reason}"
    output.mkdir()
    monkeypatch.setattr(module, "MarioJerkTask", task_class)
    args = _parse_args(
        [
            "Level1-1",
            "--output",
            str(output),
            "--transitions",
            "2",
            "--lanes",
            "1",
            "--log-every",
            "10",
            *extra_args,
        ]
    )

    result = module._run_training(args, _NullReporter(), threading.Event())

    assert result.stop_reason == expected_reason
    assert result.exit_code == expected_code
    assert result.policy_path is not None and result.policy_path.exists()
    final_row = json.loads(output.joinpath("episodes.jsonl").read_text())
    assert final_row["stop_reason"] == expected_reason


@pytest.mark.parametrize(("module", "extra_args"), TRAINERS)
def test_safe_stop_without_candidate_writes_final_metrics_but_no_policy(
    module, extra_args: list[str], tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / module.__name__
    output.mkdir()
    monkeypatch.setattr(module, "MarioJerkTask", _FakeFailureTask)
    args = _parse_args(
        [
            "Level1-1",
            "--output",
            str(output),
            "--transitions",
            "4",
            "--lanes",
            "1",
            "--log-every",
            "10",
            *extra_args,
        ]
    )
    stop_event = threading.Event()
    stop_event.set()

    result = module._run_training(args, _NullReporter(), stop_event)

    assert result.exit_code == 130
    assert result.stop_reason == "user"
    assert result.policy_path is None
    assert not list(output.glob("*.zip"))
    final_row = json.loads(output.joinpath("episodes.jsonl").read_text())
    assert final_row["phase"] == "final"
    assert final_row["stop_reason"] == "user"


@pytest.mark.parametrize(("module", "extra_args"), TRAINERS)
def test_safe_stop_with_candidate_saves_replayable_policy(
    module, extra_args: list[str], tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / module.__name__
    output.mkdir()
    monkeypatch.setattr(module, "MarioJerkTask", _FakeFailureTask)
    args = _parse_args(
        [
            "Level1-1",
            "--output",
            str(output),
            "--transitions",
            "4",
            "--lanes",
            "1",
            "--log-every",
            "10",
            *extra_args,
        ]
    )
    stop_event = threading.Event()

    result = module._run_training(
        args, _StopAfterCandidateReporter(stop_event), stop_event
    )

    assert result.exit_code == 130
    assert result.stop_reason == "user"
    assert result.policy_path is not None
    assert result.policy_path.exists()
    assert result.play_command is not None
    final_row = json.loads(output.joinpath("episodes.jsonl").read_text())
    assert final_row["stop_reason"] == "user"
    assert final_row["best_program_steps"] > 0


def test_periodic_metric_schema_is_unchanged_and_only_final_has_stop_reason(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "periodic"
    output.mkdir()
    monkeypatch.setattr(train, "MarioJerkTask", _FakeFailureTask)
    args = _parse_args(
        [
            "Level1-1",
            "--algorithm",
            "jerk",
            "--output",
            str(output),
            "--transitions",
            "2",
            "--lanes",
            "1",
            "--log-every",
            "1",
        ]
    )

    result = train._run_training(args, _NullReporter(), threading.Event())
    rows = [
        json.loads(line)
        for line in output.joinpath("episodes.jsonl").read_text().splitlines()
    ]

    assert result.stop_reason == "budget"
    assert len(rows) == 3
    assert all("stop_reason" not in row for row in rows[:-1])
    assert rows[-1]["stop_reason"] == "budget"
