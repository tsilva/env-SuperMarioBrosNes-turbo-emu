from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
import signal
import threading

import pytest

from supermariobrosnes_turbo.training_ui import (
    PlainReporter,
    TextualReporter,
    TrainingApp,
    TrainingEvent,
    TrainingSnapshot,
    resolve_ui_mode,
    safe_sigint,
)


class _Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def _snapshot(**changes) -> TrainingSnapshot:
    values = {
        "algorithm": "Beam",
        "state": "Level1-1",
        "seed": 108,
        "lanes": 8,
        "stop_rule": "first completion",
        "output": Path("runs/test/Level1-1.zip"),
        "total_timesteps": 100,
        "beam_width": 16,
        "generation": 0,
        "beam_count": 0,
        "pending_count": 0,
        "refresh_completed": 0,
        "refresh_total": 8,
    }
    values.update(changes)
    return TrainingSnapshot(**values)


@pytest.mark.parametrize(
    ("stdin_tty", "stdout_tty", "environment", "expected"),
    [
        (True, True, {"TERM": "xterm-256color"}, "tui"),
        (False, True, {"TERM": "xterm-256color"}, "plain"),
        (True, False, {"TERM": "xterm-256color"}, "plain"),
        (True, True, {"TERM": "dumb"}, "plain"),
        (True, True, {"TERM": "xterm", "CI": "1"}, "plain"),
        (True, True, {"TERM": "xterm", "NO_COLOR": "1"}, "tui"),
    ],
)
def test_ui_auto_selection(
    stdin_tty: bool,
    stdout_tty: bool,
    environment: dict[str, str],
    expected: str,
) -> None:
    assert (
        resolve_ui_mode(
            "auto",
            stdin=_Stream(stdin_tty),
            stdout=_Stream(stdout_tty),
            environ=environment,
        )
        == expected
    )


def test_forced_tui_requires_a_usable_terminal() -> None:
    with pytest.raises(ValueError, match="use --ui plain"):
        resolve_ui_mode(
            "tui",
            stdin=_Stream(False),
            stdout=_Stream(True),
            environ={"TERM": "xterm"},
        )


def test_plain_sigint_requests_safe_stop_and_restores_handler() -> None:
    stop_event = threading.Event()
    previous = signal.getsignal(signal.SIGINT)

    with safe_sigint(stop_event):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert stop_event.is_set()

    assert signal.getsignal(signal.SIGINT) is previous


def test_plain_reporter_is_line_oriented_and_has_no_control_sequences() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("test-plain-training-reporter")
    logger.handlers = [logging.StreamHandler(stream)]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    reporter = PlainReporter(logger)
    snapshot = _snapshot(
        timesteps=50,
        elapsed=2.0,
        loop_fps=25.0,
        episodes=3,
        best_mean_reward=12.5,
        best_progress=123.0,
        best_program_steps=10,
        best_program_runs=2,
    )

    reporter.start(snapshot)
    reporter.update(snapshot)
    reporter.update(snapshot, force=True)

    output = stream.getvalue()
    assert "Beam training" in output
    assert "50.00%" in output
    assert "\x1b[" not in output
    assert "\x1b]" not in output
    assert output.count("50.00%") == 1


@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
def test_dashboard_renders_updates_and_orders_events(size: tuple[int, int]) -> None:
    async def exercise() -> None:
        app = TrainingApp(_snapshot(), None)
        async with app.run_test(size=size) as pilot:
            reporter = TextualReporter(app)
            updated = _snapshot(
                timesteps=50,
                elapsed=2.0,
                loop_fps=25.0,
                generation=2,
                beam_count=12,
                pending_count=5,
                best_progress=321.0,
            )
            reporter.update(
                updated,
                TrainingEvent("beam-refresh", "generation two", 1.0),
                force=True,
            )
            reporter.update(
                updated,
                TrainingEvent("new-best", "new best path", 2.0),
                force=True,
            )
            await pilot.pause()

            assert app.query_one("#transition-progress") is not None
            assert app.query_one("#search-stats") is not None
            assert app.query_one("#search-panel").outer_size.height >= 8
            assert app.query_one("#best-panel").outer_size.height >= 8
            assert app.query_one("#event-panel").outer_size.height == 7
            assert [event.message for event in app.events] == [
                "generation two",
                "new best path",
            ]

    asyncio.run(exercise())


@pytest.mark.parametrize("key", ["q", "ctrl+c"])
def test_dashboard_stop_keys_set_the_shared_event(key: str) -> None:
    async def exercise() -> None:
        app = TrainingApp(_snapshot(), None)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press(key)
            await pilot.pause()
            assert app.stop_event.is_set()
            assert app.events[-1].kind == "stop"

    asyncio.run(exercise())


def test_worker_exception_is_captured_after_dashboard_exit() -> None:
    error = RuntimeError("worker exploded")

    def fail(_reporter, _stop_event):
        raise error

    async def exercise() -> None:
        app = TrainingApp(_snapshot(), fail)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
        assert app.failure is error
        assert app.failure_traceback is not None
        assert "worker exploded" in app.failure_traceback

    asyncio.run(exercise())
