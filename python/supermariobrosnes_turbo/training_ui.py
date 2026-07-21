"""Shared reporting and Textual UI for the action-run trainers."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import textwrap
import time
import traceback
from typing import Any, Callable, Mapping, Optional, Protocol, TextIO

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, ProgressBar, RichLog, Static


UPDATE_INTERVAL_SECONDS = 0.2


@dataclass(frozen=True)
class TrainingSnapshot:
    """An immutable view of trainer state suitable for any reporter."""

    algorithm: str
    state: str
    seed: int
    lanes: int
    stop_rule: str
    output: Path
    total_timesteps: int
    action_set: str = "simple (7 actions)"
    timesteps: int = 0
    elapsed: float = 0.0
    loop_fps: float = 0.0
    episodes: int = 0
    successful_episodes: int = 0
    accepted: bool = False
    best_completed: bool = False
    best_mean_reward: float = 0.0
    best_progress: float = 0.0
    best_program_steps: int = 0
    best_program_runs: int = 0
    retained_count: int = 0
    locked_count: int = 0
    archive_count: int = 0
    archive_selection_count: int = 0
    archive_visit_count: int = 0
    archive_update_count: int = 0
    archive_memory_bytes: int = 0
    archive_recent_new_cell_rate: float = 0.0
    archive_recent_visit_window: int = 0
    archive_visits_per_cell: float = 0.0
    archive_replay_probability: float | None = None
    archive_selected_prefix_return_mean: float | None = None
    generation: int | None = None
    beam_count: int | None = None
    beam_width: int | None = None
    pending_count: int | None = None
    refresh_completed: int | None = None
    refresh_total: int | None = None
    status: str = "Running"
    campaign_index: int = 1
    campaign_total: int = 1
    campaign_completed: int = 0

    @property
    def eta_seconds(self) -> float | None:
        if self.loop_fps <= 0.0:
            return None
        return max(self.total_timesteps - self.timesteps, 0) / self.loop_fps


@dataclass(frozen=True)
class TrainingEvent:
    """A notable event that should bypass routine UI throttling."""

    kind: str
    message: str
    elapsed: float = 0.0
    rows: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TrainingResult:
    """The final, terminal-independent training result."""

    algorithm: str
    stop_reason: str
    exit_code: int
    accepted: bool
    elapsed: float
    timesteps: int
    episodes: int
    final_row: Mapping[str, Any]
    policy_path: Path | None
    play_command: str | None
    error_message: str | None = None
    extra_summary_rows: tuple[tuple[str, str], ...] = ()


class TrainingReporter(Protocol):
    """Reporter contract shared by the independent JERK and beam loops."""

    def start(self, snapshot: TrainingSnapshot) -> None: ...

    def update(
        self,
        snapshot: TrainingSnapshot,
        event: TrainingEvent | None = None,
        *,
        force: bool = False,
    ) -> None: ...


def format_box(title: str, rows: list[tuple[str, str]]) -> str:
    label_width = max((len(label) for label, _value in rows), default=0)
    value_width = max(88 - label_width - 2, 20)
    body: list[str] = []
    for label, value in rows:
        wrapped = textwrap.wrap(
            value,
            width=value_width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        body.append(f"{label:<{label_width}}  {wrapped[0]}")
        body.extend(f"{'':<{label_width}}  {line}" for line in wrapped[1:])
    inner_width = max([len(title) + 2, *(len(line) for line in body)])
    top = f"╭─ {title} {'─' * (inner_width - len(title) - 1)}╮"
    middle = [f"│ {line:<{inner_width}} │" for line in body]
    bottom = f"╰{'─' * (inner_width + 2)}╯"
    return "\n".join([top, *middle, bottom])


def format_elapsed(seconds: float) -> str:
    whole_seconds = max(int(seconds), 0)
    hours, remainder = divmod(whole_seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_byte_size(value: int) -> str:
    """Format a non-negative byte count using compact binary units."""
    byte_count = max(int(value), 0)
    if byte_count < 1024:
        return f"{byte_count:,} B"
    amount = float(byte_count)
    for unit in ("KiB", "MiB", "GiB"):
        amount /= 1024.0
        if amount < 1024.0:
            return f"{amount:,.1f} {unit}"
    return f"{amount / 1024.0:,.1f} TiB"


def format_progress(row: Mapping[str, Any], total_timesteps: int) -> str:
    current = int(row["timesteps"])
    total = max(int(total_timesteps), 1)
    percent = min(100.0 * current / total, 100.0)
    archive = (
        f"archive {int(row.get('archive_count', 0)):,} cells  "
        f"·  restores {int(row.get('archive_selection_count', 0)):,}"
        if str(row.get("algorithm", "")).casefold() == "go-explore"
        else (
            f"archive {int(row['retained_count']):,} "
            f"({int(row['locked_count']):,} locked)"
        )
    )
    return (
        f"  {percent:6.2f}%  {current:,} / {total:,} transitions  "
        f"·  {float(row['loop_fps']):,.0f} steps/s\n"
        f"           {int(row['episodes']):,} episodes  "
        f"·  best reward {float(row['best_mean_reward']):,.1f}  "
        f"·  progress {float(row['best_progress']):,.0f}\n"
        f"           policy {int(row['best_program_steps']):,} steps / "
        f"{int(row['best_program_runs']):,} runs  "
        f"·  {archive}"
    )


def snapshot_from_row(
    *,
    algorithm: str,
    state: str,
    seed: int,
    lanes: int,
    stop_rule: str,
    output: Path,
    total_timesteps: int,
    row: Mapping[str, Any],
    status: str = "Running",
    beam_width: int | None = None,
    refresh_total: int | None = None,
) -> TrainingSnapshot:
    generation = row.get("generation")
    completed = int(row.get("episodes", 0))
    refresh_completed = (
        completed % refresh_total if refresh_total and algorithm == "beam" else None
    )
    return TrainingSnapshot(
        algorithm=algorithm,
        state=state,
        seed=seed,
        lanes=lanes,
        stop_rule=stop_rule,
        output=output,
        total_timesteps=total_timesteps,
        timesteps=int(row.get("timesteps", 0)),
        elapsed=float(row.get("elapsed", 0.0)),
        loop_fps=float(row.get("loop_fps", 0.0)),
        episodes=completed,
        successful_episodes=int(row.get("successful_episodes", 0)),
        accepted=bool(row.get("accepted", False)),
        best_completed=bool(row.get("best_completed", False)),
        best_mean_reward=float(row.get("best_mean_reward", 0.0)),
        best_progress=float(row.get("best_progress", 0.0)),
        best_program_steps=int(row.get("best_program_steps", 0)),
        best_program_runs=int(row.get("best_program_runs", 0)),
        retained_count=int(row.get("retained_count", 0)),
        locked_count=int(row.get("locked_count", 0)),
        archive_count=int(row.get("archive_count", 0)),
        archive_selection_count=int(row.get("archive_selection_count", 0)),
        archive_visit_count=int(row.get("archive_visit_count", 0)),
        archive_update_count=int(row.get("archive_update_count", 0)),
        archive_memory_bytes=int(row.get("archive_memory_bytes", 0)),
        archive_recent_new_cell_rate=float(
            row.get("archive_recent_new_cell_rate", 0.0)
        ),
        archive_recent_visit_window=int(
            row.get("archive_recent_visit_window", 0)
        ),
        archive_visits_per_cell=float(row.get("archive_visits_per_cell", 0.0)),
        archive_replay_probability=row.get("archive_replay_probability"),
        archive_selected_prefix_return_mean=row.get(
            "archive_selected_prefix_return_mean"
        ),
        generation=None if generation is None else int(generation),
        beam_count=(
            None if row.get("beam_count") is None else int(row["beam_count"])
        ),
        beam_width=beam_width,
        pending_count=(
            None if row.get("pending_count") is None else int(row["pending_count"])
        ),
        refresh_completed=refresh_completed,
        refresh_total=refresh_total,
        status=status,
    )


def usable_terminal(
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    environ = os.environ if environ is None else environ
    if not stdin.isatty() or not stdout.isatty():
        return False, "stdin and stdout must both be interactive terminals"
    if environ.get("TERM", "").strip().lower() in {"", "dumb", "unknown"}:
        return False, "TERM must identify a usable terminal (not empty or dumb)"
    if any(environ.get(name) for name in ("CI", "GITHUB_ACTIONS", "BUILD_NUMBER")):
        return False, "a continuous-integration environment was detected"
    return True, ""


def resolve_ui_mode(
    requested: str,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    usable, reason = usable_terminal(stdin=stdin, stdout=stdout, environ=environ)
    if requested == "auto":
        return "tui" if usable else "plain"
    if requested == "tui" and not usable:
        raise ValueError(f"cannot start --ui tui: {reason}; use --ui plain")
    return requested


@contextmanager
def safe_sigint(stop_event: threading.Event):
    """Turn Ctrl-C into a safe-stop request while synchronous training runs."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)


class PlainReporter:
    """The existing line-oriented formatter for logs, pipes, and CI."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def start(self, snapshot: TrainingSnapshot) -> None:
        rows = [
            ("State", snapshot.state),
            ("Action set", snapshot.action_set),
            ("Parallel lanes", f"{snapshot.lanes:,}"),
        ]
        if snapshot.campaign_total > 1:
            rows.append(
                (
                    "Level",
                    f"{snapshot.campaign_index:,} / {snapshot.campaign_total:,}",
                )
            )
        if snapshot.beam_width is not None:
            rows.append(("Beam width", f"{snapshot.beam_width:,}"))
        rows.extend(
            [
                ("Budget", f"{snapshot.total_timesteps:,} transitions"),
                ("Stop rule", snapshot.stop_rule),
                ("Policy", str(snapshot.output)),
            ]
        )
        self.logger.info(
            "\n%s", format_box(f"{snapshot.algorithm} training", rows)
        )

    def update(
        self,
        snapshot: TrainingSnapshot,
        event: TrainingEvent | None = None,
        *,
        force: bool = False,
    ) -> None:
        if event is not None:
            if event.kind != "success":
                return
            if event.rows:
                self.logger.info(
                    "\n%s", format_box(event.message, list(event.rows))
                )
            else:
                self.logger.info("%s", event.message)
            return
        if not force:
            return
        self.logger.info(
            "%s", format_progress(snapshot.__dict__, snapshot.total_timesteps)
        )


class NullReporter:
    def start(self, snapshot: TrainingSnapshot) -> None:
        del snapshot

    def update(
        self,
        snapshot: TrainingSnapshot,
        event: TrainingEvent | None = None,
        *,
        force: bool = False,
    ) -> None:
        del snapshot, event, force


class _SnapshotMessage(Message):
    def __init__(self, snapshot: TrainingSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot


class _EventMessage(Message):
    def __init__(self, event: TrainingEvent) -> None:
        super().__init__()
        self.event = event


class _FinishedMessage(Message):
    def __init__(self, result: TrainingResult) -> None:
        super().__init__()
        self.result = result


class _FailedMessage(Message):
    def __init__(self, error: BaseException, traceback_text: str) -> None:
        super().__init__()
        self.error = error
        self.traceback_text = traceback_text


class TextualReporter:
    """Worker-side reporter that only posts thread-safe app messages."""

    def __init__(self, app: "TrainingApp") -> None:
        self.app = app
        self._last_update = float("-inf")

    def start(self, snapshot: TrainingSnapshot) -> None:
        self.app.post_message(_SnapshotMessage(snapshot))

    def update(
        self,
        snapshot: TrainingSnapshot,
        event: TrainingEvent | None = None,
        *,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        immediate_snapshot = force and (
            event is None
            or event.kind in {"success", "checkpoint", "stop", "complete", "error"}
        )
        if immediate_snapshot or now - self._last_update >= UPDATE_INTERVAL_SECONDS:
            self._last_update = now
            self.app.post_message(_SnapshotMessage(snapshot))
        if event is not None:
            self.app.post_message(_EventMessage(event))


Runner = Callable[[TrainingReporter, threading.Event], TrainingResult]


class TrainingApp(App[Optional[TrainingResult]]):
    """Full-screen live dashboard for either action-run trainer."""

    TITLE = "Mario Training Dashboard"
    BINDINGS = [
        ("q", "request_stop", "Safe stop"),
        ("ctrl+c", "request_stop", "Safe stop"),
    ]
    CSS = """
    Screen { layout: vertical; }
    #run-header { height: 3; padding: 0 1; background: $primary-darken-2; }
    #progress-panel { height: 5; padding: 0 1; border: round $primary; }
    #progress-details { height: 1; }
    #level-progress-panel { display: none; height: 5; padding: 0 1; border: round $secondary; }
    #level-progress-details { height: 1; }
    #panels { height: 1fr; min-height: 8; }
    .panel { width: 1fr; padding: 0 1; border: round $accent; }
    .panel-title { height: 1; text-style: bold; color: $accent; }
    #event-panel { height: 7; padding: 0 1; border: round $secondary; }
    #event-log { height: 1fr; }
    Footer { height: 1; }
    """

    def __init__(
        self,
        initial: TrainingSnapshot,
        runner: Runner | None,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.initial = initial
        self.runner = runner
        self.stop_event = stop_event or threading.Event()
        self.result: TrainingResult | None = None
        self.failure: BaseException | None = None
        self.failure_traceback: str | None = None
        self.events: deque[TrainingEvent] = deque(maxlen=8)

    def compose(self) -> ComposeResult:
        yield Static(id="run-header")
        with Vertical(id="progress-panel"):
            yield Static("Transitions", classes="panel-title")
            yield ProgressBar(
                total=max(self.initial.total_timesteps, 1),
                show_eta=False,
                id="transition-progress",
            )
            yield Static(id="progress-details")
        with Vertical(id="level-progress-panel"):
            yield Static("Level campaign", classes="panel-title")
            yield ProgressBar(
                total=max(self.initial.campaign_total, 1),
                show_eta=False,
                id="level-progress",
            )
            yield Static(id="level-progress-details")
        with Horizontal(id="panels"):
            with Vertical(classes="panel", id="search-panel"):
                yield Static("Search", classes="panel-title")
                yield Static(id="search-stats")
            with Vertical(classes="panel", id="best-panel"):
                yield Static("Best path", classes="panel-title")
                yield Static(id="best-stats")
        with Vertical(id="event-panel"):
            yield Static("Recent events", classes="panel-title")
            yield RichLog(markup=True, wrap=True, id="event-log")
        yield Footer()

    def on_mount(self) -> None:
        self._render_snapshot(self.initial)
        if self.runner is not None:
            self.run_worker(self._worker, thread=True, exclusive=True)

    def _worker(self) -> None:
        assert self.runner is not None
        try:
            result = self.runner(TextualReporter(self), self.stop_event)
        except BaseException as error:
            self.post_message(_FailedMessage(error, traceback.format_exc()))
        else:
            self.post_message(_FinishedMessage(result))

    def action_request_stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        event = TrainingEvent("stop", "Safe stop requested; finishing current step")
        self.events.append(event)
        self._render_events()
        self.query_one("#progress-details", Static).update(
            "Stopping safely after the current vector step…"
        )

    def on__snapshot_message(self, message: _SnapshotMessage) -> None:
        self._render_snapshot(message.snapshot)
        if self.events:
            self._render_events()

    def on__event_message(self, message: _EventMessage) -> None:
        self.events.append(message.event)
        if message.event.kind in {"success", "checkpoint", "stop", "complete", "error"}:
            self._render_events()

    def on__finished_message(self, message: _FinishedMessage) -> None:
        self.result = message.result
        self.exit(message.result)

    def on__failed_message(self, message: _FailedMessage) -> None:
        self.failure = message.error
        self.failure_traceback = message.traceback_text
        self.exit(None)

    def _render_snapshot(self, snapshot: TrainingSnapshot) -> None:
        self.query_one("#run-header", Static).update(
            f"[b]{snapshot.algorithm}[/b]  {snapshot.state}  ·  seed {snapshot.seed}  "
            f"·  {snapshot.lanes} lanes\n"
            f"stop: {snapshot.stop_rule}  ·  output: {snapshot.output}"
        )
        self.query_one("#transition-progress", ProgressBar).update(
            total=max(snapshot.total_timesteps, 1),
            progress=min(snapshot.timesteps, snapshot.total_timesteps),
        )
        eta = "—" if snapshot.eta_seconds is None else format_elapsed(snapshot.eta_seconds)
        self.query_one("#progress-details", Static).update(
            f"{snapshot.timesteps:,} / {snapshot.total_timesteps:,}  ·  "
            f"elapsed {format_elapsed(snapshot.elapsed)}  ·  ETA {eta}  ·  "
            f"{snapshot.loop_fps:,.0f} steps/s  ·  {snapshot.status}"
        )
        level_panel = self.query_one("#level-progress-panel", Vertical)
        if snapshot.campaign_total > 1:
            level_panel.styles.display = "block"
            self.query_one("#level-progress", ProgressBar).update(
                total=snapshot.campaign_total,
                progress=min(snapshot.campaign_completed, snapshot.campaign_total),
            )
            self.query_one("#level-progress-details", Static).update(
                f"{snapshot.campaign_completed:,} / {snapshot.campaign_total:,} processed  ·  "
                f"current {snapshot.campaign_index:,}: {snapshot.state}"
            )
        else:
            level_panel.styles.display = "none"
        if snapshot.generation is not None:
            refresh = (
                "—"
                if snapshot.refresh_total is None
                else f"{snapshot.refresh_completed}/{snapshot.refresh_total}"
            )
            search = (
                f"Generation       {snapshot.generation:,}\n"
                f"Beam occupancy  {snapshot.beam_count or 0:,} / "
                f"{snapshot.beam_width or 0:,}\n"
                f"Pending          {snapshot.pending_count or 0:,}\n"
                f"Refresh cycle    {refresh}\n"
                f"Episodes         {snapshot.episodes:,}"
            )
        elif snapshot.algorithm.casefold() == "go-explore":
            search = (
                f"Archive cells    {snapshot.archive_count:,} · "
                f"{format_byte_size(snapshot.archive_memory_bytes)}\n"
                f"Cell restores   {snapshot.archive_selection_count:,}\n"
                f"Cell visits     {snapshot.archive_visit_count:,} · "
                f"{snapshot.archive_visits_per_cell:,.1f}/cell\n"
                f"New-cell rate   {snapshot.archive_recent_new_cell_rate:.1%} / "
                f"{snapshot.archive_recent_visit_window:,} visits\n"
                f"Archive updates {snapshot.archive_update_count:,}\n"
                f"Episodes         {snapshot.episodes:,}"
            )
        else:
            replay = snapshot.archive_replay_probability
            prefix_return = snapshot.archive_selected_prefix_return_mean
            search = (
                f"Archive          {snapshot.retained_count:,}\n"
                f"Locked           {snapshot.locked_count:,}\n"
                f"Replay chance    {'—' if replay is None else f'{replay:.1%}'}\n"
                f"Replay return    {'—' if prefix_return is None else f'{prefix_return:,.1f}'}\n"
                f"Episodes         {snapshot.episodes:,}"
            )
        self.query_one("#search-stats", Static).update(search)
        completion = "complete" if snapshot.best_completed else "not complete"
        return_label = (
            "Score-first return"
            if snapshot.algorithm.casefold() == "go-explore"
            else "Shaped return"
        )
        self.query_one("#best-stats", Static).update(
            f"Status           {completion}  ·  {snapshot.successful_episodes:,} successes\n"
            f"{return_label:<20}{snapshot.best_mean_reward:,.1f}\n"
            f"Best progress    {snapshot.best_progress:,.0f}\n"
            f"Program steps    {snapshot.best_program_steps:,}\n"
            f"Program runs     {snapshot.best_program_runs:,}"
        )

    def _render_events(self) -> None:
        log = self.query_one("#event-log", RichLog)
        log.clear()
        for event in self.events:
            timestamp = format_elapsed(event.elapsed)
            log.write(Text.from_markup(f"[dim]{timestamp:>9}[/dim]  {event.message}"))


def run_training_app(
    initial: TrainingSnapshot,
    runner: Runner,
    *,
    stop_event: threading.Event | None = None,
) -> TrainingResult:
    app = TrainingApp(initial, runner, stop_event=stop_event)
    result = app.run()
    if app.failure is not None:
        setattr(app.failure, "training_traceback", app.failure_traceback)
        raise app.failure
    if result is None:
        raise RuntimeError("training dashboard exited without a result")
    return result


def print_summary(result: TrainingResult, logger: logging.Logger) -> None:
    if result.stop_reason == "user":
        outcome = "stopped by user"
    elif result.accepted:
        outcome = "level completed"
    else:
        outcome = "budget exhausted"
    rows = [
        ("Result", outcome),
        ("Elapsed", format_elapsed(result.elapsed)),
        ("Transitions", f"{result.timesteps:,}"),
        ("Episodes", f"{result.episodes:,}"),
        ("Best reward", f"{float(result.final_row['best_mean_reward']):,.1f}"),
        ("Progress", f"{float(result.final_row['best_progress']):,.0f}"),
        (
            "Policy size",
            f"{int(result.final_row['best_program_steps']):,} steps / "
            f"{int(result.final_row['best_program_runs']):,} runs",
        ),
        *result.extra_summary_rows,
    ]
    rows.append(
        ("Saved", "no candidate policy" if result.policy_path is None else str(result.policy_path))
    )
    logger.info(
        "\n%s", format_box(f"{result.algorithm} training complete", rows)
    )
    if result.play_command is not None:
        logger.info("\nPlay the policy:\n\n  %s\n", result.play_command)


def report_failure_traceback(error: BaseException) -> None:
    stored = getattr(error, "training_traceback", None)
    if stored:
        sys.stderr.write(stored)
    else:
        traceback.print_exception(type(error), error, error.__traceback__)
