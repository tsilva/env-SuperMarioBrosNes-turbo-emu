#!/usr/bin/env python3
"""Small controller for the autoresearch-speed loop."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from benchmark_workload import canonical_noop_env_args
    from dotenv_utils import require_arg_or_env_or_dotenv_path
    from run_git_ref_benchmark import (
        AUTORESEARCH_ROOT_ENV,
        LOCAL_RESULTS_SUBDIR,
        RESULTS_TSV_COLUMNS,
    )
except ModuleNotFoundError:
    from scripts.benchmark_workload import canonical_noop_env_args
    from scripts.dotenv_utils import require_arg_or_env_or_dotenv_path
    from scripts.run_git_ref_benchmark import (
        AUTORESEARCH_ROOT_ENV,
        LOCAL_RESULTS_SUBDIR,
        RESULTS_TSV_COLUMNS,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "run_git_ref_benchmark.py"
BENCHMARK_SPS_SCRIPT = REPO_ROOT / "scripts" / "benchmark_sps.py"
DIAGNOSE_STEPS = "5000"
DIAGNOSE_REPEATS = "3"
DIAGNOSE_WARMUP = "500"
PROFILE_STEPS = "2000"
PROFILE_REPEATS = "1"
PROFILE_WARMUP = "100"
SCREEN_STEPS = "5000"
SCREEN_REPEATS = "1"
SCREEN_MEASURED_PAIRS = "3"
STACK_ACCEPT_STEPS = "30000"
STACK_ACCEPT_REPEATS = "2"
STACK_ACCEPT_WARMUPS = "1"
STACK_ACCEPT_MEASURED_PAIRS = "7"
PROBE_STEPS = "2000"
PROBE_REPEATS = "1"
PROBE_WARMUP = "50"
ACCEPT_STEPS = "50000"
ACCEPT_REPEATS = "3"
DEDICATED_ACCEPT_MEASURED_PAIRS = "3"
DEDICATED_CALIBRATE_MEASURED_INVOCATIONS = "11"


def autoresearch_root(value: str | Path | None = None) -> Path:
    return require_arg_or_env_or_dotenv_path(
        AUTORESEARCH_ROOT_ENV,
        "autoresearch root",
        value,
        must_be_dir=True,
    )


def build_benchmark_command(
    kind: str,
    refs: list[str],
    extra_args: list[str],
    *,
    full: bool = False,
) -> list[str]:
    if kind == "screen":
        if len(refs) != 2:
            raise SystemExit("screen requires baseline_ref and candidate_ref")
        command = [sys.executable, str(BENCHMARK_SCRIPT), refs[0], refs[1]]
        command += [
            "--steps",
            SCREEN_STEPS,
            "--repeats",
            SCREEN_REPEATS,
            "--warmups",
            "0",
            "--max-measured-invocations",
            SCREEN_MEASURED_PAIRS,
        ]
    elif kind == "accept":
        if len(refs) != 2:
            raise SystemExit("accept requires baseline_ref and candidate_ref")
        command = [sys.executable, str(BENCHMARK_SCRIPT), refs[0], refs[1]]
        command += ["--steps", ACCEPT_STEPS, "--repeats", ACCEPT_REPEATS]
        if not full:
            command += ["--max-measured-invocations", DEDICATED_ACCEPT_MEASURED_PAIRS]
    elif kind == "accept-stack":
        if len(refs) != 2:
            raise SystemExit("accept-stack requires baseline_ref and candidate_ref")
        command = [sys.executable, str(BENCHMARK_SCRIPT), refs[0], refs[1]]
        command += [
            "--steps",
            STACK_ACCEPT_STEPS,
            "--repeats",
            STACK_ACCEPT_REPEATS,
            "--warmups",
            STACK_ACCEPT_WARMUPS,
            "--max-measured-invocations",
            STACK_ACCEPT_MEASURED_PAIRS,
        ]
    elif kind == "calibrate":
        if len(refs) != 1:
            raise SystemExit("calibrate requires exactly one ref")
        command = [sys.executable, str(BENCHMARK_SCRIPT), refs[0], "--single"]
        command += ["--steps", ACCEPT_STEPS, "--repeats", ACCEPT_REPEATS]
        if not full:
            command += [
                "--max-measured-invocations",
                DEDICATED_CALIBRATE_MEASURED_INVOCATIONS,
            ]
    else:
        raise SystemExit(f"unknown benchmark kind: {kind}")
    return command + extra_args


def build_diagnose_command(root: Path, *, profile: bool, quick: bool = False) -> list[str]:
    output = root / "benchmarks" / (
        "local-profile-benchmark.json" if profile else "local-diagnosis.json"
    )
    steps = "1000" if quick else DIAGNOSE_STEPS
    repeats = "1" if quick else DIAGNOSE_REPEATS
    warmup = "20" if quick else DIAGNOSE_WARMUP
    if profile:
        steps = PROFILE_STEPS
        repeats = PROFILE_REPEATS
        warmup = PROFILE_WARMUP
    command = [
        sys.executable,
        str(BENCHMARK_SPS_SCRIPT),
        *canonical_noop_env_args(),
        "--steps",
        steps,
        "--repeats",
        repeats,
        "--warmup",
        warmup,
    ]
    if profile:
        command += ["--profile-output", str(root / "benchmarks" / "local-profile.json")]
    return command + ["--json", "--output-json", str(output)]


def build_probe_command(root: Path) -> list[str]:
    output = root / "benchmarks" / "local-probe.json"
    return [
        sys.executable,
        str(BENCHMARK_SPS_SCRIPT),
        *canonical_noop_env_args(),
        "--steps",
        PROBE_STEPS,
        "--repeats",
        PROBE_REPEATS,
        "--warmup",
        PROBE_WARMUP,
        "--json",
        "--output-json",
        str(output),
    ]


def run_command(
    command: list[str],
    *,
    dry_run: bool,
    env_defaults: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    prefix = []
    for key, value in (env_defaults or {}).items():
        if key not in env:
            env[key] = value
            prefix.append(f"{key}={shlex.quote(value)}")
    command_text = shlex.join(command)
    print(" ".join(prefix + [command_text]))
    if dry_run:
        return 0
    return subprocess.run(command, check=False, env=env).returncode


def benchmark_index_path(root: Path) -> Path:
    return root / "benchmarks" / LOCAL_RESULTS_SUBDIR / "index.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON file is not an object: {path}")
    return payload


def gate_truths(payload: dict[str, Any], key: str) -> bool:
    gates = payload.get(key)
    return isinstance(gates, dict) and all(value is True for value in gates.values())


def infer_status(aggregate: dict[str, Any]) -> str:
    tier = aggregate.get("benchmark_tier")
    decision = aggregate.get("decision")
    limit_stop_reason = aggregate.get("limit_stop_reason")
    ratio = aggregate.get("median_pair_ratio")
    if tier == "local_triage":
        if isinstance(ratio, (int, float)) and ratio >= 1.03:
            return "triage_promote"
        if isinstance(ratio, (int, float)) and ratio < 1.01:
            return "triage_discard"
        return "inconclusive"
    if tier == "stack_acceptance":
        load_ok = aggregate.get("load_gate_passed") is True or aggregate.get(
            "load_gate_ignored_for_validity"
        ) is True
        faster = aggregate.get("candidate_faster_pairs")
        if (
            aggregate.get("validity_passed") is True
            and load_ok
            and isinstance(ratio, (int, float))
            and ratio >= 1.03
            and isinstance(faster, int)
            and faster >= 6
        ):
            return "keep_stack"
        if decision == "converged_no_meaningful_win" or (
            isinstance(ratio, (int, float)) and ratio <= 1.0
        ):
            return "discard_stack"
        return "inconclusive"
    if tier != "local_acceptance":
        return "inconclusive"

    load_ok = aggregate.get("load_gate_passed") is True or aggregate.get(
        "load_gate_ignored_for_validity"
    ) is True
    if (
        decision == "converged_candidate_win"
        and aggregate.get("validity_passed") is True
        and load_ok
    ):
        return "keep"

    ci = aggregate.get("pair_ratio_bootstrap_ci95")
    ci_low_ok = isinstance(ci, list) and len(ci) >= 1 and float(ci[0]) >= 1.0
    faster = aggregate.get("candidate_faster_pairs")
    needed = aggregate.get("candidate_faster_pairs_required_for_win")
    faster_ok = isinstance(faster, int) and isinstance(needed, int) and faster >= needed
    if (
        limit_stop_reason is None
        and load_ok
        and gate_truths(aggregate, "stability_gates")
        and isinstance(ratio, (int, float))
        and ratio > 1.0
        and ci_low_ok
        and faster_ok
    ):
        return "keep_small_gain"

    if decision == "converged_no_meaningful_win" or (
        isinstance(ratio, (int, float)) and ratio <= 1.0
    ):
        return "discard"
    return "inconclusive"


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def row_from_event(event: dict[str, Any]) -> list[str]:
    return [json_cell(event.get(column)) for column in RESULTS_TSV_COLUMNS]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_tsv(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a") as handle:
        if write_header:
            handle.write("\t".join(RESULTS_TSV_COLUMNS) + "\n")
        handle.write("\t".join(row_from_event(event)) + "\n")


def event_from_aggregate(
    aggregate_path: Path,
    *,
    status: str | None,
    description: str,
    artifact: str | None,
) -> dict[str, Any]:
    aggregate = load_json(aggregate_path)
    refs = aggregate.get("refs") if isinstance(aggregate.get("refs"), dict) else {}
    shas = aggregate.get("shas") if isinstance(aggregate.get("shas"), dict) else {}
    inferred_status = status or infer_status(aggregate)
    event = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "event": "benchmark_result",
        "epoch": aggregate.get("run_name"),
        "commit": shas.get("candidate") or shas.get("ref"),
        "baseline_commit": shas.get("baseline"),
        "ref": refs.get("candidate") or refs.get("ref"),
        "baseline_ref": refs.get("baseline"),
        "status": inferred_status,
        "description": description,
        "artifact": artifact or str(aggregate_path),
        "aggregate_path": str(aggregate_path),
    }
    for column in RESULTS_TSV_COLUMNS:
        if column not in event and column in aggregate:
            event[column] = aggregate[column]
    limits = aggregate.get("benchmark_limits")
    if "benchmark_limits" not in event and limits is not None:
        event["benchmark_limits"] = limits
    return event


def record_aggregate(
    root: Path,
    aggregate_path: Path,
    *,
    status: str | None = None,
    description: str = "",
    artifact: str | None = None,
) -> dict[str, Any]:
    event = event_from_aggregate(
        aggregate_path,
        status=status,
        description=description,
        artifact=artifact,
    )
    append_jsonl(root / "events.jsonl", event)
    append_tsv(root / "results.tsv", event)
    return event


def record(args: argparse.Namespace) -> int:
    root = autoresearch_root(args.autoresearch_root)
    event = record_aggregate(
        root,
        args.aggregate,
        status=args.status,
        description=args.description or "",
        artifact=args.artifact,
    )
    print(
        json.dumps(
            {
                "status": event["status"],
                "event": str(root / "events.jsonl"),
                "results": str(root / "results.tsv"),
            },
            indent=2,
        )
    )
    return 0


def init(args: argparse.Namespace) -> int:
    root = autoresearch_root(args.autoresearch_root)
    directories = [
        root / "benchmarks",
        root / "states" / "SuperMarioBros-Nes-v0",
        root / "candidates",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    files_written = []
    current = root / "current.json"
    if not current.exists():
        current.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "initialized",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        files_written.append(str(current))
    ideas = root / "ideas.md"
    if not ideas.exists():
        ideas.write_text("# Autoresearch Ideas\n\n")
        files_written.append(str(ideas))
    scratchpad = root / "scratchpad.md"
    if not scratchpad.exists():
        scratchpad.write_text("# Autoresearch Scratchpad\n\n")
        files_written.append(str(scratchpad))
    results = root / "results.tsv"
    if not results.exists() or results.stat().st_size == 0:
        results.write_text("\t".join(RESULTS_TSV_COLUMNS) + "\n")
        files_written.append(str(results))
    print(
        json.dumps(
            {
                "root": str(root),
                "directories": [str(directory) for directory in directories],
                "files_written": files_written,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def read_jsonl_tail(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records[-limit:]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def read_tsv_tail(path: Path, limit: int = 5) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:]]
    return rows[-limit:]


def status_payload(root: Path, *, limit: int) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    latest_results = read_tsv_tail(root / "results.tsv", limit=limit)
    local_index = read_jsonl_tail(benchmark_index_path(root), limit=limit)
    return {
        "root": str(root),
        "branch": branch,
        "git_status_short": git_status,
        "paths": {
            "results": str(root / "results.tsv"),
            "ideas": str(root / "ideas.md"),
            "scratchpad": str(root / "scratchpad.md"),
            "current": str(root / "current.json"),
            "benchmark_index": str(benchmark_index_path(root)),
        },
        "latest_results": latest_results,
        "latest_benchmarks": local_index,
        "next": infer_next_action(latest_results, local_index),
    }


def status(args: argparse.Namespace) -> int:
    root = autoresearch_root(args.autoresearch_root)
    payload = status_payload(root, limit=args.limit)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def next_action(args: argparse.Namespace) -> int:
    root = autoresearch_root(args.autoresearch_root)
    payload = status_payload(root, limit=args.limit)
    latest_results = payload["latest_results"]
    latest_benchmarks = payload["latest_benchmarks"]
    print(
        json.dumps(
            {
                "root": payload["root"],
                "next": payload["next"],
                "latest_result": latest_results[-1] if latest_results else None,
                "latest_benchmark": latest_benchmarks[-1] if latest_benchmarks else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def infer_next_action(
    latest_results: list[dict[str, str]],
    latest_benchmarks: list[dict[str, Any]],
) -> str:
    latest_result_epoch = latest_results[-1].get("epoch") if latest_results else None
    latest_benchmark = latest_benchmarks[-1] if latest_benchmarks else None
    if latest_benchmark and latest_benchmark.get("run_name") != latest_result_epoch:
        local_dir = latest_benchmark.get("local_result_dir")
        if local_dir:
            aggregate = Path(local_dir) / "aggregate.json"
            return f"Run `autoresearch.py record {aggregate}` for the latest unrecorded benchmark."
    if not latest_results and not latest_benchmarks:
        return (
            "Run `autoresearch.py probe` or `autoresearch.py diagnose --quick` "
            "before exact-ref screening."
        )
    latest_status = latest_results[-1].get("status") if latest_results else None
    if latest_status == "triage_promote":
        return "Run `autoresearch.py checks` and then `autoresearch.py accept <baseline> <candidate>`."
    if latest_status in {"triage_discard", "discard"}:
        return "Record the lesson, then pick the next idea before another probe."
    if latest_status in {"keep", "keep_small_gain", "keep_stack"}:
        return "Update the accepted baseline and refresh calibration when the host is quiet."
    return "Inspect the latest aggregate and decide whether to discard, probe deeper, or screen."


def auto_record_latest_benchmark(root: Path, before_run_names: set[str]) -> dict[str, Any] | None:
    records = read_jsonl(benchmark_index_path(root))
    new_records = [
        record
        for record in records
        if isinstance(record.get("run_name"), str)
        and record["run_name"] not in before_run_names
    ]
    if not new_records:
        print("auto_record: no finalized benchmark aggregate found")
        return None
    latest = new_records[-1]
    local_dir = latest.get("local_result_dir")
    if not isinstance(local_dir, str):
        print("auto_record: latest benchmark index row has no local_result_dir")
        return None
    aggregate_path = Path(local_dir) / "aggregate.json"
    if not aggregate_path.exists():
        print(f"auto_record: aggregate missing at {aggregate_path}")
        return None
    event = record_aggregate(root, aggregate_path)
    print(
        json.dumps(
            {
                "auto_recorded": str(aggregate_path),
                "status": event["status"],
                "results": str(root / "results.tsv"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return event


def run_benchmark_command(
    root: Path,
    command: list[str],
    *,
    dry_run: bool,
    auto_record: bool,
) -> int:
    before_run_names = {
        record["run_name"]
        for record in read_jsonl(benchmark_index_path(root))
        if isinstance(record.get("run_name"), str)
    }
    returncode = run_command(command, dry_run=dry_run)
    if returncode == 0 and auto_record and not dry_run:
        auto_record_latest_benchmark(root, before_run_names)
    return returncode


def check_commands(*, quick: bool, surface: str) -> list[list[str]]:
    if not quick:
        return [
            ["cargo", "fmt", "--check"],
            ["cargo", "check", "--release"],
            [sys.executable, "-m", "maturin", "develop", "--release"],
            ["make", "test"],
        ]
    commands = [["cargo", "fmt", "--check"]]
    if surface in {"auto", "native"}:
        commands.append(["cargo", "check", "--release"])
    if surface in {"auto", "benchmark"}:
        commands.append(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_autoresearch.py",
                "tests/test_run_git_ref_benchmark.py",
                "tests/test_benchmark_stats.py",
            ]
        )
    if surface == "python":
        commands.append([sys.executable, "-m", "pytest", "tests/test_autoresearch.py"])
    return commands


def checks(args: argparse.Namespace) -> int:
    for command in check_commands(quick=args.quick, surface=args.surface):
        print(shlex.join(command))
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return 0


def benchmark_extra_args(args: argparse.Namespace) -> list[str]:
    extra_args = list(args.extra_args)
    if "--dry-run" in extra_args:
        args.dry_run = True
        extra_args = [arg for arg in extra_args if arg != "--dry-run"]
    if "--full" in extra_args:
        args.full = True
        extra_args = [arg for arg in extra_args if arg != "--full"]
    if "--no-record" in extra_args:
        args.no_record = True
        extra_args = [arg for arg in extra_args if arg != "--no-record"]
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return extra_args


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--autoresearch-root")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--autoresearch-root")
    status_parser.add_argument("--limit", type=int, default=5)

    next_parser = subparsers.add_parser("next")
    next_parser.add_argument("--autoresearch-root")
    next_parser.add_argument("--limit", type=int, default=5)

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--autoresearch-root")
    probe_parser.add_argument("--dry-run", action="store_true")

    diagnose_parser = subparsers.add_parser("diagnose")
    diagnose_parser.add_argument("--autoresearch-root")
    diagnose_parser.add_argument("--profile", action="store_true")
    diagnose_parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the old smoke-sized 1000-step single-repeat diagnosis.",
    )
    diagnose_parser.add_argument("--dry-run", action="store_true")

    for name in ("screen", "accept", "accept-stack"):
        benchmark_parser = subparsers.add_parser(name)
        benchmark_parser.add_argument("baseline_ref")
        benchmark_parser.add_argument("candidate_ref")
        if name == "accept":
            benchmark_parser.add_argument(
                "--full",
                action="store_true",
                help="Run the full uncapped acceptance ladder instead of the dedicated-host cap.",
            )
        benchmark_parser.add_argument("--dry-run", action="store_true")
        benchmark_parser.add_argument(
            "--no-record",
            action="store_true",
            help="Skip automatic ledger recording after a successful finalized benchmark.",
        )
        benchmark_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("ref")
    calibrate_parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full uncapped single-ref ladder instead of the dedicated-host cap.",
    )
    calibrate_parser.add_argument("--dry-run", action="store_true")
    calibrate_parser.add_argument(
        "--no-record",
        action="store_true",
        help="Skip automatic ledger recording after a successful finalized benchmark.",
    )
    calibrate_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    checks_parser = subparsers.add_parser("checks")
    checks_parser.add_argument("--dry-run", action="store_true")
    checks_parser.add_argument("--quick", action="store_true")
    checks_parser.add_argument(
        "--surface",
        choices=("auto", "native", "python", "benchmark"),
        default="auto",
    )

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("aggregate", type=Path)
    record_parser.add_argument("--autoresearch-root")
    record_parser.add_argument("--status")
    record_parser.add_argument("--description")
    record_parser.add_argument("--artifact")

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.command == "init":
        return init(args)
    if args.command == "status":
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        return status(args)
    if args.command == "next":
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        return next_action(args)
    if args.command == "probe":
        root = autoresearch_root(args.autoresearch_root)
        return run_command(
            build_probe_command(root),
            dry_run=args.dry_run,
            env_defaults={"RAYON_NUM_THREADS": "12"},
        )
    if args.command == "diagnose":
        root = autoresearch_root(args.autoresearch_root)
        return run_command(
            build_diagnose_command(root, profile=args.profile, quick=args.quick),
            dry_run=args.dry_run,
            env_defaults={"RAYON_NUM_THREADS": "12"},
        )
    if args.command in {"screen", "accept", "accept-stack"}:
        root = autoresearch_root(None)
        command = build_benchmark_command(
            args.command,
            [args.baseline_ref, args.candidate_ref],
            benchmark_extra_args(args),
            full=getattr(args, "full", False),
        )
        return run_benchmark_command(
            root,
            command,
            dry_run=args.dry_run,
            auto_record=not args.no_record,
        )
    if args.command == "calibrate":
        root = autoresearch_root(None)
        command = build_benchmark_command(
            args.command,
            [args.ref],
            benchmark_extra_args(args),
            full=args.full,
        )
        return run_benchmark_command(
            root,
            command,
            dry_run=args.dry_run,
            auto_record=not args.no_record,
        )
    if args.command == "checks":
        return checks(args)
    if args.command == "record":
        return record(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
