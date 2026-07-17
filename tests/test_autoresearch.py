from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.autoresearch import (
    BENCHMARK_SCRIPT,
    auto_record_latest_benchmark,
    benchmark_extra_args,
    build_probe_command,
    build_benchmark_command,
    build_diagnose_command,
    check_commands,
    event_from_aggregate,
    infer_next_action,
    infer_status,
    record,
    run_command,
)
from scripts.run_git_ref_benchmark import RESULTS_TSV_COLUMNS


def test_screen_and_accept_commands_are_canonical() -> None:
    screen = build_benchmark_command("screen", ["base", "cand"], [])
    accept_stack = build_benchmark_command("accept-stack", ["base", "cand"], [])
    accept = build_benchmark_command("accept", ["base", "cand"], ["--force-busy"])
    accept_full = build_benchmark_command(
        "accept", ["base", "cand"], ["--force-busy"], full=True
    )
    calibrate = build_benchmark_command("calibrate", ["main"], [])

    assert screen == [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "base",
        "cand",
        "--steps",
        "5000",
        "--repeats",
        "1",
        "--warmups",
        "0",
        "--max-measured-invocations",
        "3",
    ]
    assert accept_stack == [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "base",
        "cand",
        "--steps",
        "30000",
        "--repeats",
        "2",
        "--warmups",
        "1",
        "--max-measured-invocations",
        "7",
    ]
    assert accept == [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "base",
        "cand",
        "--steps",
        "50000",
        "--repeats",
        "3",
        "--max-measured-invocations",
        "3",
        "--force-busy",
    ]
    assert accept_full == [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "base",
        "cand",
        "--steps",
        "50000",
        "--repeats",
        "3",
        "--force-busy",
    ]
    assert calibrate == [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "main",
        "--single",
        "--steps",
        "50000",
        "--repeats",
        "3",
        "--max-measured-invocations",
        "11",
    ]


def test_benchmark_extra_args_accepts_dry_run_after_refs() -> None:
    class Args:
        extra_args = ["--dry-run", "--full", "--no-record", "--", "--force-busy"]
        dry_run = False
        full = False
        no_record = False

    assert benchmark_extra_args(Args) == ["--force-busy"]
    assert Args.dry_run is True
    assert Args.full is True
    assert Args.no_record is True


def test_diagnose_command_writes_under_autoresearch_root(tmp_path: Path) -> None:
    command = build_diagnose_command(tmp_path, profile=True)
    default_command = build_diagnose_command(tmp_path, profile=False)
    quick_command = build_diagnose_command(tmp_path, profile=False, quick=True)

    assert "--profile-output" in command
    assert str(tmp_path / "benchmarks" / "local-profile.json") in command
    assert str(tmp_path / "benchmarks" / "local-profile-benchmark.json") in command
    assert "--steps" in default_command
    assert default_command[default_command.index("--steps") + 1] == "5000"
    assert default_command[default_command.index("--repeats") + 1] == "3"
    assert default_command[default_command.index("--warmup") + 1] == "500"
    assert quick_command[quick_command.index("--steps") + 1] == "1000"
    assert quick_command[quick_command.index("--repeats") + 1] == "1"
    assert quick_command[quick_command.index("--warmup") + 1] == "20"


def test_probe_command_is_cheaper_than_diagnose(tmp_path: Path) -> None:
    command = build_probe_command(tmp_path)

    assert command[command.index("--steps") + 1] == "2000"
    assert command[command.index("--repeats") + 1] == "1"
    assert command[command.index("--warmup") + 1] == "50"
    assert str(tmp_path / "benchmarks" / "local-probe.json") in command


def test_quick_checks_are_surface_scoped() -> None:
    native = check_commands(quick=True, surface="native")
    benchmark = check_commands(quick=True, surface="benchmark")
    full = check_commands(quick=False, surface="auto")

    assert native == [["cargo", "fmt", "--check"], ["cargo", "check", "--release"]]
    assert benchmark[0] == ["cargo", "fmt", "--check"]
    assert "tests/test_run_git_ref_benchmark.py" in benchmark[1]
    assert [sys.executable, "-m", "maturin", "develop", "--release"] in full
    assert ["make", "test"] in full


def test_infer_next_action_promotes_screen_results() -> None:
    assert "probe" in infer_next_action([], [])
    assert "accept" in infer_next_action([{"status": "triage_promote"}], [])
    assert "next idea" in infer_next_action([{"status": "discard"}], [])
    assert "record" in infer_next_action(
        [],
        [{"run_name": "run-1", "local_result_dir": "/tmp/run-1"}],
    )


def test_run_command_prints_env_defaults_for_dry_run(capsys, monkeypatch) -> None:
    monkeypatch.delenv("AUTORESEARCH_TEST_ENV_DEFAULT", raising=False)

    assert (
        run_command(
            ["python", "-V"],
            dry_run=True,
            env_defaults={"AUTORESEARCH_TEST_ENV_DEFAULT": "1"},
        )
        == 0
    )
    assert capsys.readouterr().out == "AUTORESEARCH_TEST_ENV_DEFAULT=1 python -V\n"


def test_infer_status_uses_benchmark_decision_fields() -> None:
    assert infer_status({"benchmark_tier": "local_triage", "median_pair_ratio": 1.04}) == "triage_promote"
    assert infer_status({"benchmark_tier": "local_triage", "median_pair_ratio": 1.0}) == "triage_discard"
    assert (
        infer_status(
            {
                "benchmark_tier": "local_acceptance",
                "decision": "converged_candidate_win",
                "validity_passed": True,
                "load_gate_passed": True,
            }
        )
        == "keep"
    )
    assert (
        infer_status(
            {
                "benchmark_tier": "local_acceptance",
                "decision": "continue",
                "load_gate_passed": True,
                "stability_gates": {"stable": True},
                "median_pair_ratio": 1.001,
                "pair_ratio_bootstrap_ci95": [1.0, 1.002],
                "candidate_faster_pairs": 3,
                "candidate_faster_pairs_required_for_win": 3,
            }
        )
        == "keep_small_gain"
    )


def test_infer_status_handles_stack_acceptance() -> None:
    assert (
        infer_status(
            {
                "benchmark_tier": "stack_acceptance",
                "validity_passed": True,
                "load_gate_passed": True,
                "median_pair_ratio": 1.031,
                "candidate_faster_pairs": 6,
            }
        )
        == "keep_stack"
    )
    assert (
        infer_status(
            {
                "benchmark_tier": "stack_acceptance",
                "decision": "continue",
                "load_gate_passed": True,
                "median_pair_ratio": 1.0,
                "candidate_faster_pairs": 4,
            }
        )
        == "discard_stack"
    )
    assert (
        infer_status(
            {
                "benchmark_tier": "stack_acceptance",
                "decision": "continue",
                "validity_passed": False,
                "load_gate_passed": True,
                "median_pair_ratio": 1.025,
                "candidate_faster_pairs": 5,
            }
        )
        == "inconclusive"
    )


def test_record_appends_events_and_results_tsv(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    root.mkdir()
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "run_name": "benchmark-compare-test",
                "benchmark_tier": "local_triage",
                "mode": "paired_compare_fixed_local",
                "workload_hash": "abc123",
                "refs": {"baseline": "main", "candidate": "HEAD"},
                "shas": {"baseline": "1" * 40, "candidate": "2" * 40},
                "median_pair_ratio": 1.04,
                "measured_pairs": 3,
                "expected_rom_sha256": "expected",
                "rom_sha256": "actual",
            }
        )
        + "\n"
    )

    class Args:
        aggregate = aggregate_path
        autoresearch_root = str(root)
        status = None
        description = "screen passed"
        artifact = None

    assert record(Args) == 0
    event = json.loads((root / "events.jsonl").read_text().splitlines()[0])
    rows = (root / "results.tsv").read_text().splitlines()

    assert event["status"] == "triage_promote"
    assert event["commit"] == "2" * 40
    assert rows[0].split("\t") == list(RESULTS_TSV_COLUMNS)
    header = rows[0].split("\t")
    result = rows[1].split("\t")
    assert len(header) == len(result)
    values = dict(zip(header, result))
    assert values["status"] == "triage_promote"
    assert values["median_pair_ratio"] == "1.04"
    assert values["description"] == "screen passed"


def test_auto_record_latest_benchmark_appends_new_index_entry(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    local_dir = root / "benchmarks" / "local-results" / "run-new"
    local_dir.mkdir(parents=True)
    aggregate_path = local_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "run_name": "run-new",
                "benchmark_tier": "local_triage",
                "refs": {"baseline": "main", "candidate": "HEAD"},
                "shas": {"baseline": "1" * 40, "candidate": "2" * 40},
                "median_pair_ratio": 1.04,
            }
        )
        + "\n"
    )
    index_path = root / "benchmarks" / "local-results" / "index.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "run_name": "run-new",
                "local_result_dir": str(local_dir),
            }
        )
        + "\n"
    )

    event = auto_record_latest_benchmark(root, before_run_names=set())
    rows = (root / "results.tsv").read_text().splitlines()

    assert event is not None
    assert event["status"] == "triage_promote"
    assert rows[0].split("\t") == list(RESULTS_TSV_COLUMNS)
    assert "triage_promote" in rows[1]


def test_event_from_aggregate_allows_status_override(tmp_path: Path) -> None:
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps({"benchmark_tier": "local_diagnosis"}) + "\n")

    event = event_from_aggregate(
        aggregate_path,
        status="discard",
        description="manual rejection",
        artifact=None,
    )

    assert event["status"] == "discard"
    assert event["description"] == "manual rejection"
