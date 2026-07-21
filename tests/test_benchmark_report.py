from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_report


def benchmark_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        python=Path("python"),
        benchmark_script=Path("scripts/benchmark_sps.py"),
        rom_path=tmp_path / "rom.nes",
        state_dir=None,
        steps=100,
        repeats=3,
        warmup=10,
        force_busy=False,
        max_start_load=4.0,
        load_poll_seconds=0.01,
        max_load_wait_seconds=1.0,
    )


def payload(backend: str, shape: int, *, sps: float = 1000.0) -> dict[str, object]:
    vectorization = "native" if backend == "turbo" else "gymnasium.AsyncVectorEnv"
    return {
        "backend": backend,
        "package": {
            "name": (
                "supermariobrosnes-turbo" if backend == "turbo" else "stable-retro-turbo"
            ),
            "version": "test",
            "import": "supermariobrosnes_turbo" if backend == "turbo" else "stable_retro",
        },
        "config": {
            "rom_sha256": "rom-sha",
            "num_envs": shape,
            "steps": 100,
            "repeats": 3,
            "warmup": 10,
            "frame_skip": 4,
            "frame_stack": 4,
            "frame_maxpool": False,
            "grayscale": True,
            "crop_top": 32,
            "crop_bottom": 0,
            "obs_crop_mode": "mask",
            "resize_width": 84,
            "resize_height": 84,
            "obs_resize_algorithm": "area",
            "obs_layout": "chw",
            "action_set": "basic",
            "action": None,
            "actions": ["noop", "right", "right_b", "right_a"],
            "action_seed": 0,
            "state": None,
            "states": ["Level1-1", "Level1-2", "Level1-3", "Level1-4"],
            "lane_states": ["Level1-1"] * shape,
            "include_info": True,
            "terminate_on_flag": False,
            "termination": "provider_native",
            "start_game": False,
            "vectorization": vectorization,
        },
        "observation": {
            "shape": [shape, 4, 84, 84],
            "dtype": "uint8",
            "bytes": shape * 4 * 84 * 84,
            "mib": shape * 4 * 84 * 84 / 1024**2,
        },
        "load": {"enabled": True, "load_ok": True},
        "runs": [{"env_steps_per_sec": sps - 1}, {"env_steps_per_sec": sps}, {"env_steps_per_sec": sps + 1}],
    }


def test_parse_shapes_and_pair_order() -> None:
    assert benchmark_report.parse_shapes("1,16,32") == (1, 16, 32)
    assert benchmark_report.pair_order(0, 0) == ("turbo", "stable-retro")
    assert benchmark_report.pair_order(0, 1) == ("stable-retro", "turbo")
    assert benchmark_report.pair_order(1, 0) == ("stable-retro", "turbo")

    with pytest.raises(Exception, match="positive integers"):
        benchmark_report.parse_shapes("1,0")
    with pytest.raises(Exception, match="duplicates"):
        benchmark_report.parse_shapes("1,1")


def test_benchmark_command_delegates_to_canonical_script(tmp_path: Path) -> None:
    args = benchmark_args(tmp_path)
    command = benchmark_report.benchmark_command(
        args,
        backend="stable-retro",
        shape=32,
        output_json=tmp_path / "raw.json",
    )

    assert command[:2] == ["python", "scripts/benchmark_sps.py"]
    assert command[command.index("--num-envs") + 1] == "32"
    assert "--stable-retro-baseline" in command
    assert "--skip-load-preflight" in command
    assert "--max-start-load" not in command
    assert command[command.index("--warmup") + 1] == "10"
    assert command.count("10") == 1
    assert "--frame-stack" in command
    assert "--obs-crop-mode" in command


def test_load_preflight_waits_once_until_host_is_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = benchmark_args(tmp_path)
    loads = iter(((6.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    monkeypatch.setattr(benchmark_report.os, "getloadavg", lambda: next(loads))
    monkeypatch.setattr(benchmark_report.time, "sleep", lambda _seconds: None)

    result = benchmark_report.wait_for_load_headroom(args)

    assert result["enabled"] is True
    assert result["initial_1min"] == 6.0
    assert result["accepted_1min"] == 3.0
    assert result["max_start_load"] == 4.0
    assert result["load_ok"] is True
    output = capsys.readouterr().out
    assert output.count("waiting_for_load") == 1
    assert output.count("load_preflight") == 1


def test_forced_busy_preflight_is_recorded_as_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = benchmark_args(tmp_path)
    args.force_busy = True
    monkeypatch.setattr(benchmark_report.os, "getloadavg", lambda: (8.0, 0.0, 0.0))

    result = benchmark_report.wait_for_load_headroom(args)

    assert result["enabled"] is False
    assert result["initial_1min"] == 8.0
    assert result["load_ok"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--pairs", "9", "--output-dir", "/tmp/report", "--steps", "50000"],
        ["--pairs", "9", "--output-dir=/tmp/report", "--steps", "50000"],
    ],
)
def test_reproduction_args_omit_one_use_output_directory(arguments: list[str]) -> None:
    assert benchmark_report.reproduction_args(arguments) == [
        "--pairs",
        "9",
        "--steps",
        "50000",
    ]


def test_pair_validation_requires_identical_workload() -> None:
    turbo = payload("turbo", 4)
    stable = payload("stable-retro", 4)

    benchmark_report.validate_matched_pair(
        turbo,
        stable,
        turbo_path=Path("turbo.json"),
        stable_path=Path("stable.json"),
    )

    stable["config"]["action_seed"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="action_seed"):
        benchmark_report.validate_matched_pair(
            turbo,
            stable,
            turbo_path=Path("turbo.json"),
            stable_path=Path("stable.json"),
        )


def test_invocation_stats_line_reports_run_distribution() -> None:
    line = benchmark_report.format_invocation_stats(
        payload("turbo", 1, sps=1_000.0), Path("turbo.json")
    )

    assert line == (
        "median_sps=1000.0 mean_sps=1000.0 stdev_sps=1.0 cv_pct=0.10 "
        "min_sps=999.0 max_sps=1001.0 repeats=3"
    )


def test_shape_aggregation_uses_paired_ratios_and_claim_gates() -> None:
    pairs = [
        {
            "turbo_median_sps": 10_000.0 + index,
            "stable_retro_median_sps": 2_000.0,
            "speedup": (10_000.0 + index) / 2_000.0,
            "load_ok": True,
        }
        for index in range(7)
    ]

    result = benchmark_report.aggregate_shape(
        shape=16,
        pairs=pairs,
        minimum_speedup=2.0,
        minimum_pairs_for_claim=5,
        bootstrap_samples=1_000,
        load_gate_enforced=True,
    )

    assert result["turbo_invocation_median_sps"]["median"] == 10_003.0
    assert result["stable_retro_invocation_median_sps"]["median"] == 2_000.0
    assert result["pair_speedup"]["median"] == pytest.approx(5.0015)
    assert result["pair_speedup_bootstrap_ci95"][0] > 1.0
    assert result["claim_passed"] is True


def test_report_renders_result_and_validity() -> None:
    shape_result = benchmark_report.aggregate_shape(
        shape=1,
        pairs=[
            {
                "turbo_median_sps": 10_000.0,
                "stable_retro_median_sps": 2_000.0,
                "speedup": 5.0,
                "load_ok": True,
            }
            for _ in range(5)
        ],
        minimum_speedup=2.0,
        minimum_pairs_for_claim=5,
        bootstrap_samples=100,
        load_gate_enforced=True,
    )
    aggregate = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "claim_passed": True,
        "settings": {"minimum_speedup": 2.0},
        "packages": {
            "turbo": {"name": "supermariobrosnes-turbo", "version": "0.3.0"},
            "stable-retro": {
                "name": "stable-retro-turbo",
                "version": "1.0.1.post33",
            },
        },
        "results": [shape_result],
        "source": {
            "commit": "abc123",
            "working_tree_diff_sha256": "diff-sha",
        },
        "correctness": {"passed": True},
        "validity": {
            "source_clean": True,
            "load_gate_enforced": True,
            "all_shape_claims_passed": True,
        },
        "system": {
            "platform": "test-platform",
            "machine": "test-machine",
            "processor": "test-cpu",
            "logical_cpus": 8,
            "python": "3.14 test",
        },
        "load_preflight": {
            "initial_1min": 2.0,
            "accepted_1min": 2.0,
            "max_start_load": 4.0,
        },
        "reproduction_command": "make benchmark-report",
    }

    report = benchmark_report.render_report(aggregate)

    assert "**PASS:**" in report
    assert "5.00x" in report
    assert "Exact ROM-backed parity checks: **PASS**" in report
    assert "make benchmark-report" in report
