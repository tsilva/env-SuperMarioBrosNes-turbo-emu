from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_git_ref_benchmark import (
    ACTION_NAMES,
    ACTION_SEED,
    BenchmarkPlan,
    BenchmarkRef,
    EXPECTED_SMB_ROM_SHA256,
    RESULTS_TSV_COLUMNS,
    STATE_NAMES,
    append_index,
    aggregate_with_extra_load_snapshot,
    aggregate_single,
    base_aggregate,
    benchmark_command,
    benchmark_tier,
    build_plan,
    cap_checkpoints,
    decide_mode,
    execute,
    load_ok_for_validity,
    load_gate_stop_reason,
    load_raw,
    load_snapshot_shell,
    measured_invocation_limit_applies,
    parse_args,
    parse_load1,
    prepare_source_cache,
    prepared_source_dir,
    prepared_source_is_usable,
    require_load_gate,
    require_wall_clock_budget,
    run_invocation,
    run_compare,
    source_cache_root_for_plan,
    link_prepared_source,
    uv_sync_command,
    validate_rom_hash,
)


def benchmark_raw_config(
    plan: BenchmarkPlan,
    args: SimpleNamespace,
    *,
    steps: int | None = None,
    repeats: int | None = None,
) -> dict[str, object]:
    return {
        "rom_path": plan.rom_path,
        "rom_sha256": EXPECTED_SMB_ROM_SHA256,
        "rayon_num_threads": 12,
        "num_envs": 16,
        "steps": args.steps if steps is None else steps,
        "repeats": args.repeats if repeats is None else repeats,
        "warmup": 100,
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
        "action_set": "simple",
        "action": None,
        "actions": list(ACTION_NAMES),
        "action_seed": ACTION_SEED,
        "state": None,
        "states": list(STATE_NAMES),
        "lane_states": [STATE_NAMES[index % len(STATE_NAMES)] for index in range(16)],
        "state_dir": plan.state_dir,
        "include_info": True,
        "terminate_on_flag": False,
        "termination": "provider_native",
        "start_game": False,
        "vectorization": "native",
    }


def write_raw(path: Path, values: list[float], config: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            {
                "package": {
                    "name": "supermariobrosnes-turbo",
                    "version": "0.0.test",
                    "import": "supermariobrosnes_turbo",
                },
                "config": config,
                "runs": [{"env_steps_per_sec": value} for value in values],
            }
        )
        + "\n"
    )


def valid_package_metadata() -> dict[str, object]:
    return {
        "name": "supermariobrosnes-turbo",
        "version": "0.0.test",
        "import": "supermariobrosnes_turbo",
    }


def write_benchmark_dotenv(tmp_path: Path) -> None:
    autoresearch_root = tmp_path / "autoresearch"
    autoresearch_root.mkdir()
    (tmp_path / ".env").write_text(
        f"AUTORESEARCH_ROOT_PATH={autoresearch_root}\n"
    )


def test_decide_mode_rejects_multi_ref_single() -> None:
    try:
        decide_mode(["a", "b"], single=True)
    except SystemExit as exc:
        assert "--single requires exactly one ref" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_parse_args_reads_retro_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "retro-data"
    rom_path = data_root / "stable" / "SuperMarioBros-Nes-v0" / "rom.nes"
    rom_path.parent.mkdir(parents=True)
    rom_path.write_bytes(b"placeholder")
    monkeypatch.setenv("RETRO_DATA_PATH", str(data_root))
    write_benchmark_dotenv(tmp_path)

    args = parse_args(["--single", "HEAD", "--dry-run"])

    assert args.rom_path == str(rom_path)


def test_parse_args_dry_run_allows_planned_missing_rom_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    rom_path = tmp_path / "planned" / "SuperMarioBros.nes"
    write_benchmark_dotenv(tmp_path)

    args = parse_args(["--single", "HEAD", "--dry-run", "--rom-path", str(rom_path)])

    assert args.rom_path == str(rom_path)


def test_parse_args_real_run_rejects_missing_rom_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    rom_path = tmp_path / "planned" / "SuperMarioBros.nes"
    write_benchmark_dotenv(tmp_path)

    with pytest.raises(SystemExit, match="ROM path does not exist"):
        parse_args(["--single", "HEAD", "--rom-path", str(rom_path)])


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--steps", "0", "--steps must be positive"),
        ("--repeats", "0", "--repeats must be positive"),
        ("--warmups", "-1", "--warmups must be non-negative"),
        ("--max-measured-invocations", "0", "--max-measured-invocations must be positive"),
        ("--max-wall-clock-minutes", "0", "--max-wall-clock-minutes must be positive"),
    ],
)
def test_parse_args_rejects_invalid_benchmark_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    rom_path = tmp_path / "SuperMarioBros.nes"
    rom_path.write_bytes(b"placeholder")
    write_benchmark_dotenv(tmp_path)

    with pytest.raises(SystemExit, match=message):
        parse_args(["--single", "HEAD", "--dry-run", "--rom-path", str(rom_path), flag, value])


def test_parse_load1_extracts_unix_load_average() -> None:
    assert parse_load1("load average: 1.23, 4.56, 7.89") == 1.23
    assert parse_load1(" 15:41 up 10 days,  load average: 0.42, 0.50, 0.60") == 0.42
    assert parse_load1("no load here") is None


def test_load_snapshot_shell_closes_group_once() -> None:
    shell = load_snapshot_shell("/tmp/load.txt")

    assert "; } > /tmp/load.txt" in shell
    assert "}}" not in shell


def test_uv_sync_command_includes_common_user_tool_paths() -> None:
    command = uv_sync_command()

    assert "$HOME/.local/bin" in command
    assert "$HOME/.cargo/bin" in command
    assert command.endswith("uv sync --frozen --no-dev")


def test_prepare_source_cache_reuses_synced_commit_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "candidate.tar.gz"
    archive.write_bytes(b"archive bytes")
    plan = BenchmarkPlan(
        mode="compare",
        run_name="cache-test",
        run_dir=str(tmp_path / "benchmarks" / "runs" / "cache-test"),
        refs=[],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(3,),
        warmups=0,
        measured_cap=3,
    )
    ref = BenchmarkRef("candidate", "cand", "2" * 40, archive)
    calls: list[str] = []

    def fake_run_stream(_args: object, _plan: BenchmarkPlan, shell: str) -> None:
        calls.append(shell)
        prepared_dir = prepared_source_dir(plan, ref)
        tmp_dir = prepared_dir.with_name(f"{prepared_dir.name}.tmp")
        (tmp_dir / ".venv" / "bin").mkdir(parents=True)
        site_packages = tmp_dir / ".venv" / "lib" / "python3.14" / "site-packages"
        site_packages.mkdir(parents=True)
        (tmp_dir / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
        (site_packages / "supermariobrosnes_turbo.pth").write_text(
            str(tmp_dir / "python") + "\n"
        )

    monkeypatch.setattr("scripts.run_git_ref_benchmark.target_run_stream", fake_run_stream)

    cache_dir = prepare_source_cache(SimpleNamespace(), plan, ref)
    cached_again = prepare_source_cache(SimpleNamespace(), plan, ref)
    link_prepared_source(plan, ref, cache_dir)

    assert source_cache_root_for_plan(plan) == tmp_path / "benchmarks" / "prepared-sources"
    assert cache_dir == cached_again
    assert len(calls) == 1
    assert prepared_source_is_usable(cache_dir, ref)
    pth = (
        cache_dir
        / ".venv"
        / "lib"
        / "python3.14"
        / "site-packages"
        / "supermariobrosnes_turbo.pth"
    )
    assert pth.read_text() == str(cache_dir / "python") + "\n"
    assert (Path(plan.run_dir) / "sources" / "candidate").is_symlink()

    pth.write_text(str(cache_dir.with_name(f"{cache_dir.name}.tmp") / "python") + "\n")
    assert not prepared_source_is_usable(cache_dir, ref)


def test_cap_checkpoints_treats_limit_as_upper_bound() -> None:
    assert cap_checkpoints((7, 11, 15, 21, 31), None) == (7, 11, 15, 21, 31)
    assert cap_checkpoints((7, 11, 15, 21, 31), 3) == (3,)
    assert cap_checkpoints((7, 11, 15, 21, 31), 11) == (7, 11)
    assert cap_checkpoints((7, 11, 15, 21, 31), 40) == (7, 11, 15, 21, 31)


def test_measured_invocation_limit_reason_only_applies_when_shortened(tmp_path: Path) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="limit-reason-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(7, 11, 15, 21, 31),
        warmups=2,
        measured_cap=31,
    )

    assert measured_invocation_limit_applies(
        SimpleNamespace(max_measured_invocations=3), plan
    )
    assert not measured_invocation_limit_applies(
        SimpleNamespace(max_measured_invocations=31), plan
    )
    assert not measured_invocation_limit_applies(
        SimpleNamespace(max_measured_invocations=40), plan
    )


def test_benchmark_tier_labels_official_and_triage_shapes(tmp_path: Path) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="tier-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(7, 11, 15, 21, 31),
        warmups=2,
        measured_cap=31,
    )

    assert benchmark_tier(
        SimpleNamespace(steps=50000, repeats=3, max_measured_invocations=None),
        plan,
    ) == "local_acceptance"
    assert benchmark_tier(
        SimpleNamespace(steps=5000, repeats=1, max_measured_invocations=3),
        BenchmarkPlan(
            mode=plan.mode,
            run_name=plan.run_name,
            run_dir=plan.run_dir,
            refs=plan.refs,
            rom_path=plan.rom_path,
            state_dir=plan.state_dir,
            checkpoints=(3,),
            warmups=0,
            measured_cap=3,
        ),
    ) == "local_triage"
    assert benchmark_tier(
        SimpleNamespace(steps=5000, repeats=1, max_measured_invocations=None),
        plan,
    ) == "local_diagnosis"


def test_benchmark_tier_labels_stack_acceptance_shape(tmp_path: Path) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="stack-tier-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(3, 5, 7),
        warmups=1,
        measured_cap=7,
    )

    assert benchmark_tier(
        SimpleNamespace(steps=30000, repeats=2, max_measured_invocations=7),
        plan,
    ) == "stack_acceptance"


def test_stack_acceptance_plan_uses_short_checkpoint_ladder(tmp_path: Path) -> None:
    plan = build_plan(
        SimpleNamespace(
            refs=["1" * 40, "2" * 40],
            single=False,
            run_root=str(tmp_path / "runs-root"),
            state_dir=str(tmp_path / "states"),
            rom_path=str(tmp_path / "SuperMarioBros.nes"),
            steps=30000,
            repeats=2,
            warmups=1,
            max_measured_invocations=7,
        )
    )

    assert plan.checkpoints == (3, 5, 7)
    assert plan.measured_cap == 7


def test_benchmark_command_pins_canonical_workload_flags(tmp_path: Path) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="command-flags-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(7,),
        warmups=2,
        measured_cap=7,
    )

    args = SimpleNamespace(max_load=4.0)
    command = benchmark_command(
        args,
        plan,
        "candidate",
        "measured-candidate-00",
        steps=50000,
        repeats=3,
    )

    for expected in (
        "--num-envs 16",
        "--steps 50000 --repeats 3",
        "--warmup 100",
        "--frame-skip 4",
        "--frame-stack 4",
        "--crop-top 32",
        "--crop-bottom 0",
        "--resize-width 84",
        "--resize-height 84",
        "--states Level1-1,Level1-2,Level1-3,Level1-4",
        "--action-set simple --actions noop,right,right_b,right_a",
        "--action-seed 0",
        "--no-start-game",
    ):
        assert expected in command


def test_run_invocation_waits_for_load_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="load-wait-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(1,),
        warmups=0,
        measured_cap=1,
    )
    args = SimpleNamespace(
        force_busy=False,
        max_load=4.0,
        max_wall_clock_minutes=None,
    )
    loads = iter([(3.9, ""), (3.8, ""), (3.3, "")])
    labels: list[str] = []
    sleeps: list[float] = []
    commands: list[str] = []

    def fake_capture_load(_args: object, _plan: BenchmarkPlan, label: str) -> tuple[float, str]:
        labels.append(label)
        return next(loads)

    monkeypatch.setattr("scripts.run_git_ref_benchmark.capture_load", fake_capture_load)
    monkeypatch.setattr("scripts.run_git_ref_benchmark.time.sleep", sleeps.append)
    monkeypatch.setattr(
        "scripts.run_git_ref_benchmark.target_run_stream",
        lambda _args, _plan, shell: commands.append(shell),
    )

    run_invocation(
        args,
        plan,
        "candidate",
        "measured-candidate-00",
        steps=50000,
        repeats=3,
        start_time=0.0,
    )

    assert labels == [
        "before-measured-candidate-00",
        "before-measured-candidate-00-retry-01",
        "before-measured-candidate-00-retry-02",
    ]
    assert sleeps == [15.0, 15.0]
    assert len(commands) == 1


def test_execute_dry_run_reports_tier_and_workload_hash(tmp_path: Path) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="dry-run-tier-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(7, 11, 15, 21, 31),
        warmups=2,
        measured_cap=31,
    )
    args = SimpleNamespace(
        dry_run=True,
        steps=50000,
        repeats=3,
        max_load=1.0,
        force_busy=False,
        max_measured_invocations=None,
        max_wall_clock_minutes=None,
    )

    payload = execute(args, plan)

    assert payload["benchmark_tier"] == "local_acceptance"
    assert payload["workload"]["obs_resize_algorithm"] == "area"
    assert len(payload["workload_hash"]) == 64
    assert payload["planned_workload_hash"] == payload["workload_hash"]
    assert payload["workload_hash_scope"] == "planned_without_rom_or_state_file_hashes"


def test_force_busy_overrides_load_for_validity_only() -> None:
    assert not load_ok_for_validity(SimpleNamespace(force_busy=False, max_load=1.0), [2.0])
    assert load_ok_for_validity(SimpleNamespace(force_busy=True, max_load=1.0), [2.0])


def test_load_gate_blocks_missing_or_busy_load_before_expensive_phase() -> None:
    args = SimpleNamespace(force_busy=False, max_load=1.0)

    with pytest.raises(SystemExit, match="benchmark load unavailable before measured phase"):
        require_load_gate(args, None, "measured phase")
    with pytest.raises(SystemExit, match="benchmark load 2.00 meets or exceeds max 1.00 before measured phase"):
        require_load_gate(args, 2.0, "measured phase")
    with pytest.raises(SystemExit, match="benchmark load 1.00 meets or exceeds max 1.00 before measured phase"):
        require_load_gate(args, 1.0, "measured phase")

    require_load_gate(SimpleNamespace(force_busy=True, max_load=1.0), 2.0, "measured phase")


def test_checkpoint_load_gate_failure_gets_stop_reason() -> None:
    args = SimpleNamespace(force_busy=False, max_load=1.0)

    assert load_gate_stop_reason(args, 2.0) == "load_gate_failed"
    assert load_gate_stop_reason(args, None) == "load_gate_failed"
    assert load_gate_stop_reason(args, 1.0) == "load_gate_failed"
    assert load_gate_stop_reason(args, 0.5) is None
    assert load_gate_stop_reason(SimpleNamespace(force_busy=True, max_load=1.0), 2.0) is None


def test_wall_clock_budget_error_names_phase() -> None:
    args = SimpleNamespace(max_wall_clock_minutes=0.001)

    with pytest.raises(SystemExit, match="wall-clock limit exhausted before source preparation"):
        require_wall_clock_budget(args, start_time=0.0, phase="source preparation")


def test_wall_clock_budget_allows_unlimited_runs() -> None:
    args = SimpleNamespace(max_wall_clock_minutes=None)

    require_wall_clock_budget(args, start_time=0.0, phase="source preparation")


def test_validate_rom_hash_rejects_wrong_rom_bytes(tmp_path: Path) -> None:
    rom_path = tmp_path / "not-the-rom.nes"
    rom_path.write_bytes(b"not the supported rom")

    with pytest.raises(SystemExit, match="ROM SHA-256 mismatch"):
        validate_rom_hash(str(rom_path))


def test_campaign_results_header_tracks_benchmark_identity_and_rom_digest(
    tmp_path: Path,
) -> None:
    aggregate = {
        "benchmark_tier": "local_acceptance",
        "workload_hash": "workload-sha",
        "measured_pair_details": [],
        "expected_rom_sha256": "expected-rom-sha",
        "rom_sha256": "actual-rom-sha",
        "state_sha256": {"Level1-1": "state-sha"},
    }
    append_index(
        "header-contract-test",
        tmp_path / "runs" / "header-contract-test",
        aggregate,
        tmp_path,
    )
    record = json.loads((tmp_path / "index.jsonl").read_text())

    assert record["benchmark_tier"] == "local_acceptance"
    assert record["workload_hash"] == "workload-sha"
    assert record["expected_rom_sha256"] == "expected-rom-sha"
    assert record["rom_sha256"] == "actual-rom-sha"
    assert record["state_sha256"] == {"Level1-1": "state-sha"}


def test_skill_delegates_results_tsv_header_to_benchmark_record_names() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    skill_text = (repo_root / ".codex" / "skills" / "autoresearch-speed" / "SKILL.md").read_text()

    assert "scripts/autoresearch.py record" in skill_text
    assert "RESULTS_TSV_COLUMNS" in skill_text
    assert "\t".join(RESULTS_TSV_COLUMNS) not in skill_text


def test_aggregate_single_uses_convergence_helper(tmp_path: Path) -> None:
    run_dir = tmp_path / "local-single-test"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    medians = [
        1000.0,
        1002.0,
        1001.0,
        1003.0,
        1001.0,
        1002.0,
        1000.0,
        1001.0,
        1002.0,
        1001.0,
        1000.0,
    ]
    plan = BenchmarkPlan(
        mode="single",
        run_name="local-single-test",
        run_dir=str(run_dir),
        refs=[
            BenchmarkRef(
                role="ref",
                ref="HEAD",
                sha="1" * 40,
                archive=tmp_path / "ref-111111111111.tar.gz",
            )
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(5, 8, 11, 15, 21, 31),
        warmups=2,
        measured_cap=31,
    )
    args = SimpleNamespace(
        max_load=99.0,
        force_busy=False,
        steps=50000,
        repeats=3,
        max_measured_invocations=None,
        max_wall_clock_minutes=None,
    )
    config = benchmark_raw_config(plan, args)
    smoke_config = benchmark_raw_config(plan, args, steps=1000, repeats=1)
    write_raw(raw_dir / "smoke-ref.json", [1000.0], smoke_config)
    for index in range(plan.warmups):
        write_raw(raw_dir / f"warmup-ref-{index:02d}.json", [1000.0], config)
    for index, value in enumerate(medians):
        write_raw(
            raw_dir / f"measured-ref-{index:02d}.json",
            [value - 1.0, value, value + 1.0],
            config,
        )

    aggregate = aggregate_single(
        args,
        plan,
        measured_count=11,
        load_values=[0.5, 0.4],
        load_labels=["before-setup", "before-measured"],
    )

    assert aggregate["mode"] == "single_ref_fixed_local"
    assert aggregate["benchmark_tier"] == "local_diagnosis"
    assert len(aggregate["workload_hash"]) == 64
    assert aggregate["decision"] == "converged"
    assert aggregate["official_median_sps"] == 1001.0
    assert aggregate["checkpoint_trace"][-1]["count"] == 11
    assert aggregate["load_gate_passed"]
    assert not aggregate["load_gate_ignored_for_validity"]
    assert aggregate["smoke_raw_files"] == [{"file": "raw/smoke-ref.json", "tier": "smoke"}]
    assert aggregate["warmup_raw_files"] == [
        "raw/warmup-ref-00.json",
        "raw/warmup-ref-01.json",
    ]
    assert aggregate["setup_only_raw_files"] == [
        {"file": "raw/smoke-ref.json", "tier": "smoke"},
        {"file": "raw/warmup-ref-00.json", "tier": "warmup"},
        {"file": "raw/warmup-ref-01.json", "tier": "warmup"},
    ]
    assert aggregate["load_1min_labels"] == ["before-setup", "before-measured"]
    assert aggregate["load_1min_by_label"] == {"before-setup": 0.5, "before-measured": 0.4}
    assert aggregate["workload"]["rom_path"] == plan.rom_path
    assert aggregate["workload"]["state_dir"] == plan.state_dir
    assert aggregate["workload"]["warmup"] == 100
    assert aggregate["workload"]["crop_bottom"] == 0
    assert aggregate["workload"]["obs_crop_mode"] == "mask"
    assert aggregate["workload"]["obs_resize_algorithm"] == "area"
    assert aggregate["workload"]["expected_rom_sha256"] == EXPECTED_SMB_ROM_SHA256
    assert aggregate["workload"]["rom_sha256"] is None
    assert aggregate["workload"]["state_sha256"] == {
        "Level1-1": None,
        "Level1-2": None,
        "Level1-3": None,
        "Level1-4": None,
    }


def test_workload_hash_includes_state_file_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="state-hash-workload-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path="/roms/SuperMarioBros.nes",
        state_dir="/states/SuperMarioBros-Nes-v0",
        checkpoints=(7,),
        warmups=2,
        measured_cap=7,
    )
    args = SimpleNamespace(
        steps=50000,
        repeats=3,
        max_load=1.0,
        force_busy=False,
        max_measured_invocations=None,
        max_wall_clock_minutes=None,
    )
    file_hash_sets = iter(
        [
            {
                "/roms/SuperMarioBros.nes": EXPECTED_SMB_ROM_SHA256,
                "/states/SuperMarioBros-Nes-v0/Level1-1.state": "state-a",
                "/states/SuperMarioBros-Nes-v0/Level1-2.state": "state-b",
                "/states/SuperMarioBros-Nes-v0/Level1-3.state": "state-c",
                "/states/SuperMarioBros-Nes-v0/Level1-4.state": "state-d",
            },
            {
                "/roms/SuperMarioBros.nes": EXPECTED_SMB_ROM_SHA256,
                "/states/SuperMarioBros-Nes-v0/Level1-1.state": "state-a",
                "/states/SuperMarioBros-Nes-v0/Level1-2.state": "state-b",
                "/states/SuperMarioBros-Nes-v0/Level1-3.state": "state-c",
                "/states/SuperMarioBros-Nes-v0/Level1-4.state": "changed-state-d",
            },
        ]
    )

    monkeypatch.setattr(
        "scripts.run_git_ref_benchmark.target_file_hashes",
        lambda *_args: next(file_hash_sets),
    )

    first = base_aggregate(args, plan, [0.5])
    second = base_aggregate(args, plan, [0.5])

    assert first["workload"]["state_sha256"]["Level1-4"] == "state-d"
    assert second["workload"]["state_sha256"]["Level1-4"] == "changed-state-d"
    assert first["workload_hash"] != second["workload_hash"]


def test_load_raw_rejects_measured_workload_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "local-mismatch-test"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    plan = BenchmarkPlan(
        mode="single",
        run_name="local-mismatch-test",
        run_dir=str(run_dir),
        refs=[
            BenchmarkRef(
                role="ref",
                ref="HEAD",
                sha="1" * 40,
                archive=tmp_path / "ref-111111111111.tar.gz",
            )
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(5,),
        warmups=0,
        measured_cap=5,
    )
    args = SimpleNamespace(steps=50000, repeats=3)
    config = benchmark_raw_config(plan, args)
    config["steps"] = 5000
    write_raw(raw_dir / "measured-ref-00.json", [1000.0, 1001.0, 999.0], config)

    with pytest.raises(SystemExit, match="workload mismatch: config.steps=5000 expected 50000"):
        load_raw(args, plan, "measured-ref-00")


def test_load_raw_rejects_extra_workload_config_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "local-extra-config-test"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    plan = BenchmarkPlan(
        mode="single",
        run_name="local-extra-config-test",
        run_dir=str(run_dir),
        refs=[
            BenchmarkRef(
                role="ref",
                ref="HEAD",
                sha="1" * 40,
                archive=tmp_path / "ref-111111111111.tar.gz",
            )
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(5,),
        warmups=0,
        measured_cap=5,
    )
    args = SimpleNamespace(steps=50000, repeats=3)
    config = benchmark_raw_config(plan, args)
    config["unexpected"] = "extra"
    write_raw(raw_dir / "measured-ref-00.json", [1000.0, 1001.0, 999.0], config)

    with pytest.raises(SystemExit, match="workload mismatch: unexpected config key\\(s\\): unexpected"):
        load_raw(args, plan, "measured-ref-00")


def test_load_raw_rejects_package_identity_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "local-package-mismatch-test"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    plan = BenchmarkPlan(
        mode="single",
        run_name="local-package-mismatch-test",
        run_dir=str(run_dir),
        refs=[
            BenchmarkRef(
                role="ref",
                ref="HEAD",
                sha="1" * 40,
                archive=tmp_path / "ref-111111111111.tar.gz",
            )
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(5,),
        warmups=0,
        measured_cap=5,
    )
    args = SimpleNamespace(steps=50000, repeats=3)
    config = benchmark_raw_config(plan, args)
    path = raw_dir / "measured-ref-00.json"
    payload = {
        "package": {
            "name": "wrong-package",
            "version": "1.0.0",
            "import": "stable_retro",
        },
        "config": config,
        "runs": [{"env_steps_per_sec": 1000.0}],
    }
    path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(SystemExit, match="package mismatch: package.name"):
        load_raw(args, plan, "measured-ref-00")

    payload["package"] = {
        "name": "supermariobrosnes-turbo",
        "version": None,
        "import": "supermariobrosnes_turbo",
    }
    path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(SystemExit, match="package.version must be a string"):
        load_raw(args, plan, "measured-ref-00")


def test_load_raw_rejects_invalid_benchmark_samples(tmp_path: Path) -> None:
    run_dir = tmp_path / "local-invalid-samples-test"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    plan = BenchmarkPlan(
        mode="single",
        run_name="local-invalid-samples-test",
        run_dir=str(run_dir),
        refs=[
            BenchmarkRef(
                role="ref",
                ref="HEAD",
                sha="1" * 40,
                archive=tmp_path / "ref-111111111111.tar.gz",
            )
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(5,),
        warmups=0,
        measured_cap=5,
    )
    args = SimpleNamespace(steps=50000, repeats=3)
    config = benchmark_raw_config(plan, args)
    path = raw_dir / "measured-ref-00.json"

    path.write_text("[]\n")
    with pytest.raises(SystemExit, match="is not a JSON object"):
        load_raw(args, plan, "measured-ref-00")

    path.write_text(json.dumps({"package": valid_package_metadata(), "config": config, "runs": []}) + "\n")
    with pytest.raises(SystemExit, match="missing non-empty benchmark runs"):
        load_raw(args, plan, "measured-ref-00")

    path.write_text(json.dumps({"package": valid_package_metadata(), "config": config, "runs": [{"env_steps_per_sec": 0.0}]}) + "\n")
    with pytest.raises(SystemExit, match="non-positive env_steps_per_sec"):
        load_raw(args, plan, "measured-ref-00")

    path.write_text(json.dumps({"package": valid_package_metadata(), "config": config, "runs": [{"env_steps_per_sec": float("nan")}]}) + "\n")
    with pytest.raises(SystemExit, match="non-finite env_steps_per_sec"):
        load_raw(args, plan, "measured-ref-00")


def test_run_compare_checks_wall_clock_before_each_half_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="half-pair-limit-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(2,),
        warmups=0,
        measured_cap=2,
    )
    args = SimpleNamespace(
        force_busy=False,
        max_load=1.0,
        max_wall_clock_minutes=1.0,
        max_measured_invocations=None,
        steps=50000,
        repeats=3,
    )
    invocations: list[str] = []
    checks = iter([False, False, False, False, False, True])

    def fake_wall_clock_limit_exceeded(*_args: object) -> bool:
        return next(checks)

    def fake_run_invocation(
        _args: object,
        _plan: BenchmarkPlan,
        role: str,
        output_name: str,
        *_positional: object,
        **_keyword: object,
    ) -> None:
        invocations.append(output_name)

    monkeypatch.setattr("scripts.run_git_ref_benchmark.capture_load", lambda *_args: (0.1, ""))
    monkeypatch.setattr(
        "scripts.run_git_ref_benchmark.wall_clock_limit_exceeded",
        fake_wall_clock_limit_exceeded,
    )
    monkeypatch.setattr("scripts.run_git_ref_benchmark.run_invocation", fake_run_invocation)
    monkeypatch.setattr(
        "scripts.run_git_ref_benchmark.aggregate_compare",
        lambda *_args, **_keyword: {"should_stop": False},
    )
    monkeypatch.setattr("scripts.run_git_ref_benchmark.write_aggregate", lambda *_args: None)

    aggregate = run_compare(args, plan, start_time=0.0)

    assert invocations == [
        "smoke-baseline",
        "smoke-candidate",
        "measured-baseline-00",
        "measured-candidate-00",
        "measured-candidate-01",
    ]
    assert aggregate["limit_stop_reason"] == "max_wall_clock_minutes"
    assert aggregate["discarded_incomplete_pair_raw_files"] == [
        "raw/measured-candidate-01.json"
    ]


def test_run_compare_blocks_measured_pairs_when_pre_measured_load_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="busy-before-measured-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(2,),
        warmups=0,
        measured_cap=2,
    )
    args = SimpleNamespace(
        force_busy=False,
        max_load=1.0,
        max_wall_clock_minutes=0.0,
        max_measured_invocations=None,
        steps=50000,
        repeats=3,
    )
    invocations: list[str] = []
    wall_checks = iter([False, False, True])

    def fake_run_invocation(
        _args: object,
        _plan: BenchmarkPlan,
        _role: str,
        output_name: str,
        *_positional: object,
        **_keyword: object,
    ) -> None:
        invocations.append(output_name)

    monkeypatch.setattr("scripts.run_git_ref_benchmark.capture_load", lambda *_args: (2.0, ""))
    monkeypatch.setattr(
        "scripts.run_git_ref_benchmark.wall_clock_limit_exceeded",
        lambda *_args: next(wall_checks),
    )
    monkeypatch.setattr("scripts.run_git_ref_benchmark.run_invocation", fake_run_invocation)

    with pytest.raises(
        SystemExit,
        match="wall-clock limit exhausted before load cooldown before measured phase",
    ):
        run_compare(args, plan, start_time=0.0)

    assert invocations == ["smoke-baseline", "smoke-candidate"]


def test_run_compare_waits_when_load_is_high_after_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="busy-after-checkpoint-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(1, 2),
        warmups=0,
        measured_cap=2,
    )
    args = SimpleNamespace(
        force_busy=False,
        max_load=1.0,
        max_wall_clock_minutes=None,
        max_measured_invocations=None,
        steps=50000,
        repeats=3,
    )
    invocations: list[str] = []
    loads = iter([(0.1, ""), (2.0, ""), (0.5, ""), (0.1, "")])
    written: list[dict[str, object]] = []

    def fake_run_invocation(
        _args: object,
        _plan: BenchmarkPlan,
        _role: str,
        output_name: str,
        *_positional: object,
        **_keyword: object,
    ) -> None:
        invocations.append(output_name)

    monkeypatch.setattr("scripts.run_git_ref_benchmark.capture_load", lambda *_args: next(loads))
    monkeypatch.setattr("scripts.run_git_ref_benchmark.run_invocation", fake_run_invocation)
    monkeypatch.setattr(
        "scripts.run_git_ref_benchmark.aggregate_compare",
        lambda *_args, **_keyword: {"should_stop": False, "validity_passed": False},
    )
    monkeypatch.setattr(
        "scripts.run_git_ref_benchmark.write_aggregate",
        lambda _args, _plan, aggregate: written.append(dict(aggregate)),
    )

    aggregate = run_compare(args, plan, start_time=0.0)

    assert invocations == [
        "smoke-baseline",
        "smoke-candidate",
        "measured-baseline-00",
        "measured-candidate-00",
        "measured-candidate-01",
        "measured-baseline-01",
    ]
    assert aggregate["should_stop"] is False
    assert "limit_stop_reason" not in written[-1]


def test_aggregate_with_extra_load_snapshot_refreshes_validity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = BenchmarkPlan(
        mode="compare",
        run_name="after-load-refresh-test",
        run_dir=str(tmp_path / "run"),
        refs=[
            BenchmarkRef("baseline", "base", "1" * 40, tmp_path / "base.tar.gz"),
            BenchmarkRef("candidate", "cand", "2" * 40, tmp_path / "cand.tar.gz"),
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(1,),
        warmups=0,
        measured_cap=1,
    )
    args = SimpleNamespace(
        force_busy=False,
        max_load=1.0,
        max_wall_clock_minutes=None,
        max_measured_invocations=None,
        steps=50000,
        repeats=3,
    )
    original = {
        "measured_pairs": 1,
        "load_1min_labels": ["before-setup", "before-measured"],
        "load_1min_values": [0.2, 0.3],
        "limit_stop_reason": "max_measured_invocations",
    }
    calls: list[dict[str, object]] = []

    def fake_aggregate_compare(
        _args: object,
        _plan: BenchmarkPlan,
        *,
        measured_count: int,
        load_values: list[float | None],
        load_labels: list[str],
    ) -> dict[str, object]:
        calls.append(
            {
                "measured_count": measured_count,
                "load_values": list(load_values),
                "load_labels": list(load_labels),
            }
        )
        return {
            "measured_pairs": measured_count,
            "load_1min_labels": load_labels,
            "load_1min_values": load_values,
            "load_gate_passed": False,
            "validity_passed": False,
        }

    monkeypatch.setattr("scripts.run_git_ref_benchmark.aggregate_compare", fake_aggregate_compare)

    refreshed = aggregate_with_extra_load_snapshot(
        args,
        plan,
        original,
        label="after-measured",
        load_value=1.5,
    )

    assert calls == [
        {
            "measured_count": 1,
            "load_values": [0.2, 0.3, 1.5],
            "load_labels": ["before-setup", "before-measured", "after-measured"],
        }
    ]
    assert refreshed["load_gate_passed"] is False
    assert refreshed["validity_passed"] is False
    assert refreshed["limit_stop_reason"] == "load_gate_failed"
    assert refreshed["previous_limit_stop_reason"] == "max_measured_invocations"


def test_append_index_preserves_validity_metadata(tmp_path: Path) -> None:
    aggregate = {
        "mode": "paired_compare_fixed_local",
        "benchmark_tier": "local_acceptance",
        "refs": {"baseline": "main", "candidate": "HEAD"},
        "shas": {"baseline": "1" * 40, "candidate": "2" * 40},
        "workload_hash": "abcdef",
        "measured_pairs": 2,
        "measured_pair_details": [{"pair_index": 0}, {"pair_index": 1}],
        "median_pair_ratio": 1.02,
        "mean_pair_ratio": 1.01,
        "pair_ratio_bootstrap_ci95": [1.0, 1.03],
        "candidate_faster_pairs": 2,
        "candidate_faster_pairs_required_for_win": 2,
        "validity_passed": True,
        "load_gate_passed": False,
        "load_gate_ignored_for_validity": True,
        "limit_stop_reason": "max_measured_invocations",
        "previous_limit_stop_reason": "load_gate_failed",
        "benchmark_limits": {"max_measured_invocations": 3, "max_wall_clock_minutes": None},
        "discarded_incomplete_pair_raw_files": ["raw/measured-candidate-01.json"],
        "expected_rom_sha256": "expected",
        "rom_sha256": "actual",
        "state_sha256": {"Level1-1": "state-sha"},
        "decision": "needs_more_samples",
    }

    append_index(
        "metadata-test",
        tmp_path / "runs" / "metadata-test",
        aggregate,
        tmp_path,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert records[0]["load_gate_passed"] is False
    assert records[0]["benchmark_tier"] == "local_acceptance"
    assert records[0]["workload_hash"] == "abcdef"
    assert records[0]["measured_pairs"] == 2
    assert records[0]["mean_pair_ratio"] == 1.01
    assert records[0]["pair_ratio_bootstrap_ci95"] == [1.0, 1.03]
    assert records[0]["candidate_faster_pairs"] == 2
    assert records[0]["candidate_faster_pairs_required_for_win"] == 2
    assert records[0]["load_gate_ignored_for_validity"] is True
    assert records[0]["limit_stop_reason"] == "max_measured_invocations"
    assert records[0]["previous_limit_stop_reason"] == "load_gate_failed"
    assert records[0]["benchmark_limits"]["max_measured_invocations"] == 3
    assert records[0]["discarded_incomplete_pair_raw_files"] == [
        "raw/measured-candidate-01.json"
    ]
    assert records[0]["setup_only_raw_files"] is None
    assert records[0]["expected_rom_sha256"] == "expected"
    assert records[0]["rom_sha256"] == "actual"
    assert records[0]["state_sha256"] == {"Level1-1": "state-sha"}
