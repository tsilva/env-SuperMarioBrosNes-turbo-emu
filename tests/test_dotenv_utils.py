from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmark_rom import EXPECTED_SMB_ROM_SHA256
from scripts.benchmark_sps import build_result as build_native_benchmark_result
from scripts.benchmark_sps import DEFAULT_ACTION_SEED
from scripts.benchmark_sps import DEFAULT_BENCHMARK_ACTIONS
from scripts.benchmark_sps import load_preflight as native_load_preflight
from scripts.benchmark_sps import package_metadata as native_package_metadata
from scripts.benchmark_sps import resolve_verified_rom_path as resolve_native_verified_rom_path
from scripts.dotenv_utils import dotenv_value, env_or_dotenv_path
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import (
    aggregate as smb_pypi_aggregate,
)
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import (
    cached_aggregate_is_usable as smb_cached_aggregate_is_usable,
)
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import (
    parse_args as parse_smb_pypi_args,
)
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import (
    pypi_version_info_from_json as smb_pypi_version_info_from_json,
)
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import (
    run_invocations as run_smb_pypi_invocations,
)
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import (
    require_measured_load_gate as require_smb_pypi_load_gate,
)
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import state_hashes
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import (
    stable_hash as smb_pypi_stable_hash,
)
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import workload as smb_pypi_workload
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import (
    validate_args as validate_smb_pypi_args,
)
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import (
    write_manifest_and_index as write_smb_manifest_and_index,
)
from supermariobrosnes_turbo import default_rom_path


def test_dotenv_value_supports_export_and_quotes(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# local secrets\n"
        "export EXAMPLE_PATH='~/example/data'\n"
    )

    assert dotenv_value("EXAMPLE_PATH", dotenv) == "~/example/data"


def test_env_or_dotenv_path_prefers_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("EXAMPLE_PATH=/from/dotenv\n")
    monkeypatch.setenv("EXAMPLE_PATH", "/from/env")

    assert env_or_dotenv_path("EXAMPLE_PATH", dotenv) == Path("/from/env")


def test_package_default_rom_path_ignores_dotenv_rom_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RETRO_DATA_PATH", raising=False)
    monkeypatch.delenv("ROM_PATH", raising=False)
    (tmp_path / ".env").write_text("ROM_PATH=/tmp/from-dotenv.nes\n")
    monkeypatch.setattr("supermariobrosnes_turbo.roms._stable_retro_rom_path", lambda _game: None)
    monkeypatch.setattr("supermariobrosnes_turbo.roms.package_data_path", lambda: tmp_path / "data")

    assert default_rom_path() is None


def test_direct_benchmark_entrypoints_reject_wrong_rom_sha(tmp_path: Path) -> None:
    rom_path = tmp_path / "wrong-rom.nes"
    rom_path.write_bytes(b"not the expected rom")

    with pytest.raises(SystemExit, match="ROM SHA-256 mismatch"):
        resolve_native_verified_rom_path(rom_path)


def test_direct_benchmark_results_record_rom_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAYON_NUM_THREADS", "12")
    rom_path = tmp_path / "SuperMarioBros.nes"
    rom_path.write_bytes(b"placeholder")
    expected_sha = hashlib.sha256(b"placeholder").hexdigest()
    obs = np.zeros((2, 4, 84, 84), dtype=np.uint8)
    runs = [
        {
            "elapsed_s": 1.0,
            "batch_steps_per_sec": 10.0,
            "env_steps_per_sec": 20.0,
            "emulated_frames_per_sec": 80.0,
        }
    ]

    native_args = SimpleNamespace(
        num_envs=2,
        steps=10,
        repeats=1,
        warmup=0,
        frame_skip=4,
        frame_stack=4,
        rgb=False,
        crop_top=32,
        crop_bottom=0,
        obs_crop_mode="mask",
        resize_width=84,
        resize_height=84,
        action_set="basic",
        action=None,
        parsed_actions=DEFAULT_BENCHMARK_ACTIONS,
        action_seed=DEFAULT_ACTION_SEED,
        state=None,
        parsed_states=("Level1-1",),
        state_dir=None,
        include_info=True,
        terminate_on_flag=False,
        no_start_game=False,
    )
    native = build_native_benchmark_result(
        native_args,
        obs,
        runs,
        ("Level1-1", "Level1-1"),
        {"enabled": False, "start_1min": None, "max_start_load": None, "load_ok": True},
        rom_path,
    )

    assert native["config"]["rom_path"] == str(rom_path)
    assert native["config"]["rom_sha256"] == expected_sha
    assert native["config"]["rayon_num_threads"] == 12
    assert native["config"]["obs_crop_mode"] == "mask"
    assert native["config"]["obs_resize_algorithm"] == "area"
    assert native["config"]["termination"] == "provider_native"
    assert native["package"] == native_package_metadata()


def test_native_benchmark_load_preflight_uses_strict_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(skip_load_preflight=False, max_start_load=4.0)

    monkeypatch.setattr("scripts.benchmark_sps.os.getloadavg", lambda: (4.0, 3.0, 2.0))
    with pytest.raises(SystemExit, match="1-minute load 4.00 meets or exceeds --max-start-load 4.00"):
        native_load_preflight(args)

    monkeypatch.setattr("scripts.benchmark_sps.os.getloadavg", lambda: (3.99, 3.0, 2.0))
    assert native_load_preflight(args)["load_ok"] is True


def test_native_benchmark_rejects_missing_rom_before_hashing(tmp_path: Path) -> None:
    missing_rom = tmp_path / "missing.nes"

    with pytest.raises(SystemExit, match=f"ROM path does not exist: {missing_rom}"):
        resolve_native_verified_rom_path(missing_rom)


def test_pypi_benchmark_wrappers_read_retro_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "retro-data"
    rom_path = data_root / "stable" / "SuperMarioBros-Nes-v0" / "rom.nes"
    rom_path.parent.mkdir(parents=True)
    rom_path.write_bytes(b"placeholder")
    monkeypatch.setenv("RETRO_DATA_PATH", str(data_root))
    autoresearch_root = tmp_path / "autoresearch"
    autoresearch_root.mkdir()
    (tmp_path / ".env").write_text(
        f'AUTORESEARCH_ROOT_PATH="{autoresearch_root}"\n'
    )

    assert parse_smb_pypi_args(["--version", "1.0.0"]).rom_path == str(rom_path)


def test_pypi_workload_hash_inputs_include_rom_digest() -> None:
    state_sha256 = {
        "Level1-1": "sha-Level1-1",
        "Level1-2": "sha-Level1-2",
        "Level1-3": "sha-Level1-3",
        "Level1-4": "sha-Level1-4",
    }
    args = SimpleNamespace(
        python="3.14",
        rom_path="/roms/SuperMarioBros.nes",
        rom_sha256=EXPECTED_SMB_ROM_SHA256,
        state_dir="/states/SuperMarioBros-Nes-v0",
        state_sha256=state_sha256,
        num_envs=16,
        num_threads=12,
        steps=50000,
        repeats=3,
        warmup=100,
        warmup_invocations=2,
        measured_invocations=11,
    )

    smb_payload = smb_pypi_workload(args, "0.1.2")
    assert smb_payload["expected_rom_sha256"] == EXPECTED_SMB_ROM_SHA256
    assert smb_payload["rom_sha256"] == EXPECTED_SMB_ROM_SHA256
    assert smb_payload["obs_crop_mode"] == "mask"
    assert smb_payload["termination"] == "provider_native"
    assert smb_payload["state_sha256"] == state_sha256


def test_pypi_version_info_uses_requested_release_urls() -> None:
    data = {
        "info": {"version": "2.0.0"},
        "releases": {
            "1.0.0": [{"filename": "pkg-1.0.0.whl"}],
            "2.0.0": [{"filename": "pkg-2.0.0.whl"}],
        },
    }

    smb = smb_pypi_version_info_from_json(data, "1.0.0")
    assert smb["name"] == "supermariobrosnes-turbo"
    assert smb["import"] == "supermariobrosnes_turbo"
    assert smb["version"] == "1.0.0"
    assert smb["urls"] == [{"filename": "pkg-1.0.0.whl"}]

    with pytest.raises(SystemExit, match="version '9.9.9' not found"):
        smb_pypi_version_info_from_json(data, "9.9.9")


def test_pypi_smb_state_hashes_require_all_state_files(tmp_path: Path) -> None:
    state_dir = tmp_path / "states"
    state_dir.mkdir()
    for name in ("Level1-1", "Level1-2", "Level1-3", "Level1-4"):
        (state_dir / f"{name}.state").write_bytes(f"{name}-bytes".encode())

    hashes = state_hashes(state_dir)

    assert set(hashes) == {"Level1-1", "Level1-2", "Level1-3", "Level1-4"}
    assert all(len(value) == 64 for value in hashes.values())

    (state_dir / "Level1-4.state").unlink()
    with pytest.raises(SystemExit, match="state file does not exist"):
        state_hashes(state_dir)


def smb_raw_config(workload_payload: dict[str, object]) -> dict[str, object]:
    states = list(workload_payload["states"])
    return {
        "rom_path": workload_payload["rom_path"],
        "rom_sha256": workload_payload["rom_sha256"],
        "state_dir": workload_payload["state_dir"],
        "rayon_num_threads": workload_payload["num_threads"],
        "num_envs": workload_payload["num_envs"],
        "steps": workload_payload["steps"],
        "repeats": workload_payload["repeats"],
        "warmup": workload_payload["warmup"],
        "frame_skip": workload_payload["frame_skip"],
        "frame_stack": workload_payload["frame_stack"],
        "frame_maxpool": False,
        "grayscale": workload_payload["grayscale"],
        "crop_top": workload_payload["crop_top"],
        "crop_bottom": workload_payload["crop_bottom"],
        "obs_crop_mode": workload_payload["obs_crop_mode"],
        "resize_width": workload_payload["resize"][0],
        "resize_height": workload_payload["resize"][1],
        "obs_resize_algorithm": workload_payload["obs_resize_algorithm"],
        "obs_layout": "chw",
        "action_set": workload_payload["action_set"],
        "action": workload_payload["action"],
        "actions": workload_payload["actions"],
        "action_seed": workload_payload["action_seed"],
        "state": None,
        "states": states,
        "lane_states": [states[index % len(states)] for index in range(workload_payload["num_envs"])],
        "include_info": workload_payload["include_info"],
        "terminate_on_flag": False,
        "termination": workload_payload["termination"],
        "start_game": False,
        "vectorization": "native",
    }


def write_pypi_raw_bundle(
    cache_dir: Path,
    payload: dict[str, object],
    *,
    runs: list[dict[str, float]] | None = None,
    load_before: str = "10:00 up 1 day, load average: 0.50, 0.40, 0.30\n",
    load_after: str = "10:00 up 1 day, load average: 0.60, 0.40, 0.30\n",
) -> None:
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_payload = dict(payload)
    raw_payload["runs"] = (
        runs
        if runs is not None
        else [
            {"env_steps_per_sec": 1000.0},
            {"env_steps_per_sec": 1001.0},
            {"env_steps_per_sec": 999.0},
        ]
    )
    for name in ("warmup-pypi-00.json", "measured-pypi-00.json"):
        (raw_dir / name).write_text(json.dumps(raw_payload) + "\n")
    (raw_dir / "load-before-measured.txt").write_text(load_before)
    (raw_dir / "load-after-measured.txt").write_text(load_after)


def smb_raw_package(version: str = "1.0.0") -> dict[str, str]:
    return {
        "name": "supermariobrosnes-turbo",
        "version": version,
        "import": "supermariobrosnes_turbo",
    }


def test_pypi_aggregates_reject_raw_workload_mismatches(tmp_path: Path) -> None:
    cases = (
        (
            smb_pypi_aggregate,
            smb_pypi_workload,
            smb_raw_config,
            smb_raw_package(),
            SimpleNamespace(
                python="3.14",
                rom_path="/roms/SuperMarioBros.nes",
                rom_sha256=EXPECTED_SMB_ROM_SHA256,
                state_dir="/states/SuperMarioBros-Nes-v0",
                state_sha256={
                    "Level1-1": "sha-Level1-1",
                    "Level1-2": "sha-Level1-2",
                    "Level1-3": "sha-Level1-3",
                    "Level1-4": "sha-Level1-4",
                },
                num_envs=16,
                num_threads=12,
                steps=50000,
                repeats=3,
                warmup=100,
                warmup_invocations=1,
                measured_invocations=1,
                force_busy=False,
            ),
        ),
    )

    for aggregate, make_workload, make_config, package, args in cases:
        cache_dir = tmp_path / aggregate.__module__
        run_dir = tmp_path / f"{aggregate.__module__}-run"
        workload_payload = make_workload(args, "1.0.0")
        payload = {
            "package": package,
            "config": make_config(workload_payload),
            "runs": [],
        }
        write_pypi_raw_bundle(cache_dir, payload)

        aggregate(args, cache_dir, run_dir, package, workload_payload)

        payload["config"] = dict(payload["config"])
        payload["config"]["rom_sha256"] = "wrong"
        write_pypi_raw_bundle(cache_dir, payload)

        with pytest.raises(SystemExit, match="workload mismatch: config.rom_sha256"):
            aggregate(args, cache_dir, run_dir, package, workload_payload)

        payload["config"] = make_config(workload_payload)
        payload["config"]["unexpected"] = "extra"
        write_pypi_raw_bundle(cache_dir, payload)

        with pytest.raises(SystemExit, match="workload mismatch: unexpected config key\\(s\\): unexpected"):
            aggregate(args, cache_dir, run_dir, package, workload_payload)

        payload["package"] = dict(package)
        payload["package"]["version"] = "wrong"
        payload["config"] = make_config(workload_payload)
        write_pypi_raw_bundle(cache_dir, payload)

        with pytest.raises(SystemExit, match="package mismatch: package.version"):
            aggregate(args, cache_dir, run_dir, package, workload_payload)

        payload["package"] = package
        with pytest.raises(SystemExit, match="aggregate package mismatch: package.version"):
            aggregate(
                args,
                cache_dir,
                run_dir,
                {**package, "version": "wrong"},
                workload_payload,
            )

        write_pypi_raw_bundle(cache_dir, payload, runs=[])
        with pytest.raises(SystemExit, match="missing non-empty benchmark runs"):
            aggregate(args, cache_dir, run_dir, package, workload_payload)

        write_pypi_raw_bundle(cache_dir, payload, runs=[{"env_steps_per_sec": 0.0}])
        with pytest.raises(SystemExit, match="non-positive env_steps_per_sec"):
            aggregate(args, cache_dir, run_dir, package, workload_payload)

        write_pypi_raw_bundle(cache_dir, payload, runs=[{"env_steps_per_sec": float("nan")}])
        with pytest.raises(SystemExit, match="non-finite env_steps_per_sec"):
            aggregate(args, cache_dir, run_dir, package, workload_payload)


def test_pypi_force_busy_keeps_validity_separate_from_load_gate(tmp_path: Path) -> None:
    cases = (
        (
            smb_pypi_aggregate,
            smb_pypi_workload,
            smb_raw_config,
            smb_raw_package(),
            SimpleNamespace(
                python="3.14",
                rom_path="/roms/SuperMarioBros.nes",
                rom_sha256=EXPECTED_SMB_ROM_SHA256,
                state_dir="/states/SuperMarioBros-Nes-v0",
                state_sha256={
                    "Level1-1": "sha-Level1-1",
                    "Level1-2": "sha-Level1-2",
                    "Level1-3": "sha-Level1-3",
                    "Level1-4": "sha-Level1-4",
                },
                num_envs=16,
                num_threads=12,
                steps=50000,
                repeats=3,
                warmup=100,
                warmup_invocations=1,
                measured_invocations=1,
                force_busy=True,
            ),
        ),
    )

    for aggregate, make_workload, make_config, package, args in cases:
        cache_dir = tmp_path / f"{aggregate.__module__}-busy"
        run_dir = tmp_path / f"{aggregate.__module__}-busy-run"
        workload_payload = make_workload(args, "1.0.0")
        write_pypi_raw_bundle(
            cache_dir,
            {"package": package, "config": make_config(workload_payload), "runs": []},
            load_before="10:00 up 1 day, load average: 9.00, 8.00, 7.00\n",
            load_after="10:00 up 1 day, load average: 9.50, 8.00, 7.00\n",
        )

        payload = aggregate(args, cache_dir, run_dir, package, workload_payload)

        assert payload["validity_gates"]["load_below_4"] is False
        assert payload["load_gate_passed"] is False
        assert payload["load_gate_ignored_for_validity"] is True
        assert payload["validity_passed"] is True


@pytest.mark.parametrize(
    ("validator", "field", "value", "message"),
    [
        (validate_smb_pypi_args, "measured_invocations", 0, "--measured-invocations must be positive"),
        (validate_smb_pypi_args, "warmup", -1, "--warmup must be non-negative"),
    ],
)
def test_pypi_wrappers_validate_numeric_args(
    validator: object,
    field: str,
    value: int,
    message: str,
) -> None:
    args = SimpleNamespace(
        num_envs=16,
        num_threads=12,
        steps=50000,
        repeats=3,
        warmup=100,
        warmup_invocations=1,
        measured_invocations=1,
    )
    setattr(args, field, value)

    with pytest.raises(SystemExit, match=message):
        validator(args)


def test_pypi_indexes_preserve_rom_digest(tmp_path: Path) -> None:
    state_sha256 = {
        "Level1-1": "sha-Level1-1",
        "Level1-2": "sha-Level1-2",
        "Level1-3": "sha-Level1-3",
        "Level1-4": "sha-Level1-4",
    }
    for writer, package in ((write_smb_manifest_and_index, "supermariobrosnes-turbo"),):
        cache_root = tmp_path / package
        cache_dir = cache_root / "1.0.0" / "abcdef"
        cache_dir.mkdir(parents=True)
        aggregate = {
            "package": {"version": "1.0.0"},
            "workload_hash": "abcdef",
            "official_median_sps": 123.0,
            "mean_invocation_median_sps": 122.0,
            "validity_passed": True,
            "load_gate_passed": True,
            "load_gate_ignored_for_validity": False,
            "workload": {
                "expected_rom_sha256": EXPECTED_SMB_ROM_SHA256,
                "rom_sha256": EXPECTED_SMB_ROM_SHA256,
            },
        }
        if writer is write_smb_manifest_and_index:
            aggregate["workload"]["state_sha256"] = state_sha256

        writer(SimpleNamespace(local_cache_root=cache_root), cache_dir, aggregate)

        records = [
            line
            for line in (cache_root / "index.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 1
        assert '"load_gate_passed": true' in records[0]
        assert '"load_gate_ignored_for_validity": false' in records[0]
        assert f'"expected_rom_sha256": "{EXPECTED_SMB_ROM_SHA256}"' in records[0]
        assert f'"rom_sha256": "{EXPECTED_SMB_ROM_SHA256}"' in records[0]
        if writer is write_smb_manifest_and_index:
            assert '"state_sha256": {' in records[0]
            assert '"Level1-4": "sha-Level1-4"' in records[0]


def test_pypi_cache_hits_require_valid_aggregate(tmp_path: Path) -> None:
    state_sha256 = {
        "Level1-1": "sha-Level1-1",
        "Level1-2": "sha-Level1-2",
        "Level1-3": "sha-Level1-3",
        "Level1-4": "sha-Level1-4",
    }
    cases = (
        (
            smb_cached_aggregate_is_usable,
            smb_pypi_stable_hash,
            smb_pypi_workload,
            smb_raw_config,
            smb_raw_package(),
            {"name": "supermariobrosnes-turbo", "import": "supermariobrosnes_turbo"},
            SimpleNamespace(
                python="3.14",
                rom_path="/roms/SuperMarioBros.nes",
                rom_sha256=EXPECTED_SMB_ROM_SHA256,
                state_dir="/states/SuperMarioBros-Nes-v0",
                state_sha256=state_sha256,
                num_envs=16,
                num_threads=12,
                steps=50000,
                repeats=3,
                warmup=100,
                warmup_invocations=1,
                measured_invocations=1,
            ),
        ),
    )

    def write_aggregate(
        path: Path,
        workload_hash: str,
        package_identity: dict[str, object],
        workload: dict[str, object],
        raw_package: dict[str, object],
        raw_config: dict[str, object],
        **overrides: object,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_pypi_raw_bundle(
            path.parent,
            {"package": raw_package, "config": raw_config, "runs": []},
        )
        payload = {
            "validity_passed": True,
            "load_gate_passed": True,
            "load_gate_ignored_for_validity": False,
            "package": {**package_identity, "version": workload.get("version")},
            "workload_hash": workload_hash,
            "workload": workload,
            "measured_invocation_count": 1,
            "warmup_invocation_count": 1,
            "measured_invocations": [
                {
                    "file": "raw/measured-pypi-00.json",
                    "mean_env_steps_per_sec": 1000.0,
                    "median_env_steps_per_sec": 1000.0,
                    "samples_env_steps_per_sec": [1000.0, 1001.0, 999.0],
                }
            ],
            "warmup_raw_files": ["raw/warmup-pypi-00.json"],
            "official_median_sps": 1000.0,
            "mean_invocation_median_sps": 1000.0,
        }
        payload.update(overrides)
        path.write_text(json.dumps(payload) + "\n")

    for is_usable, hasher, make_workload, make_config, raw_package, package_identity, args in cases:
        aggregate = tmp_path / is_usable.__module__ / "aggregate.json"
        workload = make_workload(args, "1.0.0")
        raw_config = make_config(workload)
        workload_hash = hasher(workload)

        write_aggregate(aggregate, workload_hash, package_identity, workload, raw_package, raw_config)
        assert is_usable(aggregate, workload_hash)

        (aggregate.parent / "raw" / "measured-pypi-00.json").unlink()
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            raw_config,
            load_gate_ignored_for_validity=True,
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(aggregate, "wrong", package_identity, workload, raw_package, raw_config)
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            raw_config,
            package={},
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            raw_config,
            package={**package_identity, "version": "0.9.0"},
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            raw_config,
            package={**package_identity, "name": "wrong", "version": workload.get("version")},
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            raw_config,
            package={**package_identity, "import": "wrong", "version": workload.get("version")},
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(aggregate, workload_hash, package_identity, {}, raw_package, raw_config)
        assert not is_usable(aggregate, workload_hash)

        bad_rom = dict(workload)
        bad_rom["rom_sha256"] = "wrong"
        write_aggregate(aggregate, workload_hash, package_identity, bad_rom, raw_package, raw_config)
        assert not is_usable(aggregate, workload_hash)

        mutated_workload = dict(workload)
        mutated_workload["unexpected"] = "extra"
        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            mutated_workload,
            raw_package,
            raw_config,
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            {**raw_config, "rom_sha256": "wrong"},
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            raw_config,
            measured_invocations=[{"file": "../escape.json"}],
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            raw_config,
            measured_invocation_count=2,
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            raw_config,
            official_median_sps=999999.0,
        )
        assert not is_usable(aggregate, workload_hash)

        write_aggregate(
            aggregate,
            workload_hash,
            package_identity,
            workload,
            raw_package,
            raw_config,
            measured_invocations=[
                {
                    "file": "raw/measured-pypi-00.json",
                    "mean_env_steps_per_sec": 1000.0,
                    "median_env_steps_per_sec": 1000.0,
                    "samples_env_steps_per_sec": [1000.0],
                }
            ],
        )
        assert not is_usable(aggregate, workload_hash)

        aggregate.write_text('{"validity_passed": false}\n')
        assert not is_usable(aggregate, workload_hash)

        aggregate.write_text("{not json}\n")
        assert not is_usable(aggregate, workload_hash)

        aggregate.write_text("[]\n")
        assert not is_usable(aggregate, workload_hash)

        assert not is_usable(tmp_path / f"{is_usable.__module__}-missing.json", workload_hash)


def test_pypi_smb_cache_hits_require_state_hashes(tmp_path: Path) -> None:
    aggregate = tmp_path / "smb-cache.json"
    workload = {
        "version": "1.0.0",
        "expected_rom_sha256": EXPECTED_SMB_ROM_SHA256,
        "rom_sha256": EXPECTED_SMB_ROM_SHA256,
    }
    workload_hash = smb_pypi_stable_hash(workload)
    aggregate.write_text(
        json.dumps(
            {
                "validity_passed": True,
                "load_gate_passed": True,
                "load_gate_ignored_for_validity": False,
                "package": {
                    "name": "supermariobrosnes-turbo",
                    "import": "supermariobrosnes_turbo",
                    "version": "1.0.0",
                },
                "workload_hash": workload_hash,
                "workload": workload,
            }
        )
        + "\n"
    )

    assert not smb_cached_aggregate_is_usable(aggregate, workload_hash)


def test_pypi_wrappers_capture_load_after_warmups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        rom_path="/roms/SuperMarioBros.nes",
        state_dir="/states/SuperMarioBros-Nes-v0",
        num_envs=16,
        num_threads=12,
        steps=50000,
        repeats=3,
        warmup=100,
        warmup_invocations=1,
        measured_invocations=1,
        force_busy=True,
    )

    for runner, run_symbol in (
        (run_smb_pypi_invocations, "scripts.run_pypi_supermariobrosnes_turbo_benchmark.run"),
    ):
        commands: list[str] = []

        def fake_run(cmd: list[str], *args: object, **kwargs: object) -> object:
            commands.append(cmd[-1])
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        monkeypatch.setattr(run_symbol, fake_run)
        runner(args, tmp_path / runner.__module__)

        warmup_index = next(
            index for index, command in enumerate(commands) if "warmup-pypi-00.json" in command
        )
        measured_index = next(
            index for index, command in enumerate(commands) if "measured-pypi-00.json" in command
        )
        before_warmup_index = next(
            index for index, command in enumerate(commands) if "load-before-warmup.txt" in command
        )
        before_measured_index = next(
            index for index, command in enumerate(commands) if "load-before-measured.txt" in command
        )

        assert before_warmup_index < warmup_index < before_measured_index < measured_index
        measured_command = commands[measured_index]
        for expected in (
            "--warmup 100",
            "--frame-skip 4",
            "--frame-stack 4",
            "--crop-top 32",
            "--crop-bottom 0",
            "--obs-crop-mode mask",
            "--resize-width 84",
            "--resize-height 84",
            "Level1-1,Level1-2,Level1-3,Level1-4",
        ):
            assert expected in measured_command
        for expected in (
            "--action-set basic",
            "--actions",
            "noop,right,right_b,right_a",
            "--action-seed 0",
            "--no-start-game",
        ):
            assert expected in measured_command


def test_pypi_measured_load_gate_blocks_busy_or_unknown_load(tmp_path: Path) -> None:
    load_path = tmp_path / "load-before-measured.txt"

    for load_gate in (require_smb_pypi_load_gate,):
        load_path.write_text("10:00 up 1 day, load average: 4.01, 3.00, 2.00\n")
        with pytest.raises(SystemExit, match="benchmark load 4.01 meets or exceeds max 4.00"):
            load_gate(SimpleNamespace(force_busy=False), load_path)

        load_gate(SimpleNamespace(force_busy=True), load_path)

        load_path.write_text("10:00 up 1 day, load average: 4.00, 3.00, 2.00\n")
        with pytest.raises(SystemExit, match="benchmark load 4.00 meets or exceeds max 4.00"):
            load_gate(SimpleNamespace(force_busy=False), load_path)

        load_path.write_text("10:00 up 1 day, load average: 0.50, 0.40, 0.30\n")
        load_gate(SimpleNamespace(force_busy=False), load_path)

        load_path.write_text("load text unavailable\n")
        with pytest.raises(SystemExit, match="benchmark load unavailable before measured phase"):
            load_gate(SimpleNamespace(force_busy=False), load_path)

        load_path.write_text("10:00 up 1 day, load average: not-a-number, 0.40, 0.30\n")
        with pytest.raises(SystemExit, match="benchmark load unavailable before measured phase"):
            load_gate(SimpleNamespace(force_busy=False), load_path)
