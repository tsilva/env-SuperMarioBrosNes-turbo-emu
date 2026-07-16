#!/usr/bin/env python3
"""Run and cache the PyPI supermariobrosnes-turbo local benchmark baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from benchmark_rom import EXPECTED_SMB_ROM_SHA256, validate_rom_hash
    from benchmark_workload import (
        CANONICAL_ACTION_NAMES,
        CANONICAL_ACTION_SEED,
        CANONICAL_ACTION_SET,
        CANONICAL_CROP_BOTTOM,
        CANONICAL_CROP_TOP,
        CANONICAL_FRAME_SKIP,
        CANONICAL_FRAME_STACK,
        CANONICAL_NUM_ENVS,
        CANONICAL_OBS_CROP_MODE,
        CANONICAL_RESIZE_HEIGHT,
        CANONICAL_RESIZE_WIDTH,
        CANONICAL_STATE_NAMES,
        canonical_env_args,
        shell_args,
    )
    from dotenv_utils import require_arg_or_env_or_dotenv_path
except ModuleNotFoundError:
    from scripts.benchmark_rom import EXPECTED_SMB_ROM_SHA256, validate_rom_hash
    from scripts.benchmark_workload import (
        CANONICAL_ACTION_NAMES,
        CANONICAL_ACTION_SEED,
        CANONICAL_ACTION_SET,
        CANONICAL_CROP_BOTTOM,
        CANONICAL_CROP_TOP,
        CANONICAL_FRAME_SKIP,
        CANONICAL_FRAME_STACK,
        CANONICAL_NUM_ENVS,
        CANONICAL_OBS_CROP_MODE,
        CANONICAL_RESIZE_HEIGHT,
        CANONICAL_RESIZE_WIDTH,
        CANONICAL_STATE_NAMES,
        canonical_env_args,
        shell_args,
    )
    from scripts.dotenv_utils import require_arg_or_env_or_dotenv_path

from supermariobrosnes_turbo import resolve_required_rom_path


AUTORESEARCH_ROOT_ENV = "AUTORESEARCH_ROOT_PATH"
BENCHMARK_ROOT_SUBDIR = Path("benchmarks")
BENCHMARK_STATE_SUBDIR = Path("states") / "SuperMarioBros-Nes-v0"
LOCAL_RESULTS_SUBDIR = Path("local-results")
PYPI_CACHE_SUBDIR = LOCAL_RESULTS_SUBDIR / "pypi-supermariobrosnes-turbo"
DEFAULT_STATES = CANONICAL_STATE_NAMES
DEFAULT_ACTIONS = CANONICAL_ACTION_NAMES
DEFAULT_ACTION_SEED = CANONICAL_ACTION_SEED
PACKAGE = "supermariobrosnes-turbo"
IMPORT_PACKAGE = "supermariobrosnes_turbo"
PYPI_JSON = f"https://pypi.org/pypi/{PACKAGE}/json"
MAX_LOAD = 4.0


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hashes(state_dir: str | Path) -> dict[str, str]:
    root = Path(state_dir)
    hashes: dict[str, str] = {}
    for name in DEFAULT_STATES:
        path = root / f"{name}.state"
        if not path.exists():
            raise SystemExit(f"state file does not exist: {path}")
        if not path.is_file():
            raise SystemExit(f"state path is not a file: {path}")
        hashes[name] = sha256_path(path)
    return hashes


def pypi_version_info_from_json(
    data: dict[str, Any],
    requested_version: str | None = None,
) -> dict[str, Any]:
    version = requested_version or data["info"]["version"]
    if version not in data["releases"]:
        raise SystemExit(f"{PACKAGE} version {version!r} not found in PyPI release metadata")
    return {
        "name": PACKAGE,
        "import": IMPORT_PACKAGE,
        "version": version,
        "urls": data["releases"].get(version, []),
        "queried_url": PYPI_JSON,
    }


def pypi_version_info(requested_version: str | None = None) -> dict[str, Any]:
    with urllib.request.urlopen(PYPI_JSON, timeout=30) as response:
        data = json.load(response)
    return pypi_version_info_from_json(data, requested_version)


def pypi_latest() -> dict[str, Any]:
    return pypi_version_info()


def workload(args: argparse.Namespace, version: str) -> dict[str, Any]:
    return {
        "package": PACKAGE,
        "version": version,
        "python": args.python,
        "rom_path": args.rom_path,
        "expected_rom_sha256": EXPECTED_SMB_ROM_SHA256,
        "rom_sha256": args.rom_sha256,
        "state_dir": args.state_dir,
        "state_sha256": args.state_sha256,
        "num_envs": args.num_envs,
        "num_threads": args.num_threads,
        "steps": args.steps,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "warmup_invocations": args.warmup_invocations,
        "measured_invocations": args.measured_invocations,
        "frame_skip": CANONICAL_FRAME_SKIP,
        "frame_stack": CANONICAL_FRAME_STACK,
        "grayscale": True,
        "crop_top": CANONICAL_CROP_TOP,
        "crop_bottom": CANONICAL_CROP_BOTTOM,
        "obs_crop_mode": CANONICAL_OBS_CROP_MODE,
        "resize": [CANONICAL_RESIZE_WIDTH, CANONICAL_RESIZE_HEIGHT],
        "states": list(DEFAULT_STATES),
        "action_set": CANONICAL_ACTION_SET,
        "action": None,
        "actions": list(DEFAULT_ACTIONS),
        "action_seed": DEFAULT_ACTION_SEED,
        "include_info": True,
        "obs_resize_algorithm": "area",
        "termination": "provider_native",
    }


def stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_paths(args: argparse.Namespace, version: str, workload_hash: str) -> tuple[Path, Path]:
    cache_dir = args.local_cache_root / version / workload_hash[:16]
    return cache_dir, cache_dir / "aggregate.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "num_envs",
        "num_threads",
        "steps",
        "repeats",
        "warmup_invocations",
        "measured_invocations",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")


def cached_aggregate_is_usable(path: Path, expected_workload_hash: str) -> bool:
    try:
        aggregate = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    workload = aggregate.get("workload", {})
    package = aggregate.get("package", {})
    return (
        aggregate.get("validity_passed") is True
        and aggregate.get("workload_hash") == expected_workload_hash
        and isinstance(workload, dict)
        and stable_hash(workload) == expected_workload_hash
        and isinstance(package, dict)
        and package.get("name") == PACKAGE
        and package.get("import") == IMPORT_PACKAGE
        and isinstance(package.get("version"), str)
        and package.get("version") == workload.get("version")
        and aggregate.get("load_gate_passed") is True
        and aggregate.get("load_gate_ignored_for_validity") is not True
        and workload.get("expected_rom_sha256") == EXPECTED_SMB_ROM_SHA256
        and workload.get("rom_sha256") == EXPECTED_SMB_ROM_SHA256
        and isinstance(workload.get("state_sha256"), dict)
        and all(workload["state_sha256"].get(name) for name in DEFAULT_STATES)
        and cached_raw_files_are_usable(path.parent, aggregate, workload)
    )


def safe_cache_file(cache_dir: Path, relative_path: str) -> Path | None:
    path = cache_dir / relative_path
    try:
        path.resolve().relative_to(cache_dir.resolve())
    except ValueError:
        return None
    return path


def cached_raw_files_are_usable(
    cache_dir: Path,
    aggregate: dict[str, Any],
    workload_payload: dict[str, Any],
) -> bool:
    measured = aggregate.get("measured_invocations")
    warmups = aggregate.get("warmup_raw_files")
    if not isinstance(measured, list) or not measured or not isinstance(warmups, list):
        return False
    if aggregate.get("measured_invocation_count") != len(measured):
        return False
    if aggregate.get("warmup_invocation_count") != len(warmups):
        return False
    measured_medians: list[float] = []
    for item in measured:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            return False
        path = safe_cache_file(cache_dir, item["file"])
        if path is None:
            return False
        try:
            payload = load_json(path)
            require_raw_payload_matches_workload(payload, workload_payload, path)
            samples = env_steps_per_sec_samples(payload, path)
        except (OSError, ValueError, json.JSONDecodeError, SystemExit):
            return False
        sample_median = median(samples)
        measured_medians.append(sample_median)
        if not samples_match(item.get("samples_env_steps_per_sec"), samples):
            return False
        if not float_matches(item.get("median_env_steps_per_sec"), sample_median):
            return False
        if not float_matches(item.get("mean_env_steps_per_sec"), mean(samples)):
            return False
    for item in warmups:
        if not isinstance(item, str):
            return False
        path = safe_cache_file(cache_dir, item)
        if path is None:
            return False
        try:
            require_raw_payload_matches_workload(load_json(path), workload_payload, path)
        except (OSError, ValueError, json.JSONDecodeError, SystemExit):
            return False
    if not float_matches(aggregate.get("official_median_sps"), median(measured_medians)):
        return False
    if not float_matches(aggregate.get("mean_invocation_median_sps"), mean(measured_medians)):
        return False
    return True


def float_matches(value: object, expected: float) -> bool:
    try:
        actual = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)


def samples_match(value: object, expected: list[float]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    return all(float_matches(actual, sample) for actual, sample in zip(value, expected, strict=True))


def median(values: list[float]) -> float:
    return statistics.median(values)


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summary(values: list[float]) -> dict[str, float]:
    avg = mean(values)
    sd = stdev(values)
    return {
        "mean": avg,
        "median": median(values),
        "stdev": sd,
        "min": min(values),
        "max": max(values),
        "cv": sd / avg if avg else 0.0,
    }


def bootstrap_ci_median(values: list[float], n: int = 20000, seed: int = 12345) -> list[float]:
    rng = random.Random(seed)
    boot = []
    for _ in range(n):
        boot.append(median([values[rng.randrange(len(values))] for _ in values]))
    boot.sort()
    return [boot[int(0.025 * n)], boot[int(0.975 * n) - 1]]


def outliers(values: list[float]) -> dict[str, list[int]]:
    if len(values) < 4:
        return {"iqr_invocation_median_indices": [], "mad_invocation_median_indices": []}
    sorted_values = sorted(values)
    q1 = statistics.median(sorted_values[: len(sorted_values) // 2])
    q3 = statistics.median(sorted_values[(len(sorted_values) + 1) // 2 :])
    iqr = q3 - q1
    iqr_indices = [
        i for i, value in enumerate(values) if value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr
    ]
    med = median(values)
    mad = median([abs(value - med) for value in values])
    mad_indices = (
        []
        if mad == 0
        else [i for i, value in enumerate(values) if abs(value - med) / (1.4826 * mad) > 3.5]
    )
    return {"iqr_invocation_median_indices": iqr_indices, "mad_invocation_median_indices": mad_indices}


def parse_load1(text: str | None) -> float | None:
    if not text or "load average:" not in text:
        return None
    try:
        return float(text.split("load average:", 1)[1].split(",", 1)[0].strip())
    except ValueError:
        return None


def require_raw_payload_matches_workload(
    payload: dict[str, Any],
    workload_payload: dict[str, Any],
    path: Path,
) -> None:
    package = payload.get("package")
    if not isinstance(package, dict):
        raise SystemExit(f"{path} is missing package metadata")
    expected_package = {
        "name": PACKAGE,
        "version": workload_payload["version"],
        "import": IMPORT_PACKAGE,
    }
    package_mismatches = [
        f"package.{key}={package.get(key)!r} expected {value!r}"
        for key, value in expected_package.items()
        if package.get(key) != value
    ]
    if package_mismatches:
        raise SystemExit(f"{path} package mismatch: " + "; ".join(package_mismatches))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise SystemExit(f"{path} is missing benchmark config")
    expected = {
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
        "state": None,
        "states": workload_payload["states"],
        "lane_states": [
            workload_payload["states"][index % len(workload_payload["states"])]
            for index in range(int(workload_payload["num_envs"]))
        ],
        "action_set": workload_payload["action_set"],
        "action": workload_payload["action"],
        "actions": workload_payload["actions"],
        "action_seed": workload_payload["action_seed"],
        "include_info": workload_payload["include_info"],
        "terminate_on_flag": False,
        "termination": workload_payload["termination"],
        "start_game": False,
        "vectorization": "native",
    }
    mismatches = [
        f"config.{key}={config.get(key)!r} expected {value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    extra_keys = sorted(set(config) - set(expected))
    if extra_keys:
        names = ", ".join(str(key) for key in extra_keys)
        raise SystemExit(f"{path} workload mismatch: unexpected config key(s): {names}")
    if mismatches:
        raise SystemExit(f"{path} workload mismatch: " + "; ".join(mismatches))
    env_steps_per_sec_samples(payload, path)


def env_steps_per_sec_samples(payload: dict[str, Any], path: Path) -> list[float]:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit(f"{path} is missing non-empty benchmark runs")
    samples: list[float] = []
    for index, run_payload in enumerate(runs):
        if not isinstance(run_payload, dict):
            raise SystemExit(f"{path} run {index} is not an object")
        try:
            sample = float(run_payload["env_steps_per_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{path} run {index} has invalid env_steps_per_sec") from exc
        if not math.isfinite(sample):
            raise SystemExit(f"{path} run {index} has non-finite env_steps_per_sec")
        if sample <= 0.0:
            raise SystemExit(f"{path} run {index} has non-positive env_steps_per_sec")
        samples.append(sample)
    return samples


def require_package_info_matches_workload(
    package_info: dict[str, Any],
    workload_payload: dict[str, Any],
) -> None:
    expected = {
        "name": PACKAGE,
        "import": IMPORT_PACKAGE,
        "version": workload_payload["version"],
    }
    mismatches = [
        f"package.{key}={package_info.get(key)!r} expected {value!r}"
        for key, value in expected.items()
        if package_info.get(key) != value
    ]
    if mismatches:
        raise SystemExit("aggregate package mismatch: " + "; ".join(mismatches))


def require_measured_load_gate(args: argparse.Namespace, path: Path) -> None:
    if args.force_busy:
        return
    value = parse_load1(path.read_text())
    if value is None:
        raise SystemExit(
            f"benchmark load unavailable before measured phase; rerun with --force-busy to override"
        )
    if value >= MAX_LOAD:
        raise SystemExit(
            f"benchmark load {value:.2f} meets or exceeds max {MAX_LOAD:.2f} before measured phase; "
            "rerun with --force-busy to override"
        )


def make_local_run_name(version: str, workload_hash: str) -> str:
    safe_version = re.sub(r"[^A-Za-z0-9_.-]", "-", version)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return f"benchmark-pypi-supermariobrosnes-turbo-{safe_version}-{stamp}-{workload_hash[:8]}"


def local_setup(args: argparse.Namespace, run_dir: Path, version: str) -> None:
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2("scripts/benchmark_sps.py", run_dir / "scripts")
    shutil.copy2("scripts/benchmark_rom.py", run_dir / "scripts")
    shutil.copy2("scripts/benchmark_workload.py", run_dir / "scripts")
    shutil.copy2("scripts/dotenv_utils.py", run_dir / "scripts")
    setup = (
        f"cd {quote(str(run_dir))} && "
        f"uv venv --python {quote(args.python)} .venv && "
        f"uv pip install --python .venv/bin/python {quote(PACKAGE + '==' + version)}"
    )
    run(["bash", "-lc", setup])


def run_invocations(args: argparse.Namespace, run_dir: Path) -> None:
    workload_args = shell_args(canonical_env_args())
    base_cmd = (
        f"cd {quote(str(run_dir))} && "
        f"RAYON_NUM_THREADS={args.num_threads} .venv/bin/python scripts/benchmark_sps.py "
        f"--rom-path {quote(args.rom_path)} "
        f"--state-dir {quote(args.state_dir)} "
        f"{workload_args} "
        f"--steps {args.steps} --repeats {args.repeats} "
        f"--warmup {args.warmup} "
        f"--json --output-json "
    )
    run(["bash", "-lc", f"uptime > {quote(str(run_dir / 'raw' / 'load-before-warmup.txt'))}"])
    for index in range(args.warmup_invocations):
        output = run_dir / "raw" / f"warmup-pypi-{index:02d}.json"
        run(["bash", "-lc", base_cmd + quote(str(output)) + f" > {quote(str(output) + '.stdout')}"])
    load_before_measured = run_dir / "raw" / "load-before-measured.txt"
    run(["bash", "-lc", f"uptime > {quote(str(load_before_measured))}"])
    require_measured_load_gate(args, load_before_measured)
    for index in range(args.measured_invocations):
        output = run_dir / "raw" / f"measured-pypi-{index:02d}.json"
        run(["bash", "-lc", base_cmd + quote(str(output)) + f" > {quote(str(output) + '.stdout')}"])
    run(["bash", "-lc", f"uptime > {quote(str(run_dir / 'raw' / 'load-after-measured.txt'))}"])


def copy_local_tree(run_dir: Path, cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    (cache_dir / "raw").mkdir(parents=True, exist_ok=True)
    for path in sorted((run_dir / "raw").glob("*")):
        if path.suffix == ".json" and path.name.endswith(".stdout.json"):
            continue
        if path.suffix in {".json", ".txt"}:
            shutil.copy2(path, cache_dir / "raw" / path.name)


def aggregate(
    args: argparse.Namespace,
    cache_dir: Path,
    run_dir: Path,
    version_info: dict[str, Any],
    workload_payload: dict[str, Any],
) -> dict[str, Any]:
    require_package_info_matches_workload(version_info, workload_payload)
    measured_files = [
        cache_dir / "raw" / f"measured-pypi-{index:02d}.json"
        for index in range(args.measured_invocations)
    ]
    warmup_files = [
        cache_dir / "raw" / f"warmup-pypi-{index:02d}.json"
        for index in range(args.warmup_invocations)
    ]
    measured = []
    all_samples = []
    for path in measured_files:
        result = load_json(path)
        require_raw_payload_matches_workload(result, workload_payload, path)
        samples = env_steps_per_sec_samples(result, path)
        all_samples.extend(samples)
        measured.append(
            {
                "file": str(path.relative_to(cache_dir)),
                "mean_env_steps_per_sec": mean(samples),
                "median_env_steps_per_sec": median(samples),
                "samples_env_steps_per_sec": samples,
            }
        )
    for path in warmup_files:
        require_raw_payload_matches_workload(load_json(path), workload_payload, path)
    medians = [item["median_env_steps_per_sec"] for item in measured]
    ci = bootstrap_ci_median(medians)
    median_summary = summary(medians)
    all_sample_summary = summary(all_samples)
    diagnostics = outliers(medians)
    load_before = (cache_dir / "raw" / "load-before-measured.txt").read_text()
    load_after = (cache_dir / "raw" / "load-after-measured.txt").read_text()
    load_values = {
        "before": parse_load1(load_before),
        "after": parse_load1(load_after),
    }
    gates = {
        "invocation_median_cv_below_0_75_percent": median_summary["cv"] < 0.0075,
        "all_sample_cv_below_1_25_percent": all_sample_summary["cv"] < 0.0125,
        "bootstrap_ci_width_below_0_75_percent": (ci[1] - ci[0]) / median(medians) < 0.0075,
        "no_iqr_mad_outliers": not diagnostics["iqr_invocation_median_indices"]
        and not diagnostics["mad_invocation_median_indices"],
        "load_below_4": all(value is not None and value < 4 for value in load_values.values()),
    }
    load_gate_passed = gates["load_below_4"]
    validity_passed = all(
        value for key, value in gates.items() if key != "load_below_4"
    ) and (load_gate_passed or args.force_busy)
    aggregate_payload = {
        "mode": "pypi_supermariobrosnes_turbo_fixed_local",
        "package": version_info,
        "workload": workload_payload,
        "workload_hash": stable_hash(workload_payload),
        "execution_target": "local_machine",
        "local_run_dir": str(run_dir),
        "measured_invocation_count": args.measured_invocations,
        "warmup_invocation_count": args.warmup_invocations,
        "measured_invocations": measured,
        "warmup_raw_files": [str(path.relative_to(cache_dir)) for path in warmup_files],
        "official_median_sps": median(medians),
        "mean_invocation_median_sps": mean(medians),
        "bootstrap_ci95_invocation_median_sps": ci,
        "run_median_summary": median_summary,
        "all_sample_summary": all_sample_summary,
        "outlier_diagnostics": diagnostics,
        "load_snapshots": {"texts": {"before": load_before, "after": load_after}, "load1_values": load_values},
        "load_gate_passed": load_gate_passed,
        "load_gate_ignored_for_validity": bool(args.force_busy),
        "validity_gates": gates,
        "validity_passed": validity_passed,
        "tier": "full_official",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_dir / "aggregate.json").write_text(json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n")
    return aggregate_payload


def write_manifest_and_index(args: argparse.Namespace, cache_dir: Path, aggregate_payload: dict[str, Any]) -> None:
    files = []
    for path in sorted(cache_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            if path.suffix == ".json":
                load_json(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"path": str(path.relative_to(cache_dir)), "bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "copied_files": files,
        "retention": "local cache is durable; local pypi run dir is purged after copy",
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    index_path = args.local_cache_root / "index.jsonl"
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "version": aggregate_payload["package"]["version"],
        "workload_hash": aggregate_payload["workload_hash"],
        "cache_dir": str(cache_dir),
        "official_median_sps": aggregate_payload["official_median_sps"],
        "mean_invocation_median_sps": aggregate_payload["mean_invocation_median_sps"],
        "validity_passed": aggregate_payload["validity_passed"],
        "load_gate_passed": aggregate_payload.get("load_gate_passed"),
        "load_gate_ignored_for_validity": aggregate_payload.get("load_gate_ignored_for_validity"),
        "expected_rom_sha256": aggregate_payload["workload"].get("expected_rom_sha256"),
        "rom_sha256": aggregate_payload["workload"].get("rom_sha256"),
        "state_sha256": aggregate_payload["workload"].get("state_sha256"),
    }
    existing = []
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            if not line.strip():
                continue
            old = json.loads(line)
            if not (old.get("version") == record["version"] and old.get("workload_hash") == record["workload_hash"]):
                existing.append(json.dumps(old, sort_keys=True))
    with index_path.open("w") as handle:
        for line in existing:
            handle.write(line + "\n")
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def purge_local(run_root: Path, run_dir: Path) -> None:
    try:
        run_dir.relative_to(run_root / "runs")
    except ValueError as exc:
        raise SystemExit(f"Refusing to purge path outside benchmark runs: {run_dir}") from exc
    shutil.rmtree(run_dir, ignore_errors=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None, help="PyPI version; defaults to latest from PyPI JSON.")
    parser.add_argument("--python", default="3.14")
    parser.add_argument(
        "--rom-path",
        default=None,
        help="ROM path on the benchmark machine. Defaults to Stable Retro-compatible discovery.",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help=(
            "State directory on the benchmark machine. Defaults to "
            f"{AUTORESEARCH_ROOT_ENV}/{BENCHMARK_STATE_SUBDIR}."
        ),
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Root for temporary benchmark runs. Defaults to "
            f"{AUTORESEARCH_ROOT_ENV}/{BENCHMARK_ROOT_SUBDIR}."
        ),
    )
    parser.add_argument(
        "--local-cache-root",
        type=Path,
        default=None,
        help=(
            "Durable cache for copied PyPI benchmark results. Defaults to "
            f"{AUTORESEARCH_ROOT_ENV}/{BENCHMARK_ROOT_SUBDIR}/{PYPI_CACHE_SUBDIR}."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=CANONICAL_NUM_ENVS)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--warmup-invocations", type=int, default=2)
    parser.add_argument("--measured-invocations", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-busy", action="store_true")
    parser.add_argument("--keep-run-dir", action="store_true")
    args = parser.parse_args(argv)
    try:
        rom_path = resolve_required_rom_path(args.rom_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not rom_path.exists():
        raise SystemExit(f"ROM path does not exist: {rom_path}")
    if not rom_path.is_file():
        raise SystemExit(f"ROM path is not a file: {rom_path}")
    args.rom_path = str(rom_path.resolve())
    autoresearch_root = require_arg_or_env_or_dotenv_path(
        AUTORESEARCH_ROOT_ENV,
        "autoresearch root",
        must_be_dir=True,
    )
    args.run_root = (
        Path(args.run_root).expanduser()
        if args.run_root
        else autoresearch_root / BENCHMARK_ROOT_SUBDIR
    ).resolve(strict=False)
    args.local_cache_root = (
        args.local_cache_root.expanduser()
        if args.local_cache_root
        else args.run_root / PYPI_CACHE_SUBDIR
    ).resolve(strict=False)
    state_dir = Path(args.state_dir).expanduser() if args.state_dir else autoresearch_root / BENCHMARK_STATE_SUBDIR
    args.state_dir = str(state_dir.resolve(strict=False))
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.rom_sha256 = validate_rom_hash(args.rom_path)
    args.state_sha256 = state_hashes(args.state_dir)
    version_info = pypi_version_info(args.version)
    workload_payload = workload(args, version_info["version"])
    workload_hash = stable_hash(workload_payload)
    cache_dir, aggregate_path = cache_paths(args, version_info["version"], workload_hash)
    if aggregate_path.exists() and not args.force and cached_aggregate_is_usable(aggregate_path, workload_hash):
        print(json.dumps({"cache_hit": True, "aggregate": str(aggregate_path), "workload_hash": workload_hash}, indent=2))
        return 0

    run_name = make_local_run_name(version_info["version"], workload_hash)
    run_dir = args.run_root / "runs" / run_name
    try:
        local_setup(args, run_dir, version_info["version"])
        run_invocations(args, run_dir)
        copy_local_tree(run_dir, cache_dir)
        aggregate_payload = aggregate(args, cache_dir, run_dir, version_info, workload_payload)
        write_manifest_and_index(args, cache_dir, aggregate_payload)
    except BaseException:
        if not args.keep_run_dir:
            try:
                purge_local(args.run_root, run_dir)
            except Exception as cleanup_error:
                print(f"warning: failed to purge interrupted local run {run_dir}: {cleanup_error}", file=sys.stderr)
        raise
    if not args.keep_run_dir:
        purge_local(args.run_root, run_dir)
    print(
        json.dumps(
            {
                "cache_hit": False,
                "aggregate": str(cache_dir / "aggregate.json"),
                "workload_hash": workload_hash,
                "validity_passed": aggregate_payload["validity_passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
