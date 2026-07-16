#!/usr/bin/env python3
"""Run fixed local git-ref benchmarks with sequential convergence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

try:
    from benchmark_stats import (
        DEFAULT_COMPARISON_CHECKPOINTS,
        DEFAULT_SINGLE_CHECKPOINTS,
        comparison_convergence,
        median,
        single_ref_convergence,
        summary,
    )
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
        CANONICAL_START_GAME,
        CANONICAL_TERMINATE_ON_FLAG,
        canonical_env_args,
        shell_args,
    )
    from dotenv_utils import require_arg_or_env_or_dotenv_path
except ModuleNotFoundError:
    from scripts.benchmark_stats import (
        DEFAULT_COMPARISON_CHECKPOINTS,
        DEFAULT_SINGLE_CHECKPOINTS,
        comparison_convergence,
        median,
        single_ref_convergence,
        summary,
    )
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
        CANONICAL_START_GAME,
        CANONICAL_TERMINATE_ON_FLAG,
        canonical_env_args,
        shell_args,
    )
    from scripts.dotenv_utils import require_arg_or_env_or_dotenv_path

from supermariobrosnes_turbo import resolve_required_rom_path


AUTORESEARCH_ROOT_ENV = "AUTORESEARCH_ROOT_PATH"
BENCHMARK_ROOT_SUBDIR = Path("benchmarks")
BENCHMARK_STATE_SUBDIR = Path("states") / "SuperMarioBros-Nes-v0"
STATE_NAMES = CANONICAL_STATE_NAMES
ACTION_NAMES = CANONICAL_ACTION_NAMES
ACTION_SEED = CANONICAL_ACTION_SEED
PACKAGE_NAME = "supermariobrosnes-turbo"
IMPORT_PACKAGE = "supermariobrosnes_turbo"
ARCHIVE_SUBDIR = Path("local-archives")
LOCAL_RESULTS_SUBDIR = Path("local-results")
PREPARED_SOURCES_SUBDIR = Path("prepared-sources")
RESULTS_TSV_COLUMNS = (
    "epoch",
    "commit",
    "baseline_commit",
    "mode",
    "benchmark_tier",
    "workload_hash",
    "measured_invocation_count",
    "measured_pairs",
    "official_median_sps",
    "mean_invocation_median_sps",
    "bootstrap_ci95_invocation_median_sps",
    "median_pair_ratio",
    "mean_pair_ratio",
    "pair_ratio_bootstrap_ci95",
    "candidate_faster_pairs",
    "candidate_faster_pairs_required_for_win",
    "validity_passed",
    "load_gate_passed",
    "load_gate_ignored_for_validity",
    "limit_stop_reason",
    "previous_limit_stop_reason",
    "benchmark_limits",
    "discarded_incomplete_pair_raw_files",
    "expected_rom_sha256",
    "rom_sha256",
    "state_sha256",
    "decision",
    "status",
    "description",
    "artifact",
)

Mode = Literal["single", "compare"]
STACK_ACCEPTANCE_CHECKPOINTS = (3, 5, 7)
LOAD_GATE_START_MARGIN = 0.85
LOAD_GATE_POLL_SECONDS = 15.0


@dataclass(frozen=True)
class BenchmarkRef:
    role: str
    ref: str
    sha: str
    archive: Path

    @property
    def short_sha(self) -> str:
        return self.sha[:12]


@dataclass(frozen=True)
class BenchmarkPlan:
    mode: Mode
    run_name: str
    run_dir: str
    refs: list[BenchmarkRef]
    rom_path: str
    state_dir: str
    checkpoints: tuple[int, ...]
    warmups: int
    measured_cap: int


def run(
    cmd: list[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_stream(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def quote(value: str | os.PathLike[str]) -> str:
    return shlex.quote(str(value))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def benchmark_tier(args: argparse.Namespace, plan: BenchmarkPlan) -> str:
    if (
        plan.mode == "compare"
        and args.steps == 50000
        and args.repeats == 3
        and plan.warmups == 2
    ):
        return "local_acceptance"
    if (
        plan.mode == "compare"
        and args.steps == 5000
        and args.repeats == 1
        and plan.warmups == 0
        and args.max_measured_invocations == 3
    ):
        return "local_triage"
    if (
        plan.mode == "compare"
        and args.steps == 30000
        and args.repeats == 2
        and plan.warmups == 1
        and args.max_measured_invocations == 7
    ):
        return "stack_acceptance"
    return "local_diagnosis"


def is_stack_acceptance_shape(args: argparse.Namespace, mode: Mode, warmups: int) -> bool:
    return (
        mode == "compare"
        and args.steps == 30000
        and args.repeats == 2
        and warmups == 1
        and args.max_measured_invocations == 7
    )


def resolve_ref(ref: str) -> str:
    return run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).stdout.strip()


def default_main_ref() -> str:
    return run(["git", "rev-parse", "--verify", "main^{commit}"]).stdout.strip()


def archive_ref(role: str, sha: str, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"{role}-{sha[:12]}.tar.gz"
    completed = subprocess.run(
        ["git", "archive", "--format=tar", sha],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(completed.stdout)
    return path


def decide_mode(refs: list[str], *, single: bool) -> tuple[Mode, list[tuple[str, str]]]:
    if single:
        if len(refs) != 1:
            raise SystemExit("--single requires exactly one ref")
        return "single", [("ref", refs[0])]
    if len(refs) == 1:
        return "compare", [("baseline", default_main_ref()), ("candidate", refs[0])]
    if len(refs) == 2:
        return "compare", [("baseline", refs[0]), ("candidate", refs[1])]
    raise SystemExit("pass one ref with --single, one candidate ref, or baseline candidate")


def make_run_name(mode: Mode, refs: list[BenchmarkRef]) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if mode == "single":
        return f"benchmark-single-{stamp}-R{refs[0].short_sha}"
    baseline = next(ref for ref in refs if ref.role == "baseline")
    candidate = next(ref for ref in refs if ref.role == "candidate")
    return f"benchmark-compare-{stamp}-B{baseline.short_sha}-C{candidate.short_sha}"


def build_plan(args: argparse.Namespace) -> BenchmarkPlan:
    mode, role_refs = decide_mode(args.refs, single=args.single)
    root = Path(args.run_root)
    archive_dir = root / ARCHIVE_SUBDIR
    refs = []
    for role, ref in role_refs:
        sha = ref if re.fullmatch(r"[0-9a-fA-F]{40}", ref) else resolve_ref(ref)
        archive = archive_dir / f"{role}-{sha[:12]}.tar.gz"
        refs.append(BenchmarkRef(role=role, ref=ref, sha=sha, archive=archive))

    run_name = make_run_name(mode, refs)
    run_dir = str(root / "runs" / run_name)
    state_dir = str(args.state_dir)
    default_checkpoints = (
        DEFAULT_SINGLE_CHECKPOINTS if mode == "single" else DEFAULT_COMPARISON_CHECKPOINTS
    )
    warmups = args.warmups if args.warmups is not None else 2
    if is_stack_acceptance_shape(args, mode, warmups):
        default_checkpoints = STACK_ACCEPTANCE_CHECKPOINTS
    checkpoints = cap_checkpoints(default_checkpoints, args.max_measured_invocations)
    return BenchmarkPlan(
        mode=mode,
        run_name=run_name,
        run_dir=run_dir,
        refs=refs,
        rom_path=args.rom_path,
        state_dir=state_dir,
        checkpoints=checkpoints,
        warmups=warmups,
        measured_cap=checkpoints[-1],
    )


def target_run(args: argparse.Namespace, plan: BenchmarkPlan, shell: str) -> str:
    return run(["bash", "-lc", shell]).stdout


def target_run_stream(args: argparse.Namespace, plan: BenchmarkPlan, shell: str) -> None:
    run_stream(["bash", "-lc", shell])


def target_write(args: argparse.Namespace, plan: BenchmarkPlan, path: str, text: str) -> None:
    local = Path(path)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text)


def target_read(args: argparse.Namespace, plan: BenchmarkPlan, path: str) -> str:
    return Path(path).read_text()


def target_exists(args: argparse.Namespace, plan: BenchmarkPlan, path: str) -> bool:
    cmd = f"test -e {quote(path)}"
    return run(["bash", "-lc", cmd], check=False).returncode == 0


def parse_load1(uptime_text: str) -> float | None:
    if "load average:" not in uptime_text:
        return None
    try:
        return float(uptime_text.split("load average:", 1)[1].split(",", 1)[0].strip())
    except ValueError:
        return None


def load_snapshot_shell(raw_path: str) -> str:
    return (
        "set -e; "
        f"{{ uname -n; uptime; "
        "sysctl -n hw.ncpu 2>/dev/null || nproc; "
        "sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | sed -n '1,40p'; "
        "ps -Ao pid,pcpu,pmem,comm,args | sort -k2 -nr | head -20; } "
        f"> {quote(raw_path)}"
    )


def capture_load(args: argparse.Namespace, plan: BenchmarkPlan, label: str) -> tuple[float | None, str]:
    raw_path = f"{plan.run_dir}/raw/load-{label}.txt"
    shell = load_snapshot_shell(raw_path)
    target_run(args, plan, shell)
    text = target_read(args, plan, raw_path)
    return parse_load1(text), text


def load_ok(load_values: list[float | None], max_load: float) -> bool:
    return all(value is not None and value < max_load for value in load_values)


def load_ok_for_validity(args: argparse.Namespace, load_values: list[float | None]) -> bool:
    if args.force_busy:
        return True
    return load_ok(load_values, args.max_load)


def require_load_gate(args: argparse.Namespace, load_value: float | None, phase: str) -> None:
    if args.force_busy:
        return
    if load_value is None:
        raise SystemExit(
            f"benchmark load unavailable before {phase}; rerun with --force-busy to override"
        )
    if load_value >= args.max_load:
        raise SystemExit(
            f"benchmark load {load_value:.2f} meets or exceeds max {args.max_load:.2f} before {phase}; "
            "rerun with --force-busy to override"
        )


def cooldown_load_label(base_label: str, attempt: int) -> str:
    return base_label if attempt == 0 else f"{base_label}-retry-{attempt:02d}"


def wait_for_load_headroom(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    base_label: str,
    phase: str,
    *,
    start_time: float | None = None,
) -> tuple[str, float | None]:
    target_load = args.max_load * LOAD_GATE_START_MARGIN
    attempt = 0
    while True:
        label = cooldown_load_label(base_label, attempt)
        load_value, _ = capture_load(args, plan, label)
        if load_value is None:
            require_load_gate(args, load_value, phase)
        if load_value < target_load:
            return label, load_value
        if start_time is not None:
            require_wall_clock_budget(args, start_time, f"load cooldown before {phase}")
        time.sleep(LOAD_GATE_POLL_SECONDS)
        attempt += 1


def wait_for_invocation_load_gate(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    output_name: str,
    *,
    start_time: float | None = None,
) -> None:
    if args.force_busy:
        return
    wait_for_load_headroom(
        args,
        plan,
        f"before-{output_name}",
        output_name,
        start_time=start_time,
    )


def capture_load_gate_snapshot(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    label: str,
    phase: str,
    *,
    start_time: float,
) -> tuple[str, float | None]:
    if args.force_busy:
        return label, capture_load(args, plan, label)[0]
    return wait_for_load_headroom(args, plan, label, phase, start_time=start_time)


def load_gate_stop_reason(args: argparse.Namespace, load_value: float | None) -> str | None:
    if args.force_busy:
        return None
    if load_value is None or load_value >= args.max_load:
        return "load_gate_failed"
    return None


def cap_checkpoints(checkpoints: tuple[int, ...], cap: int | None) -> tuple[int, ...]:
    if cap is None:
        return checkpoints
    if cap >= checkpoints[-1]:
        return checkpoints
    capped = [checkpoint for checkpoint in checkpoints if checkpoint <= cap]
    if not capped or capped[-1] != cap:
        capped.append(cap)
    return tuple(capped)


def measured_invocation_limit_applies(args: argparse.Namespace, plan: BenchmarkPlan) -> bool:
    if args.max_measured_invocations is None:
        return False
    default_max = (
        DEFAULT_SINGLE_CHECKPOINTS[-1]
        if plan.mode == "single"
        else DEFAULT_COMPARISON_CHECKPOINTS[-1]
    )
    return args.max_measured_invocations < default_max


def wall_clock_limit_exceeded(args: argparse.Namespace, start_time: float) -> bool:
    if args.max_wall_clock_minutes is None:
        return False
    return time.monotonic() - start_time >= args.max_wall_clock_minutes * 60.0


def require_wall_clock_budget(args: argparse.Namespace, start_time: float, phase: str) -> None:
    if wall_clock_limit_exceeded(args, start_time):
        raise SystemExit(f"wall-clock limit exhausted before {phase}")


def ensure_states(args: argparse.Namespace, plan: BenchmarkPlan) -> None:
    target_run(args, plan, f"mkdir -p {quote(plan.state_dir)}")
    missing = [
        name
        for name in STATE_NAMES
        if not target_exists(args, plan, f"{plan.state_dir}/{name}.state")
    ]
    if not missing:
        return
    if args.state_source is None:
        missing_files = ", ".join(f"{name}.state" for name in missing)
        raise SystemExit(
            f"missing benchmark state files in {plan.state_dir}: {missing_files}. "
            f"Populate the state cache under {AUTORESEARCH_ROOT_ENV} or pass --state-source."
        )
    if not args.state_source.is_dir():
        raise SystemExit(f"benchmark state source is not a directory: {args.state_source}")
    for name in missing:
        source = args.state_source / f"{name}.state"
        if not source.exists():
            raise SystemExit(f"missing state source {source}")
    for name in missing:
        shutil.copy2(args.state_source / f"{name}.state", Path(plan.state_dir) / f"{name}.state")


def create_archives(plan: BenchmarkPlan) -> list[BenchmarkRef]:
    archived = []
    for ref in plan.refs:
        path = archive_ref(ref.role, ref.sha, ref.archive.parent)
        archived.append(BenchmarkRef(ref.role, ref.ref, ref.sha, path))
    return archived


def uv_sync_command() -> str:
    return 'env PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" uv sync --frozen --no-dev'


def prepare_sources(args: argparse.Namespace, plan: BenchmarkPlan) -> None:
    target_run(args, plan, f"mkdir -p {quote(plan.run_dir + '/archives')} {quote(plan.run_dir + '/raw')}")
    archives = Path(plan.run_dir) / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    for ref in plan.refs:
        shutil.copy2(ref.archive, archives / ref.archive.name)

    for ref in plan.refs:
        prepared_dir = prepare_source_cache(args, plan, ref)
        link_prepared_source(plan, ref, prepared_dir)


def benchmark_run_root_for_plan(plan: BenchmarkPlan) -> Path:
    run_dir = Path(plan.run_dir)
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent
    return run_dir.parent


def source_cache_root_for_plan(plan: BenchmarkPlan) -> Path:
    return benchmark_run_root_for_plan(plan) / PREPARED_SOURCES_SUBDIR


def prepared_source_dir(plan: BenchmarkPlan, ref: BenchmarkRef) -> Path:
    return source_cache_root_for_plan(plan) / f"sha-{ref.short_sha}"


def prepared_source_manifest(path: Path) -> Path:
    return path / ".autoresearch-prepared-source.json"


def prepared_source_editable_pths(path: Path) -> list[Path]:
    site_packages_root = path / ".venv" / "lib"
    if not site_packages_root.is_dir():
        return []
    return list(site_packages_root.glob(f"python*/site-packages/{IMPORT_PACKAGE}.pth"))


def prepared_source_has_valid_editable_pth(path: Path) -> bool:
    pths = prepared_source_editable_pths(path)
    if not pths:
        return True
    expected = str(path / "python")
    return any(pth.read_text().strip() == expected for pth in pths)


def repair_prepared_source_editable_pth(path: Path) -> None:
    expected = str(path / "python") + "\n"
    for pth in prepared_source_editable_pths(path):
        pth.write_text(expected)


def prepared_source_is_usable(path: Path, ref: BenchmarkRef) -> bool:
    manifest_path = prepared_source_manifest(path)
    python_path = path / ".venv" / "bin" / "python"
    if not path.is_dir() or not manifest_path.is_file() or not python_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        manifest.get("sha") == ref.sha
        and manifest.get("archive_sha256") == sha256_path(ref.archive)
        and prepared_source_has_valid_editable_pth(path)
    )


def prepare_source_cache(args: argparse.Namespace, plan: BenchmarkPlan, ref: BenchmarkRef) -> Path:
    cache_dir = prepared_source_dir(plan, ref)
    if prepared_source_is_usable(cache_dir, ref):
        return cache_dir

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shell = (
        f"mkdir -p {quote(tmp_dir)} && "
        f"tar -xzf {quote(ref.archive)} -C {quote(tmp_dir)} && "
        f"cd {quote(tmp_dir)} && {uv_sync_command()}"
    )
    target_run_stream(args, plan, shell)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sha": ref.sha,
        "archive_sha256": sha256_path(ref.archive),
        "uv_sync_command": uv_sync_command(),
    }
    prepared_source_manifest(tmp_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    shutil.rmtree(cache_dir, ignore_errors=True)
    tmp_dir.rename(cache_dir)
    repair_prepared_source_editable_pth(cache_dir)
    return cache_dir


def link_prepared_source(plan: BenchmarkPlan, ref: BenchmarkRef, prepared_dir: Path) -> None:
    sources_root = Path(plan.run_dir) / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    source_dir = sources_root / ref.role
    if source_dir.is_symlink() or source_dir.is_file():
        source_dir.unlink()
    elif source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.symlink_to(prepared_dir, target_is_directory=True)


def benchmark_command(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    role: str,
    output_name: str,
    *,
    steps: int,
    repeats: int,
) -> str:
    source_dir = f"{plan.run_dir}/sources/{role}"
    output = f"{plan.run_dir}/raw/{output_name}.json"
    workload_args = shell_args(canonical_env_args())
    return (
        f"cd {quote(source_dir)} && "
        "RAYON_NUM_THREADS=12 .venv/bin/python scripts/benchmark_sps.py "
        f"--rom-path {quote(plan.rom_path)} "
        f"--state-dir {quote(plan.state_dir)} "
        f"{workload_args} "
        f"--steps {steps} --repeats {repeats} "
        "--warmup 100 "
        f"--max-start-load {args.max_load} "
        f"--json --output-json {quote(output)} "
        f"> {quote(output + '.stdout.json')}"
    )


def run_invocation(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    role: str,
    output_name: str,
    *,
    steps: int,
    repeats: int,
    start_time: float | None = None,
) -> None:
    wait_for_invocation_load_gate(args, plan, output_name, start_time=start_time)
    target_run_stream(
        args,
        plan,
        benchmark_command(args, plan, role, output_name, steps=steps, repeats=repeats),
    )


def require_raw_payload_matches_plan(
    payload: dict[str, Any],
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    name: str,
    *,
    steps: int | None = None,
    repeats: int | None = None,
) -> None:
    package = payload.get("package")
    if not isinstance(package, dict):
        raise SystemExit(f"raw/{name}.json is missing package metadata")
    expected_package = {
        "name": PACKAGE_NAME,
        "import": IMPORT_PACKAGE,
    }
    package_mismatches = [
        f"package.{key}={package.get(key)!r} expected {value!r}"
        for key, value in expected_package.items()
        if package.get(key) != value
    ]
    if package_mismatches:
        raise SystemExit(f"raw/{name}.json package mismatch: " + "; ".join(package_mismatches))
    if not isinstance(package.get("version"), str):
        raise SystemExit(f"raw/{name}.json package.version must be a string")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise SystemExit(f"raw/{name}.json is missing benchmark config")
    expected_steps = args.steps if steps is None else steps
    expected_repeats = args.repeats if repeats is None else repeats
    expected = {
        "rom_path": plan.rom_path,
        "rom_sha256": EXPECTED_SMB_ROM_SHA256,
        "rayon_num_threads": 12,
        "num_envs": CANONICAL_NUM_ENVS,
        "steps": expected_steps,
        "repeats": expected_repeats,
        "warmup": 100,
        "frame_skip": CANONICAL_FRAME_SKIP,
        "frame_stack": CANONICAL_FRAME_STACK,
        "frame_maxpool": False,
        "grayscale": True,
        "crop_top": CANONICAL_CROP_TOP,
        "crop_bottom": CANONICAL_CROP_BOTTOM,
        "obs_crop_mode": CANONICAL_OBS_CROP_MODE,
        "resize_width": CANONICAL_RESIZE_WIDTH,
        "resize_height": CANONICAL_RESIZE_HEIGHT,
        "obs_resize_algorithm": "area",
        "obs_layout": "chw",
        "action_set": CANONICAL_ACTION_SET,
        "action": None,
        "actions": list(ACTION_NAMES),
        "action_seed": ACTION_SEED,
        "state": None,
        "states": list(STATE_NAMES),
        "lane_states": [
            STATE_NAMES[index % len(STATE_NAMES)] for index in range(CANONICAL_NUM_ENVS)
        ],
        "state_dir": plan.state_dir,
        "include_info": True,
        "terminate_on_flag": CANONICAL_TERMINATE_ON_FLAG,
        "termination": "provider_native",
        "start_game": CANONICAL_START_GAME,
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
        raise SystemExit(f"raw/{name}.json workload mismatch: unexpected config key(s): {names}")
    if mismatches:
        raise SystemExit(f"raw/{name}.json workload mismatch: " + "; ".join(mismatches))


def env_steps_per_sec_samples(payload: dict[str, Any], label: str) -> list[float]:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit(f"{label} is missing non-empty benchmark runs")
    samples: list[float] = []
    for index, run_payload in enumerate(runs):
        if not isinstance(run_payload, dict):
            raise SystemExit(f"{label} run {index} is not an object")
        try:
            sample = float(run_payload["env_steps_per_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{label} run {index} has invalid env_steps_per_sec") from exc
        if not math.isfinite(sample):
            raise SystemExit(f"{label} run {index} has non-finite env_steps_per_sec")
        if sample <= 0.0:
            raise SystemExit(f"{label} run {index} has non-positive env_steps_per_sec")
        samples.append(sample)
    return samples


def load_raw(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    name: str,
    *,
    steps: int | None = None,
    repeats: int | None = None,
) -> dict[str, Any]:
    payload = json.loads(target_read(args, plan, f"{plan.run_dir}/raw/{name}.json"))
    if not isinstance(payload, dict):
        raise SystemExit(f"raw/{name}.json is not a JSON object")
    require_raw_payload_matches_plan(
        payload,
        args,
        plan,
        name,
        steps=steps,
        repeats=repeats,
    )
    env_steps_per_sec_samples(payload, f"raw/{name}.json")
    return payload


def raw_file_record(args: argparse.Namespace, plan: BenchmarkPlan, name: str, *, tier: str) -> dict[str, str]:
    load_raw(
        args,
        plan,
        name,
        steps=1000 if tier == "smoke" else args.steps,
        repeats=1 if tier == "smoke" else args.repeats,
    )
    return {"file": f"raw/{name}.json", "tier": tier}


def invocation_median(payload: dict[str, Any]) -> float:
    return median(env_steps_per_sec_samples(payload, "raw payload"))


def invocation_mean(payload: dict[str, Any]) -> float:
    samples = env_steps_per_sec_samples(payload, "raw payload")
    return sum(samples) / len(samples)


def all_samples(payloads: list[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for payload in payloads:
        samples.extend(env_steps_per_sec_samples(payload, "raw payload"))
    return samples


def aggregate_single(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    *,
    measured_count: int,
    load_values: list[float | None],
    load_labels: list[str] | None = None,
) -> dict[str, Any]:
    payloads = [load_raw(args, plan, f"measured-ref-{index:02d}") for index in range(measured_count)]
    medians = [invocation_median(payload) for payload in payloads]
    samples = all_samples(payloads)
    convergence = single_ref_convergence(
        medians,
        samples,
        load_ok=load_ok_for_validity(args, load_values),
        checkpoints=plan.checkpoints,
    )
    measured = [
        {
            "file": f"raw/measured-ref-{index:02d}.json",
            "mean_env_steps_per_sec": invocation_mean(payload),
            "median_env_steps_per_sec": invocation_median(payload),
            "samples_env_steps_per_sec": env_steps_per_sec_samples(payload, "raw payload"),
        }
        for index, payload in enumerate(payloads)
    ]
    ref = plan.refs[0]
    smoke_raw_files = [raw_file_record(args, plan, "smoke-ref", tier="smoke")]
    warmup_raw_files = [
        raw_file_record(args, plan, f"warmup-ref-{index:02d}", tier="warmup")
        for index in range(plan.warmups)
    ]
    return {
        **base_aggregate(args, plan, load_values, load_labels=load_labels),
        "mode": "single_ref_fixed_local",
        "refs": {"ref": ref.ref},
        "shas": {"ref": ref.sha},
        "measured_invocations": measured,
        "smoke_raw_files": smoke_raw_files,
        "warmup_raw_files": [record["file"] for record in warmup_raw_files],
        "setup_only_raw_files": smoke_raw_files + warmup_raw_files,
        "load_gate_passed": load_ok(load_values, args.max_load),
        "load_gate_ignored_for_validity": bool(args.force_busy),
        **convergence,
    }


def aggregate_compare(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    *,
    measured_count: int,
    load_values: list[float | None],
    load_labels: list[str] | None = None,
) -> dict[str, Any]:
    pairs = []
    baseline_medians = []
    candidate_medians = []
    baseline_samples = []
    candidate_samples = []
    for index in range(measured_count):
        baseline_payload = load_raw(args, plan, f"measured-baseline-{index:02d}")
        candidate_payload = load_raw(args, plan, f"measured-candidate-{index:02d}")
        baseline_median = invocation_median(baseline_payload)
        candidate_median = invocation_median(candidate_payload)
        baseline_medians.append(baseline_median)
        candidate_medians.append(candidate_median)
        baseline_samples.extend(env_steps_per_sec_samples(baseline_payload, "raw payload"))
        candidate_samples.extend(env_steps_per_sec_samples(candidate_payload, "raw payload"))
        pairs.append(
            {
                "pair_index": index,
                "baseline_file": f"raw/measured-baseline-{index:02d}.json",
                "candidate_file": f"raw/measured-candidate-{index:02d}.json",
                "baseline_median_env_steps_per_sec": baseline_median,
                "candidate_median_env_steps_per_sec": candidate_median,
                "pair_ratio": candidate_median / baseline_median,
            }
        )
    pair_ratios = [pair["pair_ratio"] for pair in pairs]
    convergence = comparison_convergence(
        pair_ratios,
        load_ok=load_ok_for_validity(args, load_values),
        checkpoints=plan.checkpoints,
    )
    baseline = next(ref for ref in plan.refs if ref.role == "baseline")
    candidate = next(ref for ref in plan.refs if ref.role == "candidate")
    smoke_raw_files = [
        raw_file_record(args, plan, "smoke-baseline", tier="smoke"),
        raw_file_record(args, plan, "smoke-candidate", tier="smoke"),
    ]
    warmup_raw_files = [
        raw_file_record(args, plan, f"warmup-baseline-{index:02d}", tier="warmup")
        for index in range(plan.warmups)
    ] + [
        raw_file_record(args, plan, f"warmup-candidate-{index:02d}", tier="warmup")
        for index in range(plan.warmups)
    ]
    return {
        **base_aggregate(args, plan, load_values, load_labels=load_labels),
        "mode": "paired_compare_fixed_local",
        "refs": {"baseline": baseline.ref, "candidate": candidate.ref},
        "shas": {"baseline": baseline.sha, "candidate": candidate.sha},
        "measured_pair_details": pairs,
        "smoke_raw_files": smoke_raw_files,
        "warmup_raw_files": [record["file"] for record in warmup_raw_files],
        "setup_only_raw_files": smoke_raw_files + warmup_raw_files,
        "baseline_run_median_summary": summary(baseline_medians),
        "candidate_run_median_summary": summary(candidate_medians),
        "baseline_all_sample_summary": summary(baseline_samples),
        "candidate_all_sample_summary": summary(candidate_samples),
        "paired_gain_percent": (convergence["median_pair_ratio"] - 1.0) * 100.0,
        "load_gate_passed": load_ok(load_values, args.max_load),
        "load_gate_ignored_for_validity": bool(args.force_busy),
        **convergence,
    }


def archive_hashes(plan: BenchmarkPlan) -> dict[str, str]:
    return {ref.role: sha256_path(ref.archive) for ref in plan.refs if ref.archive.exists()}


def target_file_hashes(args: argparse.Namespace, plan: BenchmarkPlan, paths: list[str]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for path in paths:
        if not target_exists(args, plan, path):
            hashes[path] = None
            continue
        hashes[path] = sha256_path(Path(path))
    return hashes


def workload_payload(args: argparse.Namespace, plan: BenchmarkPlan) -> dict[str, Any]:
    return {
        "rom_path": plan.rom_path,
        "state_dir": plan.state_dir,
        "rayon_num_threads": 12,
        "num_envs": CANONICAL_NUM_ENVS,
        "steps": args.steps,
        "repeats": args.repeats,
        "warmup": 100,
        "frame_skip": CANONICAL_FRAME_SKIP,
        "frame_stack": CANONICAL_FRAME_STACK,
        "grayscale": True,
        "crop_top": CANONICAL_CROP_TOP,
        "crop_bottom": CANONICAL_CROP_BOTTOM,
        "obs_crop_mode": CANONICAL_OBS_CROP_MODE,
        "resize": [CANONICAL_RESIZE_WIDTH, CANONICAL_RESIZE_HEIGHT],
        "states": list(STATE_NAMES),
        "action_set": CANONICAL_ACTION_SET,
        "action": None,
        "actions": list(ACTION_NAMES),
        "action_seed": ACTION_SEED,
        "obs_resize_algorithm": "area",
        "include_info": True,
        "terminate_on_flag": CANONICAL_TERMINATE_ON_FLAG,
        "start_game": CANONICAL_START_GAME,
    }


def base_aggregate(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    load_values: list[float | None],
    *,
    load_labels: list[str] | None = None,
) -> dict[str, Any]:
    if load_labels is None:
        load_labels = [f"load-{index}" for index in range(len(load_values))]
    state_paths = [f"{plan.state_dir}/{name}.state" for name in STATE_NAMES]
    file_hashes = target_file_hashes(args, plan, [plan.rom_path, *state_paths])
    rom_sha256 = file_hashes.get(plan.rom_path)
    state_sha256 = {
        name: file_hashes.get(f"{plan.state_dir}/{name}.state") for name in STATE_NAMES
    }
    workload = {
        **workload_payload(args, plan),
        "expected_rom_sha256": EXPECTED_SMB_ROM_SHA256,
        "rom_sha256": rom_sha256,
        "state_sha256": state_sha256,
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_tier": benchmark_tier(args, plan),
        "execution_target": "local_machine",
        "target_run_dir": plan.run_dir,
        "run_name": plan.run_name,
        "local_git_status_short": run(["git", "status", "--short"]).stdout.splitlines(),
        "dirty_local_files_excluded": True,
        "workload": workload,
        "workload_hash": stable_hash(workload),
        "sequential_policy": {
            "warmups": plan.warmups,
            "checkpoints": list(plan.checkpoints),
            "max_measured_samples": plan.measured_cap,
        },
        "source_archive_sha256": archive_hashes(plan),
        "expected_rom_sha256": EXPECTED_SMB_ROM_SHA256,
        "rom_sha256": rom_sha256,
        "state_sha256": state_sha256,
        "load_1min_labels": load_labels,
        "load_1min_values": load_values,
        "load_1min_by_label": dict(zip(load_labels, load_values, strict=False)),
        "load_policy": {
            "max_load": args.max_load,
            "force_busy": bool(args.force_busy),
        },
        "benchmark_limits": {
            "max_measured_invocations": args.max_measured_invocations,
            "max_wall_clock_minutes": args.max_wall_clock_minutes,
        },
        "command": {
            "benchmark": "RAYON_NUM_THREADS=12 .venv/bin/python scripts/benchmark_sps.py ...",
            "stats_helper": "scripts/benchmark_stats.py",
        },
    }


def write_aggregate(args: argparse.Namespace, plan: BenchmarkPlan, aggregate: dict[str, Any]) -> None:
    text = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    target_write(args, plan, f"{plan.run_dir}/aggregate.json", text)


def run_single(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    start_time: float,
    setup_load_snapshots: list[tuple[str, float | None]] | None = None,
) -> dict[str, Any]:
    require_wall_clock_budget(args, start_time, "single-ref smoke")
    run_invocation(
        args,
        plan,
        "ref",
        "smoke-ref",
        steps=1000,
        repeats=1,
        start_time=start_time,
    )
    for index in range(plan.warmups):
        require_wall_clock_budget(args, start_time, f"single-ref warmup {index}")
        run_invocation(
            args,
            plan,
            "ref",
            f"warmup-ref-{index:02d}",
            steps=args.steps,
            repeats=args.repeats,
            start_time=start_time,
        )

    load_snapshots = list(setup_load_snapshots or [])
    before_measured_label, before_measured_load = capture_load_gate_snapshot(
        args,
        plan,
        "before-measured",
        "measured phase",
        start_time=start_time,
    )
    load_snapshots.append((before_measured_label, before_measured_load))
    load_values = [value for _label, value in load_snapshots]
    load_labels = [label for label, _value in load_snapshots]
    require_load_gate(args, before_measured_load, "measured phase")
    aggregate: dict[str, Any] = {}
    measured_count = 0
    for checkpoint in plan.checkpoints:
        while measured_count < checkpoint:
            if wall_clock_limit_exceeded(args, start_time):
                if measured_count == 0:
                    raise SystemExit("wall-clock limit exhausted before any measured invocations")
                aggregate = aggregate_single(
                    args,
                    plan,
                    measured_count=measured_count,
                    load_values=load_values,
                    load_labels=load_labels,
                )
                aggregate["limit_stop_reason"] = "max_wall_clock_minutes"
                write_aggregate(args, plan, aggregate)
                return aggregate
            run_invocation(
                args,
                plan,
                "ref",
                f"measured-ref-{measured_count:02d}",
                steps=args.steps,
                repeats=args.repeats,
                start_time=start_time,
            )
            measured_count += 1
        checkpoint_label, checkpoint_load = capture_load_gate_snapshot(
            args,
            plan,
            f"after-checkpoint-{checkpoint}",
            f"checkpoint {checkpoint}",
            start_time=start_time,
        )
        load_snapshots.append((checkpoint_label, checkpoint_load))
        load_values = [value for _label, value in load_snapshots]
        load_labels = [label for label, _value in load_snapshots]
        aggregate = aggregate_single(
            args,
            plan,
            measured_count=measured_count,
            load_values=load_values,
            load_labels=load_labels,
        )
        write_aggregate(args, plan, aggregate)
        stop_reason = load_gate_stop_reason(args, checkpoint_load)
        if stop_reason is not None:
            aggregate["limit_stop_reason"] = stop_reason
            write_aggregate(args, plan, aggregate)
            return aggregate
        if aggregate["should_stop"]:
            break
    if measured_invocation_limit_applies(args, plan) and measured_count >= plan.measured_cap:
        aggregate["limit_stop_reason"] = aggregate.get("limit_stop_reason") or "max_measured_invocations"
        write_aggregate(args, plan, aggregate)
    return aggregate


def pair_order(index: int) -> tuple[str, str]:
    return ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")


def run_compare(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    start_time: float,
    setup_load_snapshots: list[tuple[str, float | None]] | None = None,
) -> dict[str, Any]:
    require_wall_clock_budget(args, start_time, "baseline smoke")
    run_invocation(
        args,
        plan,
        "baseline",
        "smoke-baseline",
        steps=1000,
        repeats=1,
        start_time=start_time,
    )
    require_wall_clock_budget(args, start_time, "candidate smoke")
    run_invocation(
        args,
        plan,
        "candidate",
        "smoke-candidate",
        steps=1000,
        repeats=1,
        start_time=start_time,
    )
    for index in range(plan.warmups):
        for role in pair_order(index):
            require_wall_clock_budget(args, start_time, f"{role} warmup {index}")
            run_invocation(
                args,
                plan,
                role,
                f"warmup-{role}-{index:02d}",
                steps=args.steps,
                repeats=args.repeats,
                start_time=start_time,
            )

    load_snapshots = list(setup_load_snapshots or [])
    before_measured_label, before_measured_load = capture_load_gate_snapshot(
        args,
        plan,
        "before-measured",
        "measured phase",
        start_time=start_time,
    )
    load_snapshots.append((before_measured_label, before_measured_load))
    load_values = [value for _label, value in load_snapshots]
    load_labels = [label for label, _value in load_snapshots]
    require_load_gate(args, before_measured_load, "measured phase")
    aggregate: dict[str, Any] = {}
    measured_count = 0
    for checkpoint in plan.checkpoints:
        while measured_count < checkpoint:
            completed_roles: list[str] = []
            for role in pair_order(measured_count):
                if wall_clock_limit_exceeded(args, start_time):
                    if measured_count == 0:
                        raise SystemExit("wall-clock limit exhausted before any measured pairs")
                    aggregate = aggregate_compare(
                        args,
                        plan,
                        measured_count=measured_count,
                        load_values=load_values,
                        load_labels=load_labels,
                    )
                    aggregate["limit_stop_reason"] = "max_wall_clock_minutes"
                    if completed_roles:
                        aggregate["discarded_incomplete_pair_raw_files"] = [
                            f"raw/measured-{completed_role}-{measured_count:02d}.json"
                            for completed_role in completed_roles
                        ]
                    write_aggregate(args, plan, aggregate)
                    return aggregate
                run_invocation(
                    args,
                    plan,
                    role,
                    f"measured-{role}-{measured_count:02d}",
                    steps=args.steps,
                    repeats=args.repeats,
                    start_time=start_time,
                )
                completed_roles.append(role)
            measured_count += 1
        checkpoint_label, checkpoint_load = capture_load_gate_snapshot(
            args,
            plan,
            f"after-checkpoint-{checkpoint}",
            f"checkpoint {checkpoint}",
            start_time=start_time,
        )
        load_snapshots.append((checkpoint_label, checkpoint_load))
        load_values = [value for _label, value in load_snapshots]
        load_labels = [label for label, _value in load_snapshots]
        aggregate = aggregate_compare(
            args,
            plan,
            measured_count=measured_count,
            load_values=load_values,
            load_labels=load_labels,
        )
        write_aggregate(args, plan, aggregate)
        stop_reason = load_gate_stop_reason(args, checkpoint_load)
        if stop_reason is not None:
            aggregate["limit_stop_reason"] = stop_reason
            write_aggregate(args, plan, aggregate)
            return aggregate
        if aggregate["should_stop"]:
            break
    if measured_invocation_limit_applies(args, plan) and measured_count >= plan.measured_cap:
        aggregate["limit_stop_reason"] = aggregate.get("limit_stop_reason") or "max_measured_invocations"
        write_aggregate(args, plan, aggregate)
    return aggregate


def finalize_local(plan: BenchmarkPlan) -> Path:
    run_dir = Path(plan.run_dir)
    local_results_root = local_results_root_for_plan(plan)
    local_dir = local_results_root / plan.run_name
    if local_dir.exists():
        shutil.rmtree(local_dir)
    raw_dir = local_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "aggregate.json", local_dir / "aggregate.json")
    for path in sorted((run_dir / "raw").glob("*")):
        if path.suffix == ".json" and path.name.endswith(".stdout.json"):
            continue
        if path.suffix in {".json", ".txt"}:
            shutil.copy2(path, raw_dir / path.name)
    copied_files = []
    for path in sorted(local_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            if path.suffix == ".json":
                json.loads(path.read_text())
            copied_files.append(
                {
                    "path": str(path.relative_to(local_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    if not any(item["path"].startswith("raw/") for item in copied_files):
        raise SystemExit("No raw benchmark files were copied; refusing local cleanup")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local_run_dir": str(run_dir),
        "local_result_dir": str(local_dir),
        "copied_files": copied_files,
        "cleanup_policy": {
            "local_bulk_removed": ["sources/", "archives/", "raw/*.stdout.json"],
        },
    }
    (local_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    aggregate = json.loads((local_dir / "aggregate.json").read_text())
    append_index(plan.run_name, local_dir, aggregate, local_results_root)
    shutil.rmtree(run_dir / "sources", ignore_errors=True)
    shutil.rmtree(run_dir / "archives", ignore_errors=True)
    for path in (run_dir / "raw").glob("*.stdout.json"):
        path.unlink()
    return local_dir


def local_results_root_for_plan(plan: BenchmarkPlan) -> Path:
    run_dir = Path(plan.run_dir)
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent / LOCAL_RESULTS_SUBDIR
    return run_dir.parent / LOCAL_RESULTS_SUBDIR


def append_index(
    run_name: str,
    local_dir: Path,
    aggregate: dict[str, Any],
    local_results_root: Path,
) -> None:
    local_results_root.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "local_result_dir": str(local_dir),
        "mode": aggregate.get("mode"),
        "benchmark_tier": aggregate.get("benchmark_tier"),
        "refs": aggregate.get("refs"),
        "shas": aggregate.get("shas"),
        "workload_hash": aggregate.get("workload_hash"),
        "measured_invocation_count": aggregate.get("measured_invocation_count"),
        "measured_pairs": aggregate.get(
            "measured_pairs", len(aggregate.get("measured_pair_details", []))
        ),
        "official_median_sps": aggregate.get("official_median_sps"),
        "mean_invocation_median_sps": aggregate.get("mean_invocation_median_sps"),
        "bootstrap_ci95_invocation_median_sps": aggregate.get(
            "bootstrap_ci95_invocation_median_sps"
        ),
        "median_pair_ratio": aggregate.get("median_pair_ratio"),
        "mean_pair_ratio": aggregate.get("mean_pair_ratio"),
        "pair_ratio_bootstrap_ci95": aggregate.get("pair_ratio_bootstrap_ci95"),
        "candidate_faster_pairs": aggregate.get("candidate_faster_pairs"),
        "candidate_faster_pairs_required_for_win": aggregate.get(
            "candidate_faster_pairs_required_for_win"
        ),
        "validity_passed": aggregate.get("validity_passed"),
        "load_gate_passed": aggregate.get("load_gate_passed"),
        "load_gate_ignored_for_validity": aggregate.get("load_gate_ignored_for_validity"),
        "limit_stop_reason": aggregate.get("limit_stop_reason"),
        "previous_limit_stop_reason": aggregate.get("previous_limit_stop_reason"),
        "benchmark_limits": aggregate.get("benchmark_limits"),
        "setup_only_raw_files": aggregate.get("setup_only_raw_files"),
        "discarded_incomplete_pair_raw_files": aggregate.get(
            "discarded_incomplete_pair_raw_files"
        ),
        "expected_rom_sha256": aggregate.get("expected_rom_sha256"),
        "rom_sha256": aggregate.get("rom_sha256"),
        "state_sha256": aggregate.get("state_sha256"),
        "decision": aggregate.get("decision"),
    }
    index_path = local_results_root / "index.jsonl"
    existing = []
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line)
            if parsed.get("run_name") != run_name:
                existing.append(json.dumps(parsed, sort_keys=True))
    with index_path.open("w") as handle:
        for line in existing:
            handle.write(line + "\n")
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def aggregate_with_extra_load_snapshot(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    aggregate: dict[str, Any],
    *,
    label: str,
    load_value: float | None,
) -> dict[str, Any]:
    load_labels = list(aggregate.get("load_1min_labels", []))
    load_values = list(aggregate.get("load_1min_values", []))
    load_labels.append(label)
    load_values.append(load_value)
    if plan.mode == "single":
        measured_count = int(aggregate["measured_invocation_count"])
        refreshed = aggregate_single(
            args,
            plan,
            measured_count=measured_count,
            load_values=load_values,
            load_labels=load_labels,
        )
    else:
        measured_count = int(aggregate["measured_pairs"])
        refreshed = aggregate_compare(
            args,
            plan,
            measured_count=measured_count,
            load_values=load_values,
            load_labels=load_labels,
        )
    for key in ("limit_stop_reason", "discarded_incomplete_pair_raw_files"):
        if key in aggregate:
            refreshed[key] = aggregate[key]
    stop_reason = load_gate_stop_reason(args, load_value)
    if stop_reason is not None:
        if "limit_stop_reason" in refreshed:
            refreshed["previous_limit_stop_reason"] = refreshed["limit_stop_reason"]
        refreshed["limit_stop_reason"] = stop_reason
    return refreshed


def execute(args: argparse.Namespace, plan: BenchmarkPlan) -> dict[str, Any]:
    start_time = time.monotonic()
    if args.dry_run:
        workload = workload_payload(args, plan)
        planned_workload_hash = stable_hash(workload)
        payload = {
            "dry_run": True,
            "plan": plan_to_json(plan),
            "benchmark_tier": benchmark_tier(args, plan),
            "workload": workload,
            "workload_hash": planned_workload_hash,
            "planned_workload_hash": planned_workload_hash,
            "workload_hash_scope": "planned_without_rom_or_state_file_hashes",
            "load_policy": {
                "max_load": args.max_load,
                "force_busy": bool(args.force_busy),
            },
            "benchmark_limits": {
                "max_measured_invocations": args.max_measured_invocations,
                "max_wall_clock_minutes": args.max_wall_clock_minutes,
            },
            "git_status": run(["git", "status", "--short"]).stdout.splitlines(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    validate_rom_hash(args.rom_path)
    require_wall_clock_budget(args, start_time, "source archive creation")
    archived_refs = create_archives(plan)
    plan = BenchmarkPlan(
        mode=plan.mode,
        run_name=plan.run_name,
        run_dir=plan.run_dir,
        refs=archived_refs,
        rom_path=plan.rom_path,
        state_dir=plan.state_dir,
        checkpoints=plan.checkpoints,
        warmups=plan.warmups,
        measured_cap=plan.measured_cap,
    )
    target_run(args, plan, f"mkdir -p {quote(plan.run_dir + '/raw')}")
    require_wall_clock_budget(args, start_time, "state setup")
    ensure_states(args, plan)
    initial_load, _ = capture_load(args, plan, "before-setup")
    require_load_gate(args, initial_load, "source preparation")
    require_wall_clock_budget(args, start_time, "source preparation")
    prepare_sources(args, plan)
    aggregate = (
        run_single(args, plan, start_time, [("before-setup", initial_load)])
        if plan.mode == "single"
        else run_compare(args, plan, start_time, [("before-setup", initial_load)])
    )
    after_measured_label, after_measured_load = capture_load_gate_snapshot(
        args,
        plan,
        "after-measured",
        "after measured phase",
        start_time=start_time,
    )
    aggregate = aggregate_with_extra_load_snapshot(
        args,
        plan,
        aggregate,
        label=after_measured_label,
        load_value=after_measured_load,
    )
    write_aggregate(args, plan, aggregate)
    if not args.no_finalize:
        finalize_local(plan)
    return aggregate


def plan_to_json(plan: BenchmarkPlan) -> dict[str, Any]:
    return {
        "mode": plan.mode,
        "run_name": plan.run_name,
        "run_dir": plan.run_dir,
        "refs": [
            {"role": ref.role, "ref": ref.ref, "sha": ref.sha, "archive": str(ref.archive)}
            for ref in plan.refs
        ],
        "rom_path": plan.rom_path,
        "state_dir": plan.state_dir,
        "warmups": plan.warmups,
        "checkpoints": list(plan.checkpoints),
        "measured_cap": plan.measured_cap,
    }


def resolve_rom_path_for_args(args: argparse.Namespace) -> str:
    try:
        path = resolve_required_rom_path(args.rom_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.dry_run:
        if not path.exists():
            raise SystemExit(f"ROM path does not exist: {path}")
        if not path.is_file():
            raise SystemExit(f"ROM path is not a file: {path}")
    return str(path.resolve(strict=False))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refs", nargs="+", help="single ref, candidate ref, or baseline candidate")
    parser.add_argument("--single", action="store_true", help="Benchmark one ref only.")
    parser.add_argument(
        "--rom-path",
        default=None,
        help="ROM path on the benchmark machine. Defaults to Stable Retro-compatible discovery.",
    )
    parser.add_argument(
        "--state-dir",
        help=(
            "State directory on the benchmark machine. Defaults to "
            f"{AUTORESEARCH_ROOT_ENV}/{BENCHMARK_STATE_SUBDIR}."
        ),
    )
    parser.add_argument(
        "--state-source",
        type=Path,
        help=(
            "Optional source directory for missing state files. By default, "
            f"states are read from {AUTORESEARCH_ROOT_ENV}/{BENCHMARK_STATE_SUBDIR}."
        ),
    )
    parser.add_argument(
        "--run-root",
        help=(
            "Root for temporary benchmark runs. Defaults to "
            f"{AUTORESEARCH_ROOT_ENV}/{BENCHMARK_ROOT_SUBDIR}."
        ),
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--max-load", type=float)
    parser.add_argument("--force-busy", action="store_true")
    parser.add_argument(
        "--max-measured-invocations",
        type=int,
        help="Stop after at most this many measured invocations or comparison pairs.",
    )
    parser.add_argument(
        "--max-wall-clock-minutes",
        type=float,
        help="Stop before starting another benchmark phase after this many minutes.",
    )
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.rom_path = resolve_rom_path_for_args(args)

    autoresearch_root = require_arg_or_env_or_dotenv_path(
        AUTORESEARCH_ROOT_ENV,
        "autoresearch root",
        must_be_dir=True,
    )
    run_root = Path(args.run_root).expanduser() if args.run_root else autoresearch_root / BENCHMARK_ROOT_SUBDIR
    args.run_root = str(run_root.resolve(strict=False))
    state_dir = Path(args.state_dir).expanduser() if args.state_dir else autoresearch_root / BENCHMARK_STATE_SUBDIR
    args.state_dir = str(state_dir.resolve(strict=False))
    args.state_source = args.state_source.resolve(strict=False) if args.state_source else None
    if args.max_load is None:
        args.max_load = max(os.cpu_count() or 1, 1) / 3
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.warmups is not None and args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.max_measured_invocations is not None and args.max_measured_invocations <= 0:
        raise SystemExit("--max-measured-invocations must be positive")
    if args.max_wall_clock_minutes is not None and args.max_wall_clock_minutes <= 0:
        raise SystemExit("--max-wall-clock-minutes must be positive")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    execute(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
