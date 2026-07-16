#!/usr/bin/env python3
"""Produce a paired Turbo-versus-upstream-Stable-Retro benchmark report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence

from supermariobrosnes_turbo import default_rom_path

try:
    from benchmark_stats import bootstrap_ci_median, env_steps_per_sec_samples, median, summary
    from benchmark_workload import canonical_env_args
    from dotenv_utils import env_or_dotenv_path
except ModuleNotFoundError:
    from scripts.benchmark_stats import (
        bootstrap_ci_median,
        env_steps_per_sec_samples,
        median,
        summary,
    )
    from scripts.benchmark_workload import canonical_env_args
    from scripts.dotenv_utils import env_or_dotenv_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_sps.py"
DEFAULT_PARITY_TEST = (
    REPO_ROOT
    / "tests"
    / "test_supermariobrosnes_turbo_vec_env_parity.py"
)
PROTOCOL = "paired_alternating_turbo_vs_upstream_stable_retro_v1"
BACKENDS = ("turbo", "stable-retro")
MATCHED_CONFIG_KEYS = (
    "rom_sha256",
    "num_envs",
    "steps",
    "repeats",
    "warmup",
    "frame_skip",
    "frame_stack",
    "frame_maxpool",
    "grayscale",
    "crop_top",
    "crop_bottom",
    "obs_crop_mode",
    "resize_width",
    "resize_height",
    "obs_resize_algorithm",
    "obs_layout",
    "action_set",
    "action",
    "actions",
    "action_seed",
    "state",
    "states",
    "lane_states",
    "include_info",
    "terminate_on_flag",
    "termination",
    "start_game",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_shapes(raw: str) -> tuple[int, ...]:
    try:
        shapes = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--shapes must contain comma-separated integers") from exc
    if not shapes or any(shape <= 0 for shape in shapes):
        raise argparse.ArgumentTypeError("--shapes must contain positive integers")
    if len(set(shapes)) != len(shapes):
        raise argparse.ArgumentTypeError("--shapes must not contain duplicates")
    return shapes


def default_output_dir() -> Path | None:
    root = env_or_dotenv_path("AUTORESEARCH_ROOT_PATH")
    if root is None:
        return None
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return root / "benchmark-reports" / f"turbo-vs-stable-retro-{stamp}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=parse_shapes, default=(1, 16, 32))
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--minimum-speedup", type=float, default=2.0)
    parser.add_argument("--minimum-pairs-for-claim", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-start-load", type=float, default=4.0)
    parser.add_argument("--load-poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-load-wait-seconds", type=float, default=900.0)
    parser.add_argument("--rom-path", type=Path, default=default_rom_path())
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--benchmark-script", type=Path, default=DEFAULT_BENCHMARK_SCRIPT)
    parser.add_argument("--skip-correctness-checks", action="store_true")
    parser.add_argument("--force-busy", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty source tree, but mark the report invalid for a publishable claim.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for field in (
        "pairs",
        "steps",
        "repeats",
        "minimum_pairs_for_claim",
        "bootstrap_samples",
    ):
        if getattr(args, field) <= 0:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive")
    for field in ("warmup_pairs", "warmup"):
        if getattr(args, field) < 0:
            raise SystemExit(f"--{field.replace('_', '-')} must be non-negative")
    for field in (
        "minimum_speedup",
        "max_start_load",
        "load_poll_seconds",
        "max_load_wait_seconds",
    ):
        value = float(getattr(args, field))
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"--{field.replace('_', '-')} must be a positive finite number")
    if args.rom_path is None:
        raise SystemExit(
            "ROM path required; pass --rom-path or import it with "
            "`python -m supermariobrosnes_turbo.import /path/to/roms`"
        )
    if not args.benchmark_script.is_file():
        raise SystemExit(f"benchmark script does not exist: {args.benchmark_script}")


def run_capture(
    command: Sequence[str], *, cwd: Path = REPO_ROOT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_source_identity() -> dict[str, Any]:
    commit_result = run_capture(["git", "rev-parse", "HEAD"])
    status_result = run_capture(["git", "status", "--porcelain=v1"])
    diff_result = run_capture(["git", "diff", "--binary", "HEAD"])
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    status = status_result.stdout.splitlines() if status_result.returncode == 0 else []
    diff_bytes = diff_result.stdout.encode() if diff_result.returncode == 0 else b""
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status,
        "working_tree_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def optional_command_value(command: Sequence[str]) -> str | None:
    result = run_capture(command)
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def system_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
    }
    if platform.system() == "Darwin":
        metadata.update(
            {
                "cpu_brand": optional_command_value(["sysctl", "-n", "machdep.cpu.brand_string"]),
                "physical_cpus": optional_command_value(["sysctl", "-n", "hw.physicalcpu"]),
                "memory_bytes": optional_command_value(["sysctl", "-n", "hw.memsize"]),
            }
        )
    return metadata


def pair_order(shape_index: int, pair_index: int) -> tuple[str, str]:
    return BACKENDS if (shape_index + pair_index) % 2 == 0 else tuple(reversed(BACKENDS))


def canonical_args_for_shape(shape: int) -> list[str]:
    arguments = canonical_env_args()
    index = arguments.index("--num-envs")
    arguments[index + 1] = str(shape)
    return arguments


def benchmark_command(
    args: argparse.Namespace,
    *,
    backend: str,
    shape: int,
    output_json: Path,
) -> list[str]:
    command = [
        str(args.python),
        str(args.benchmark_script),
        *canonical_args_for_shape(shape),
        "--rom-path",
        str(args.rom_path),
        "--steps",
        str(args.steps),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--output-json",
        str(output_json),
        "--json",
    ]
    if args.state_dir is not None:
        command.extend(("--state-dir", str(args.state_dir)))
    command.append("--skip-load-preflight")
    if backend == "stable-retro":
        command.append("--stable-retro-baseline")
    return command


def wait_for_load_headroom(args: argparse.Namespace) -> dict[str, Any]:
    load_available = hasattr(os, "getloadavg")
    if args.force_busy:
        current = None
        if load_available:
            try:
                current = os.getloadavg()[0]
            except OSError:
                pass
        return {
            "enabled": False,
            "available": current is not None,
            "initial_1min": current,
            "accepted_1min": current,
            "max_start_load": args.max_start_load,
            "waited_s": 0.0,
            "load_ok": False,
        }
    if not load_available:
        return {
            "enabled": True,
            "available": False,
            "initial_1min": None,
            "accepted_1min": None,
            "max_start_load": args.max_start_load,
            "waited_s": 0.0,
            "load_ok": True,
        }
    started = time.monotonic()
    initial_load = None
    while True:
        try:
            load = os.getloadavg()[0]
        except OSError:
            return {
                "enabled": True,
                "available": False,
                "initial_1min": initial_load,
                "accepted_1min": None,
                "max_start_load": args.max_start_load,
                "waited_s": time.monotonic() - started,
                "load_ok": True,
            }
        if initial_load is None:
            initial_load = load
        if load < args.max_start_load:
            waited = time.monotonic() - started
            print(
                f"load_preflight initial_1min={initial_load:.2f} "
                f"accepted_1min={load:.2f} max_start_load={args.max_start_load:.2f} "
                f"waited_s={waited:.1f}",
                flush=True,
            )
            return {
                "enabled": True,
                "available": True,
                "initial_1min": initial_load,
                "accepted_1min": load,
                "max_start_load": args.max_start_load,
                "waited_s": waited,
                "load_ok": True,
            }
        elapsed = time.monotonic() - started
        if elapsed >= args.max_load_wait_seconds:
            raise SystemExit(
                f"load remained at or above {args.max_start_load:.2f} for "
                f"{elapsed:.0f}s; rerun on a quiet host or use --force-busy for diagnostics"
            )
        print(
            f"waiting_for_load current={load:.2f} target_below={args.max_start_load:.2f}",
            flush=True,
        )
        time.sleep(min(args.load_poll_seconds, args.max_load_wait_seconds - elapsed))


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read benchmark JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark JSON is not an object: {path}")
    return payload


def validate_backend_payload(
    payload: dict[str, Any], *, backend: str, shape: int, path: Path
) -> None:
    if payload.get("backend") != backend:
        raise ValueError(f"{path} backend={payload.get('backend')!r}, expected {backend!r}")
    package = payload.get("package")
    expected_package = "supermariobrosnes-turbo" if backend == "turbo" else "stable-retro"
    if not isinstance(package, dict) or package.get("name") != expected_package:
        raise ValueError(f"{path} package metadata does not identify {expected_package}")
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("num_envs") != shape:
        raise ValueError(f"{path} does not report num_envs={shape}")
    expected_shape = [shape, 4, 84, 84]
    observation = payload.get("observation")
    if not isinstance(observation, dict) or observation.get("shape") != expected_shape:
        raise ValueError(f"{path} observation shape does not match {expected_shape}")
    if observation.get("dtype") != "uint8":
        raise ValueError(f"{path} observation dtype is not uint8")
    env_steps_per_sec_samples(payload, path)


def validate_matched_pair(
    turbo: dict[str, Any], stable: dict[str, Any], *, turbo_path: Path, stable_path: Path
) -> None:
    turbo_config = turbo.get("config")
    stable_config = stable.get("config")
    if not isinstance(turbo_config, dict) or not isinstance(stable_config, dict):
        raise ValueError("paired payloads must contain config objects")
    mismatches = [
        key
        for key in MATCHED_CONFIG_KEYS
        if turbo_config.get(key) != stable_config.get(key)
    ]
    if mismatches:
        raise ValueError(
            f"workload mismatch between {turbo_path} and {stable_path}: {', '.join(mismatches)}"
        )
    if turbo.get("observation") != stable.get("observation"):
        raise ValueError(
            f"observation metadata mismatch between {turbo_path} and {stable_path}"
        )


def invocation_median(payload: dict[str, Any], path: Path) -> float:
    return median(env_steps_per_sec_samples(payload, path))


def format_invocation_stats(payload: dict[str, Any], path: Path) -> str:
    samples = env_steps_per_sec_samples(payload, path)
    stats = summary(samples)
    return (
        f"median_sps={stats['median']:.1f} "
        f"mean_sps={stats['mean']:.1f} "
        f"stdev_sps={stats['stdev']:.1f} "
        f"cv_pct={stats['cv'] * 100:.2f} "
        f"min_sps={stats['min']:.1f} "
        f"max_sps={stats['max']:.1f} "
        f"repeats={len(samples)}"
    )


def execute_invocation(
    args: argparse.Namespace,
    *,
    backend: str,
    shape: int,
    output_json: Path,
) -> dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    command = benchmark_command(args, backend=backend, shape=shape, output_json=output_json)
    started = time.perf_counter()
    result = run_capture(command)
    elapsed = time.perf_counter() - started
    log_path = output_json.with_suffix(".log")
    log_path.write_text(
        f"command: {shlex.join(command)}\n"
        f"returncode: {result.returncode}\n"
        f"wall_time_s: {elapsed:.6f}\n"
        "stdout:\n"
        f"{result.stdout}\n"
        "stderr:\n"
        f"{result.stderr}\n"
    )
    if result.returncode != 0:
        raise SystemExit(f"benchmark invocation failed; see {log_path}")
    payload = read_json_object(output_json)
    validate_backend_payload(payload, backend=backend, shape=shape, path=output_json)
    return payload


def correctness_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        "-m",
        "pytest",
        "-q",
        "-m",
        "retro_oracle",
        str(DEFAULT_PARITY_TEST),
    ]


def run_correctness_checks(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if args.skip_correctness_checks:
        return {"executed": False, "passed": False, "reason": "explicitly skipped"}
    command = correctness_command(args)
    started = time.perf_counter()
    result = run_capture(command)
    elapsed = time.perf_counter() - started
    payload = {
        "executed": True,
        "passed": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "wall_time_s": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (output_dir / "correctness.json").write_text(json.dumps(payload, indent=2) + "\n")
    if result.returncode != 0:
        raise SystemExit(f"correctness checks failed; see {output_dir / 'correctness.json'}")
    return payload


def aggregate_shape(
    *,
    shape: int,
    pairs: list[dict[str, Any]],
    minimum_speedup: float,
    minimum_pairs_for_claim: int,
    bootstrap_samples: int,
    load_gate_enforced: bool,
) -> dict[str, Any]:
    turbo_medians = [float(pair["turbo_median_sps"]) for pair in pairs]
    stable_medians = [float(pair["stable_retro_median_sps"]) for pair in pairs]
    ratios = [float(pair["speedup"]) for pair in pairs]
    ci = bootstrap_ci_median(ratios, n=bootstrap_samples, seed=12_345 + shape)
    load_ok = all(bool(pair["load_ok"]) for pair in pairs)
    faster_pairs = sum(ratio > 1.0 for ratio in ratios)
    faster_pairs_required = math.ceil(len(ratios) * 0.75)
    gates = {
        "minimum_pair_count_met": len(pairs) >= minimum_pairs_for_claim,
        "median_speedup_at_least_threshold": median(ratios) >= minimum_speedup,
        "bootstrap_ci95_lower_above_1": ci[0] > 1.0,
        "turbo_faster_in_at_least_75_percent_of_pairs": faster_pairs >= faster_pairs_required,
        "load_gate_enforced": load_gate_enforced,
        "all_measured_load_checks_passed": load_ok,
    }
    return {
        "num_envs": shape,
        "pair_count": len(pairs),
        "turbo_invocation_median_sps": summary(turbo_medians),
        "stable_retro_invocation_median_sps": summary(stable_medians),
        "pair_speedup": summary(ratios),
        "pair_speedup_bootstrap_ci95": ci,
        "turbo_faster_pairs": faster_pairs,
        "turbo_faster_pairs_required": faster_pairs_required,
        "claim_gates": gates,
        "claim_passed": all(gates.values()),
        "pairs": pairs,
    }


def collect_shape(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    shape: int,
    shape_index: int,
    measured: bool,
    load_gate_passed: bool,
) -> list[dict[str, Any]]:
    pair_count = args.pairs if measured else args.warmup_pairs
    phase = "measured" if measured else "warmup"
    collected: list[dict[str, Any]] = []
    for pair_index in range(pair_count):
        order = pair_order(shape_index, pair_index)
        payloads: dict[str, dict[str, Any]] = {}
        paths: dict[str, Path] = {}
        for backend in order:
            filename = f"{phase}-pair-{pair_index + 1:02d}-{backend}.json"
            path = output_dir / "raw" / f"envs-{shape}" / filename
            prefix = (
                f"shape={shape} phase={phase} pair={pair_index + 1}/{pair_count} "
                f"backend={backend}"
            )
            print(prefix, end=" ", flush=True)
            paths[backend] = path
            try:
                payloads[backend] = execute_invocation(
                    args, backend=backend, shape=shape, output_json=path
                )
            except SystemExit:
                print("status=failed", flush=True)
                raise
            print(format_invocation_stats(payloads[backend], path), flush=True)
        validate_matched_pair(
            payloads["turbo"],
            payloads["stable-retro"],
            turbo_path=paths["turbo"],
            stable_path=paths["stable-retro"],
        )
        turbo_median = invocation_median(payloads["turbo"], paths["turbo"])
        stable_median = invocation_median(payloads["stable-retro"], paths["stable-retro"])
        print(
            f"shape={shape} phase={phase} pair={pair_index + 1}/{pair_count} "
            f"turbo_median_sps={turbo_median:.1f} "
            f"stable_retro_median_sps={stable_median:.1f} "
            f"speedup={turbo_median / stable_median:.2f}x",
            flush=True,
        )
        collected.append(
            {
                "pair": pair_index + 1,
                "order": list(order),
                "turbo_file": str(paths["turbo"].relative_to(output_dir)),
                "stable_retro_file": str(paths["stable-retro"].relative_to(output_dir)),
                "turbo_median_sps": turbo_median,
                "stable_retro_median_sps": stable_median,
                "speedup": turbo_median / stable_median,
                "load_ok": load_gate_passed,
                "rom_sha256": payloads["turbo"]["config"]["rom_sha256"],
                "turbo_package": payloads["turbo"]["package"],
                "stable_retro_package": payloads["stable-retro"]["package"],
            }
        )
    return collected


def format_float(value: float) -> str:
    return f"{value:,.1f}"


def reproduction_args(argv: Sequence[str]) -> list[str]:
    """Return experiment arguments without a one-use report destination."""
    result: list[str] = []
    skip_next = False
    for argument in argv:
        if skip_next:
            skip_next = False
            continue
        if argument == "--output-dir":
            skip_next = True
            continue
        if argument.startswith("--output-dir="):
            continue
        result.append(argument)
    return result


def render_report(aggregate: dict[str, Any]) -> str:
    settings = aggregate["settings"]
    validity = aggregate["validity"]
    lines = [
        "# Turbo vs Upstream Stable Retro SPS Benchmark",
        "",
        f"Generated: `{aggregate['created_at']}`",
        "",
        "## Verdict",
        "",
    ]
    if aggregate["claim_passed"]:
        lines.append(
            f"**PASS:** Turbo exceeded the predeclared {settings['minimum_speedup']:.2f}x "
            "median-speedup threshold at every rollout shape, with every paired 95% "
            "bootstrap interval above 1.0x."
        )
    else:
        lines.append(
            "**NOT ESTABLISHED:** one or more correctness, provenance, load, sample-count, "
            "or speedup gates did not pass. The measurements remain diagnostic evidence."
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Envs | Turbo median SPS | Stable Retro median SPS | Median paired speedup | 95% CI | Pairs | Claim |",
            "|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for result in aggregate["results"]:
        ci = result["pair_speedup_bootstrap_ci95"]
        lines.append(
            f"| {result['num_envs']} | "
            f"{format_float(result['turbo_invocation_median_sps']['median'])} | "
            f"{format_float(result['stable_retro_invocation_median_sps']['median'])} | "
            f"{result['pair_speedup']['median']:.2f}x | "
            f"{ci[0]:.2f}x–{ci[1]:.2f}x | {result['pair_count']} | "
            f"{'PASS' if result['claim_passed'] else 'FAIL'} |"
        )
    source = aggregate["source"]
    correctness = aggregate["correctness"]
    packages = aggregate["packages"]
    load_preflight = aggregate["load_preflight"]
    lines.extend(
        [
            "",
            "## Matched workload",
            "",
            "- One backend per invocation; paired order alternates Turbo/Stable and Stable/Turbo.",
            "- Canonical ROM and round-robin `Level1-1` through `Level1-4` lane states.",
            "- Deterministic sampled actions: `noop`, `right`, `right_b`, `right_a`, seed `0`.",
            "- Frame skip 4, no max-pooling, four-frame stack, integer grayscale, top-32 HUD mask, integer area resize to 84x84, CHW `uint8`.",
            "- Timed work includes vector stepping, preprocessing, IPC, infos, and selective terminal-lane resets.",
            "- Construction, initial reset, action generation, and warmup are outside measured time.",
            "",
            "## Validity",
            "",
            f"- Exact ROM-backed parity checks: **{'PASS' if correctness['passed'] else 'FAIL'}**",
            f"- Clean source tree: **{'PASS' if validity['source_clean'] else 'FAIL'}**",
            f"- Session-start load gate: **{'PASS' if validity['load_gate_enforced'] else 'FAIL'}**",
            f"- Initial/accepted 1-minute load: `{load_preflight['initial_1min']}` / `{load_preflight['accepted_1min']}` (limit `{load_preflight['max_start_load']}`)",
            f"- All shape claims passed: **{'PASS' if validity['all_shape_claims_passed'] else 'FAIL'}**",
            f"- Git commit: `{source.get('commit')}`",
            f"- Working-tree diff SHA-256: `{source['working_tree_diff_sha256']}`",
            "",
            "## Environment",
            "",
            f"- Platform: `{aggregate['system']['platform']}`",
            f"- Machine: `{aggregate['system']['machine']}`",
            f"- CPU: `{aggregate['system'].get('cpu_brand') or aggregate['system'].get('processor')}`",
            f"- Logical CPUs: `{aggregate['system']['logical_cpus']}`",
            f"- Python: `{aggregate['system']['python'].splitlines()[0]}`",
            f"- Turbo package: `{packages['turbo']['name']}=={packages['turbo']['version']}`",
            f"- Baseline package: `{packages['stable-retro']['name']}=={packages['stable-retro']['version']}`",
            "- Stable Retro vectorization: one upstream scalar `RetroEnv` worker per lane under Gymnasium `AsyncVectorEnv`; process and IPC overhead are included.",
            "",
            "## Reproduction",
            "",
            "```bash",
            aggregate["reproduction_command"],
            "```",
            "",
            "Raw invocation JSON and logs are under `raw/`; machine-readable statistics are in `aggregate.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    validate_args(args)
    output_dir = args.output_dir or default_output_dir()
    if output_dir is None:
        raise SystemExit(
            "output directory required; pass --output-dir or set AUTORESEARCH_ROOT_PATH in .env"
        )
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = git_source_identity()
    if source["dirty"] and not args.allow_dirty:
        raise SystemExit(
            "refusing a publishable benchmark from a dirty source tree; commit/stash changes "
            "or pass --allow-dirty for diagnostic evidence"
        )
    load_preflight = wait_for_load_headroom(args)
    load_gate_passed = bool(load_preflight["enabled"] and load_preflight["load_ok"])
    correctness = run_correctness_checks(args, output_dir)
    results = []
    warmup_records: dict[str, Any] = {}
    for shape_index, shape in enumerate(args.shapes):
        warmup_records[str(shape)] = collect_shape(
            args,
            output_dir=output_dir,
            shape=shape,
            shape_index=shape_index,
            measured=False,
            load_gate_passed=load_gate_passed,
        )
        pairs = collect_shape(
            args,
            output_dir=output_dir,
            shape=shape,
            shape_index=shape_index,
            measured=True,
            load_gate_passed=load_gate_passed,
        )
        results.append(
            aggregate_shape(
                shape=shape,
                pairs=pairs,
                minimum_speedup=args.minimum_speedup,
                minimum_pairs_for_claim=args.minimum_pairs_for_claim,
                bootstrap_samples=args.bootstrap_samples,
                load_gate_enforced=bool(load_preflight["enabled"]),
            )
        )

    validity = {
        "source_clean": not source["dirty"],
        "correctness_checks_passed": bool(correctness["passed"]),
        "load_gate_enforced": bool(load_preflight["enabled"]),
        "all_shape_claims_passed": all(result["claim_passed"] for result in results),
    }
    command = [
        str(args.python),
        str(Path(__file__).resolve()),
        *reproduction_args(raw_argv),
    ]
    first_pair = results[0]["pairs"][0]
    aggregate = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "claim_passed": all(validity.values()),
        "validity": validity,
        "settings": {
            "shapes": list(args.shapes),
            "pairs": args.pairs,
            "warmup_pairs": args.warmup_pairs,
            "steps": args.steps,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "minimum_speedup": args.minimum_speedup,
            "minimum_pairs_for_claim": args.minimum_pairs_for_claim,
            "bootstrap_samples": args.bootstrap_samples,
            "max_start_load": args.max_start_load,
            "load_poll_seconds": args.load_poll_seconds,
            "max_load_wait_seconds": args.max_load_wait_seconds,
            "force_busy": args.force_busy,
        },
        "packages": {
            "turbo": first_pair["turbo_package"],
            "stable-retro": first_pair["stable_retro_package"],
        },
        "source": source,
        "system": system_metadata(),
        "load_preflight": load_preflight,
        "correctness": correctness,
        "warmup_pairs": warmup_records,
        "results": results,
        "reproduction_command": shlex.join(command),
    }
    aggregate_path = output_dir / "aggregate.json"
    report_path = output_dir / "report.md"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_report(aggregate))
    print(f"aggregate={aggregate_path}")
    print(f"report={report_path}")
    print(f"claim_passed={aggregate['claim_passed']}")
    return 0 if aggregate["claim_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
