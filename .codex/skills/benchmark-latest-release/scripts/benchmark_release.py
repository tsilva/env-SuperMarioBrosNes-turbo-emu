#!/usr/bin/env python3
"""Fail-closed helpers for official TurboBench evidence publication."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT = "env-supermariobrosnes-turbo-emu"
BASELINE = "stable-retro"
BASELINE_VERSION = "1.0.1"
PROFILE = "supermario/canonical-v1"
TURBOBENCH_RESULT_VERSION = "1.0.1"
TURBOBENCH_VERIFIER_VERSION = "1.0.2"
HF_REPO_ID = "tsilva/env-supermariobrosnes-turbo-emu-benchmarks"
INDEX_SCHEMA = "env-supermariobrosnes-turbo-emu.benchmark-index/v1"
ROM_SHA256 = "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de"
CANONICAL_CPU = "AMD Ryzen 5 7600X"
QUARANTINE = timedelta(days=7)
QUARANTINE_EXEMPT_PROJECTS = frozenset({PROJECT})
PYPI_URL = "https://pypi.org/pypi/{project}/json"


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_time(value: Any) -> datetime:
    text = str(value or "").replace("Z", "+00:00")
    if not text:
        raise ValueError("PyPI artifact has no upload timestamp")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _quarantine_for_project(project: str) -> timedelta:
    normalized = re.sub(r"[-_.]+", "-", project).casefold()
    return timedelta(0) if normalized in QUARANTINE_EXEMPT_PROJECTS else QUARANTINE


def _request_pypi(project: str) -> dict[str, Any]:
    request = urllib.request.Request(
        PYPI_URL.format(project=project),
        headers={"User-Agent": "env-supermariobrosnes-turbo-emu-benchmark-publisher/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot query PyPI for {project}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("PyPI returned a non-object payload")
    return payload


def latest_release(project: str, *, require_eligible: bool) -> dict[str, Any]:
    payload = _request_pypi(project)
    version = str(payload.get("info", {}).get("version", "")).strip()
    if not version:
        raise ValueError(f"PyPI did not report a latest version for {project}")
    files = [
        item
        for item in payload.get("releases", {}).get(version, [])
        if isinstance(item, dict) and not item.get("yanked")
    ]
    if not files:
        raise ValueError(
            f"PyPI latest version {project}=={version} has no non-yanked files"
        )
    wheel_names = [str(item.get("filename", "")) for item in files]
    matching_wheels = [
        name
        for name in wheel_names
        if name.endswith(".whl")
        and "cp39-abi3" in name
        and "manylinux" in name
        and "x86_64" in name
    ]
    if not matching_wheels:
        raise ValueError(
            f"PyPI latest version {project}=={version} has no Linux x86-64 cp39-abi3 wheel"
        )
    uploaded = min(
        _parse_time(item.get("upload_time_iso_8601") or item.get("upload_time"))
        for item in files
    )
    quarantine = _quarantine_for_project(project)
    eligible_at = uploaded + quarantine
    now = _now_utc()
    result = {
        "project": project,
        "version": version,
        "uploaded_at": _iso(uploaded),
        "eligible_at": _iso(eligible_at),
        "eligible": now >= eligible_at,
        "quarantine_exempt": quarantine == timedelta(0),
        "linux_x86_64_cp39_abi3_wheels": matching_wheels,
        "pypi_url": f"https://pypi.org/project/{project}/{version}/",
    }
    if require_eligible and not result["eligible"]:
        raise ValueError(
            f"{project}=={version} is the latest release but remains inside TurboBench's "
            f"seven-day quarantine until {result['eligible_at']}"
        )
    return result


def _command(value: str) -> list[str]:
    command = shlex.split(value)
    if not command:
        raise ValueError("TurboBench command cannot be empty")
    return command


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _verify_with_turbobench(bundle: Path, command_text: str) -> dict[str, Any]:
    command = _command(command_text)
    version = _run([*command, "--version"])
    if (
        version.returncode
        or version.stdout.strip() != f"turbobench {TURBOBENCH_VERIFIER_VERSION}"
    ):
        detail = version.stderr.strip() or version.stdout.strip() or "command failed"
        raise ValueError(
            f"expected turbobench {TURBOBENCH_VERIFIER_VERSION}; got {detail!r} "
            f"from {command_text!r}"
        )
    verified = _run([*command, "verify", str(bundle)])
    try:
        payload = json.loads(verified.stdout)
    except json.JSONDecodeError as exc:
        detail = verified.stderr.strip() or verified.stdout.strip()
        raise ValueError(f"TurboBench verify did not return JSON: {detail}") from exc
    if (
        verified.returncode
        or not isinstance(payload, dict)
        or not payload.get("passed")
    ):
        errors = payload.get("errors", []) if isinstance(payload, dict) else []
        raise ValueError(f"TurboBench verification failed: {errors}")
    return payload


def _safe_component(value: str, label: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _bundle_entry(
    bundle: Path,
    version: str,
    command_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verified = _verify_with_turbobench(bundle, command_text)
    result = _json(bundle / "result.json")
    lock = _json(bundle / "resolved-lock.json")
    manifest = _json(bundle / "manifest.json")

    if result.get("profile", {}).get("id") != PROFILE:
        raise ValueError(f"expected profile {PROFILE}")
    if result.get("tool", {}).get("version") != TURBOBENCH_RESULT_VERSION:
        raise ValueError(
            f"bundle was not created by TurboBench {TURBOBENCH_RESULT_VERSION}"
        )
    if result.get("claim", {}).get("status") != "official":
        raise ValueError("bundle claim is not official")
    if result.get("validity", {}).get("passed") is not True:
        raise ValueError("bundle validity did not pass")
    if result.get("comparison", {}).get("outcome") == "inconclusive":
        raise ValueError(
            "inconclusive result cannot become the latest public package claim"
        )

    comparison = result.get("comparison", {})
    left = comparison.get("left", {})
    right = comparison.get("right", {})
    if (left.get("provider"), left.get("version")) != (PROJECT, version):
        raise ValueError(f"candidate is not {PROJECT}=={version}")
    if (right.get("provider"), right.get("version")) != (BASELINE, BASELINE_VERSION):
        raise ValueError(f"baseline is not {BASELINE}=={BASELINE_VERSION}")
    if left.get("source_identity") != f"pypi:{PROJECT}=={version}":
        raise ValueError("candidate is not the exact PyPI artifact")
    if right.get("source_identity") != f"pypi:{BASELINE}=={BASELINE_VERSION}":
        raise ValueError("baseline is not the exact PyPI artifact")

    providers = lock.get("providers", {})
    if providers.get("left", {}).get("source_kind") != "pypi":
        raise ValueError("candidate lock source_kind is not pypi")
    if providers.get("right", {}).get("source_kind") != "pypi":
        raise ValueError("baseline lock source_kind is not pypi")
    if lock.get("python_minor") != "3.14":
        raise ValueError("provider Python minor is not 3.14")

    shapes = comparison.get("shapes", {})
    if set(shapes) != {"1", "16", "32"}:
        raise ValueError(f"expected shapes 1, 16, and 32; got {sorted(shapes)}")

    host = result.get("system", {}).get("host", {})
    if host.get("os") != "Linux" or str(host.get("architecture", "")).lower() not in {
        "x86_64",
        "amd64",
    }:
        raise ValueError("bundle was not produced on the canonical x86-64 Linux host")
    if CANONICAL_CPU not in str(host.get("cpu", "")):
        raise ValueError(f"bundle CPU is not the canonical {CANONICAL_CPU}")

    media = bundle / "media"
    if media.is_dir() and any(path.is_file() for path in media.rglob("*")):
        raise ValueError("public evidence bundle must not contain gameplay media")
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("bundle contains a symbolic link")
    if any(
        path.is_file() and path.suffix.casefold() == ".nes"
        for path in bundle.rglob("*")
    ):
        raise ValueError("bundle contains a ROM file")
    if any(item.get("sha256") == ROM_SHA256 for item in manifest.get("artifacts", [])):
        raise ValueError("bundle contains an artifact matching the canonical ROM")

    bundle_id = str(verified.get("bundle_id", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", bundle_id):
        raise ValueError("bundle ID is missing or invalid")
    shape_results = []
    for shape in ("1", "16", "32"):
        statistics = shapes[shape]["statistics"]
        shape_results.append(
            {
                "envs": int(shape),
                "candidate_median_sps": statistics["median_left_sps"],
                "baseline_median_sps": statistics["median_right_sps"],
                "median_paired_ratio": statistics[
                    "median_paired_ratio_left_over_right"
                ],
                "bootstrap_95_ci": statistics["bootstrap"]["ci"],
            }
        )
    entry = {
        "package_version": version,
        "candidate": {
            "provider": PROJECT,
            "version": version,
            "source_identity": left["source_identity"],
            "artifact_sha256": left["artifact_sha256"],
        },
        "baseline": {
            "provider": BASELINE,
            "version": BASELINE_VERSION,
            "source_identity": right["source_identity"],
            "artifact_sha256": right["artifact_sha256"],
        },
        "profile": result["profile"],
        "tool": result["tool"],
        "host": host,
        "bundle_id": bundle_id,
        "artifact_count": verified["artifact_count"],
        "shape_results": shape_results,
        "published_at": _iso(datetime.now(timezone.utc)),
    }
    return entry, verified


def _load_index(path: Path | None, repo_id: str) -> dict[str, Any]:
    if path is None:
        return {"schema": INDEX_SCHEMA, "repo_id": repo_id, "entries": []}
    index = _json(path)
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError(f"unsupported benchmark index schema in {path}")
    if index.get("repo_id") != repo_id:
        raise ValueError(
            f"benchmark index belongs to {index.get('repo_id')!r}, not {repo_id!r}"
        )
    if not isinstance(index.get("entries"), list):
        raise ValueError("benchmark index entries must be a list")
    return index


def _render_card(index: dict[str, Any]) -> str:
    entries = index["entries"]
    latest_id = index["latest_bundle_id"]
    latest = next(item for item in entries if item["bundle_id"] == latest_id)
    path = latest["path"]
    lines = [
        "---",
        'pretty_name: "env-SuperMarioBrosNes-turbo-emu TurboBench Evidence"',
        "tags:",
        "- reinforcement-learning",
        "- benchmark",
        "- gymnasium",
        "- emulator",
        "---",
        "",
        "# env-SuperMarioBrosNes-turbo-emu TurboBench Evidence",
        "",
        "Official, first-party TurboBench evidence for published "
        "`env-supermariobrosnes-turbo-emu` releases compared with `stable-retro==1.0.1`.",
        "Every bundle is portable, hash-bound, and independently checkable with "
        "`turbobench verify`; this verifies integrity and consistency, not independent "
        "reproduction or author identity.",
        "",
        "No ROM, local asset path, secret, or gameplay media is included.",
        "",
        f"## Latest: {latest['package_version']}",
        "",
        f"Bundle ID: `{latest['bundle_id']}`",
        "",
        f"![Shape-local median SPS]({path}/chart.svg)",
        "",
        "| Envs | Turbo median SPS | Stable Retro median SPS | Paired speedup | 95% paired bootstrap CI |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for shape in latest["shape_results"]:
        lower, upper = shape["bootstrap_95_ci"]
        lines.append(
            f"| {shape['envs']} | {shape['candidate_median_sps']:,.1f} | "
            f"{shape['baseline_median_sps']:,.1f} | {shape['median_paired_ratio']:.4f}x | "
            f"{lower:.4f}x–{upper:.4f}x |"
        )
    lines.extend(
        [
            "",
            f"[Read the complete generated report]({path}/report.md)",
            "",
            "## Published bundles",
            "",
            "| Package | Baseline | Bundle ID | Evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for entry in reversed(entries):
        lines.append(
            f"| `{PROJECT}=={entry['package_version']}` | "
            f"`{BASELINE}=={BASELINE_VERSION}` | `{entry['bundle_id']}` | "
            f"[report]({entry['path']}/report.md) |"
        )
    lines.extend(
        [
            "",
            "## Verify a tagged publication",
            "",
            "```bash",
            f"hf download {index['repo_id']} --type dataset \\",
            f"  --revision v{latest['package_version']} --local-dir ./benchmark-evidence",
            "turbobench verify \\",
            f"  ./benchmark-evidence/{path}",
            "```",
            "",
            "The candidate package, baseline, Python runtime, workload profile, raw paired "
            "measurements, statistical calculations, and host record are retained inside each bundle.",
            "",
        ]
    )
    return "\n".join(lines)


def stage(args: argparse.Namespace) -> dict[str, Any]:
    bundle = args.bundle.expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"bundle directory does not exist: {bundle}")
    version = _safe_component(args.version, "version")
    release = latest_release(PROJECT, require_eligible=True)
    if release["version"] != version:
        raise ValueError(
            f"refusing to stage {PROJECT}=={version}; newest stable PyPI release is "
            f"{release['version']}"
        )
    repo_id = args.repo_id
    entry, verified = _bundle_entry(bundle, version, args.turbobench_command)
    index = _load_index(args.existing_index, repo_id)
    existing = [
        item for item in index["entries"] if item.get("package_version") == version
    ]
    if existing and any(
        item.get("bundle_id") != entry["bundle_id"] for item in existing
    ):
        raise ValueError(
            f"package version {version} is already published with a different bundle ID"
        )
    if existing:
        entry = existing[0]
    else:
        index["entries"].append(entry)

    relative = (
        Path("bundles")
        / f"v{version}"
        / f"vs-{BASELINE}-{BASELINE_VERSION}"
        / entry["bundle_id"]
    )
    entry["path"] = relative.as_posix()
    index["latest_version"] = version
    index["latest_bundle_id"] = entry["bundle_id"]

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite staging path: {output}")
    output.mkdir(parents=True)
    destination = output / relative
    shutil.copytree(bundle, destination)
    copied = _verify_with_turbobench(destination, args.turbobench_command)
    if copied.get("bundle_id") != entry["bundle_id"]:
        raise ValueError("copied bundle ID changed during staging")
    _write_json(output / "benchmark-index.json", index)
    (output / "README.md").write_text(_render_card(index), encoding="utf-8")
    return {
        "output": str(output),
        "repo_id": repo_id,
        "version": version,
        "bundle_id": entry["bundle_id"],
        "bundle_path": relative.as_posix(),
        "artifact_count": verified["artifact_count"],
    }


def verify_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    index = _load_index(root / "benchmark-index.json", args.repo_id)
    matches = [
        item for item in index["entries"] if item.get("package_version") == args.version
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one index entry for {args.version}; found {len(matches)}"
        )
    entry = matches[0]
    bundle = root / str(entry["path"])
    checked, verified = _bundle_entry(bundle, args.version, args.turbobench_command)
    if checked["bundle_id"] != entry.get("bundle_id"):
        raise ValueError("downloaded bundle ID does not match benchmark-index.json")
    return {
        "root": str(root),
        "version": args.version,
        "bundle_id": checked["bundle_id"],
        "artifact_count": verified["artifact_count"],
        "passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    latest = commands.add_parser(
        "latest", help="resolve the newest stable PyPI release"
    )
    latest.add_argument("--project", default=PROJECT)
    latest.add_argument("--require-eligible", action="store_true")

    stage_parser = commands.add_parser(
        "stage", help="validate and stage one HF publication"
    )
    stage_parser.add_argument("--bundle", type=Path, required=True)
    stage_parser.add_argument("--version", required=True)
    stage_parser.add_argument("--output", type=Path, required=True)
    stage_parser.add_argument("--existing-index", type=Path)
    stage_parser.add_argument("--repo-id", default=HF_REPO_ID)
    stage_parser.add_argument(
        "--turbobench-command",
        default=os.environ.get("TURBOBENCH_COMMAND", "turbobench"),
    )

    verify = commands.add_parser(
        "verify-snapshot", help="verify a downloaded HF snapshot"
    )
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--version", required=True)
    verify.add_argument("--repo-id", default=HF_REPO_ID)
    verify.add_argument(
        "--turbobench-command",
        default=os.environ.get("TURBOBENCH_COMMAND", "turbobench"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "latest":
            result = latest_release(
                args.project, require_eligible=args.require_eligible
            )
        elif args.command == "stage":
            result = stage(args)
        else:
            result = verify_snapshot(args)
    except ValueError as exc:
        print(f"benchmark_release: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
