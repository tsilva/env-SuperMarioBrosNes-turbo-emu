#!/usr/bin/env python3
"""Deterministic helpers for SuperMarioBros-Nes-turbo release wheel builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[4]
VERSION_PATH = REPO_ROOT / "VERSION.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CARGO_TOML = REPO_ROOT / "Cargo.toml"
CARGO_LOCK = REPO_ROOT / "Cargo.lock"
UV_LOCK = REPO_ROOT / "uv.lock"
PYTHON = Path(sys.executable)
PACKAGE_NAME = "supermariobrosnes-turbo"
IMPORT_NAME = "supermariobrosnes_turbo"
EXTENSION_NAME = "_supermariobrosnes_turbo"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+|\.dev\d+)?$")
RELEASE_PLATFORMS = (
    "macos-arm64",
    "macos-x86_64",
    "linux-x86_64",
    "linux-aarch64",
    "windows-x86_64",
)

IGNORED_DIR_NAMES_ANYWHERE = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "target",
}
IGNORED_ROOT_DIR_NAMES = {
    "artifacts",
    "build",
    "dist",
    "target-release",
    "target-release-linux",
}
IGNORED_FILE_SUFFIXES = {".o", ".a", ".so", ".dylib", ".d", ".pyc", ".pyo"}
ROM_SUFFIXES = {".nes", ".sfc", ".smc", ".gb", ".gbc", ".gen", ".sms", ".bin"}
EXPECTED_SMB_ROM_SHA256 = "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de"


def release_temp_dir() -> Path:
    configured = os.environ.get("RELEASE_BUILD_TMPDIR")
    if configured:
        root = Path(configured)
    else:
        root = Path("/private/tmp")
        if not root.exists():
            root = Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root


def read_pyproject() -> dict[str, object]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def pyproject_name() -> str:
    return str(read_pyproject()["project"]["name"])  # type: ignore[index]


def read_version() -> str:
    return VERSION_PATH.read_text(encoding="utf-8").strip()


def pyproject_version() -> str:
    return str(read_pyproject()["project"]["version"])  # type: ignore[index]


def section_version(path: Path, section: str, *, package_name: str | None = None) -> str:
    current_section: str | None = None
    current_name: str | None = None
    in_matching_package = package_name is None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[[") and stripped.endswith("]]"):
            current_section = stripped[2:-2].strip()
            current_name = None
            in_matching_package = package_name is None
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            current_name = None
            in_matching_package = package_name is None
            continue
        if current_section != section:
            continue
        if stripped.startswith("name = "):
            current_name = stripped.split("=", 1)[1].strip().strip('"')
            if package_name is not None:
                in_matching_package = current_name == package_name
            continue
        if in_matching_package and stripped.startswith("version = "):
            return stripped.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"could not find version in [{section}] of {path}")


def cargo_version() -> str:
    return section_version(CARGO_TOML, "package")


def cargo_lock_version() -> str:
    return section_version(CARGO_LOCK, "package", package_name=PACKAGE_NAME)


def validate_version(version: str) -> None:
    if VERSION_RE.match(version) is None:
        raise SystemExit(f"unsupported version format: {version!r}")


def split_release(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if match is None:
        raise SystemExit(
            f"cannot compute major/minor/patch bump from non-final version {version!r}; pass --to"
        )
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def next_version(version: str, part: str) -> str:
    major, minor, patch = split_release(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(part)


def version_sort_key(version: str) -> tuple[int, int, int, int]:
    validate_version(version)
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?$", version)
    if match is None:
        raise ValueError(f"cannot sort non-final release version: {version!r}")
    major, minor, patch, post = match.groups()
    return int(major), int(minor), int(patch), int(post or 0)


def replace_section_version(path: Path, section: str, version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    current_section: str | None = None
    changed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and not stripped.startswith("[["):
            current_section = stripped[1:-1].strip()
            continue
        if current_section == section and stripped.startswith("version = "):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'version = "{version}"{newline}'
            changed = True
            break
    if not changed:
        raise RuntimeError(f"could not replace version in [{section}] of {path}")
    path.write_text("".join(lines), encoding="utf-8")


def replace_package_version(path: Path, package_name: str, version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_package = False
    matching_package = False
    changed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[[") and stripped.endswith("]]"):
            in_package = stripped[2:-2].strip() == "package"
            matching_package = False
            continue
        if not in_package:
            continue
        if stripped.startswith("name = "):
            matching_package = stripped.split("=", 1)[1].strip().strip('"') == package_name
            continue
        if matching_package and stripped.startswith("version = "):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'version = "{version}"{newline}'
            changed = True
            break
    if not changed:
        raise RuntimeError(f"could not replace {package_name!r} version in {path}")
    path.write_text("".join(lines), encoding="utf-8")


def write_version(version: str) -> None:
    VERSION_PATH.write_text(f"{version}\n", encoding="utf-8")


def check_version(args: argparse.Namespace) -> None:
    versions = {
        "version_txt": read_version(),
        "pyproject": pyproject_version(),
        "cargo_toml": cargo_version(),
        "cargo_lock": cargo_lock_version(),
    }
    expected = args.version
    failures = []
    if pyproject_name() != PACKAGE_NAME:
        failures.append(f"pyproject package name is {pyproject_name()!r}, expected {PACKAGE_NAME!r}")
    if len(set(versions.values())) != 1:
        failures.append(f"version mismatch: {versions}")
    if expected is not None and set(versions.values()) != {expected}:
        failures.append(f"expected version {expected!r}, saw {versions}")
    result = {"package": pyproject_name(), "versions": versions}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


def run_capture(args_list: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args_list,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout.strip()


def check_tools(args: argparse.Namespace) -> None:
    checks = {
        "cargo": run_capture(["cargo", "--version"]),
        "docker": run_capture(["docker", "--version"]),
        "maturin": run_capture([str(PYTHON), "-m", "maturin", "--version"]),
        "cibuildwheel": run_capture(
            [
                str(PYTHON),
                "-c",
                "from importlib.metadata import version; print('cibuildwheel ' + version('cibuildwheel'))",
            ]
        ),
        "twine": run_capture([str(PYTHON), "-m", "twine", "--version"]),
    }
    result = {
        name: {"ok": code == 0, "output": output}
        for name, (code, output) in checks.items()
    }
    print(json.dumps(result, indent=2))
    missing = [name for name, item in result.items() if not item["ok"]]
    if missing:
        raise SystemExit(f"missing release tooling: {', '.join(missing)}")


def bump_version(args: argparse.Namespace) -> None:
    target = args.to or next_version(read_version(), args.part)
    validate_version(target)
    if args.write:
        write_version(target)
        replace_section_version(PYPROJECT, "project", target)
        replace_section_version(CARGO_TOML, "package", target)
        replace_package_version(CARGO_LOCK, PACKAGE_NAME, target)
        replace_package_version(UV_LOCK, PACKAGE_NAME, target)
    print(target)


def sync_version(args: argparse.Namespace) -> None:
    version = read_version()
    validate_version(version)
    replace_section_version(PYPROJECT, "project", version)
    replace_section_version(CARGO_TOML, "package", version)
    replace_package_version(CARGO_LOCK, PACKAGE_NAME, version)
    replace_package_version(UV_LOCK, PACKAGE_NAME, version)
    print(version)


def fetch_pypi_project() -> dict[str, object]:
    url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def release_has_non_yanked_file(files: object) -> bool:
    if not isinstance(files, list):
        return False
    return any(isinstance(file, dict) and not file.get("yanked", False) for file in files)


def latest_non_yanked_pypi_version(releases: object) -> str | None:
    if not isinstance(releases, dict):
        return None
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    for version, files in releases.items():
        if not isinstance(version, str) or not release_has_non_yanked_file(files):
            continue
        try:
            candidates.append((version_sort_key(version), version))
        except ValueError:
            continue
    if not candidates:
        return None
    return max(candidates)[1]


def check_pypi(args: argparse.Namespace) -> None:
    validate_version(args.version)
    try:
        data = fetch_pypi_project()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(json.dumps({"package": PACKAGE_NAME, "exists": False, "version_exists": False}, indent=2))
            return
        raise
    releases = data.get("releases", {})
    exists = args.version in releases and bool(releases[args.version])
    print(json.dumps({"package": PACKAGE_NAME, "version": args.version, "version_exists": exists}, indent=2))
    if exists:
        raise SystemExit(f"{PACKAGE_NAME} {args.version} already exists on PyPI")


def latest_pypi(args: argparse.Namespace) -> None:
    try:
        data = fetch_pypi_project()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(json.dumps({"package": PACKAGE_NAME, "exists": False, "latest_non_yanked": None}, indent=2))
            return
        raise
    latest = latest_non_yanked_pypi_version(data.get("releases"))
    info = data.get("info", {})
    info_version = info.get("version") if isinstance(info, dict) else None
    result = {
        "package": PACKAGE_NAME,
        "exists": True,
        "latest_non_yanked": latest,
        "pypi_info_version": info_version,
    }
    print(json.dumps(result, indent=2))
    if args.fail_if_mismatch and latest != info_version:
        raise SystemExit(f"PyPI info.version {info_version!r} does not match latest non-yanked {latest!r}")


def should_ignore(rel: Path) -> bool:
    if any(part in IGNORED_DIR_NAMES_ANYWHERE for part in rel.parts):
        return True
    if len(rel.parts) == 1 and (
        rel.name in IGNORED_ROOT_DIR_NAMES or rel.name.startswith("target-release-")
    ):
        return True
    if rel.name.startswith("wheelhouse"):
        return True
    if rel.suffix in IGNORED_FILE_SUFFIXES:
        return True
    return False


def copy_clean_tree(destination: Path, *, force: bool = False) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} already exists; pass --force to replace it")
        shutil.rmtree(destination)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        base = Path(directory)
        for name in names:
            rel = (base / name).relative_to(REPO_ROOT)
            if should_ignore(rel):
                ignored.add(name)
        return ignored

    shutil.copytree(REPO_ROOT, destination, symlinks=True, ignore=ignore)


def find_contamination(root: Path) -> dict[str, list[str]]:
    compiled: list[str] = []
    roms: list[str] = []
    pycache: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts:
            pycache.append(rel.as_posix())
        if not path.is_file():
            continue
        if rel.suffix in IGNORED_FILE_SUFFIXES:
            compiled.append(rel.as_posix())
        if rel.suffix.lower() in ROM_SUFFIXES:
            roms.append(rel.as_posix())
    return {"compiled_artifacts": compiled, "rom_payloads": roms, "pycache": pycache}


def fail_on_contamination(root: Path) -> None:
    contamination = find_contamination(root)
    failures = {key: value for key, value in contamination.items() if value}
    if failures:
        print(json.dumps(failures, indent=2), file=sys.stderr)
        raise SystemExit(f"{root} is not a clean release source copy")


def wheelhouse(version: str, platform_name: str) -> Path:
    return REPO_ROOT / f"wheelhouse-v{version}-{platform_name}"


def cargo_target_dir(platform_name: str, root: Path = REPO_ROOT) -> Path:
    legacy = {
        "macos": "target-release",
        "linux": "target-release-linux",
    }
    if platform_name in legacy:
        return root / legacy[platform_name]
    if platform_name not in RELEASE_PLATFORMS:
        raise ValueError(f"unknown platform: {platform_name}")
    return root / f"target-release-{platform_name}"


def linux_build_env(platform_name: str, root: Path = REPO_ROOT) -> dict[str, str]:
    if platform_name not in {"linux-x86_64", "linux-aarch64"}:
        raise ValueError(f"not a Linux release platform: {platform_name}")
    arch = platform_name.removeprefix("linux-")
    target_dir = cargo_target_dir(platform_name, root).resolve()
    return {
        "CIBW_BUILD": f"cp39-manylinux_{arch}",
        "CIBW_ARCHS_LINUX": arch,
        "CIBW_SKIP": "*-musllinux_*",
        "CIBW_BEFORE_ALL_LINUX": "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal",
        "CIBW_CONTAINER_ENGINE": f"docker; create_args: --volume={target_dir}:/cargo-target",
        "CIBW_ENVIRONMENT_LINUX": (
            'PATH="$HOME/.cargo/bin:$PATH" '
            "CARGO_NET_GIT_FETCH_WITH_CLI=true "
            "CARGO_TARGET_DIR=/cargo-target"
        ),
    }


def prepare_sources(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    validate_version(version)
    root = args.root or Path(
        tempfile.mkdtemp(
            prefix=f"supermariobrosnes-turbo-v{version}-builds.",
            dir=release_temp_dir(),
        )
    )
    root = root.resolve()
    macos_src = root / "macos-src"
    linux_src = root / "linux-src-clean"
    copy_clean_tree(macos_src, force=args.force)
    copy_clean_tree(linux_src, force=args.force)
    fail_on_contamination(macos_src)
    fail_on_contamination(linux_src)
    print(
        json.dumps(
            {
                "version": version,
                "root": str(root),
                "macos_src": str(macos_src),
                "linux_src_clean": str(linux_src),
                "macos_wheelhouse": str(wheelhouse(version, "macos")),
                "linux_wheelhouse": str(wheelhouse(version, "linux")),
            },
            indent=2,
        )
    )


def shell_quote(value: str | Path) -> str:
    import shlex

    return shlex.quote(str(value))


def build_commands(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    validate_version(version)
    macos_src = Path(args.macos_src) if args.macos_src else Path("<macos-src>")
    linux_src = Path(args.linux_src) if args.linux_src else Path("<linux-src-clean>")
    macos_out = wheelhouse(version, "macos")
    linux_out = wheelhouse(version, "linux")
    print("# macOS arm64")
    print(f"cd {shell_quote(macos_src)}")
    print(
        "MACOSX_DEPLOYMENT_TARGET=14.0 "
        "ARCHFLAGS='-arch arm64' "
        f"CARGO_TARGET_DIR={shell_quote(cargo_target_dir('macos', macos_src))} "
        f"{shell_quote(PYTHON)} -m maturin build --release --out {shell_quote(macos_out)}"
    )
    print()
    print("# Linux x86_64 manylinux")
    print(f"cd {shell_quote(linux_src)}")
    linux_env = linux_build_env("linux-x86_64", linux_src)
    print(
        " ".join(f"{key}={shell_quote(value)}" for key, value in linux_env.items())
        + f" {shell_quote(PYTHON)} -m cibuildwheel --platform linux --output-dir {shell_quote(linux_out)}"
    )


def build_platform(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    validate_version(version)
    output_dir = wheelhouse(version, args.platform)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    target_dir = cargo_target_dir(args.platform)
    target_dir.mkdir(parents=True, exist_ok=True)
    if args.platform.startswith("macos-"):
        arch = args.platform.removeprefix("macos-")
        env.update(
            {
                "MACOSX_DEPLOYMENT_TARGET": "14.0" if arch == "arm64" else "13.0",
                "ARCHFLAGS": f"-arch {arch}",
                "CARGO_TARGET_DIR": str(target_dir),
            }
        )
        run([str(PYTHON), "-m", "maturin", "build", "--release", "--out", str(output_dir)], env=env)
        return
    if args.platform.startswith("windows-"):
        env["CARGO_TARGET_DIR"] = str(target_dir)
        run([str(PYTHON), "-m", "maturin", "build", "--release", "--out", str(output_dir)], env=env)
        return
    env.update(linux_build_env(args.platform))
    run(
        [str(PYTHON), "-m", "cibuildwheel", "--platform", "linux", "--output-dir", str(output_dir)],
        env=env,
    )


def wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def audit_wheel(wheel: Path, version: str) -> dict[str, object]:
    names = wheel_names(wheel)
    extension_entries = [
        name
        for name in names
        if name.startswith(f"{IMPORT_NAME}/{EXTENSION_NAME}") and name.endswith((".so", ".pyd"))
    ]
    metadata_entries = [name for name in names if name.endswith(".dist-info/METADATA")]
    rom_payloads = [name for name in names if Path(name).suffix.lower() in ROM_SUFFIXES]
    pycache_entries = [name for name in names if "__pycache__" in Path(name).parts or name.endswith(".pyc")]
    license_entries = [name for name in names if name.endswith("/LICENSE")]
    notice_entries = [name for name in names if name.endswith("/NOTICE.md")]
    entry_point_entries = [
        name for name in names if name.endswith(".dist-info/entry_points.txt")
    ]
    entry_points = ""
    if len(entry_point_entries) == 1:
        with zipfile.ZipFile(wheel) as zf:
            entry_points = zf.read(entry_point_entries[0]).decode("utf-8")
    checks = {
        "version_in_filename": version in wheel.name,
        "abi3_wheel": "abi3" in wheel.name,
        "has_package_init": f"{IMPORT_NAME}/__init__.py" in names,
        "has_env_source": f"{IMPORT_NAME}/env.py" in names,
        "has_unified_cli_source": f"{IMPORT_NAME}/cli.py" in names,
        "has_training_source": f"{IMPORT_NAME}/training.py" in names,
        "has_playback_source": f"{IMPORT_NAME}/state_playback.py" in names,
        "has_unified_entry_point": (
            "smb-turbo=supermariobrosnes_turbo.cli:main" in entry_points.replace(" ", "")
        ),
        "no_legacy_entry_points": not any(
            command in entry_points
            for command in (
                "smb-turbo-import",
                "smb-turbo-play",
                "smb-turbo-train",
            )
        ),
        "has_extension": bool(extension_entries),
        "has_metadata": bool(metadata_entries),
        "has_license": bool(license_entries),
        "has_notice": bool(notice_entries),
        "has_py_typed": f"{IMPORT_NAME}/py.typed" in names,
        "no_rom_payloads": not rom_payloads,
        "no_pycache": not pycache_entries,
    }
    return {
        "wheel": str(wheel),
        "extension_entries": extension_entries,
        "metadata_entries": metadata_entries,
        "license_entries": license_entries,
        "notice_entries": notice_entries,
        "entry_point_entries": entry_point_entries,
        "rom_payloads": rom_payloads,
        "pycache_entries": pycache_entries,
        "checks": checks,
    }


def assert_audit_passed(results: list[dict[str, object]]) -> None:
    failures: dict[str, list[str]] = {}
    for result in results:
        checks = result["checks"]
        assert isinstance(checks, dict)
        failed = [key for key, value in checks.items() if not value]
        if failed:
            failures[str(result["wheel"])] = failed
    if failures:
        print(json.dumps(results, indent=2), file=sys.stderr)
        raise SystemExit(f"wheel audit failed: {failures}")


def wheel_release_platform(wheel: Path) -> str | None:
    name = wheel.name
    markers = {
        "macos-arm64": ("macosx", "arm64"),
        "macos-x86_64": ("macosx", "x86_64"),
        "linux-x86_64": ("manylinux", "x86_64"),
        "linux-aarch64": ("manylinux", "aarch64"),
        "windows-x86_64": ("win_amd64",),
    }
    for platform_name, required in markers.items():
        if all(marker in name for marker in required):
            return platform_name
    return None


def assert_platform_coverage(wheels: list[Path]) -> None:
    seen = {platform for wheel in wheels if (platform := wheel_release_platform(wheel))}
    missing = sorted(set(RELEASE_PLATFORMS) - seen)
    if missing:
        raise SystemExit(f"release wheel set is missing platforms: {', '.join(missing)}")


def find_wheels(version: str) -> list[Path]:
    candidates: list[Path] = []
    for platform_name in (*RELEASE_PLATFORMS, "macos", "linux"):
        candidates.extend(wheelhouse(version, platform_name).glob(f"*{version}*.whl"))
    return sorted(set(candidates))


def audit_wheels(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    wheels = [Path(wheel) for wheel in args.wheels] if args.wheels else find_wheels(version)
    if args.require_all_platforms:
        assert_platform_coverage(wheels)
    results = [audit_wheel(wheel, version) for wheel in wheels]
    assert_audit_passed(results)
    print(json.dumps(results, indent=2))


def audit_sdist(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    sdist = args.sdist.resolve()
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
    rom_payloads = [name for name in names if Path(name).suffix.lower() in ROM_SUFFIXES]
    checks = {
        "version_in_filename": version in sdist.name,
        "has_pyproject": any(name.endswith("/pyproject.toml") for name in names),
        "has_cargo_toml": any(name.endswith("/Cargo.toml") for name in names),
        "has_license": any(name.endswith("/LICENSE") for name in names),
        "has_notice": any(name.endswith("/NOTICE.md") for name in names),
        "has_py_typed": any(name.endswith(f"/{IMPORT_NAME}/py.typed") for name in names),
        "has_packaged_states": any(name.endswith(".state") for name in names),
        "no_rom_payloads": not rom_payloads,
    }
    result = {"sdist": str(sdist), "rom_payloads": rom_payloads, "checks": checks}
    print(json.dumps(result, indent=2))
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise SystemExit(f"sdist audit failed: {failed}")


def run(args_list: list[str], **kwargs: object) -> None:
    print("+", " ".join(shell_quote(arg) for arg in args_list))
    subprocess.run(args_list, check=True, **kwargs)


def smoke_distribution(
    distribution: Path,
    python: Path,
    *,
    required_rom: Path | None = None,
) -> None:
    distribution = distribution.resolve()
    required_rom_expression = (
        repr(str(required_rom.resolve())) if required_rom is not None else "default_rom_path()"
    )
    feature_smoke_required = required_rom is not None
    with tempfile.TemporaryDirectory(prefix="supermariobrosnes-distribution-smoke.", dir=release_temp_dir()) as tmp:
        target = Path(tmp)
        environment = target / "venv"
        run([str(python), "-m", "venv", str(environment)])
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        environment_python = scripts / ("python.exe" if os.name == "nt" else "python")
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment_python),
                str(distribution),
            ]
        )
        code = f"""
import numpy as np
import {IMPORT_NAME}
from {IMPORT_NAME} import {EXTENSION_NAME}
from {IMPORT_NAME} import (
    AVAILABLE_INFO_KEYS,
    EXTRA_INFO_KEYS,
    INFO_KEYS,
    Actions,
    NES_BUTTONS,
    default_rom_path,
)
print({IMPORT_NAME}.__file__)
print({EXTENSION_NAME}.__file__)
assert {IMPORT_NAME}.__file__.startswith({str(environment)!r})
assert hasattr({IMPORT_NAME}, "SuperMarioBrosNesTurboVecEnv")
assert {IMPORT_NAME}.SuperMarioBrosNesTurboVecEnv.supports_live_snapshots is True
assert len(INFO_KEYS) == 10
assert len(EXTRA_INFO_KEYS) > 0
assert AVAILABLE_INFO_KEYS == INFO_KEYS + EXTRA_INFO_KEYS
assert hasattr({EXTENSION_NAME}, "extra_info_descriptors")
assert len({EXTENSION_NAME}.extra_info_descriptors()) == len(EXTRA_INFO_KEYS)
for method in ("extra_info_shape", "extra_info_into", "ram_shape", "ram_into"):
    assert hasattr({EXTENSION_NAME}._RetroVecEnv, method)

rom_path = {required_rom_expression}
if rom_path is None:
    assert {feature_smoke_required!r} is False
    print("ROM-backed feature smoke skipped: ABI surface passed, canonical SMB ROM is unavailable")
else:
    env = {IMPORT_NAME}.SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        rom_path=rom_path,
        num_envs=2,
        num_threads=1,
        use_restricted_actions=Actions.ALL,
        frame_skip=1,
        frame_stack=1,
        obs_grayscale=True,
        obs_resize=(84, 84),
        obs_layout="chw",
        info_filter={{
            "mode": "all",
            "keys": ["x_pos", "y_pos", "area_type", "enemy_active", "enemy_x_pos"],
        }},
    )
    try:
        _, initial_infos = env.reset()
        assert initial_infos["x_pos"].shape == (2,)
        assert initial_infos["y_pos"].dtype == np.int32
        assert initial_infos["area_type"].dtype == np.int8
        assert initial_infos["enemy_active"].shape == (2, 6)
        assert initial_infos["enemy_active"].dtype == np.bool_
        assert initial_infos["enemy_x_pos"].shape == (2, 6)
        frozen_enemy_x = initial_infos["enemy_x_pos"].copy()
        ram = env.ram()
        assert ram.shape == (2, 2048)
        assert ram.dtype == np.uint8
        assert ram.flags.writeable is False
        warmup = np.zeros((2, len(NES_BUTTONS)), dtype=np.uint8)
        _, _, _, _, step_infos = env.step(warmup)
        assert set(key for key in step_infos if not key.startswith("_")) == {{
            "x_pos", "y_pos", "area_type", "enemy_active", "enemy_x_pos"
        }}
        np.testing.assert_array_equal(initial_infos["enemy_x_pos"], frozen_enemy_x)
        handles = env.capture_snapshots(
            np.asarray([True, False], dtype=np.bool_)
        )
        assert handles[0] is not None
        assert handles[0].nbytes > 0
        assert handles[1] is None

        reset_options = {{
            "reset_mask": np.asarray([True, True], dtype=np.bool_),
            "state_indices": np.asarray([-1, -1], dtype=np.int32),
            "snapshots": [handles[0], handles[0]],
        }}
        restored, restored_infos = env.reset(options=reset_options)
        np.testing.assert_array_equal(restored[0], restored[1])
        assert restored_infos["start_source"].tolist() == [
            "snapshot",
            "snapshot",
        ]

        replay_actions = np.zeros_like(warmup)
        replay_actions[:, NES_BUTTONS.index("RIGHT")] = 1
        first = tuple(
            np.asarray(value).copy() for value in env.step(replay_actions)[:4]
        )
        env.reset(options=reset_options)
        second = env.step(replay_actions)
        for expected, actual in zip(first, second[:4]):
            np.testing.assert_array_equal(expected, actual)
    finally:
        env.close()
"""
        run([str(environment_python), "-c", code], cwd=target)
        command = scripts / ("smb-turbo.exe" if os.name == "nt" else "smb-turbo")
        for arguments in (
            ["--help"],
            ["import", "--help"],
            ["train", "--help"],
            ["play", "--help"],
        ):
            run([str(command), *arguments], cwd=target)


def resolve_smoke_wheel(path: Path) -> Path:
    wheel = path.resolve()
    if wheel.is_dir():
        candidates = sorted(wheel.glob("*.whl"))
        if len(candidates) != 1:
            raise SystemExit(f"expected one wheel in {wheel}, found {candidates}")
        wheel = candidates[0]
    return wheel


def smoke_wheel(args: argparse.Namespace) -> None:
    wheel = resolve_smoke_wheel(args.wheel)
    smoke_distribution(wheel, args.python)


def smoke_feature_wheel(args: argparse.Namespace) -> None:
    wheel = resolve_smoke_wheel(args.wheel)
    rom = args.rom.resolve()
    if not rom.is_file():
        raise SystemExit(f"feature-smoke ROM does not exist: {rom}")
    rom_digest = sha256(rom)
    if rom_digest != EXPECTED_SMB_ROM_SHA256:
        raise SystemExit(
            f"feature-smoke ROM SHA-256 must be {EXPECTED_SMB_ROM_SHA256}, got {rom_digest}"
        )
    version_text = subprocess.check_output(
        [
            str(args.python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
    ).strip()
    if version_text != "3.9":
        raise SystemExit(
            f"feature-smoke Python must be CPython 3.9, got {version_text} from {args.python}"
        )
    smoke_distribution(wheel, args.python, required_rom=rom)
    evidence = {
        "status": "passed",
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel),
        "rom_sha256": rom_digest,
        "python": str(args.python.resolve()),
        "python_version": version_text,
        "feature": "processed_research_infos",
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


def smoke_sdist(args: argparse.Namespace) -> None:
    smoke_distribution(args.sdist, args.python)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_check(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    wheels = [Path(wheel) for wheel in args.wheels] if args.wheels else find_wheels(version)
    assert_platform_coverage(wheels)
    results = [audit_wheel(wheel, version) for wheel in wheels]
    assert_audit_passed(results)
    run([str(PYTHON), "-m", "twine", "check", *[str(wheel) for wheel in wheels]])
    hashes = {str(wheel): sha256(wheel) for wheel in wheels}
    print(json.dumps({"audits": results, "sha256": hashes}, indent=2))
    print()
    print(f"{shell_quote(PYTHON)} -m twine upload --config-file .pypirc \\")
    for index, wheel in enumerate(wheels):
        suffix = " \\" if index < len(wheels) - 1 else ""
        print(f"  {shell_quote(wheel)}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-version", help="Check package identity and version consistency")
    check.add_argument("--version")
    check.set_defaults(func=check_version)

    tools = subparsers.add_parser("check-tools", help="Check release build tooling")
    tools.set_defaults(func=check_tools)

    bump = subparsers.add_parser("bump-version", help="Print or write a target version")
    bump.add_argument("--to")
    bump.add_argument("--part", choices=("major", "minor", "patch"), default="patch")
    bump.add_argument("--write", action="store_true")
    bump.set_defaults(func=bump_version)

    sync = subparsers.add_parser("sync-version", help="Write pyproject.toml and Cargo.toml from VERSION.txt")
    sync.set_defaults(func=sync_version)

    pypi = subparsers.add_parser("check-pypi", help="Fail if the target version exists on PyPI")
    pypi.add_argument("--version", required=True)
    pypi.set_defaults(func=check_pypi)

    latest = subparsers.add_parser("latest-pypi", help="Print the latest non-yanked PyPI version")
    latest.add_argument(
        "--fail-if-mismatch",
        action="store_true",
        help="Fail if PyPI info.version differs from the computed latest non-yanked release",
    )
    latest.set_defaults(func=latest_pypi)

    prepare = subparsers.add_parser("prepare-sources", help="Create clean macOS/Linux source copies")
    prepare.add_argument("--version")
    prepare.add_argument("--root", type=Path)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=prepare_sources)

    commands = subparsers.add_parser("build-commands", help="Print platform build commands")
    commands.add_argument("--version")
    commands.add_argument("--macos-src")
    commands.add_argument("--linux-src")
    commands.set_defaults(func=build_commands)

    platform = subparsers.add_parser("build-platform", help="Build one release wheel platform")
    platform.add_argument("--version")
    platform.add_argument("--platform", choices=RELEASE_PLATFORMS, required=True)
    platform.set_defaults(func=build_platform)

    audit = subparsers.add_parser("audit-wheels", help="Audit wheel contents")
    audit.add_argument("--version")
    audit.add_argument("--require-all-platforms", action="store_true")
    audit.add_argument("wheels", nargs="*")
    audit.set_defaults(func=audit_wheels)

    sdist = subparsers.add_parser("audit-sdist", help="Audit source-distribution contents")
    sdist.add_argument("sdist", type=Path)
    sdist.add_argument("--version")
    sdist.set_defaults(func=audit_sdist)

    smoke = subparsers.add_parser("smoke-wheel", help="Install and import-test a built wheel")
    smoke.add_argument("wheel", type=Path)
    smoke.add_argument("--python", type=Path, default=PYTHON)
    smoke.set_defaults(func=smoke_wheel)

    feature_smoke = subparsers.add_parser(
        "smoke-feature-wheel",
        help="Fail-closed Python 3.9 installed-wheel smoke for ROM-backed research infos",
    )
    feature_smoke.add_argument("wheel", type=Path)
    feature_smoke.add_argument("--python", type=Path, required=True)
    feature_smoke.add_argument("--rom", type=Path, required=True)
    feature_smoke.add_argument("--evidence", type=Path, required=True)
    feature_smoke.set_defaults(func=smoke_feature_wheel)

    smoke_source = subparsers.add_parser("smoke-sdist", help="Build, install, and import-test a source distribution")
    smoke_source.add_argument("sdist", type=Path)
    smoke_source.add_argument("--python", type=Path, default=PYTHON)
    smoke_source.set_defaults(func=smoke_sdist)

    final = subparsers.add_parser("final-check", help="Audit wheels, run twine check, hash, and print upload command")
    final.add_argument("--version")
    final.add_argument("wheels", nargs="*")
    final.set_defaults(func=final_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
