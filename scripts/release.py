#!/usr/bin/env python3
"""Bump, commit, tag, and push a SuperMarioBros-Nes-turbo release."""

from __future__ import annotations

import argparse
from datetime import date
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_HELPER = REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CHANGES = REPO_ROOT / "CHANGES.md"


def run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    return subprocess.run(args, cwd=REPO_ROOT, env=env, check=True, text=True)


def capture(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()


def ensure_clean() -> None:
    status = capture(["git", "status", "--short"])
    if status:
        raise SystemExit(f"release tree must be clean before bumping:\n{status}")


def upstream_ref() -> str:
    try:
        return capture(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    except subprocess.CalledProcessError as exc:
        raise SystemExit("current branch must have an upstream before cutting a release") from exc


def ensure_synced() -> tuple[str, str]:
    upstream = upstream_ref()
    if "/" not in upstream:
        raise SystemExit(f"unexpected upstream ref: {upstream}")
    remote, branch = upstream.split("/", 1)
    run(["git", "fetch", "--prune", remote])
    left_right = capture(["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
    ahead, behind = [int(part) for part in left_right.split()]
    if ahead or behind:
        raise SystemExit(
            f"current branch must be synced with {upstream} before release; "
            f"ahead={ahead} behind={behind}"
        )
    return remote, branch


def helper(*args: str) -> None:
    run([str(PYTHON), str(RELEASE_HELPER), *args])


def helper_capture(*args: str) -> str:
    return capture([str(PYTHON), str(RELEASE_HELPER), *args])


def target_version(args: argparse.Namespace) -> str:
    if args.to:
        version = args.to
    else:
        version = helper_capture("bump-version", "--part", args.part).splitlines()[-1]
    helper("check-version")
    helper("check-pypi", "--version", version)
    return version


def refresh_locks() -> None:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".uv-cache")
    run(["uv", "lock"], env=env)
    run(["cargo", "generate-lockfile"])


def promote_changelog(version: str, *, release_date: str | None = None) -> None:
    text = CHANGES.read_text(encoding="utf-8")
    prefix = "# Changelog\n\n## Unreleased\n\n"
    if not text.startswith(prefix):
        raise SystemExit("CHANGES.md must begin with an Unreleased section")
    tail = text[len(prefix) :]
    separator = tail.find("\n## ")
    if separator < 0:
        unreleased = tail.strip()
        history = ""
    else:
        unreleased = tail[:separator].strip()
        history = tail[separator + 1 :].strip()
    if not unreleased or unreleased == "- Nothing yet.":
        raise SystemExit("CHANGES.md Unreleased section must describe the release")
    if f"## {version} " in text:
        raise SystemExit(f"CHANGES.md already contains release {version}")
    released = release_date or date.today().isoformat()
    updated = (
        f"{prefix}- Nothing yet.\n\n"
        f"## {version} - {released}\n\n{unreleased}\n"
    )
    if history:
        updated += f"\n{history}\n"
    CHANGES.write_text(updated, encoding="utf-8")


def run_checks(skip_checks: bool) -> None:
    if skip_checks:
        return
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".uv-cache")
    env.setdefault("ALLOW_MISSING_ROM_TESTS", "1")
    run(["cargo", "fmt", "--check", "--all"])
    run(
        [
            "cargo",
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ]
    )
    run(["cargo", "check", "--workspace", "--release"])
    run([str(PYTHON), "scripts/check_smb_dependency_closure.py"], env=env)
    run([str(PYTHON), "-m", "maturin", "develop", "--release"], env=env)
    run(["make", "test", "PYTHON=.venv/bin/python"], env=env)


def create_commit_and_tag(version: str) -> str:
    tag = f"v{version}"
    if subprocess.run(["git", "rev-parse", "--verify", "--quiet", tag], cwd=REPO_ROOT).returncode == 0:
        raise SystemExit(f"tag already exists locally: {tag}")
    run(
        [
            "git",
            "add",
            "VERSION.txt",
            "pyproject.toml",
            "Cargo.toml",
            "Cargo.lock",
            "uv.lock",
            "CHANGES.md",
        ]
    )
    run(["git", "commit", "-m", f"Release {tag}"])
    run(["git", "tag", tag, "HEAD"])
    return tag


def push_release(remote: str, branch: str, tag: str, dry_run: bool) -> None:
    args = ["git", "push", "--atomic", remote, f"HEAD:{branch}", tag]
    if dry_run:
        args.insert(2, "--dry-run")
    run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--to", help="Exact release version, for example 0.1.4")
    group.add_argument(
        "--part",
        choices=("patch", "minor", "major"),
        default="patch",
        help="Version component to bump when --to is omitted",
    )
    parser.add_argument("--skip-checks", action="store_true", help="Skip local cargo/maturin/test gates")
    parser.add_argument("--dry-run-push", action="store_true", help="Create the commit and tag, but dry-run the push")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if not PYTHON.exists():
        raise SystemExit("expected release environment at .venv/bin/python; run `uv sync --extra dev --group dev`")
    ensure_clean()
    remote, branch = ensure_synced()
    version = target_version(args)
    helper("bump-version", "--to", version, "--write")
    release_date = date.today().isoformat()
    promote_changelog(version, release_date=release_date)
    refresh_locks()
    helper("check-version", "--version", version)
    run_checks(args.skip_checks)
    tag = create_commit_and_tag(version)
    push_release(remote, branch, tag, args.dry_run_push)
    print()
    print(f"Released {tag}: pushed {branch} and tag to {remote}.")
    print("GitHub Actions will build, validate, and publish the release distributions from the pushed tag.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
