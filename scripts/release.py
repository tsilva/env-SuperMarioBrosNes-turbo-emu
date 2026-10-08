#!/usr/bin/env python3
"""Bump, commit, tag, and push an env-SuperMarioBrosNes-turbo-emu release."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_HELPER = REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
PYTHON = Path(sys.executable)
RELEASE_FILES = (
    REPO_ROOT / "VERSION.txt",
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "Cargo.toml",
    REPO_ROOT / "Cargo.lock",
    REPO_ROOT / "uv.lock",
    REPO_ROOT / "CITATION.cff",
)
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
    if branch != "main" or capture(["git", "branch", "--show-current"]) != "main":
        raise SystemExit("publication requires synchronized main")
    return remote, branch


def helper(*args: str) -> None:
    run([str(PYTHON), str(RELEASE_HELPER), *args])


def helper_capture(*args: str) -> str:
    return capture([str(PYTHON), str(RELEASE_HELPER), *args])


def tag_exists(tag: str) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", tag],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
    ).returncode == 0


def target_version(args: argparse.Namespace) -> str:
    if args.to:
        version = args.to
    elif args.part:
        version = helper_capture("bump-version", "--part", args.part).splitlines()[-1]
    else:
        current = (REPO_ROOT / "VERSION.txt").read_text(encoding="utf-8").strip()
        version = (
            current
            if not tag_exists(f"v{current}")
            else helper_capture("bump-version", "--part", "patch").splitlines()[-1]
        )
    helper("check-version")
    helper("check-pypi", "--version", version)
    return version


def refresh_locks() -> None:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".uv-cache")
    env["UV_CONFIG_FILE"] = str(REPO_ROOT / "uv-tool.toml")
    run(["uv", "lock", "--check"], env=env)
    run(["cargo", "metadata", "--locked", "--no-deps"])


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
            "CITATION.cff",
        ]
    )
    if subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=REPO_ROOT,
    ).returncode != 0:
        run(["git", "commit", "-m", f"Release {tag}"])
    run(["git", "tag", "-a", tag, "-m", f"Release {tag}"])
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
        help="Version component to bump; by default reuse an untagged project version or bump patch",
    )
    parser.add_argument("--dry-run-push", action="store_true", help="Create the commit and tag, but dry-run the push")
    parser.add_argument("--validate", action="store_true", help="validate the pushed main commit in Actions without publication")
    args = parser.parse_args()
    if args.validate and (args.to or args.part or args.dry_run_push):
        parser.error("--validate cannot be combined with version or push options")
    return args


def validate() -> None:
    upstream = upstream_ref()
    remote, _, branch = upstream.partition("/")
    if branch != "main":
        raise SystemExit("validation requires a main upstream")
    run(["git", "fetch", remote, "main"])
    sha = capture(["git", "rev-parse", f"{remote}/main"])
    run(["gh", "workflow", "run", "release.yml", "--ref", "main", "-f", f"ref={sha}"])
    print(f"validation-sha\t{sha}")


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if args.validate:
        validate()
        return
    ensure_clean()
    remote, branch = ensure_synced()
    version = target_version(args)
    snapshots = {path: path.read_bytes() for path in RELEASE_FILES}
    try:
        helper("bump-version", "--to", version, "--write")
        refresh_locks()
        helper("check-version", "--version", version)
        run(["git", "diff", "--check"])
        tag = create_commit_and_tag(version)
    except BaseException:
        for path, contents in snapshots.items():
            path.write_bytes(contents)
        subprocess.run(["git", "reset", "--quiet"], cwd=REPO_ROOT, check=False)
        raise
    push_release(remote, branch, tag, args.dry_run_push)
    print()
    print(f"Released {tag}: pushed {branch} and tag to {remote}.")
    print("GitHub Actions will build, validate, and publish the release distributions from the pushed tag.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
