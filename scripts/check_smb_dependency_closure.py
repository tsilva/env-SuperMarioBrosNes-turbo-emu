#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = ROOT / "ci" / "smb-extension-dependencies.txt"
ROOT_PACKAGE = "supermariobrosnes-turbo"


def dependency_closure() -> list[str]:
    result = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    root_id = next(
        package_id
        for package_id, package in packages.items()
        if package["name"] == ROOT_PACKAGE
    )

    pending = [root_id]
    visited: set[str] = set()
    while pending:
        package_id = pending.pop()
        if package_id in visited:
            continue
        visited.add(package_id)
        for dependency in nodes[package_id]["deps"]:
            if any(
                kind["kind"] in (None, "normal")
                for kind in dependency["dep_kinds"]
            ):
                pending.append(dependency["pkg"])

    visited.remove(root_id)
    return sorted(
        f"{packages[package_id]['name']}=={packages[package_id]['version']}"
        for package_id in visited
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the normal Cargo dependency closure of the SMB extension"
    )
    parser.add_argument("--print", action="store_true", dest="print_only")
    args = parser.parse_args()

    actual = dependency_closure()
    if args.print_only:
        print("\n".join(actual))
        return 0

    expected = [
        line.strip()
        for line in EXPECTED_PATH.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if actual == expected:
        print(f"SMB extension dependency closure matches {EXPECTED_PATH.relative_to(ROOT)}")
        return 0

    print("SMB extension dependency closure changed:")
    print("expected:")
    print("\n".join(expected))
    print("actual:")
    print("\n".join(actual))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
