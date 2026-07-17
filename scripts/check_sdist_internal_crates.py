#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


REQUIRED_SUFFIXES = {
    "crates/nes-turbo-nrom-core/Cargo.toml",
    "crates/nes-turbo-nrom-core/src/lib.rs",
    "crates/nes-turbo-nrom-core/src/machine.rs",
    "crates/smb-turbo-driver/Cargo.toml",
    "crates/smb-turbo-driver/src/lib.rs",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that the root sdist contains its internal Rust path dependencies"
    )
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()

    with tarfile.open(args.sdist, "r:gz") as archive:
        names = archive.getnames()
    missing = sorted(
        suffix
        for suffix in REQUIRED_SUFFIXES
        if not any(name.endswith(f"/{suffix}") for name in names)
    )
    if missing:
        print("source distribution is missing internal path dependencies:")
        print("\n".join(missing))
        return 1
    print("source distribution contains both internal Rust path dependencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
