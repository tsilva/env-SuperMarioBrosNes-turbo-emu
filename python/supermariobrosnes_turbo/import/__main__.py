from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..roms import import_roms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import the supported Super Mario Bros ROM into Stable Retro-compatible data.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="ROM files, ZIP archives, or directories to search (default: current directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        destination = import_roms(args.paths)
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Imported SuperMarioBros-Nes-v0 to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
