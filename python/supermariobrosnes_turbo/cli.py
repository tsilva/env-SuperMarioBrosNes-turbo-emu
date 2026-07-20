"""Unified command-line interface for SuperMarioBros-Nes-turbo."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Optional


Command = Callable[[Optional[list[str]]], int]


def _import_command(argv: list[str] | None) -> int:
    from importlib import import_module

    command = import_module("supermariobrosnes_turbo.import.__main__")
    return int(command.main(argv, prog="smb-turbo import"))


def _train_command(argv: list[str] | None) -> int:
    from .training import main

    return main(argv, prog="smb-turbo train")


def _play_command(argv: list[str] | None) -> int:
    from .state_playback import main

    return main(argv, prog="smb-turbo play")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smb-turbo", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands: tuple[tuple[str, str, Command], ...] = (
        ("import", "import the supported ROM", _import_command),
        ("train", "train action-run policies for one or all levels", _train_command),
        ("play", "play an exact state manually or with a policy", _play_command),
    )
    for name, help_text, handler in commands:
        subparser = subparsers.add_parser(name, help=help_text, add_help=False)
        subparser.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(None if argv is None else list(argv))
    return int(args.handler(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
