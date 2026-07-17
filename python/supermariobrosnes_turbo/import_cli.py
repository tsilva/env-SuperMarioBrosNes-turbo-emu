"""Keyword-safe console entry point for the ROM importer."""

from __future__ import annotations

from importlib import import_module
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    importer = import_module("supermariobrosnes_turbo.import.__main__")
    return int(importer.main(argv))
