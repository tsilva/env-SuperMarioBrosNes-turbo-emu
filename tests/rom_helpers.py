from __future__ import annotations

import os
from pathlib import Path

import pytest

from supermariobrosnes_turbo import RETRO_DATA_PATH_ENV_VAR, default_rom_path


ALLOW_MISSING_ROM_TESTS_ENV_VAR = "ALLOW_MISSING_ROM_TESTS"


def require_rom() -> Path:
    rom_path = default_rom_path()
    if rom_path is None:
        if os.environ.get(ALLOW_MISSING_ROM_TESTS_ENV_VAR) == "1":
            pytest.skip(f"set {RETRO_DATA_PATH_ENV_VAR} to run ROM-dependent tests")
        pytest.fail(f"set {RETRO_DATA_PATH_ENV_VAR} to run ROM-dependent tests")
    if not rom_path.exists():
        if os.environ.get(ALLOW_MISSING_ROM_TESTS_ENV_VAR) == "1":
            pytest.skip(f"imported ROM does not exist: {rom_path}")
        pytest.fail(f"imported ROM does not exist: {rom_path}")
    return rom_path
