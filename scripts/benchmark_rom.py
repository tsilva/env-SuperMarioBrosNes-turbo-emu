from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_SMB_ROM_SHA256 = "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de"


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rom_hash(path: str | Path) -> str:
    expanded = Path(path).expanduser()
    digest = sha256_path(expanded)
    if digest != EXPECTED_SMB_ROM_SHA256:
        raise SystemExit(
            f"ROM SHA-256 mismatch for {expanded}: got {digest}, expected {EXPECTED_SMB_ROM_SHA256}"
        )
    return digest
