from __future__ import annotations

import hashlib
import importlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from supermariobrosnes_turbo import default_rom_path, resolve_required_rom_path
from supermariobrosnes_turbo import roms


def _write_importable_rom(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    data: bytes = b"canonical-rom",
) -> bytes:
    monkeypatch.setattr(roms, "EXPECTED_SMB_ROM_SHA256", hashlib.sha256(data).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _zip_bytes(name: str, data: bytes) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(name, data)
    return buffer.getvalue()


def test_retro_data_path_uses_stable_retro_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "retro-data"
    rom_path = root / "stable" / roms.DEFAULT_GAME / "rom.nes"
    rom_path.parent.mkdir(parents=True)
    rom_path.write_bytes(b"rom")
    monkeypatch.setenv(roms.RETRO_DATA_PATH_ENV_VAR, str(root))
    monkeypatch.setattr(roms, "_stable_retro_rom_path", lambda _game: None)

    assert default_rom_path() == rom_path
    assert resolve_required_rom_path() == rom_path


def test_package_data_tree_is_default_without_retro_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "package-data"
    rom_path = root / "stable" / roms.DEFAULT_GAME / "rom.nes"
    rom_path.parent.mkdir(parents=True)
    rom_path.write_bytes(b"rom")
    monkeypatch.delenv(roms.RETRO_DATA_PATH_ENV_VAR, raising=False)
    monkeypatch.setattr(roms, "package_data_path", lambda: root)
    monkeypatch.setattr(roms, "_stable_retro_rom_path", lambda _game: None)

    assert default_rom_path() == rom_path


def test_installed_stable_retro_tree_is_final_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stable_rom = tmp_path / "stable-retro" / "rom.nes"
    stable_rom.parent.mkdir()
    stable_rom.write_bytes(b"rom")
    monkeypatch.delenv(roms.RETRO_DATA_PATH_ENV_VAR, raising=False)
    monkeypatch.setattr(roms, "package_data_path", lambda: tmp_path / "missing")
    monkeypatch.setattr(roms, "_stable_retro_rom_path", lambda _game: stable_rom)

    assert default_rom_path() == stable_rom


def test_explicit_path_overrides_imported_rom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "retro-data"
    imported = root / "stable" / roms.DEFAULT_GAME / "rom.nes"
    imported.parent.mkdir(parents=True)
    imported.write_bytes(b"imported")
    explicit = tmp_path / "explicit.nes"
    monkeypatch.setenv(roms.RETRO_DATA_PATH_ENV_VAR, str(root))

    assert resolve_required_rom_path(explicit) == explicit


def test_legacy_rom_path_is_not_discovered_and_gets_migration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "legacy.nes"
    legacy.write_bytes(b"rom")
    monkeypatch.setenv(roms.LEGACY_ROM_PATH_ENV_VAR, str(legacy))
    monkeypatch.setenv(roms.RETRO_DATA_PATH_ENV_VAR, str(tmp_path / "empty"))
    monkeypatch.setattr(roms, "_stable_retro_rom_path", lambda _game: None)

    assert default_rom_path() is None
    with pytest.raises(ValueError, match="ROM_PATH is no longer supported"):
        resolve_required_rom_path()


def test_import_rom_from_file_is_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source" / "mario.nes"
    data = _write_importable_rom(monkeypatch, source)
    root = tmp_path / "retro-data"

    destination = roms.import_roms([source], root=root)
    first_mtime = destination.stat().st_mtime_ns
    assert destination == root / "stable" / roms.DEFAULT_GAME / "rom.nes"
    assert destination.read_bytes() == data

    assert roms.import_roms([source], root=root) == destination
    assert destination.stat().st_mtime_ns == first_mtime


def test_import_rom_recursively_from_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source" / "nested" / "mario.nes"
    data = _write_importable_rom(monkeypatch, source)

    destination = roms.import_roms([tmp_path / "source"], root=tmp_path / "data")

    assert destination.read_bytes() == data


def test_import_rom_from_nested_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"canonical-rom"
    monkeypatch.setattr(roms, "EXPECTED_SMB_ROM_SHA256", hashlib.sha256(data).hexdigest())
    archive_path = tmp_path / "roms.zip"
    archive_path.write_bytes(_zip_bytes("nested.zip", _zip_bytes("mario.nes", data)))

    destination = roms.import_roms([archive_path], root=tmp_path / "data")

    assert destination.read_bytes() == data


def test_import_rejects_nonmatching_roms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "wrong.nes"
    source.write_bytes(b"wrong")
    monkeypatch.setattr(roms, "EXPECTED_SMB_ROM_SHA256", hashlib.sha256(b"right").hexdigest())

    with pytest.raises(FileNotFoundError, match="No supported Super Mario Bros NES ROM"):
        roms.import_roms([source], root=tmp_path / "data")


def test_import_surfaces_unwritable_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "mario.nes"
    _write_importable_rom(monkeypatch, source)

    def deny_write(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("destination is read-only")

    monkeypatch.setattr(roms.tempfile, "NamedTemporaryFile", deny_write)

    with pytest.raises(PermissionError, match="destination is read-only"):
        roms.import_roms([source], root=tmp_path / "data")


def test_import_cli_uses_retro_data_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "mario.nes"
    data = _write_importable_rom(monkeypatch, source)
    root = tmp_path / "retro-data"
    monkeypatch.setenv(roms.RETRO_DATA_PATH_ENV_VAR, str(root))
    command = importlib.import_module("supermariobrosnes_turbo.import.__main__")

    assert command.main([str(source)]) == 0
    destination = root / "stable" / roms.DEFAULT_GAME / "rom.nes"
    assert destination.read_bytes() == data
    assert str(destination) in capsys.readouterr().out


def test_retro_data_path_states_are_discoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supermariobrosnes_turbo import list_available_states

    root = tmp_path / "retro-data"
    state = root / "stable" / roms.DEFAULT_GAME / "Custom.state"
    state.parent.mkdir(parents=True)
    state.write_bytes(b"state")
    monkeypatch.setenv(roms.RETRO_DATA_PATH_ENV_VAR, str(root))

    assert "Custom" in list_available_states()
