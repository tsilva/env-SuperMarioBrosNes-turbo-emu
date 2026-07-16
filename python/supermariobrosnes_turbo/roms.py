from __future__ import annotations

from collections.abc import Iterable, Iterator
import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile, ZipFile


DEFAULT_GAME = "SuperMarioBros-Nes-v0"
RETRO_DATA_PATH_ENV_VAR = "RETRO_DATA_PATH"
LEGACY_ROM_PATH_ENV_VAR = "ROM_PATH"
EXPECTED_SMB_ROM_SHA256 = "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de"
ROM_FILENAME = "rom.nes"
MAX_ARCHIVE_DEPTH = 8
MAX_ARCHIVE_ENTRY_SIZE = 16 * 1024 * 1024


def package_data_path() -> Path:
    """Return this package's default Stable Retro-compatible data root."""
    return Path(__file__).resolve().parent / "data"


def data_path() -> Path:
    """Return the configured Stable Retro data root."""
    value = os.environ.get(RETRO_DATA_PATH_ENV_VAR)
    return Path(value).expanduser() if value else package_data_path()


def game_data_path(root: str | Path | None = None, game: str = DEFAULT_GAME) -> Path:
    base = Path(root).expanduser() if root is not None else data_path()
    return base / "stable" / game


def imported_rom_path(root: str | Path | None = None, game: str = DEFAULT_GAME) -> Path:
    return game_data_path(root, game) / ROM_FILENAME


def _stable_retro_rom_path(game: str) -> Path | None:
    try:
        import stable_retro.data  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        path = stable_retro.data.get_romfile_path(
            game,
            stable_retro.data.Integrations.ALL,
        )
    except Exception:
        return None
    return Path(path) if path else None


def rom_candidates(game: str = DEFAULT_GAME) -> tuple[Path, ...]:
    """Return automatic ROM candidates in Stable Retro-compatible precedence."""
    candidates = [imported_rom_path(game=game)]
    if not candidates[0].is_file():
        stable_retro_path = _stable_retro_rom_path(game)
        if stable_retro_path is not None:
            candidates.append(stable_retro_path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        expanded = candidate.expanduser()
        key = expanded.resolve(strict=False)
        if key not in seen:
            unique.append(expanded)
            seen.add(key)
    return tuple(unique)


def default_rom_path(game: str = DEFAULT_GAME) -> Path | None:
    """Return the first imported ROM found through Stable Retro discovery."""
    return next((path for path in rom_candidates(game) if path.is_file()), None)


def _missing_rom_message(game: str) -> str:
    expected = imported_rom_path(game=game)
    searched = ", ".join(str(path) for path in rom_candidates(game))
    message = (
        f"ROM required for {game}; pass rom_path or import it with "
        f"`python -m supermariobrosnes_turbo.import /path/to/roms`. "
        f"Set {RETRO_DATA_PATH_ENV_VAR} to the data root containing "
        f"stable/{game}/{ROM_FILENAME}. Expected {expected}; searched: {searched}"
    )
    if os.environ.get(LEGACY_ROM_PATH_ENV_VAR):
        message += (
            f". {LEGACY_ROM_PATH_ENV_VAR} is no longer supported; set "
            f"{RETRO_DATA_PATH_ENV_VAR} to the data root instead"
        )
    return message


def resolve_required_rom_path(
    path: str | Path | None = None,
    game: str = DEFAULT_GAME,
) -> Path:
    """Resolve an explicit ROM path or a Stable Retro-compatible imported ROM."""
    if path is not None:
        return Path(path).expanduser()
    resolved = default_rom_path(game)
    if resolved is None:
        raise ValueError(_missing_rom_message(game))
    return resolved


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_roms(data: bytes, label: str, depth: int = 0) -> Iterator[tuple[str, bytes]]:
    if depth >= MAX_ARCHIVE_DEPTH:
        return
    try:
        with ZipFile(BytesIO(data)) as archive:
            for entry in sorted(archive.infolist(), key=lambda item: item.filename):
                if entry.is_dir() or entry.file_size > MAX_ARCHIVE_ENTRY_SIZE:
                    continue
                suffix = Path(entry.filename).suffix.lower()
                if suffix not in {".nes", ".zip"}:
                    continue
                entry_data = archive.read(entry)
                entry_label = f"{label}!{entry.filename}"
                if suffix == ".nes":
                    yield entry_label, entry_data
                else:
                    yield from _zip_roms(entry_data, entry_label, depth + 1)
    except (BadZipFile, KeyError, NotImplementedError, OSError, RuntimeError):
        return


def _source_roms(source: Path) -> Iterator[tuple[str, bytes]]:
    if source.is_dir():
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            yield from _source_roms(path)
        return
    if not source.is_file():
        return
    suffix = source.suffix.lower()
    if suffix == ".nes":
        yield str(source), source.read_bytes()
    elif suffix == ".zip":
        yield from _zip_roms(source.read_bytes(), str(source))


def find_supported_rom(sources: Iterable[str | Path]) -> tuple[str, bytes] | None:
    for raw_source in sources:
        source = Path(raw_source).expanduser()
        for label, data in _source_roms(source):
            if sha256_bytes(data) == EXPECTED_SMB_ROM_SHA256:
                return label, data
    return None


def import_roms(
    sources: Iterable[str | Path],
    *,
    root: str | Path | None = None,
) -> Path:
    """Import the canonical ROM and return its Stable Retro-compatible path."""
    source_list = tuple(sources)
    match = find_supported_rom(source_list)
    if match is None:
        rendered = ", ".join(str(Path(source).expanduser()) for source in source_list)
        raise FileNotFoundError(
            f"No supported Super Mario Bros NES ROM found in: {rendered or '<none>'}"
        )

    _label, data = match
    destination = imported_rom_path(root=root)
    if destination.is_file() and sha256_path(destination) == EXPECTED_SMB_ROM_SHA256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{ROM_FILENAME}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
