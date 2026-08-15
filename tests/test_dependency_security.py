from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def locked_versions(lockfile: Path, package: str) -> set[str]:
    text = lockfile.read_text()
    pattern = re.compile(
        rf'^name = "{re.escape(package)}"\nversion = "([^"]+)"$',
        re.MULTILINE,
    )
    return set(pattern.findall(text))


def test_rust_python_boundary_uses_patched_pyo3_family() -> None:
    assert locked_versions(ROOT / "Cargo.lock", "pyo3") == {"0.29.2"}
    assert locked_versions(ROOT / "Cargo.lock", "numpy") == {"0.29.0"}


def test_universal_python_lock_keeps_runtime_families_patched() -> None:
    lockfile = ROOT / "uv.lock"
    assert locked_versions(lockfile, "cryptography") == {"50.0.0"}
    assert locked_versions(lockfile, "filelock") == {"3.32.2"}
    assert locked_versions(lockfile, "requests") == {"2.34.2"}
    assert locked_versions(lockfile, "urllib3") == {"2.7.0"}


def test_python_39_pytest_exception_is_test_only_and_explicit() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert locked_versions(ROOT / "uv.lock", "pytest") == {"8.4.2", "9.1.1"}
    assert '"pytest>=8.4.2,<9; python_version < \'3.10\'"' in pyproject
    assert '"pytest>=9.0.3; python_version >= \'3.10\'"' in pyproject
    assert "--ignore GHSA-6w46-j5rx-g56g" in workflow
    assert "pytest" not in pyproject.split("dependencies = [", maxsplit=1)[1].split(
        "]", maxsplit=1
    )[0]


def test_playback_has_no_vulnerable_hugging_face_runtime_client() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    playback = (ROOT / "python/supermariobrosnes_turbo/policy_playback.py").read_text()

    assert re.search(r"playback = \[\s*\]", pyproject)
    assert "from huggingface_hub" not in playback
    assert 'https://huggingface.co/' in playback
