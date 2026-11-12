from __future__ import annotations

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def locked_versions(lockfile: Path, package: str) -> set[str]:
    text = lockfile.read_text()
    pattern = re.compile(
        rf'^name = "{re.escape(package)}"\nversion = "([^"]+)"$',
        re.MULTILINE,
    )
    return set(pattern.findall(text))


def test_rust_python_boundary_uses_patched_pyo3_family() -> None:
    assert locked_versions(ROOT / "Cargo.lock", "pyo3") == {"0.29.3"}
    assert locked_versions(ROOT / "Cargo.lock", "numpy") == {"0.29.0"}


def test_universal_python_lock_keeps_runtime_families_patched() -> None:
    lockfile = ROOT / "uv.lock"
    assert locked_versions(lockfile, "cryptography") == {"50.0.0"}
    assert locked_versions(lockfile, "requests") == {"2.34.2"}
    assert locked_versions(lockfile, "urllib3") == {"2.8.0"}


def test_universal_python_lock_excludes_certifi_and_filelock_advisories() -> None:
    lockfile = ROOT / "uv.lock"
    certifi = locked_versions(lockfile, "certifi")
    filelock = locked_versions(lockfile, "filelock")

    assert certifi and all(Version(v) >= Version("2026.10.19") for v in certifi)
    assert filelock
    # The Python 3.10 fallback is only locked when a dependency needs it.
    modern_filelock = filelock - {"3.20.3"}
    assert modern_filelock and all(Version(v) >= Version("4.2.1") for v in modern_filelock)


def test_security_constraints_keep_cooldown_and_python_compatibility() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    tool = tomllib.loads((ROOT / "uv-tool.toml").read_text())
    project_tool = project["tool"]["uv"]

    assert project["project"]["requires-python"] == ">=3.9"
    assert project_tool["exclude-newer"] == tool["exclude-newer"] == "7 days"
    assert project_tool["constraint-dependencies"] == tool["constraint-dependencies"]

    constraints = [Requirement(value) for value in project_tool["constraint-dependencies"]]
    for python, safe_filelock in [
        ("3.10", "3.20.3"),
        ("3.11", "4.2.1"),
        ("3.14", "4.2.1"),
    ]:
        env = {"python_version": python}
        filelock = [r for r in constraints if r.name == "filelock" and r.marker.evaluate(env)]
        certifi = [r for r in constraints if r.name == "certifi" and r.marker.evaluate(env)]
        assert len(filelock) == len(certifi) == 1
        assert safe_filelock in filelock[0].specifier
        assert "3.32.2" not in filelock[0].specifier
        assert "2026.10.19" in certifi[0].specifier
        assert "2026.10.18" not in certifi[0].specifier

    old_python = {"python_version": "3.9"}
    assert not any(
        r.name in {"certifi", "filelock"} and r.marker.evaluate(old_python)
        for r in constraints
    )


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
    playback = (ROOT / "python/env_supermariobrosnes_turbo_emu/policy_playback.py").read_text()

    assert re.search(r"playback = \[\s*\]", pyproject)
    assert "from huggingface_hub" not in playback
    assert "urllib.request.urlopen" in playback
