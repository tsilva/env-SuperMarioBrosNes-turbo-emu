from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _benchmark_release_module():
    path = (
        ROOT
        / ".codex"
        / "skills"
        / "benchmark-latest-release"
        / "scripts"
        / "benchmark_release.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload(uploaded: datetime) -> dict:
    filename = (
        "supermariobrosnes_turbo-0.6.4-cp39-abi3-"
        "manylinux_2_17_x86_64.whl"
    )
    artifact = {
        "filename": filename,
        "upload_time_iso_8601": uploaded.isoformat(),
        "yanked": False,
    }
    return {
        "info": {"version": "0.6.4"},
        "releases": {"0.6.4": [artifact]},
    }


def test_supermario_project_is_exempt_from_package_quarantine(monkeypatch) -> None:
    module = _benchmark_release_module()
    uploaded = datetime(2026, 8, 12, 12, 20, tzinfo=UTC)
    monkeypatch.setattr(module, "_request_pypi", lambda project: _payload(uploaded))
    monkeypatch.setattr(
        module, "_now_utc", lambda: datetime(2026, 8, 13, 12, 20, tzinfo=UTC)
    )

    result = module.latest_release(
        "supermariobrosnes-turbo", require_eligible=True
    )

    assert result["eligible"] is True
    assert result["quarantine_exempt"] is True
    assert result["eligible_at"] == result["uploaded_at"]


def test_other_projects_keep_the_seven_day_quarantine(monkeypatch) -> None:
    module = _benchmark_release_module()
    uploaded = datetime(2026, 8, 12, 12, 20, tzinfo=UTC)
    monkeypatch.setattr(module, "_request_pypi", lambda project: _payload(uploaded))
    monkeypatch.setattr(
        module, "_now_utc", lambda: datetime(2026, 8, 13, 12, 20, tzinfo=UTC)
    )

    with pytest.raises(ValueError, match="seven-day quarantine"):
        module.latest_release("another-project", require_eligible=True)
