import importlib.util
from pathlib import Path

import supermariobrosnes_turbo


ROOT = Path(__file__).resolve().parents[1]


def _release_module():
    path = ROOT / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("release_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_package_exposes_distribution_version():
    assert supermariobrosnes_turbo.__version__ != "0+unknown"
    assert supermariobrosnes_turbo.__version__ == (ROOT / "VERSION.txt").read_text().strip()


def test_project_policy_files_exist():
    expected = (
        "LICENSE",
        "NOTICE.md",
        "SECURITY.md",
        "SUPPORT.md",
        "GOVERNANCE.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        ".github/CODEOWNERS",
        ".github/pull_request_template.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/workflows/ci.yml",
    )
    assert all((ROOT / relative).is_file() for relative in expected)


def test_installed_commands_cover_import_train_and_play():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "[project.scripts]" in pyproject
    assert 'smb-turbo = "supermariobrosnes_turbo.cli:main"' in pyproject
    assert "smb-turbo-import" not in pyproject
    assert "smb-turbo-play" not in pyproject
    assert "smb-turbo-train" not in pyproject


def test_imported_rom_is_ignored_and_excluded_from_distributions():
    gitignore = (ROOT / ".gitignore").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "python/supermariobrosnes_turbo/data/**/rom.nes" in gitignore
    assert 'exclude = ["python/supermariobrosnes_turbo/data/**/rom.nes"]' in pyproject


def test_release_promotes_unreleased_changelog(tmp_path, monkeypatch):
    release = _release_module()
    changes = tmp_path / "CHANGES.md"
    changes.write_text(
        "# Changelog\n\n## Unreleased\n\n- New behavior.\n\n## 0.3.0 - 2026-07-14\n\n- Old behavior.\n"
    )
    monkeypatch.setattr(release, "CHANGES", changes)

    release.promote_changelog("0.3.1", release_date="2026-07-15")

    assert changes.read_text() == (
        "# Changelog\n\n## Unreleased\n\n- Nothing yet.\n\n"
        "## 0.3.1 - 2026-07-15\n\n- New behavior.\n\n"
        "## 0.3.0 - 2026-07-14\n\n- Old behavior.\n"
    )


def test_release_generates_changelog_when_unreleased_is_empty(tmp_path, monkeypatch):
    release = _release_module()
    changes = tmp_path / "CHANGES.md"
    changes.write_text(
        "# Changelog\n\n## Unreleased\n\n- Nothing yet.\n\n"
        "## 0.3.0 - 2026-07-14\n\n- Old behavior.\n"
    )
    monkeypatch.setattr(release, "CHANGES", changes)

    release.promote_changelog(
        "0.3.1",
        release_date="2026-07-15",
        generated_notes="- Improve automatic releases.",
    )

    assert "## 0.3.1 - 2026-07-15\n\n- Improve automatic releases." in changes.read_text()


def test_release_accepts_already_prepared_target_changelog(tmp_path, monkeypatch):
    release = _release_module()
    changes = tmp_path / "CHANGES.md"
    original = (
        "# Changelog\n\n## Unreleased\n\n- Nothing yet.\n\n"
        "## 0.3.1 - 2026-07-15\n\n- Prepared release.\n"
    )
    changes.write_text(original)
    monkeypatch.setattr(release, "CHANGES", changes)

    release.promote_changelog("0.3.1", generated_notes="- Generated release.")

    assert changes.read_text() == original


def test_release_folds_new_notes_into_already_prepared_target(tmp_path, monkeypatch):
    release = _release_module()
    changes = tmp_path / "CHANGES.md"
    changes.write_text(
        "# Changelog\n\n## Unreleased\n\n- Later improvement.\n\n"
        "## 0.3.1 - 2026-07-15\n\n- Prepared release.\n"
    )
    monkeypatch.setattr(release, "CHANGES", changes)

    release.promote_changelog("0.3.1", generated_notes="- Generated release.")

    assert changes.read_text() == (
        "# Changelog\n\n## Unreleased\n\n- Nothing yet.\n\n"
        "## 0.3.1 - 2026-07-15\n\n- Later improvement.\n"
        "- Prepared release.\n"
    )
