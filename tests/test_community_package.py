import importlib.util
import re
from pathlib import Path

import env_supermariobrosnes_turbo_emu


ROOT = Path(__file__).resolve().parents[1]


def _release_module():
    path = ROOT / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("release_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_package_exposes_distribution_version():
    assert env_supermariobrosnes_turbo_emu.__version__ != "0+unknown"
    assert env_supermariobrosnes_turbo_emu.__version__ == (ROOT / "VERSION.txt").read_text().strip()


def test_project_policy_files_exist():
    expected = (
        "LICENSE",
        "CITATION.cff",
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


def test_installed_commands_cover_import_and_play():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "[project.scripts]" in pyproject
    assert 'smb-turbo = "env_supermariobrosnes_turbo_emu.cli:main"' in pyproject
    assert "smb-turbo-import" not in pyproject
    assert "smb-turbo-play" not in pyproject
    assert "smb-turbo-train" not in pyproject


def test_readme_delegates_training_to_pinned_gradlab_recipes():
    readme = (ROOT / "README.md").read_text()

    assert (
        "uvx gradlab@0.1.1 train "
        "SuperMarioBros-Nes-v0/Level1-1/turbo-demo --rom "
        "/absolute/path/to/SuperMarioBros.nes"
    ) in readme
    assert (
        "uvx gradlab@0.1.1 train "
        "SuperMarioBros-Nes-v0/Level1-1/go-explore-jerk-20m --rom "
        "/absolute/path/to/SuperMarioBros.nes"
    ) in readme
    assert "SuperMarioBros-Nes-v0/Level1-1/go-explore-20m" not in readme
    assert "smb-turbo train" not in readme
    assert "train.py" not in readme


def test_readme_use_example_matches_action_batch_contract():
    readme = (ROOT / "README.md").read_text()
    use_section = readme.split("## Use from Python", maxsplit=1)[1].split(
        "## Train with GradLab", maxsplit=1
    )[0]

    assert "use_restricted_actions=Actions.ALL" in use_section
    assert 'action_batch("right", env.num_envs)' in use_section
    assert 'use_restricted_actions="basic"' not in use_section


def test_readme_leads_with_a_supported_first_run_path():
    readme = (ROOT / "README.md").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert readme.index("## Quick start") < readme.index("## What it provides")
    assert "uv tool install env-supermariobrosnes-turbo-emu" in readme
    assert "smb-turbo play --rom /absolute/path/to/SuperMarioBros.nes" in readme
    assert "discoverable SDL2 runtime" in readme
    assert "Windows PowerShell" not in readme
    assert '"Operating System :: Microsoft :: Windows"' not in pyproject


def test_readme_documents_turbo_advantages_beyond_speed():
    readme = (ROOT / "README.md").read_text()
    why = readme.split("## What it provides", maxsplit=1)[1].split(
        "## Compared with Stable Retro", maxsplit=1
    )[0]
    comparison = readme.split("## Compared with Stable Retro", maxsplit=1)[1].split(
        "## Use from Python", maxsplit=1
    )[0]

    for term in (
        "step_async()",
        "num_threads",
        "noop_reset_max",
        "sticky_action_prob",
        "reward_clip",
        "state_indices",
        "Portable snapshots",
        "Actions.DISCRETE",
        "Actions.MULTI_DISCRETE",
        "obs_copy",
        "render_lane()",
        "info_filter",
        "ram()",
        "capabilities",
        "signal_schema",
        "automatically switch",
    ):
        assert term in why

    for term in (
        "https://stable-retro.farama.org/python/",
        "not a drop-in Stable Retro replacement",
        "multiplayer",
        "RAM observations",
        "BK2 movie recording",
        "one player",
        "image observations",
    ):
        assert term in comparison


def test_api_info_table_matches_available_info_key_order():
    api = (ROOT / "API.md").read_text()
    info_section = api.split("## Research infos", maxsplit=1)[1].split(
        "The environment may also add lifecycle metadata", maxsplit=1
    )[0]
    documented_keys = tuple(
        re.findall(
            r"^\| `([^`]+)` \| (?:Legacy/default|Extra/opt-in) \|",
            info_section,
            re.MULTILINE,
        )
    )

    assert documented_keys == env_supermariobrosnes_turbo_emu.AVAILABLE_INFO_KEYS


def test_api_snapshot_example_encodes_before_closing_source():
    api = (ROOT / "API.md").read_text()
    snapshot_section = api.split("## Live snapshots", maxsplit=1)[1].split(
        "## Research infos", maxsplit=1
    )[0]

    assert "source = make_env()" in snapshot_section
    assert "destination = make_env()" in snapshot_section
    assert snapshot_section.index(
        "payloads = source.encode_snapshots"
    ) < snapshot_section.index(
        "source.close()"
    )


def test_api_documents_public_execution_and_diagnostic_controls():
    api = (ROOT / "API.md").read_text()
    execution = api.split("## Execution and episode controls", maxsplit=1)[1].split(
        "## States and selective reset", maxsplit=1
    )[0]
    profiling = api.split("## Profiling", maxsplit=1)[1]

    for term in (
        "step_async(actions)",
        "step_wait()",
        "num_threads=None",
        "noop_reset_max=N",
        "sticky_action_prob=p",
        "reward_clip=True",
    ):
        assert term in execution

    for term in (
        "enable_profiler()",
        "profiler_snapshot(top_n=64)",
        "reset_profiler()",
        "disable_profiler()",
    ):
        assert term in profiling


def test_benchmark_docs_pin_the_recorded_harness_and_current_results():
    readme = (ROOT / "README.md").read_text()
    benchmarks = (ROOT / "BENCHMARKS.md").read_text()
    commit = "917c3d70b04b54779a05f94055a748ceda524b20"
    tag_url = (
        "https://huggingface.co/datasets/tsilva/"
        "env-supermariobrosnes-turbo-emu-benchmarks/tree/v0.6.4"
    )
    bundle_url = (
        f"{tag_url}/bundles/v0.6.4/vs-stable-retro-1.0.1/"
        "65eb59b9c84d0420483a051f09df08b57d334d817671cbac685a5cd1dd11fc21"
    )

    assert "[verified `0.6.4` benchmarks](BENCHMARKS.md)" in readme
    assert "published `0.3.0` mapper 0/NROM benchmark" not in readme
    assert "media/benchmark-throughput.svg" not in readme
    assert "https://pypi.org/project/turbobench-cli/1.0.2/" in readme
    assert "https://pypi.org/project/turbobench-cli/1.0.1/" in benchmarks
    assert "https://pypi.org/project/turbobench-cli/1.0.2/" in benchmarks
    assert "turbobench-cli=2026-08-13T14:50:17Z" in benchmarks
    assert "turbobench-cli==1.0.1" in benchmarks
    assert "turbobench-cli=2026-08-13T15:41:56Z" in benchmarks
    assert "turbobench-cli==1.0.2" in benchmarks
    assert tag_url in readme
    assert bundle_url in readme
    assert tag_url in benchmarks
    assert bundle_url in benchmarks
    assert f"git checkout --detach {commit}" in benchmarks
    assert "uv run turbobench compare supermario/canonical-v1" in benchmarks
    assert "independently verified" not in benchmarks
    assert "stable-retro-turbo" not in readme.lower()
    assert "stable-retro-turbo" not in benchmarks.lower()


def test_citation_metadata_describes_latest_release():
    citation = (ROOT / "CITATION.cff").read_text()

    assert "cff-version: 1.2.0" in citation
    assert 'title: "env-SuperMarioBrosNes-turbo-emu"' in citation
    assert "family-names: Silva" in citation
    assert "given-names: Tiago" in citation
    assert "license: MIT" in citation
    assert f"version: {(ROOT / 'VERSION.txt').read_text().strip()}" in citation


def test_imported_rom_is_ignored_and_excluded_from_distributions():
    gitignore = (ROOT / ".gitignore").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "python/env_supermariobrosnes_turbo_emu/data/**/rom.nes" in gitignore
    assert 'exclude = ["python/env_supermariobrosnes_turbo_emu/data/**/rom.nes"]' in pyproject


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
