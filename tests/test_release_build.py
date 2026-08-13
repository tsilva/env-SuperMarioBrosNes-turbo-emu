import importlib.util
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10.
    import tomli as tomllib


def _release_build_module():
    root = Path(__file__).resolve().parents[1]
    path = root / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
    spec = importlib.util.spec_from_file_location("release_build", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_version_file_is_the_single_source_of_truth():
    root = Path(__file__).resolve().parents[1]

    assert (root / "VERSION.txt").read_text(encoding="utf-8").strip()
    assert "VERSION.txt" in (
        root / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
    ).read_text(encoding="utf-8")
    assert "VERSION.txt" in (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    release_script = (root / "scripts" / "release.py").read_text(encoding="utf-8")
    for release_file in (
        "VERSION.txt",
        "pyproject.toml",
        "Cargo.toml",
        "Cargo.lock",
        "uv.lock",
        "CITATION.cff",
        "CHANGES.md",
    ):
        assert f'"{release_file}"' in release_script


def test_runtime_dependency_bounds_match_the_supported_contract():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["dependencies"] == [
        "gymnasium>=1.1,<2",
        "numpy>=1.26,<3",
    ]


def test_version_bump_updates_only_the_project_entries_in_lockfiles(tmp_path):
    release_build = _release_build_module()
    lock = tmp_path / "lock.toml"
    lock.write_text(
        '[[package]]\nname = "dependency"\nversion = "9.9.9"\n\n'
        '[[package]]\nname = "supermariobrosnes-turbo"\nversion = "0.4.2"\n'
    )

    release_build.replace_package_version(lock, "supermariobrosnes-turbo", "0.4.3")

    assert 'name = "dependency"\nversion = "9.9.9"' in lock.read_text()
    assert 'name = "supermariobrosnes-turbo"\nversion = "0.4.3"' in lock.read_text()


def test_version_bump_updates_citation_release_metadata(tmp_path, monkeypatch):
    release_build = _release_build_module()
    citation = tmp_path / "CITATION.cff"
    citation.write_text("version: 0.6.2\ndate-released: 2026-07-29\n")
    monkeypatch.setattr(release_build, "CITATION", citation)

    release_build.replace_citation_release("0.6.3")

    contents = citation.read_text()
    assert "version: 0.6.3" in contents
    assert f"date-released: {release_build.date.today().isoformat()}" in contents


def test_release_validates_python_314_with_stable_abi_wheels():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8",
    )
    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")

    assert 'PYTHON_VERSION: "3.14"' in workflow
    assert 'features = ["abi3-py39", "extension-module"]' in cargo


def test_local_release_runs_the_ci_source_gates():
    root = Path(__file__).resolve().parents[1]
    release_script = (root / "scripts" / "release.py").read_text(encoding="utf-8")

    assert '"fmt", "--check", "--all"' in release_script
    assert '"clippy"' in release_script
    assert '"--workspace"' in release_script
    assert '"--all-targets"' in release_script
    assert '"--all-features"' in release_script
    assert "check_smb_dependency_closure.py" in release_script
    assert '"uv", "lock", "--check"' in release_script
    assert '"cargo", "metadata", "--locked", "--no-deps"' in release_script
    assert '"cargo", "generate-lockfile"' not in release_script


def test_release_wheel_builds_use_platform_scoped_cargo_caches():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8",
    )
    release_build = _release_build_module()

    assert release_build.RELEASE_PLATFORMS == (
        "macos-arm64",
        "linux-x86_64",
    )
    assert "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9" in workflow
    assert "path: ${{ matrix.cargo_target_dir }}" in workflow
    assert "key: cargo-target-v2-${{ matrix.platform }}-${{ steps.source.outputs.sha }}" in workflow
    assert 'run: echo "sha=$(git rev-parse HEAD)" >> "$GITHUB_OUTPUT"' in workflow
    assert "cargo-target-v2-${{ matrix.platform }}-" in workflow
    for platform_name in release_build.RELEASE_PLATFORMS:
        assert f"platform: {platform_name}" in workflow
        assert f"cargo_target_dir: target-release-{platform_name}" in workflow
    for platform_name in (
        "macos-x86_64",
        "linux-aarch64",
        "windows-x86_64",
    ):
        assert f"platform: {platform_name}" not in workflow

    assert release_build.cargo_target_dir("macos-arm64", root) == (
        root / "target-release-macos-arm64"
    )
    assert release_build.cargo_target_dir("linux-x86_64", root) == (
        root / "target-release-linux-x86_64"
    )


def test_linux_release_cache_is_mounted_into_cibuildwheel(tmp_path):
    release_build = _release_build_module()
    env = release_build.linux_build_env("linux-x86_64", tmp_path)

    assert env["CIBW_CONTAINER_ENGINE"] == (
        f"docker; create_args: --volume={(tmp_path / 'target-release-linux-x86_64').resolve()}:/cargo-target"
    )
    assert "CARGO_TARGET_DIR=/cargo-target" in env["CIBW_ENVIRONMENT_LINUX"]
    assert release_build.should_ignore(Path("target-release-linux"))
    assert release_build.should_ignore(Path("target-release-linux-aarch64"))
    assert not release_build.should_ignore(Path("nested/target-release-linux"))


def test_release_requires_all_documented_wheel_platforms(tmp_path):
    release_build = _release_build_module()
    names = (
        "pkg-0.3.0-cp39-abi3-macosx_14_0_arm64.whl",
        "pkg-0.3.0-cp39-abi3-manylinux_2_17_x86_64.whl",
    )
    wheels = []
    for name in names:
        wheel = tmp_path / name
        wheel.touch()
        wheels.append(wheel)

    release_build.assert_platform_coverage(wheels)
    assert {release_build.wheel_release_platform(wheel) for wheel in wheels} == set(
        release_build.RELEASE_PLATFORMS
    )


@pytest.mark.parametrize(
    "name",
    (
        "pkg-0.3.0-cp39-abi3-macosx_13_0_x86_64.whl",
        "pkg-0.3.0-cp39-abi3-manylinux_2_17_aarch64.whl",
        "pkg-0.3.0-cp39-abi3-win_amd64.whl",
    ),
)
def test_release_rejects_unpublished_wheel_platforms(tmp_path, name):
    release_build = _release_build_module()
    supported = (
        tmp_path / "pkg-0.3.0-cp39-abi3-macosx_14_0_arm64.whl",
        tmp_path / "pkg-0.3.0-cp39-abi3-manylinux_2_17_x86_64.whl",
    )
    unpublished = tmp_path / name
    for wheel in (*supported, unpublished):
        wheel.touch()

    with pytest.raises(SystemExit, match="contains unpublished platforms"):
        release_build.assert_platform_coverage([*supported, unpublished])


def test_latest_non_yanked_pypi_version_ignores_fully_yanked_latest_release():
    release_build = _release_build_module()
    releases = {
        "0.2.3": [{"filename": "older.whl", "yanked": False}],
        "0.2.4": [{"filename": "current.whl", "yanked": False}],
        "0.3.0": [
            {"filename": "macos.whl", "yanked": True},
            {"filename": "linux.whl", "yanked": True},
        ],
    }

    assert release_build.latest_non_yanked_pypi_version(releases) == "0.2.4"


def test_latest_non_yanked_pypi_version_accepts_release_with_any_non_yanked_file():
    release_build = _release_build_module()
    releases = {
        "0.2.4": [{"filename": "current.whl", "yanked": False}],
        "0.2.5": [
            {"filename": "bad-platform.whl", "yanked": True},
            {"filename": "good-platform.whl", "yanked": False},
        ],
    }

    assert release_build.latest_non_yanked_pypi_version(releases) == "0.2.5"


def test_built_distribution_smoke_exercises_snapshot_replay_when_rom_is_available():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
    ).read_text(encoding="utf-8")

    for required in (
        "default_rom_path",
        "supports_live_snapshots",
        "snapshot_codec_api_version",
        "snapshot_codec_id",
        "capture_snapshots",
        "encode_snapshots",
        "decode_snapshots",
        '"snapshots": [decoded_handles[0], decoded_handles[0]]',
        'restored_infos["start_source"]',
        "np.testing.assert_array_equal(expected, actual)",
        "canonical SMB ROM is unavailable",
        "extra_info_descriptors",
        "enemy_active",
        "smoke-feature-wheel",
        "feature-smoke Python must be CPython 3.9",
        "feature-smoke ROM SHA-256",
        "supermariobrosnes-turbo.portable-v2",
        'b"SMBVEC2\\\\0"',
    ):
        assert required in source
