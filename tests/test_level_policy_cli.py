from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from supermariobrosnes_turbo import cli, state_playback, training
from supermariobrosnes_turbo.jerk import (
    policy_path_for_state,
    resolve_state_name,
    run_directory_for_state,
    validate_state_name,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("state", ["Level1-1", "Level2-1-clouds-easy", "Custom"])
def test_state_deterministically_selects_run_and_policy_names(state: str) -> None:
    assert validate_state_name(state) == state
    assert run_directory_for_state(state) == Path(f"runs/{state}")
    assert policy_path_for_state(state) == Path(f"runs/{state}/{state}.zip")


@pytest.mark.parametrize("state", ["", ".", "..", "../Level1-1", "a/b", r"a\b"])
def test_path_like_state_names_cannot_escape_policy_directory(state: str) -> None:
    with pytest.raises(ValueError, match="state name|state identifier"):
        policy_path_for_state(state)


def test_state_resolution_is_exact_and_supports_custom_names(tmp_path: Path) -> None:
    tmp_path.joinpath("Custom.state").write_bytes(b"state")

    assert resolve_state_name("Custom", state_dir=tmp_path) == "Custom"
    with pytest.raises(ValueError, match="unknown state 'custom'"):
        resolve_state_name("custom", state_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown state '1-1'"):
        resolve_state_name("1-1", state_dir=tmp_path)


def test_train_parser_uses_the_state_key_and_new_flags_only() -> None:
    parser = training.build_parser()
    args = parser.parse_args(
        [
            "Level1-1",
            "--transitions",
            "100",
            "--lanes",
            "4",
            "--continue-after-completion",
            "--overwrite",
        ]
    )

    assert args.state == "Level1-1"
    assert args.algorithm == "beam"
    assert args.transitions == 100
    assert args.lanes == 4
    assert args.continue_after_completion
    assert args.overwrite
    with pytest.raises(SystemExit):
        parser.parse_args(["Level1-1", "--timesteps", "100"])


def test_algorithm_specific_options_are_rejected() -> None:
    parser = training.build_parser()
    args = parser.parse_args(
        ["Level1-1", "--algorithm", "beam", "--retained-limit", "3"]
    )

    with pytest.raises(SystemExit):
        training._apply_algorithm_defaults(parser, args)


def test_play_parser_owns_modes_and_playback_options() -> None:
    parser = state_playback.build_parser()
    assert parser.parse_args([]).state == "Level1-1"
    args = parser.parse_args(
        ["Level1-1", "--policy", "policy.zip", "--backend", "native"]
    )

    assert args.state == "Level1-1"
    assert args.policy == "policy.zip"
    assert args.backend == "native"
    with pytest.raises(SystemExit):
        parser.parse_args(["Level1-1", "--manual", "--policy", "policy.zip"])
    with pytest.raises(SystemExit):
        parser.parse_args(["Level1-1", "--rom-path", "mario.nes"])


def test_play_without_state_starts_from_level_1_1(monkeypatch) -> None:
    played: list[str] = []

    class Player:
        def __init__(self, args) -> None:
            played.append(args.state)

        def run(self) -> None:
            pass

    monkeypatch.setattr(
        state_playback,
        "resolve_state_name",
        lambda state, **_kwargs: state,
    )
    monkeypatch.setattr(state_playback, "SdlExternalVecPlayer", Player)

    assert state_playback.main(["--manual"]) == 0
    assert played == ["Level1-1"]


def test_play_command_resolves_exact_state_policy(tmp_path: Path) -> None:
    policy = policy_path_for_state("Custom", runs_root=tmp_path)
    policy.parent.mkdir(parents=True)
    policy.touch()

    assert state_playback.resolve_state_policy("Custom", runs_dir=tmp_path) == policy
    assert state_playback.resolve_state_policy("Level1-1", runs_dir=tmp_path) is None


def test_play_prefers_legacy_beam_over_legacy_jerk(tmp_path: Path) -> None:
    beam = tmp_path / "Custom-beam" / "Custom.zip"
    jerk = tmp_path / "Custom-jerk" / "Custom.zip"
    beam.parent.mkdir()
    jerk.parent.mkdir()
    beam.touch()
    jerk.touch()

    assert state_playback.resolve_state_policy("Custom", runs_dir=tmp_path) == beam

    canonical = policy_path_for_state("Custom", runs_root=tmp_path)
    canonical.parent.mkdir()
    canonical.touch()
    assert state_playback.resolve_state_policy("Custom", runs_dir=tmp_path) == canonical


def test_unified_cli_exposes_only_import_train_and_play() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()

    assert "{import,train,play}" in help_text
    pyproject = ROOT.joinpath("pyproject.toml").read_text()
    assert 'smb-turbo = "supermariobrosnes_turbo.cli:main"' in pyproject
    assert "smb-turbo-train" not in pyproject


@pytest.mark.parametrize("launcher", ["train.py", "play.py"])
def test_root_launchers_delegate_to_package_cli(launcher: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / launcher), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Level1-1" in completed.stdout


def test_train_main_dispatches_beam(monkeypatch) -> None:
    from supermariobrosnes_turbo import beam_training

    monkeypatch.setattr(training, "resolve_state_name", lambda state, **_kwargs: state)
    monkeypatch.setattr(beam_training, "run", lambda args, _parser: int(args.algorithm == "beam"))

    assert training.main(["Level1-1", "--algorithm", "beam"]) == 1
