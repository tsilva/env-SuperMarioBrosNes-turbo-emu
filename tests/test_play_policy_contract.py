from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from supermariobrosnes_turbo import ACTION_SETS
from supermariobrosnes_turbo.jerk import ActionRun, JerkPolicy, policy_path_for_level

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import play_policy  # noqa: E402


def test_jerk_checkpoint_uses_native_lightweight_contract() -> None:
    args = play_policy.parse_args(["final_model.zip"])

    play_policy.apply_checkpoint_defaults(args, Path("final_model.zip"))

    assert args.backend == "native"
    assert args.scale == 2
    assert args.max_pool_frames is False
    assert args.crop_mode == "remove"


def test_explicit_stable_retro_backend_is_preserved() -> None:
    args = play_policy.parse_args(["policy.json", "--backend", "stable-retro"])

    play_policy.apply_checkpoint_defaults(args, Path("policy.json"))

    assert args.backend == "stable-retro"
    assert args.max_pool_frames is False
    assert args.crop_mode == "remove"


def test_stable_retro_supports_mask_crop_through_shared_preprocessing() -> None:
    args = play_policy.parse_args(
        ["policy.json", "--backend", "stable-retro", "--crop-mode", "mask"]
    )

    play_policy.apply_checkpoint_defaults(args, Path("policy.json"))

    assert args.backend == "stable-retro"
    assert args.crop_mode == "mask"


def test_level_counters_map_to_named_policy() -> None:
    assert play_policy.level_name_from_counters((0, 0)) == "Level1-1"
    assert play_policy.level_name_from_counters((0, 1)) == "Level1-2"
    assert play_policy.level_name_from_counters((1, 0)) == "Level2-1"


def test_player_activates_policy_for_new_level(tmp_path: Path) -> None:
    target_path = policy_path_for_level("Level1-2", runs_root=tmp_path)
    target_path.parent.mkdir(parents=True)
    JerkPolicy(
        action_names=ACTION_SETS["simple"],
        action_runs=(ActionRun(2, 1), ActionRun(3, 1)),
        fallback_action=0,
    ).save(target_path)

    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.args = SimpleNamespace(
        action_set="simple",
        level_policy_root=tmp_path,
    )
    player.action_names = ACTION_SETS["simple"]

    assert player.activate_level_policy((0, 1))
    assert player.current_policy_level == "Level1-2"
    assert player.model_path == target_path
    action, _state = player.model.predict(np.zeros((1, 1), dtype=np.uint8))
    assert action.tolist() == [2]


def test_player_keeps_current_policy_when_next_level_is_untrained(
    tmp_path: Path,
) -> None:
    current = object()
    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.args = SimpleNamespace(
        action_set="simple",
        level_policy_root=tmp_path,
    )
    player.action_names = ACTION_SETS["simple"]
    player.model = current

    assert not player.activate_level_policy((8, 8))
    assert player.model is current
