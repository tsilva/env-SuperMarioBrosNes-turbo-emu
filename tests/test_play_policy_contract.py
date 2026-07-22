from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from supermariobrosnes_turbo import ACTION_SETS
from supermariobrosnes_turbo.jerk import ActionRun, JerkPolicy, policy_path_for_state

from supermariobrosnes_turbo import manual_playback
from supermariobrosnes_turbo import policy_playback as play_policy


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


def test_uncapped_playback_disables_delay_and_renderer_vsync() -> None:
    args = play_policy.parse_args(["policy.json", "--fps", "max"])
    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.frame_delay_s = manual_playback.frame_delay_for_fps(args.fps)

    player.sleep_until_next_frame()

    assert args.fps is None
    assert player.frame_delay_s is None
    assert (
        manual_playback.renderer_flags_for_fps(args.fps)
        == manual_playback.SDL_RENDERER_ACCELERATED
    )
    assert (
        manual_playback.renderer_flags_for_fps(60)
        & manual_playback.SDL_RENDERER_PRESENTVSYNC
    )


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        (
            "raw",
            {
                "obs_grayscale": False,
                "frame_stack": 1,
                "maxpool_last_two": False,
                "obs_crop": None,
                "obs_resize": None,
            },
        ),
        (
            "preprocessed",
            {
                "obs_grayscale": True,
                "frame_stack": 4,
                "maxpool_last_two": False,
                "obs_crop": (32, 0, 0, 0),
                "obs_resize": (84, 84),
            },
        ),
    ],
)
def test_native_view_configures_a_directly_displayable_environment(
    monkeypatch, view: str, expected: dict[str, object]
) -> None:
    captured: dict[str, object] = {}

    class FakeEnv:
        def __init__(self, _game: str, **kwargs: object) -> None:
            captured.update(kwargs)

        def seed(self, seed: int) -> None:
            captured["seed"] = seed

    monkeypatch.setattr(play_policy, "SuperMarioBrosNesTurboVecEnv", FakeEnv)
    args = play_policy.parse_args(
        ["policy.json", "--backend", "native", "--view", view]
    )
    play_policy.apply_checkpoint_defaults(args, Path("policy.json"))
    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.args = args
    player.current_state = args.state
    player.rom_path = Path("mario.nes")

    player.make_env()

    for key, value in expected.items():
        assert captured[key] == value


def test_native_raw_view_displays_temporally_stable_rgb_render() -> None:
    rendered = np.full((224, 240, 3), 37, dtype=np.uint8)
    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.args = SimpleNamespace(backend="native", view="raw")
    player.env = SimpleNamespace(render=lambda: rendered)
    player.obs = np.zeros((1, 3, 224, 240), dtype=np.uint8)

    assert player.current_display_frame() is rendered


def test_stable_retro_raw_view_uses_native_visible_rgb_dimensions(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeEnv:
        def seed(self, seed: int) -> None:
            captured["seed"] = seed

    def create_env(**kwargs: object) -> FakeEnv:
        captured.update(kwargs)
        return FakeEnv()

    monkeypatch.setattr(play_policy, "create_stable_retro_vector_env", create_env)
    args = play_policy.parse_args(
        ["policy.json", "--backend", "stable-retro", "--view", "raw"]
    )
    play_policy.apply_checkpoint_defaults(args, Path("policy.json"))
    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.args = args
    player.current_state = args.state
    player.rom_path = Path("mario.nes")

    player.make_env()

    preprocessing = captured["preprocessing"]
    assert isinstance(preprocessing, play_policy.PreprocessingConfig)
    assert preprocessing.grayscale is False
    assert preprocessing.frame_stack == 1
    assert preprocessing.maxpool_last_two is False
    assert preprocessing.crop_top == 0
    assert preprocessing.crop_bottom == 0
    assert preprocessing.resize_width == 240
    assert preprocessing.resize_height == 224


def test_level_counters_map_to_named_policy() -> None:
    assert play_policy.level_name_from_counters((0, 0)) == "Level1-1"
    assert play_policy.level_name_from_counters((0, 1)) == "Level1-2"
    assert play_policy.level_name_from_counters((1, 0)) == "Level2-1"


def test_player_activates_policy_for_new_level(tmp_path: Path) -> None:
    target_path = policy_path_for_state("Level1-2", runs_root=tmp_path)
    target_path.parent.mkdir(parents=True)
    JerkPolicy(
        action_names=ACTION_SETS["basic"],
        action_runs=(ActionRun(2, 1), ActionRun(3, 1)),
        fallback_action=0,
    ).save(target_path)

    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.args = SimpleNamespace(
        action_set="basic",
        backend="native",
        level_policy_root=tmp_path,
    )
    player.requested_action_set = "basic"
    player.action_names = ACTION_SETS["basic"]

    assert player.activate_level_policy((0, 1))
    assert player.current_policy_level == "Level1-2"
    assert player.model_path == target_path
    action, _state = player.model.predict(np.zeros((1, 1), dtype=np.uint8))
    assert action.tolist() == [2]


def test_player_infers_each_automatic_level_policy_action_set(
    tmp_path: Path,
) -> None:
    target_path = policy_path_for_state("Level8-4", runs_root=tmp_path)
    target_path.parent.mkdir(parents=True)
    JerkPolicy(
        action_names=ACTION_SETS["standard"],
        action_runs=(ActionRun(7, 1),),
        fallback_action=0,
    ).save(target_path)

    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.args = SimpleNamespace(
        action_set=None,
        backend="native",
        level_policy_root=tmp_path,
    )
    player.requested_action_set = None
    player.action_names = ACTION_SETS["basic"]

    assert player.activate_level_policy((7, 3))
    assert player.action_names == ACTION_SETS["standard"]
    assert player.action_masks.shape == (8, 9)
    action, _state = player.model.predict(np.zeros((1, 1), dtype=np.uint8))
    assert action.tolist() == [7]


def test_player_rejects_automatic_policy_outside_explicit_action_set(
    tmp_path: Path,
) -> None:
    target_path = policy_path_for_state("Level8-4", runs_root=tmp_path)
    target_path.parent.mkdir(parents=True)
    JerkPolicy(
        action_names=ACTION_SETS["standard"],
        action_runs=(ActionRun(7, 1),),
        fallback_action=0,
    ).save(target_path)

    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.args = SimpleNamespace(
        action_set="basic",
        backend="native",
        level_policy_root=tmp_path,
    )
    player.requested_action_set = "basic"
    player.action_names = ACTION_SETS["basic"]

    with pytest.raises(ValueError, match="does not match --action-set='basic'"):
        player.activate_level_policy((7, 3))


def test_player_keeps_current_policy_when_next_level_is_untrained(
    tmp_path: Path,
) -> None:
    current = object()
    player = play_policy.SdlPolicyPlayer.__new__(play_policy.SdlPolicyPlayer)
    player.args = SimpleNamespace(
        action_set="basic",
        level_policy_root=tmp_path,
    )
    player.action_names = ACTION_SETS["basic"]
    player.model = current

    assert not player.activate_level_policy((8, 8))
    assert player.model is current
