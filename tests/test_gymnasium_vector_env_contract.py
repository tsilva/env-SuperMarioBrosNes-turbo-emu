from __future__ import annotations

import inspect

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv

from rom_helpers import require_rom
from supermariobrosnes_turbo import (
    Actions,
    NES_BUTTONS,
    SuperMarioBrosNesTurboVecEnv,
    list_available_states,
)
from supermariobrosnes_turbo.env import _normalize_initial_state_config

CANONICAL_LEVEL_STATES = tuple(
    f"Level{world}-{level}"
    for world in range(1, 9)
    for level in range(1, 5)
)


def make_env(**kwargs: object) -> SuperMarioBrosNesTurboVecEnv:
    return SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state=kwargs.pop("state", "Level1-1"),
        rom_path=require_rom(),
        num_envs=kwargs.pop("num_envs", 2),
        use_restricted_actions=Actions.ALL,
        frame_skip=kwargs.pop("frame_skip", 4),
        frame_stack=kwargs.pop("frame_stack", 1),
        obs_grayscale=kwargs.pop("obs_grayscale", True),
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_resize_algorithm="area",
        obs_layout=kwargs.pop("obs_layout", "chw"),
        **kwargs,
    )


def noop(num_envs: int) -> np.ndarray:
    return np.zeros((num_envs, len(NES_BUTTONS)), dtype=np.uint8)


def test_public_surface_is_manual_reset_only() -> None:
    signature = inspect.signature(SuperMarioBrosNesTurboVecEnv)
    for name in ("done_on", "autoreset_mode", "done_on_info"):
        assert name not in signature.parameters
    for name in ("set_state_policy", "set_state_sampling_weights", "state_sampling_weights"):
        assert not hasattr(SuperMarioBrosNesTurboVecEnv, name)
    assert SuperMarioBrosNesTurboVecEnv.metadata["autoreset_mode"] is AutoresetMode.DISABLED
    with pytest.raises(TypeError, match="done_on"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", done_on=[])
    with pytest.raises(TypeError, match="autoreset_mode"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", autoreset_mode="Disabled")


def test_named_simple_action_set_is_discrete_and_maps_indices() -> None:
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        rom_path=require_rom(),
        num_envs=2,
        action_set="simple",
        frame_skip=1,
        frame_stack=1,
        obs_grayscale=True,
        obs_resize=(1, 1),
    )
    try:
        assert env.action_set == "simple"
        assert env.action_meanings == (
            "noop",
            "right",
            "right_b",
            "right_a",
            "right_a_b",
            "a",
            "left",
        )
        assert env.single_action_space == spaces.Discrete(7)
        assert env.action_space == spaces.MultiDiscrete([7, 7])
        env.reset(seed=0)
        env.step(np.asarray([0, 6], dtype=np.int64))
        with pytest.raises(ValueError, match="action_set='simple'"):
            env.step(np.asarray([0, 7], dtype=np.int64))
    finally:
        env.close()


def test_constructor_state_forms_and_packaged_inventory() -> None:
    assert {"Level1-1", "Level8-4", "Level2-1-clouds"} <= set(list_available_states())
    _states, names, weights = _normalize_initial_state_config(
        {"Level1-1": 0.0, "Level1-2": 3.0}, None, num_envs=2
    )
    assert names == ("Level1-1", "Level1-2")
    assert weights == [0.0, 1.0]
    with pytest.raises(ValueError, match="positive finite"):
        _normalize_initial_state_config(
            {"Level1-1": 0.0, "Level1-2": 0.0}, None, num_envs=2
        )


def test_all_canonical_packaged_states_load_and_step() -> None:
    assert set(CANONICAL_LEVEL_STATES) <= set(list_available_states())
    env = make_env(
        state=list(CANONICAL_LEVEL_STATES),
        num_envs=len(CANONICAL_LEVEL_STATES),
        frame_skip=1,
    )
    expected_level_hi = np.repeat(np.arange(8, dtype=np.int32), 4)
    expected_level_lo = np.tile(np.arange(4, dtype=np.int32), 8)
    try:
        obs, infos = env.reset(seed=0)
        assert obs.shape == (len(CANONICAL_LEVEL_STATES), 1, 84, 84)
        assert env.active_states() == CANONICAL_LEVEL_STATES
        np.testing.assert_array_equal(infos["levelHi"], expected_level_hi)
        np.testing.assert_array_equal(infos["levelLo"], expected_level_lo)

        _obs, _rewards, terminated, truncated, infos = env.step(
            noop(len(CANONICAL_LEVEL_STATES))
        )
        assert not np.any(terminated)
        assert not np.any(truncated)
        np.testing.assert_array_equal(infos["levelHi"], expected_level_hi)
        np.testing.assert_array_equal(infos["levelLo"], expected_level_lo)
    finally:
        env.close()


def test_reset_step_raw_signals_preprocessing_and_reward_clip() -> None:
    env = make_env(reward_clip=(-0.25, 0.25))
    try:
        assert isinstance(env, VectorEnv)
        assert env.autoreset_mode is AutoresetMode.DISABLED
        assert env.metadata["autoreset_mode"] is AutoresetMode.DISABLED
        obs, infos = env.reset(seed=123)
        assert obs.shape == (2, 1, 84, 84)
        assert "lives" in infos
        obs, rewards, terminated, truncated, infos = env.step(noop(2))
        assert obs.shape == (2, 1, 84, 84)
        assert rewards.dtype == np.float32
        assert np.all((-0.25 <= rewards) & (rewards <= 0.25))
        assert terminated.dtype == truncated.dtype == np.bool_
        for key in ("x_pos", "lives", "levelHi", "levelLo", "score", "time", "scrolling"):
            assert key in infos
            assert f"_{key}" in infos
    finally:
        env.close()


def test_masked_reset_isolates_unselected_lane_and_tracks_active_state() -> None:
    env = make_env(state=["Level1-1", "Level1-2"])
    try:
        obs, _ = env.reset(seed=9)
        env.step(noop(2))
        lane_zero = env._obs[0].copy()
        mask = np.array([False, True], dtype=np.bool_)
        starts = np.array([-1, 0], dtype=np.int32)
        reset_obs, infos = env.reset(options={"reset_mask": mask, "start_indices": starts})
        np.testing.assert_array_equal(env._obs[0], lane_zero)
        np.testing.assert_array_equal(reset_obs[0], lane_zero)
        assert env.active_states()[0] == "Level1-1"
        assert env.active_states()[1] == "Level1-1"
        assert bool(infos["_lives"][1])
        assert not bool(infos["_lives"][0])
    finally:
        env.close()


def test_reset_mask_validation() -> None:
    env = make_env(num_envs=1)
    try:
        with pytest.raises(TypeError, match="NumPy array"):
            env.reset(options={"reset_mask": [True]})
        with pytest.raises(TypeError, match="dtype"):
            env.reset(options={"reset_mask": np.array([1], dtype=np.int8)})
        with pytest.raises(ValueError, match="at least one"):
            env.reset(options={"reset_mask": np.array([False], dtype=np.bool_)})
    finally:
        env.close()


def test_safe_view_preserves_previous_observation() -> None:
    env = make_env(num_envs=1, obs_copy="safe_view")
    try:
        first, _ = env.reset()
        frozen = first.copy()
        env.step(noop(1))
        np.testing.assert_array_equal(first, frozen)
    finally:
        env.close()


def test_rgb_render_is_independent_of_policy_preprocessing() -> None:
    env = make_env(num_envs=1, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        frames = env.render()
        assert obs.shape == (1, 1, 84, 84)
        assert frames is not None
        assert frames.shape == (224, 240, 3)
    finally:
        env.close()


def test_native_game_over_blocks_until_manual_reset() -> None:
    env = make_env(num_envs=1, frame_skip=4, info_filter="all")
    try:
        env.reset()
        for _ in range(8_000):
            _obs, _reward, terminated, truncated, infos = env.step(noop(1))
            if bool(terminated[0] or truncated[0]):
                assert "final_obs" not in infos
                assert "final_info" not in infos
                with pytest.raises(RuntimeError, match="reset"):
                    env.step(noop(1))
                mask = np.array([True], dtype=np.bool_)
                env.reset(options={"reset_mask": mask})
                env.step(noop(1))
                return
        pytest.fail("native game-over did not occur within probe budget")
    finally:
        env.close()
