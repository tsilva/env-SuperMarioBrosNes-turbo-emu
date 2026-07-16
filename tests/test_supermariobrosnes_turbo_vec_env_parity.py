from __future__ import annotations

import importlib.metadata
import inspect

import numpy as np
import pytest

from rom_helpers import require_rom
from scripts.benchmark_sps import (
    GAME,
    PreprocessingConfig,
    create_stable_retro_vector_env,
    named_action_mask,
)
from supermariobrosnes_turbo import (
    Actions,
    NES_BUTTONS,
    SuperMarioBrosNesTurboVecEnv,
)
from supermariobrosnes_turbo import _supermariobrosnes_turbo as native


def test_oracle_is_upstream_stable_retro() -> None:
    assert importlib.metadata.version("stable-retro") == "1.0.1"


def test_native_binding_removed_lifecycle_and_policy_mutators() -> None:
    for name in (
        "done_on_info",
        "terminal_observations",
        "terminal_infos",
        "set_initial_states",
    ):
        assert not hasattr(native._RetroVecEnv, name)


def test_public_signature_preserves_vector_features() -> None:
    params = inspect.signature(SuperMarioBrosNesTurboVecEnv).parameters
    for name in (
        "state",
        "num_threads",
        "obs_copy",
        "obs_resize",
        "obs_crop",
        "obs_grayscale",
        "frame_skip",
        "frame_stack",
        "maxpool_last_two",
        "noop_reset_max",
        "sticky_action_prob",
        "reward_clip",
        "info_filter",
    ):
        assert name in params
    for name in ("done_on", "autoreset_mode"):
        assert name not in params


@pytest.mark.retro_oracle
@pytest.mark.parametrize("num_envs", [1, 4])
def test_upstream_oracle_exact_short_sequence_parity(num_envs: int) -> None:
    rom_path = require_rom()
    states = [f"Level1-{index + 1}" for index in range(num_envs)]
    preprocessing = PreprocessingConfig(4, 4, True, 32, 0, "mask", 84, 84)
    retro_env = create_stable_retro_vector_env(
        rom_path=rom_path,
        lane_state_names=states,
        preprocessing=preprocessing,
        asynchronous=True,
    )
    fast_env = SuperMarioBrosNesTurboVecEnv(
        GAME,
        state=states,
        rom_path=rom_path,
        num_envs=num_envs,
        use_restricted_actions=Actions.ALL,
        frame_skip=4,
        frame_stack=4,
        obs_grayscale=True,
        obs_crop=(32, 0, 0, 0),
        obs_crop_mode="mask",
        obs_resize=(84, 84),
        obs_resize_algorithm="area",
        obs_layout="chw",
        maxpool_last_two=False,
    )
    action_names = ("noop", "right", "right_b", "right_a") * 4
    try:
        retro_obs, _ = retro_env.reset()
        fast_obs, _ = fast_env.reset()
        assert retro_obs.shape == fast_obs.shape == (num_envs, 4, 84, 84)
        assert retro_obs.dtype == fast_obs.dtype == np.uint8
        np.testing.assert_array_equal(retro_obs, fast_obs)
        public_masks = [named_action_mask(name, retro_env.buttons) for name in action_names]
        button_indices = {
            button: index for index, button in enumerate(retro_env.buttons) if button is not None
        }
        for buttons in (
            ("UP", "A"),
            ("DOWN", "B"),
            ("LEFT", "A", "B"),
            ("SELECT", "START"),
            ("UP", "DOWN", "LEFT", "RIGHT", "A", "B"),
        ):
            mask = np.zeros((len(retro_env.buttons),), dtype=np.uint8)
            for button in buttons:
                mask[button_indices[button]] = 1
            public_masks.append(mask)
        for public_mask in public_masks:
            retro_action = np.repeat(public_mask[None, :], num_envs, axis=0)
            fast_action = retro_action.copy()
            retro_obs, retro_rewards, retro_terminated, retro_truncated, _ = retro_env.step(
                retro_action
            )
            fast_obs, fast_rewards, fast_terminated, fast_truncated, _ = fast_env.step(
                fast_action
            )
            np.testing.assert_array_equal(retro_obs, fast_obs)
            np.testing.assert_array_equal(retro_rewards, fast_rewards)
            np.testing.assert_array_equal(retro_terminated, fast_terminated)
            np.testing.assert_array_equal(retro_truncated, fast_truncated)
    finally:
        retro_env.close()
        fast_env.close()


def test_action_and_layout_runtime_smoke() -> None:
    env = SuperMarioBrosNesTurboVecEnv(
        GAME,
        state="Level1-1",
        rom_path=require_rom(),
        num_envs=1,
        use_restricted_actions=Actions.ALL,
        obs_layout="hwc",
        obs_grayscale=False,
        obs_resize=(96, 112),
        obs_crop=(16, 8, 4, 4),
        frame_skip=2,
        frame_stack=2,
        maxpool_last_two=True,
        noop_reset_max=2,
        sticky_action_prob=0.1,
    )
    try:
        obs, _ = env.reset(seed=5)
        assert obs.shape == (1, 96, 112, 6)
        actions = np.zeros((1, len(NES_BUTTONS)), dtype=np.uint8)
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        assert next_obs.shape == obs.shape
        assert rewards.shape == terminated.shape == truncated.shape == (1,)
        assert "lives" in infos and "levelHi" in infos and "levelLo" in infos
    finally:
        env.close()
