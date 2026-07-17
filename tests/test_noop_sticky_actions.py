from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from supermariobrosnes_turbo import (
    ACTION_BUTTONS,
    ACTION_MEANINGS,
    BUTTON_TO_INDEX,
    NES_BUTTONS,
    Actions,
    SuperMarioBrosNesTurboVecEnv,
)
from rom_helpers import require_rom


def action_masks(names: list[str]) -> np.ndarray:
    masks = np.zeros((len(names), len(NES_BUTTONS)), dtype=np.uint8)
    for env_idx, name in enumerate(names):
        for button in ACTION_BUTTONS[name]:
            masks[env_idx, BUTTON_TO_INDEX[button]] = 1
    return masks


def make_env(
    rom_path: Path,
    *,
    num_envs: int = 4,
    frame_skip: int = 1,
    frame_stack: int = 1,
    seed: int = 1,
    noop_reset_max: int = 0,
    sticky_action_prob: float = 0.0,
) -> SuperMarioBrosNesTurboVecEnv:
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        rom_path=rom_path,
        num_envs=num_envs,
        use_restricted_actions=Actions.ALL,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        obs_grayscale=True,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_resize_algorithm="area",
        obs_layout="chw",
        noop_reset_max=noop_reset_max,
        sticky_action_prob=sticky_action_prob,
    )
    env.seed(seed)
    return env


def assert_step_equal(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    assert len(actual) == len(expected)
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)


def step_arrays(
    env: SuperMarioBrosNesTurboVecEnv,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs, rewards, terminated, truncated, _infos = env.step(actions)
    return obs, rewards, terminated, truncated


def reset_obs(env: SuperMarioBrosNesTurboVecEnv) -> np.ndarray:
    obs, _infos = env.reset()
    return obs


def test_noop_and_sticky_validation_runs_before_rom_load() -> None:
    missing_rom = "/definitely/missing/SuperMarioBros.nes"
    with pytest.raises(ValueError, match="noop_reset_max"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", rom_path=missing_rom, noop_reset_max=-1)
    with pytest.raises(ValueError, match="sticky_action_prob"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", rom_path=missing_rom, sticky_action_prob=-0.01)
    with pytest.raises(ValueError, match="sticky_action_prob"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", rom_path=missing_rom, sticky_action_prob=1.01)


def test_noop_and_sticky_properties_are_exposed() -> None:
    rom_path = require_rom()
    env = make_env(rom_path, noop_reset_max=2, sticky_action_prob=0.25)

    assert env.noop_reset_max == 2
    assert env.sticky_action_prob == pytest.approx(0.25)


def test_noop_reset_max_one_matches_explicit_single_noop_step() -> None:
    rom_path = require_rom()
    baseline = make_env(rom_path, num_envs=1, seed=1, noop_reset_max=0)
    reset_noop = make_env(rom_path, num_envs=1, seed=1, noop_reset_max=1)

    baseline.reset()
    expected_obs = step_arrays(baseline, action_masks(["noop"]))[0]
    actual_obs = reset_obs(reset_noop)

    np.testing.assert_array_equal(actual_obs, expected_obs)


def test_noop_reset_is_seed_deterministic_across_envs() -> None:
    rom_path = require_rom()
    first = make_env(rom_path, num_envs=8, seed=17, noop_reset_max=2)
    second = make_env(rom_path, num_envs=8, seed=17, noop_reset_max=2)

    np.testing.assert_array_equal(reset_obs(first), reset_obs(second))


def test_full_sticky_actions_reuse_initial_noop_action() -> None:
    rom_path = require_rom()
    sticky = make_env(rom_path, num_envs=4, seed=7, sticky_action_prob=1.0)
    explicit_noop = make_env(rom_path, num_envs=4, seed=7)

    np.testing.assert_array_equal(reset_obs(sticky), reset_obs(explicit_noop))
    right_actions = action_masks(["right"] * 4)
    noop_actions = action_masks(["noop"] * 4)
    for _ in range(8):
        assert_step_equal(step_arrays(sticky, right_actions), step_arrays(explicit_noop, noop_actions))


def test_stochastic_sticky_actions_are_seed_deterministic() -> None:
    rom_path = require_rom()
    first = make_env(rom_path, num_envs=8, seed=1234, sticky_action_prob=0.5)
    second = make_env(rom_path, num_envs=8, seed=1234, sticky_action_prob=0.5)
    action_trace = [
        action_masks([ACTION_MEANINGS[(step + lane) % len(ACTION_MEANINGS)] for lane in range(8)])
        for step in range(12)
    ]

    np.testing.assert_array_equal(reset_obs(first), reset_obs(second))
    for actions in action_trace:
        assert_step_equal(step_arrays(first, actions), step_arrays(second, actions))


def test_disabled_noop_and_sticky_match_default_env() -> None:
    rom_path = require_rom()
    default = make_env(rom_path, num_envs=8, seed=99)
    explicit_disabled = make_env(
        rom_path,
        num_envs=8,
        seed=99,
        noop_reset_max=0,
        sticky_action_prob=0.0,
    )
    action_trace = [
        action_masks([ACTION_MEANINGS[(step * 3 + lane) % len(ACTION_MEANINGS)] for lane in range(8)])
        for step in range(10)
    ]

    np.testing.assert_array_equal(reset_obs(default), reset_obs(explicit_disabled))
    for actions in action_trace:
        assert_step_equal(step_arrays(default, actions), step_arrays(explicit_disabled, actions))
