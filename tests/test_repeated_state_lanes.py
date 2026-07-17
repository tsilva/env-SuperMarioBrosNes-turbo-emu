from __future__ import annotations

from pathlib import Path

import numpy as np

from supermariobrosnes_turbo import (
    ACTION_MEANINGS,
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    action_batch,
)
from rom_helpers import require_rom
GROUP_STATES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")


def make_env(rom_path: Path, state: str | list[str], num_envs: int) -> SuperMarioBrosNesTurboVecEnv:
    state_config = (
        {"state_catalog": tuple(dict.fromkeys(state))}
        if isinstance(state, list)
        else {"state": state}
    )
    return SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        **state_config,
        rom_path=rom_path,
        num_envs=num_envs,
        use_restricted_actions=Actions.ALL,
        frame_skip=4,
        frame_stack=4,
        obs_grayscale=True,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_resize_algorithm="area",
        obs_layout="chw",
    )


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


def reset_obs(
    env: SuperMarioBrosNesTurboVecEnv,
    state_indices: np.ndarray | None = None,
) -> np.ndarray:
    options = None if state_indices is None else {"state_indices": state_indices}
    obs, _infos = env.reset(options=options)
    return obs


def test_repeated_state_lanes_match_independent_lane_references() -> None:
    rom_path = require_rom()
    lane_states = [GROUP_STATES[index % len(GROUP_STATES)] for index in range(16)]
    vector_env = make_env(rom_path, lane_states, num_envs=16)
    refs = [make_env(rom_path, state, num_envs=1) for state in GROUP_STATES]

    lane_indices = np.asarray(
        [GROUP_STATES.index(state) for state in lane_states], dtype=np.int32
    )
    vector_obs = reset_obs(vector_env, lane_indices)
    ref_obs = [reset_obs(ref) for ref in refs]
    for lane, state in enumerate(lane_states):
        ref_index = GROUP_STATES.index(state)
        np.testing.assert_array_equal(vector_obs[lane], ref_obs[ref_index][0])

    for action_name in ("noop", "noop", "right", "noop"):
        vector_result = step_arrays(vector_env, action_batch(action_name, 16))
        ref_results = [
            step_arrays(ref, action_batch(action_name, 1)) for ref in refs
        ]
        for lane, state in enumerate(lane_states):
            ref_index = GROUP_STATES.index(state)
            for actual_array, expected_array in zip(vector_result, ref_results[ref_index]):
                np.testing.assert_array_equal(actual_array[lane], expected_array[0])


def test_repeated_state_lanes_match_references_after_autoreset() -> None:
    rom_path = require_rom()
    lane_states = [GROUP_STATES[index % len(GROUP_STATES)] for index in range(16)]
    vector_env = make_env(rom_path, lane_states, num_envs=16)
    refs = [make_env(rom_path, state, num_envs=1) for state in GROUP_STATES]

    lane_indices = np.asarray(
        [GROUP_STATES.index(state) for state in lane_states], dtype=np.int32
    )
    reset_obs(vector_env, lane_indices)
    for ref in refs:
        reset_obs(ref)

    saw_done = False
    for _ in range(900):
        vector_result = step_arrays(vector_env, action_batch("noop", 16))
        ref_results = [step_arrays(ref, action_batch("noop", 1)) for ref in refs]
        saw_done |= bool(np.any(vector_result[2]) or np.any(vector_result[3]))
        for lane, state in enumerate(lane_states):
            ref_index = GROUP_STATES.index(state)
            for actual_array, expected_array in zip(vector_result, ref_results[ref_index]):
                np.testing.assert_array_equal(actual_array[lane], expected_array[0])
        if saw_done:
            return

    raise AssertionError("expected repeated-state noop rollout to trigger autoreset")


def test_repeated_state_lanes_match_independent_env_with_divergent_actions() -> None:
    rom_path = require_rom()
    lane_states = [GROUP_STATES[index % len(GROUP_STATES)] for index in range(16)]
    vector_env = make_env(rom_path, lane_states, num_envs=16)
    independent = make_env(rom_path, lane_states, num_envs=16)
    lane_indices = np.asarray(
        [GROUP_STATES.index(state) for state in lane_states], dtype=np.int32
    )
    vector_env.reset(options={"state_indices": lane_indices})
    independent.reset(options={"state_indices": lane_indices})

    actions = action_batch("noop", 16)
    actions[4] = action_batch("right", 1)[0]

    assert_step_equal(step_arrays(vector_env, actions), step_arrays(independent, actions))
    assert_step_equal(step_arrays(vector_env, actions), step_arrays(independent, actions))
