from __future__ import annotations

from pathlib import Path

import numpy as np

from env_supermariobrosnes_turbo_emu import ACTION_BUTTONS, BUTTON_TO_INDEX, NES_BUTTONS, Actions
from env_supermariobrosnes_turbo_emu import EnvSuperMarioBrosNesTurboEmuVecEnv, default_rom_path, resolve_required_rom_path


DEFAULT_ROM = default_rom_path()
INFO_ARRAYS = (
    ("x_pos", np.uint16),
    ("coins", np.uint8),
    ("level_hi", np.int16),
    ("level_lo", np.int16),
    ("lives", np.int16),
    ("score", np.uint32),
    ("scrolling", np.int16),
    ("time", np.uint16),
    ("xscroll_hi", np.uint8),
    ("xscroll_lo", np.uint8),
)


def make_env(rom_path: Path, num_envs: int) -> EnvSuperMarioBrosNesTurboEmuVecEnv:
    return EnvSuperMarioBrosNesTurboEmuVecEnv(
        "SuperMarioBros-Nes-v0",
        rom_path=rom_path.expanduser(),
        num_envs=num_envs,
        use_restricted_actions=Actions.ALL,
        frame_skip=4,
        obs_grayscale=True,
        frame_stack=4,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_resize_algorithm="area",
        obs_layout="chw",
        obs_copy="copy",
    )


def action_batch(names: str | list[str], num_envs: int) -> np.ndarray:
    if isinstance(names, str):
        names = [names] * num_envs
    actions = np.zeros((num_envs, len(NES_BUTTONS)), dtype=np.uint8)
    for env_idx, name in enumerate(names):
        for button in ACTION_BUTTONS[name]:
            actions[env_idx, BUTTON_TO_INDEX[button]] = 1
    return actions


def step_uniform(
    env: EnvSuperMarioBrosNesTurboEmuVecEnv, action: str, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actions = action_batch(action, env.num_envs)
    result = None
    for _ in range(count):
        result = env.step(actions)[:4]
    assert result is not None
    return result


def assert_uniform_info(vec: EnvSuperMarioBrosNesTurboEmuVecEnv, one: EnvSuperMarioBrosNesTurboEmuVecEnv) -> None:
    for attr, dtype in INFO_ARRAYS:
        vec_values = getattr(vec, attr)
        one_values = getattr(one, attr)
        np.testing.assert_array_equal(
            vec_values,
            np.full((vec.num_envs,), one_values[0], dtype=dtype),
        )


def assert_lane_info(vec: EnvSuperMarioBrosNesTurboEmuVecEnv, lane: int, ref: EnvSuperMarioBrosNesTurboEmuVecEnv) -> None:
    for attr, _dtype in INFO_ARRAYS:
        assert getattr(vec, attr)[lane] == getattr(ref, attr)[0]


def reset_obs(env: EnvSuperMarioBrosNesTurboEmuVecEnv) -> np.ndarray:
    obs, _infos = env.reset()
    return obs


def check_uniform_lanes_match_single_lane_reference(rom_path: Path) -> None:
    vec = make_env(rom_path, 16)
    one = make_env(rom_path, 1)
    obs_vec = reset_obs(vec)
    obs_one = reset_obs(one)
    assert obs_vec.shape == (16, 4, 84, 84)
    assert obs_vec.dtype == np.uint8
    np.testing.assert_array_equal(obs_vec[0], obs_one[0])

    for action, count in (
        ("noop", 30),
        ("start", 8),
        ("noop", 30),
        ("noop", 20),
        ("right", 5),
        ("noop", 10),
    ):
        vec_obs, vec_rewards, vec_terminated, vec_truncated = step_uniform(vec, action, count)
        one_obs, one_rewards, one_terminated, one_truncated = step_uniform(one, action, count)
        np.testing.assert_array_equal(vec_obs[0], one_obs[0])
        np.testing.assert_array_equal(vec_rewards, np.full((16,), one_rewards[0], dtype=np.float32))
        np.testing.assert_array_equal(
            vec_terminated, np.full((16,), one_terminated[0], dtype=np.bool_)
        )
        np.testing.assert_array_equal(
            vec_truncated, np.full((16,), one_truncated[0], dtype=np.bool_)
        )
        assert_uniform_info(vec, one)
        for lane in range(1, 16):
            np.testing.assert_array_equal(vec_obs[0], vec_obs[lane])


def check_divergent_actions_match_independent_lane_references(rom_path: Path) -> None:
    vec = make_env(rom_path, 8)
    refs = [make_env(rom_path, 1) for _ in range(8)]
    vec.reset()
    for ref in refs:
        ref.reset()

    for action, count in (("noop", 30), ("start", 8), ("noop", 30), ("noop", 5)):
        step_uniform(vec, action, count)
        for ref in refs:
            step_uniform(ref, action, count)

    action_names = ["noop", "right", "a", "start", "noop", "right", "a", "start"]
    actions = action_batch(action_names, 8)
    obs, rewards, terminated, truncated = vec.step(actions)[:4]
    for lane, ref in enumerate(refs):
        ref_obs, ref_rewards, ref_terminated, ref_truncated = ref.step(action_batch(action_names[lane], 1))[:4]
        np.testing.assert_array_equal(obs[lane], ref_obs[0])
        assert rewards[lane] == ref_rewards[0]
        assert terminated[lane] == ref_terminated[0]
        assert truncated[lane] == ref_truncated[0]
        assert_lane_info(vec, lane, ref)


def main() -> None:
    rom_path = resolve_required_rom_path(DEFAULT_ROM)
    check_uniform_lanes_match_single_lane_reference(rom_path)
    check_divergent_actions_match_independent_lane_references(rom_path)
    print("equivalence=ok obs_shape=(16, 4, 84, 84) obs_dtype=uint8")


if __name__ == "__main__":
    main()
