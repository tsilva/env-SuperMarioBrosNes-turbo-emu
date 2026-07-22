from __future__ import annotations

import gzip
import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys

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

CANONICAL_LEVEL_STATES = tuple(
    f"Level{world}-{level}"
    for world in range(1, 9)
    for level in range(1, 5)
)


def make_env(**kwargs: object) -> SuperMarioBrosNesTurboVecEnv:
    state_kwargs = (
        {}
        if "state_catalog" in kwargs and "state" not in kwargs
        else {"state": kwargs.pop("state", "Level1-1")}
    )
    return SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        **state_kwargs,
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
    for name in (
        "set_state_policy",
        "set_state_sampling_weights",
        "state_sampling_weights",
        "initial_state_names",
        "active_states",
    ):
        assert not hasattr(SuperMarioBrosNesTurboVecEnv, name)
    assert SuperMarioBrosNesTurboVecEnv.metadata["autoreset_mode"] is AutoresetMode.DISABLED
    with pytest.raises(TypeError, match="done_on"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", done_on=[])
    with pytest.raises(TypeError, match="autoreset_mode"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", autoreset_mode="Disabled")


@pytest.mark.parametrize("num_threads", [0, -1])
def test_num_threads_must_be_positive_before_rom_loading(num_threads: int) -> None:
    with pytest.raises(ValueError, match="num_threads must be > 0"):
        SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            rom_path="/definitely/missing/SuperMarioBros-Nes-v0.nes",
            num_threads=num_threads,
        )


def test_num_threads_reports_effective_native_limit() -> None:
    automatic = make_env(num_envs=4)
    sequential = make_env(num_envs=4, num_threads=1)
    capped = make_env(num_envs=2, num_threads=10_000)
    single_lane = make_env(num_envs=1, num_threads=10_000)
    try:
        assert 1 <= automatic.num_threads <= automatic.num_envs
        assert sequential.num_threads == 1
        assert 1 <= capped.num_threads <= capped.num_envs
        assert single_lane.num_threads == 1
        assert automatic._core.num_threads == automatic.num_threads
        assert sequential._core.num_threads == sequential.num_threads
        assert capped._core.num_threads == capped.num_threads
    finally:
        automatic.close()
        sequential.close()
        capped.close()
        single_lane.close()


def test_automatic_num_threads_honors_rayon_environment_in_fresh_process() -> None:
    child_env = os.environ.copy()
    child_env["RAYON_NUM_THREADS"] = "2"
    child_env["SMB_TEST_ROM"] = str(require_rom())
    script = """
import os
from supermariobrosnes_turbo import SuperMarioBrosNesTurboVecEnv

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    state="Level1-1",
    rom_path=os.environ["SMB_TEST_ROM"],
    num_envs=4,
    frame_skip=1,
    frame_stack=1,
    obs_grayscale=True,
    obs_resize=(1, 1),
)
try:
    print(env.num_threads)
finally:
    env.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "2"


def test_explicit_thread_counts_preserve_deterministic_lane_contract() -> None:
    states = ["Level1-1", "Level1-2", "Level1-3", "Level1-4"]
    sequential = make_env(state_catalog=states, num_envs=4, num_threads=1, frame_skip=1)
    parallel = make_env(state_catalog=states, num_envs=4, num_threads=4, frame_skip=1)

    def assert_result_equal(actual: tuple[object, ...], expected: tuple[object, ...]) -> None:
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            if isinstance(actual_value, dict):
                assert isinstance(expected_value, dict)
                assert actual_value.keys() == expected_value.keys()
                for key in actual_value:
                    np.testing.assert_array_equal(actual_value[key], expected_value[key])
            else:
                np.testing.assert_array_equal(actual_value, expected_value)

    try:
        assert_result_equal(sequential.reset(seed=73), parallel.reset(seed=73))
        actions = noop(4)
        right = NES_BUTTONS.index("RIGHT")
        button_a = NES_BUTTONS.index("A")
        for lane in range(4):
            actions[lane, right] = lane % 2
            actions[lane, button_a] = lane // 2
        for _ in range(8):
            assert_result_equal(sequential.step(actions), parallel.step(actions))

        mask = np.array([True, False, True, False], dtype=np.bool_)
        starts = np.array([1, -1, 0, -1], dtype=np.int32)
        options = {"reset_mask": mask, "state_indices": starts}
        assert_result_equal(sequential.reset(options=options), parallel.reset(options=options))
        np.testing.assert_array_equal(
            sequential.active_state_indices(), parallel.active_state_indices()
        )
        assert_result_equal(sequential.step(actions), parallel.step(actions))
    finally:
        sequential.close()
        parallel.close()


def test_named_basic_action_set_is_discrete_and_maps_indices() -> None:
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        rom_path=require_rom(),
        num_envs=2,
        use_restricted_actions="basic",
        frame_skip=1,
        frame_stack=1,
        obs_grayscale=True,
        obs_resize=(1, 1),
    )
    try:
        assert env.action_preset == "basic"
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
        with pytest.raises(ValueError, match="action_preset='basic'"):
            env.step(np.asarray([0, 7], dtype=np.int64))
    finally:
        env.close()


def test_removed_constructor_action_set_is_rejected() -> None:
    with pytest.raises(TypeError, match="action_set"):
        SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            action_set="basic",
        )


def test_constructor_state_forms_and_packaged_inventory() -> None:
    assert {"Level1-1", "Level8-4", "Level2-1-clouds"} <= set(list_available_states())
    env = make_env(state_catalog=("Level1-1", "Level1-2"))
    try:
        assert env.state_catalog == ("Level1-1", "Level1-2")
        np.testing.assert_array_equal(env.active_state_indices(), [0, 0])
    finally:
        env.close()
    with pytest.raises(TypeError, match="single state"):
        make_env(state=["Level1-1", "Level1-2"])
    with pytest.raises(TypeError, match="single state"):
        make_env(state={"Level1-1": 0.25, "Level1-2": 0.75})
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_env(state="Level1-1", state_catalog=("Level1-1",))
    with pytest.raises(ValueError, match="at least one"):
        make_env(state_catalog=())
    with pytest.raises(ValueError, match="duplicate"):
        make_env(state_catalog=("Level1-1", "Level1-1"))
    with pytest.raises(FileNotFoundError, match="could not resolve"):
        make_env(state_catalog=("DefinitelyMissingState",))


def test_all_canonical_packaged_states_load_and_step() -> None:
    assert set(CANONICAL_LEVEL_STATES) <= set(list_available_states())
    env = make_env(
        state_catalog=CANONICAL_LEVEL_STATES,
        num_envs=len(CANONICAL_LEVEL_STATES),
        frame_skip=1,
    )
    expected_level_hi = np.repeat(np.arange(8, dtype=np.int32), 4)
    expected_level_lo = np.tile(np.arange(4, dtype=np.int32), 8)
    try:
        state_indices = np.arange(len(CANONICAL_LEVEL_STATES), dtype=np.int32)
        obs, infos = env.reset(seed=0, options={"state_indices": state_indices})
        assert obs.shape == (len(CANONICAL_LEVEL_STATES), 1, 84, 84)
        assert env.state_catalog == CANONICAL_LEVEL_STATES
        np.testing.assert_array_equal(env.active_state_indices(), state_indices)
        np.testing.assert_array_equal(infos["state_index"], state_indices)
        assert np.all(infos["_state_index"])
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
    env = make_env(state_catalog=("Level1-1", "Level1-2"))
    try:
        obs, _ = env.reset(
            seed=9,
            options={"state_indices": np.array([0, 1], dtype=np.int32)},
        )
        env.step(noop(2))
        lane_zero = env._obs[0].copy()
        mask = np.array([False, True], dtype=np.bool_)
        starts = np.array([-1, 0], dtype=np.int32)
        reset_obs, infos = env.reset(options={"reset_mask": mask, "state_indices": starts})
        np.testing.assert_array_equal(env._obs[0], lane_zero)
        np.testing.assert_array_equal(reset_obs[0], lane_zero)
        np.testing.assert_array_equal(env.active_state_indices(), [0, 0])
        np.testing.assert_array_equal(infos["state_index"], [0, 0])
        np.testing.assert_array_equal(infos["_state_index"], mask)
        assert bool(infos["_lives"][1])
        assert not bool(infos["_lives"][0])
    finally:
        env.close()


def test_live_snapshots_support_mixed_resets_cross_lane_fanout_and_exact_replay() -> None:
    env = make_env(
        state_catalog=("Level1-1", "Level1-2"),
        num_envs=3,
        frame_skip=1,
        frame_stack=2,
    )
    try:
        env.reset(
            options={"state_indices": np.asarray([0, 1, 0], dtype=np.int32)}
        )
        env.step(noop(3))
        handles = env.capture_snapshots(
            np.asarray([True, False, False], dtype=np.bool_)
        )
        assert handles[0] is not None
        assert handles[0].nbytes > 0
        assert handles[1:] == (None, None)
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(handles[0])

        env.step(noop(3))
        mask = np.asarray([True, True, True], dtype=np.bool_)
        starts = np.asarray([-1, 1, -1], dtype=np.int32)
        reset_obs, infos = env.reset(
            options={
                "reset_mask": mask,
                "state_indices": starts,
                "snapshots": [handles[0], None, handles[0]],
            }
        )
        np.testing.assert_array_equal(reset_obs[0], reset_obs[2])
        np.testing.assert_array_equal(env.active_state_indices(), [0, 1, 0])
        assert infos["start_source"].tolist() == [
            "snapshot",
            "environment",
            "snapshot",
        ]
        np.testing.assert_array_equal(infos["_start_source"], mask)

        actions = noop(3)
        actions[0, NES_BUTTONS.index("RIGHT")] = 1
        actions[2, NES_BUTTONS.index("RIGHT")] = 1
        first = env.step(actions)
        np.testing.assert_array_equal(first[0][0], first[0][2])
        assert first[1][0] == first[1][2]
        assert first[2][0] == first[2][2]
        assert first[3][0] == first[3][2]

        env.reset(
            options={
                "reset_mask": mask,
                "state_indices": starts,
                "snapshots": [handles[0], None, handles[0]],
            }
        )
        second = env.step(actions)
        for first_value, second_value in zip(first[:4], second[:4], strict=True):
            np.testing.assert_array_equal(first_value, second_value)
    finally:
        env.close()


def test_live_snapshot_lifecycle_owner_and_async_validation_are_atomic() -> None:
    env = make_env(num_envs=2, frame_skip=1)
    mask = np.asarray([True, False], dtype=np.bool_)
    with pytest.raises(RuntimeError, match="initial reset"):
        env.capture_snapshots(mask)
    env.reset()
    handles = env.capture_snapshots(mask)
    before_obs = env._obs.copy()
    before_states = env.active_state_indices().copy()

    with pytest.raises(ValueError, match="static state selector"):
        env.reset(
            options={
                "reset_mask": mask,
                "state_indices": np.asarray([0, -1], dtype=np.int32),
                "snapshots": handles,
            }
        )
    np.testing.assert_array_equal(env._obs, before_obs)
    np.testing.assert_array_equal(env.active_state_indices(), before_states)

    env.step_async(noop(2))
    with pytest.raises(RuntimeError, match="asynchronous step"):
        env.capture_snapshots(mask)
    env.step_wait_gymnasium()

    other = make_env(num_envs=2, frame_skip=1)
    try:
        other.reset()
        other_before = other._obs.copy()
        with pytest.raises(ValueError, match="different environment"):
            other.reset(
                options={
                    "reset_mask": mask,
                    "state_indices": np.asarray([-1, -1], dtype=np.int32),
                    "snapshots": handles,
                }
            )
        np.testing.assert_array_equal(other._obs, other_before)
    finally:
        other.close()

    env.close()
    with pytest.raises(RuntimeError, match="closed environment"):
        env.capture_snapshots(mask)


def test_reset_mask_validation() -> None:
    env = make_env(num_envs=1)
    try:
        with pytest.raises(TypeError, match="NumPy array"):
            env.reset(options={"reset_mask": [True]})
        with pytest.raises(TypeError, match="dtype"):
            env.reset(options={"reset_mask": np.array([1], dtype=np.int8)})
        with pytest.raises(ValueError, match="at least one"):
            env.reset(options={"reset_mask": np.array([False], dtype=np.bool_)})
        with pytest.raises(TypeError, match="state_indices.*NumPy"):
            env.reset(options={"state_indices": [0]})
        with pytest.raises(TypeError, match="state_indices.*dtype"):
            env.reset(options={"state_indices": np.array([0], dtype=np.int64)})
        with pytest.raises(ValueError, match="state_catalog"):
            env.reset(options={"state_indices": np.array([-1], dtype=np.int32)})
        with pytest.raises(ValueError, match="state_catalog"):
            env.reset(options={"state_indices": np.array([1], dtype=np.int32)})
        with pytest.raises(ValueError, match="unsupported"):
            env.reset(options={"start_indices": np.array([0], dtype=np.int32)})
    finally:
        env.close()


def test_invalid_state_index_is_rejected_atomically() -> None:
    env = make_env(state_catalog=("Level1-1", "Level1-2"), num_envs=2)
    try:
        env.reset(options={"state_indices": np.array([0, 1], dtype=np.int32)})
        env.step(noop(2))
        before_obs = env._obs.copy()
        before_states = env.active_state_indices().copy()
        with pytest.raises(ValueError, match="state_indices"):
            env.reset(
                options={
                    "reset_mask": np.array([True, True], dtype=np.bool_),
                    "state_indices": np.array([0, 9], dtype=np.int32),
                }
            )
        np.testing.assert_array_equal(env._obs, before_obs)
        np.testing.assert_array_equal(env.active_state_indices(), before_states)
    finally:
        env.close()


def test_direct_rom_reset_reports_negative_state_index() -> None:
    env = make_env(state=None, num_envs=1)
    try:
        _obs, infos = env.reset()
        assert env.state_catalog == ()
        np.testing.assert_array_equal(env.active_state_indices(), [-1])
        np.testing.assert_array_equal(infos["state_index"], [-1])
        np.testing.assert_array_equal(infos["_state_index"], [True])
        with pytest.raises(ValueError, match="non-empty state_catalog"):
            env.reset(options={"state_indices": np.array([0], dtype=np.int32)})
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


def test_rgb_render_keeps_mario_visible_during_injury_transition() -> None:
    state_path = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "supermariobrosnes_turbo"
        / "data"
        / "SuperMarioBros-Nes-v0"
        / "Level1-1.state"
    )
    state = bytearray(gzip.decompress(state_path.read_bytes()))
    ram_field = b"RAM\0" + (2048).to_bytes(4, "little")
    ram_offset = state.index(ram_field) + len(ram_field)
    state[ram_offset + 0x0756] = 1  # PlayerStatus: big
    state[ram_offset + 0x0754] = 0  # PlayerSize: big

    def transition_env(frame_skip: int) -> SuperMarioBrosNesTurboVecEnv:
        return SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            state=bytes(state),
            rom_path=require_rom(),
            num_envs=1,
            use_restricted_actions=Actions.ALL,
            render_mode="rgb_array",
            frame_skip=frame_skip,
            frame_stack=1,
            obs_grayscale=True,
            obs_resize=(8, 8),
        )

    skipped = transition_env(4)
    reference = transition_env(1)
    action = noop(1)
    action[0, NES_BUTTONS.index("B")] = 1
    action[0, NES_BUTTONS.index("RIGHT")] = 1
    saw_injury_transition = False
    saw_flicker_pair = False
    try:
        skipped.reset()
        reference.reset()
        assert skipped.render() is not None
        assert reference.render() is not None

        for _ in range(50):
            skipped.step(action)
            recent_frames: list[tuple[np.ndarray, int]] = []
            for _ in range(4):
                reference.step(action)
                frame = reference.render()
                assert frame is not None
                oam = reference._core._debug_oam(0)
                visible_sprites = sum(oam[offset] < 239 for offset in range(0, 256, 4))
                recent_frames.append((frame, visible_sprites))

            previous, current = recent_frames[-2:]
            expected = previous if previous[1] > current[1] else current
            rendered = skipped.render()
            assert rendered is not None
            np.testing.assert_array_equal(rendered, expected[0])

            ram = skipped.ram()[0]
            if ram[0x000E] == 10:  # InjuryBlink / big-to-small transition
                saw_injury_transition = True
                saw_flicker_pair |= previous[1] != current[1]

        assert saw_injury_transition
        assert saw_flicker_pair
    finally:
        skipped.close()
        reference.close()


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
