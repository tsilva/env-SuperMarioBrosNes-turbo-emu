from __future__ import annotations

import numpy as np
import pytest

from rom_helpers import require_rom
from supermariobrosnes_turbo import (
    AVAILABLE_INFO_KEYS,
    EXTRA_INFO_KEYS,
    INFO_KEYS,
    Actions,
    AreaType,
    Direction,
    GameMode,
    NES_BUTTONS,
    PlayerMotion,
    PlayerPower,
    PlayerTask,
    SuperMarioBrosNesTurboVecEnv,
)


EXPECTED_EXTRA_INFO_KEYS = (
    "area_id",
    "area_type",
    "y_pos",
    "y_screen_pos",
    "player_motion",
    "player_power",
    "is_large",
    "x_velocity",
    "y_velocity",
    "facing",
    "is_crouching",
    "is_swimming",
    "injury_timer",
    "star_timer",
    "game_mode",
    "player_task",
    "enemy_active",
    "enemy_type_id",
    "enemy_x_pos",
    "enemy_y_pos",
    "enemy_x_velocity",
    "enemy_y_velocity",
    "enemy_facing",
    "area_pointer",
    "loop_command_active",
    "loop_correct_count",
    "loop_pass_count",
)


def make_env(
    *,
    num_envs: int = 2,
    num_threads: int = 1,
    frame_skip: int = 1,
    info_filter: object = "all",
):
    return SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        rom_path=require_rom(),
        num_envs=num_envs,
        num_threads=num_threads,
        use_restricted_actions=Actions.ALL,
        frame_skip=frame_skip,
        frame_stack=1,
        obs_grayscale=True,
        obs_resize=(84, 84),
        obs_layout="chw",
        info_filter=info_filter,
    )


def noop(num_envs: int) -> np.ndarray:
    return np.zeros((num_envs, len(NES_BUTTONS)), dtype=np.uint8)


def test_public_catalog_and_enums_are_stable() -> None:
    assert INFO_KEYS == (
        "x_pos",
        "coins",
        "levelHi",
        "levelLo",
        "lives",
        "score",
        "scrolling",
        "time",
        "xscrollHi",
        "xscrollLo",
    )
    assert EXTRA_INFO_KEYS == EXPECTED_EXTRA_INFO_KEYS
    assert AVAILABLE_INFO_KEYS == INFO_KEYS + EXTRA_INFO_KEYS
    assert AreaType.UNKNOWN == -1
    assert PlayerMotion.CLIMBING == 3
    assert PlayerPower.FIRE == 2
    assert Direction.LEFT == -1
    assert GameMode.GAMEPLAY == 1
    assert PlayerTask.PLAYER_CONTROL == 8


def test_explicit_info_keys_validate_and_canonicalize_before_rom_loading() -> None:
    with pytest.raises(ValueError, match="unknown info key"):
        SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            rom_path="/definitely/missing.nes",
            info_filter={"keys": ["not_a_signal"]},
        )

    env = make_env(
        num_envs=1,
        info_filter={
            "keys": ["enemy_active", "x_pos", "area_type", "x_pos", "area_type"]
        },
    )
    try:
        _obs, infos = env.reset()
        game_keys = [key for key in infos if not key.startswith("_")]
        assert game_keys[:3] == ["x_pos", "area_type", "enemy_active"]
    finally:
        env.close()


def test_reset_infos_match_processed_ram_and_public_dtypes() -> None:
    selected = ("x_pos",) + EXTRA_INFO_KEYS
    env = make_env(info_filter={"mode": "all", "keys": selected})
    try:
        _obs, infos = env.reset()
        ram = env.ram()
        assert ram.shape == (2, 2048)
        assert ram.dtype == np.uint8
        assert not ram.flags.writeable

        np.testing.assert_array_equal(infos["area_id"], ram[:, 0x0760])
        expected_area_type = ram[:, 0x074E].astype(np.int16)
        expected_area_type[expected_area_type > 3] = -1
        np.testing.assert_array_equal(infos["area_type"], expected_area_type)
        np.testing.assert_array_equal(
            infos["y_pos"],
            (ram[:, 0x00B5].astype(np.int32) << 8) | ram[:, 0x00CE],
        )
        np.testing.assert_array_equal(infos["y_screen_pos"], ram[:, 0x03B8])
        np.testing.assert_array_equal(infos["is_large"], ram[:, 0x0754] == 0)
        np.testing.assert_array_equal(infos["is_crouching"], ram[:, 0x0714] != 0)
        np.testing.assert_array_equal(infos["is_swimming"], ram[:, 0x0704] != 0)
        np.testing.assert_array_equal(infos["area_pointer"], ram[:, 0x0750])
        np.testing.assert_array_equal(infos["loop_command_active"], ram[:, 0x0745] != 0)
        np.testing.assert_array_equal(infos["loop_correct_count"], ram[:, 0x06D9])
        np.testing.assert_array_equal(infos["loop_pass_count"], ram[:, 0x06DA])

        assert infos["area_type"].dtype == np.int8
        assert infos["y_pos"].dtype == np.int32
        assert infos["is_large"].dtype == np.bool_
        assert infos["area_pointer"].dtype == np.int16
        assert infos["loop_command_active"].dtype == np.bool_
        assert infos["enemy_active"].dtype == np.bool_
        assert infos["enemy_x_pos"].dtype == np.int32
        assert infos["enemy_x_pos"].shape == (2, 6)

        active = infos["enemy_active"]
        np.testing.assert_array_equal(infos["enemy_type_id"][~active], -1)
        np.testing.assert_array_equal(infos["enemy_x_pos"][~active], -1)
        np.testing.assert_array_equal(infos["enemy_y_pos"][~active], -1)
        np.testing.assert_array_equal(infos["enemy_x_velocity"][~active], 0)
        np.testing.assert_array_equal(infos["enemy_y_velocity"][~active], 0)
        np.testing.assert_array_equal(infos["enemy_facing"][~active], 0)
    finally:
        env.close()


def test_default_empty_and_none_filters_do_not_allocate_extra_storage() -> None:
    default = make_env(num_envs=1, info_filter="all")
    empty = make_env(num_envs=1, info_filter={"mode": "all", "keys": []})
    none = make_env(num_envs=1, info_filter="none")
    try:
        assert default._extra_info is None
        assert empty._extra_info is None
        assert none._extra_info is None
        assert default._core.extra_info_shape() == (1, 0)

        _obs, default_infos = default.reset()
        assert all(key in default_infos for key in INFO_KEYS)
        assert not any(key in default_infos for key in EXTRA_INFO_KEYS)

        _obs, empty_infos = empty.reset()
        assert "state_index" in empty_infos
        assert not any(key in empty_infos for key in AVAILABLE_INFO_KEYS)

        _obs, none_infos = none.reset()
        assert "state_index" in none_infos
        assert not any(key in none_infos for key in AVAILABLE_INFO_KEYS)
        none.step(noop(1))
        assert none.x_pos.shape == (1,)
    finally:
        default.close()
        empty.close()
        none.close()


def test_terminal_filter_skips_nonterminal_extra_extraction() -> None:
    env = make_env(
        num_envs=1,
        info_filter={"mode": "terminal", "keys": ["area_type", "enemy_active"]},
    )
    try:
        env.reset()
        assert env._extra_info is not None
        env._extra_info.fill(123)
        _obs, _reward, terminated, truncated, infos = env.step(noop(1))
        assert not bool(terminated[0] or truncated[0])
        assert infos == {}
        np.testing.assert_array_equal(env._extra_info, 123)
    finally:
        env.close()


def test_terminal_filter_emits_processed_extras_on_game_over() -> None:
    env = make_env(
        num_envs=1,
        frame_skip=4,
        info_filter={"mode": "terminal", "keys": ["game_mode", "enemy_active"]},
    )
    try:
        env.reset()
        for _ in range(8_000):
            _obs, _reward, terminated, truncated, infos = env.step(noop(1))
            if bool(terminated[0] or truncated[0]):
                assert infos["game_mode"].shape == (1,)
                assert infos["enemy_active"].shape == (1, 6)
                np.testing.assert_array_equal(infos["_game_mode"], [True])
                np.testing.assert_array_equal(infos["_enemy_active"], [True])
                return
        pytest.fail("native game-over did not occur within probe budget")
    finally:
        env.close()


def test_masked_reset_masks_extra_info_without_touching_other_lane() -> None:
    env = make_env(
        info_filter={"keys": ["y_pos", "enemy_active", "enemy_x_pos"]},
    )
    try:
        env.reset(seed=11)
        env.step(noop(2))
        before_ram = env.ram()[1].copy()
        reset_mask = np.asarray([True, False], dtype=np.bool_)
        state_indices = np.asarray([0, -1], dtype=np.int32)
        _obs, infos = env.reset(
            options={"reset_mask": reset_mask, "state_indices": state_indices}
        )
        np.testing.assert_array_equal(infos["_y_pos"], reset_mask)
        np.testing.assert_array_equal(infos["_enemy_active"], reset_mask)
        np.testing.assert_array_equal(infos["_enemy_x_pos"], reset_mask)
        np.testing.assert_array_equal(env.ram()[1], before_ram)
    finally:
        env.close()


def test_extra_info_results_are_owned_and_snapshot_reset_is_redecoded() -> None:
    keys = ["y_pos", "enemy_active", "enemy_x_pos"]
    env = make_env(num_envs=1, info_filter={"keys": keys})
    try:
        _obs, first = env.reset()
        frozen = {key: first[key].copy() for key in keys}
        handles = env.capture_snapshots(np.asarray([True], dtype=np.bool_))
        first["enemy_x_pos"].fill(12345)
        env.step(noop(1))
        _obs, restored = env.reset(
            options={
                "reset_mask": np.asarray([True], dtype=np.bool_),
                "state_indices": np.asarray([-1], dtype=np.int32),
                "snapshots": [handles[0]],
            }
        )
        for key in keys:
            np.testing.assert_array_equal(restored[key], frozen[key])
    finally:
        env.close()


def test_serial_and_parallel_extra_info_are_deterministic() -> None:
    keys = ["x_pos", "y_pos", "player_motion", "enemy_active", "enemy_x_pos"]
    serial = make_env(num_envs=4, num_threads=1, info_filter={"keys": keys})
    parallel = make_env(num_envs=4, num_threads=4, info_filter={"keys": keys})
    try:
        serial.reset(seed=7)
        parallel.reset(seed=7)
        for _ in range(5):
            serial_step = serial.step(noop(4))
            parallel_step = parallel.step(noop(4))
            for expected, actual in zip(serial_step[:4], parallel_step[:4]):
                np.testing.assert_array_equal(expected, actual)
            for key in keys:
                np.testing.assert_array_equal(serial_step[4][key], parallel_step[4][key])
    finally:
        serial.close()
        parallel.close()
