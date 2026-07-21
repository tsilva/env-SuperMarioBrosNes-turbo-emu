from __future__ import annotations

import json
from importlib import resources
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from gymnasium import spaces

from rom_helpers import require_rom
from supermariobrosnes_turbo import Actions, NES_BUTTONS, SuperMarioBrosNesTurboVecEnv
from supermariobrosnes_turbo.env import (
    ACTION_SETS,
    DISCRETE_CONTROLLER_BYTES,
    _named_action_controller_bytes,
)


PUBLIC_TO_CONTROLLER_BITS = (
    (0, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 0),
)
FILTER_GROUPS = ((0, 16, 32), (0, 64, 128), (0, 1, 256, 257))


def test_packaged_action_tables_match_stable_retro_integration_metadata() -> None:
    pytest.importorskip("stable_retro")
    stable_metadata = json.loads(
        resources.files("stable_retro")
        .joinpath("data", "stable", "SuperMarioBros-Nes-v0", "metadata.json")
        .read_text(encoding="utf-8")
    )
    turbo_metadata = json.loads(
        resources.files("supermariobrosnes_turbo")
        .joinpath("data", "SuperMarioBros-Nes-v0", "metadata.json")
        .read_text(encoding="utf-8")
    )
    if "action_sets" not in stable_metadata:
        pytest.skip("installed stable-retro-turbo predates action-table metadata")

    def tables(metadata: dict[str, object]) -> set[tuple[tuple[str, ...], ...]]:
        return {
            tuple(tuple(action) for action in table)
            for table in metadata["action_sets"].values()
        }

    assert tables(turbo_metadata) == tables(stable_metadata)


def expected_controller_byte(public_bits: int) -> int:
    controller = 0
    for public_bit, controller_bit in PUBLIC_TO_CONTROLLER_BITS:
        controller |= ((public_bits >> public_bit) & 1) << controller_bit
    return controller


def expected_filtered_bits(public_bits: int) -> int:
    filtered = 0
    for group in FILTER_GROUPS:
        group_mask = 0
        for value in group:
            group_mask |= value
        value = public_bits & group_mask
        if value in group:
            filtered |= value
    return filtered


def action_lookup(mode: str) -> np.ndarray:
    env = object.__new__(SuperMarioBrosNesTurboVecEnv)
    env.num_buttons = len(NES_BUTTONS)
    env._action_mode = mode
    return env._build_mask_to_controller_bytes()


@pytest.mark.parametrize(
    ("mode", "single_space", "vector_space"),
    [
        (Actions.ALL, spaces.MultiBinary(9), spaces.MultiBinary((2, 9))),
        (Actions.FILTERED, spaces.MultiBinary(9), spaces.MultiBinary((2, 9))),
        (Actions.DISCRETE, spaces.Discrete(36), spaces.MultiDiscrete([36, 36])),
        (
            Actions.MULTI_DISCRETE,
            spaces.MultiDiscrete([3, 3, 4]),
            spaces.Box(
                low=np.zeros((2, 3), dtype=np.int64),
                high=np.asarray([[2, 2, 3], [2, 2, 3]], dtype=np.int64),
                dtype=np.int64,
            ),
        ),
    ],
)
def test_supported_modes_expose_stable_retro_action_spaces(
    mode: Actions,
    single_space: spaces.Space,
    vector_space: spaces.Space,
) -> None:
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        rom_path=require_rom(),
        num_envs=2,
        use_restricted_actions=mode,
    )
    try:
        assert env.single_action_space == single_space
        assert env.action_space == vector_space
    finally:
        env.close()


def test_multi_discrete_components_map_like_stable_retro() -> None:
    env = object.__new__(SuperMarioBrosNesTurboVecEnv)
    env._action_mode = "MULTI_DISCRETE"
    env._BUTTON_COMBOS = FILTER_GROUPS
    env.num_envs = 2

    result = env._actions_to_controller_bytes(np.asarray([[0, 0, 0], [2, 1, 3]]))

    assert result.tolist() == [0, expected_controller_byte(32 | 64 | 257)]


def test_all_preserves_every_public_mask_except_the_unused_slot() -> None:
    lookup = action_lookup("ALL")

    assert lookup.tolist() == [expected_controller_byte(bits) for bits in range(512)]
    for bits in range(512):
        assert lookup[bits] == lookup[bits ^ (1 << 1)]
    assert len(set(int(value) for value in lookup)) == 256


def test_filtered_matches_group_filtering_for_all_masks_and_36_results() -> None:
    lookup = action_lookup("FILTERED")
    expected = [
        expected_controller_byte(expected_filtered_bits(bits)) for bits in range(512)
    ]

    assert lookup.tolist() == expected
    assert len(set(int(value) for value in lookup)) == 36
    assert set(int(value) for value in lookup) == set(
        int(value) for value in DISCRETE_CONTROLLER_BYTES
    )
    assert lookup[16 | 32] == 0
    assert lookup[16 | 32 | 256] == 1
    assert lookup[64 | 128 | 1] == 2
    assert lookup[4 | 8] == 0
    assert lookup[511] == 3


def test_discrete_indices_are_bijective_in_stable_retro_order() -> None:
    expected = []
    for action in range(36):
        vertical = FILTER_GROUPS[0][action % 3]
        horizontal = FILTER_GROUPS[1][(action // 3) % 3]
        face = FILTER_GROUPS[2][(action // 9) % 4]
        expected.append(expected_controller_byte(vertical | horizontal | face))

    assert DISCRETE_CONTROLLER_BYTES.tolist() == expected
    assert len(set(expected)) == 36


def test_named_action_sets_keep_their_controller_mappings() -> None:
    assert _named_action_controller_bytes(ACTION_SETS["basic"]).tolist() == [
        0,
        128,
        130,
        129,
        131,
        1,
        64,
    ]
    assert _named_action_controller_bytes(ACTION_SETS["right-jump"]).tolist() == [
        128,
        130,
        129,
        131,
    ]
    assert _named_action_controller_bytes(ACTION_SETS["basic-start"]).tolist() == [
        0,
        128,
        130,
        129,
        131,
        1,
        64,
        8,
    ]
    assert _named_action_controller_bytes(ACTION_SETS["standard"]).tolist() == [
        0,
        128,
        130,
        129,
        131,
        1,
        64,
        32,
    ]
    assert _named_action_controller_bytes(("start", "a", "right_b")).tolist() == [
        8,
        1,
        130,
    ]


def test_named_preset_and_inline_table_have_the_same_contract() -> None:
    rom = require_rom()
    named = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        rom_path=rom,
        num_envs=1,
        use_restricted_actions="right-jump",
    )
    inline = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        rom_path=rom,
        num_envs=1,
        use_restricted_actions=[
            ["RIGHT"],
            ["RIGHT", "B"],
            ["RIGHT", "A"],
            ["RIGHT", "A", "B"],
        ],
    )
    try:
        assert named.action_preset == "right-jump"
        assert inline.action_preset is None
        assert named.action_table == inline.action_table
        assert named.action_meanings == inline.action_meanings
        assert named.action_table_hash == inline.action_table_hash
    finally:
        named.close()
        inline.close()


def test_all_filtered_and_discrete_step_the_same_36_native_states() -> None:
    public_masks = np.zeros((36, len(NES_BUTTONS)), dtype=np.uint8)
    for action in range(36):
        public_bits = (
            FILTER_GROUPS[0][action % 3]
            | FILTER_GROUPS[1][(action // 3) % 3]
            | FILTER_GROUPS[2][(action // 9) % 4]
        )
        for button in range(len(NES_BUTTONS)):
            public_masks[action, button] = (public_bits >> button) & 1

    envs = [
        SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            state="Level1-1",
            rom_path=require_rom(),
            num_envs=36,
            use_restricted_actions=mode,
            frame_skip=1,
            frame_stack=1,
            obs_grayscale=True,
            obs_resize=(1, 1),
        )
        for mode in (Actions.ALL, Actions.FILTERED, Actions.DISCRETE)
    ]
    try:
        reset_observations = [env.reset(seed=9)[0] for env in envs]
        np.testing.assert_array_equal(reset_observations[0], reset_observations[1])
        np.testing.assert_array_equal(reset_observations[0], reset_observations[2])

        envs[0].step_async(public_masks)
        envs[1].step_async(public_masks)
        envs[2].step_async(np.arange(36, dtype=np.int64))
        assert len(set(int(value) for value in envs[2]._actions)) == 36
        np.testing.assert_array_equal(envs[0]._actions, DISCRETE_CONTROLLER_BYTES)
        np.testing.assert_array_equal(envs[1]._actions, DISCRETE_CONTROLLER_BYTES)
        np.testing.assert_array_equal(envs[2]._actions, DISCRETE_CONTROLLER_BYTES)

        step_results = [env.step_wait_gymnasium() for env in envs]
        for result_index in range(4):
            np.testing.assert_array_equal(
                step_results[0][result_index],
                step_results[1][result_index],
            )
            np.testing.assert_array_equal(
                step_results[0][result_index],
                step_results[2][result_index],
            )
    finally:
        for env in envs:
            env.close()


@pytest.mark.retro_oracle
def test_conversion_semantics_match_live_sibling_stable_retro_turbo() -> None:
    sibling = Path(__file__).resolve().parents[2] / "stable-retro-turbo"
    python = sibling / ".venv" / "bin" / "python"
    if not python.is_file():
        pytest.skip(f"live sibling Stable Retro environment not found at {python}")

    script = r"""
import json
import stable_retro as retro

def bits(mask):
    return sum(int(value) << index for index, value in enumerate(mask))

payload = {"version": getattr(retro, "__version__", "unknown")}
for name in ("ALL", "FILTERED", "DISCRETE"):
    mode = getattr(retro.Actions, name)
    env = retro.make(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        use_restricted_actions=mode,
        render_mode="rgb_array",
    )
    try:
        payload.setdefault("buttons", env.buttons)
        payload.setdefault("button_combos", env.button_combos)
        if name == "DISCRETE":
            payload[name] = [bits(env.action_to_array(action)[0]) for action in range(36)]
        else:
            payload[name] = [
                bits(env.action_to_array([(action >> index) & 1 for index in range(9)])[0])
                for action in range(512)
            ]
    finally:
        env.close()
print("ACTION_ORACLE=" + json.dumps(payload, separators=(",", ":")))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sibling)
    result = subprocess.run(
        [str(python), "-c", script],
        cwd=sibling,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    oracle_line = next(
        line.removeprefix("ACTION_ORACLE=")
        for line in result.stdout.splitlines()
        if line.startswith("ACTION_ORACLE=")
    )
    oracle = json.loads(oracle_line)

    assert oracle["version"].startswith("1.0.1")
    assert oracle["buttons"] == [
        "B",
        None,
        "SELECT",
        "START",
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "A",
    ]
    assert oracle["button_combos"] == [list(group) for group in FILTER_GROUPS]
    assert [expected_controller_byte(bits) for bits in oracle["ALL"]] == action_lookup(
        "ALL"
    ).tolist()
    assert [
        expected_controller_byte(bits) for bits in oracle["FILTERED"]
    ] == action_lookup("FILTERED").tolist()
    assert [
        expected_controller_byte(bits) for bits in oracle["DISCRETE"]
    ] == DISCRETE_CONTROLLER_BYTES.tolist()
