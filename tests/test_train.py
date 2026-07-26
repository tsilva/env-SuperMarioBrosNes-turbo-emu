from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from gymnasium import spaces

from supermariobrosnes_turbo import training as train_module
from supermariobrosnes_turbo import ACTION_SETS, PlayerMotion, PlayerTask
from supermariobrosnes_turbo.action_run import (
    ACTION_RUN_POLICY_MEMBER,
    ActionRun,
    ActionRunPolicy,
    LEGACY_JERK_POLICY_MEMBER,
    canonicalize_runs,
    load_action_run_checkpoint,
    save_action_run_checkpoint,
    truncate_runs,
)
from supermariobrosnes_turbo.training import (
    GO_EXPLORE_CELL_INFO_KEYS,
    GO_EXPLORE_CELL_KEY_BYTES,
    GO_EXPLORE_CELL_KEY_STRUCT,
    GO_EXPLORE_CELL_X_BUCKET_PIXELS,
    GO_EXPLORE_CELL_Y_BUCKET_PIXELS,
    MarioTask,
    REWARD_FUNCTION_SCORE_FIRST,
    REWARD_FUNCTION_SPEEDRUN,
    REWARD_MODE_SCORE_FIRST,
    go_explore_route_phase,
    mark_new_scroll_transitions,
    episode_boundary,
    sanitize_progress_x,
    score_first_step_cost,
    shape_step_rewards,
)


ACTIONS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left")


def test_task_uses_minimal_native_observation_and_default_action_set(
    monkeypatch,
) -> None:
    class FakeNative:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.config = kwargs
            self.action_preset = kwargs["use_restricted_actions"]
            self.action_meanings = ACTION_SETS[self.action_preset]
            self.single_action_space = spaces.Discrete(len(self.action_meanings))

    monkeypatch.setattr(train_module, "SuperMarioBrosNesTurboVecEnv", FakeNative)

    task = MarioTask(
        state="Level1-1",
        state_dir=None,
        rom_path=None,
        seed=0,
        n_envs=2,
        max_episode_steps=100,
        stall_steps=10,
        step_cost=0.1,
    )

    assert task.native.action_preset == "standard"
    assert task.native.single_action_space.n == 8
    assert task.action_names == ACTION_SETS["standard"]
    assert task.native.config["render_mode"] is None
    assert task.native.config["obs_crop"] == (0, 223, 0, 239)
    assert "obs_resize" not in task.native.config
    assert task.native.config["obs_grayscale"] is True
    assert task.native.config["obs_copy"] == "unsafe_view"
    assert task.native.config["frame_stack"] == 1
    assert task.native.config["maxpool_last_two"] is False
    assert task.native.config["noop_reset_max"] == 0
    assert task.native.config["info_filter"] == "none"


def test_task_accepts_a_state_general_down_action_set(monkeypatch) -> None:
    class FakeNative:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.config = kwargs
            self.action_preset = kwargs["use_restricted_actions"]
            self.action_meanings = ACTION_SETS[self.action_preset]

    monkeypatch.setattr(train_module, "SuperMarioBrosNesTurboVecEnv", FakeNative)

    task = MarioTask(
        state="Level8-4",
        state_dir=None,
        rom_path=None,
        seed=0,
        n_envs=2,
        max_episode_steps=100,
        stall_steps=10,
        step_cost=0.1,
        noop_reset_max=120,
        action_set="standard",
    )

    assert task.action_names == ACTION_SETS["standard"]
    assert task.native.config["noop_reset_max"] == 120


def test_go_explore_task_uses_minimal_observation_and_semantic_infos(
    monkeypatch,
) -> None:
    class FakeNative:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.config = kwargs
            self.action_meanings = ACTION_SETS[kwargs["use_restricted_actions"]]

    monkeypatch.setattr(train_module, "SuperMarioBrosNesTurboVecEnv", FakeNative)

    task = MarioTask(
        state="Level1-1",
        state_dir=None,
        rom_path=None,
        seed=0,
        n_envs=2,
        max_episode_steps=100,
        stall_steps=10,
        step_cost=0.1,
        go_explore_cells=True,
    )

    assert task.native.config["obs_crop"] == (0, 223, 0, 239)
    assert "obs_crop_mode" not in task.native.config
    assert "obs_resize" not in task.native.config
    assert task.native.config["obs_grayscale"] is True
    assert task.native.config["frame_stack"] == 1
    assert task.native.config["info_filter"] == {
        "mode": "all",
        "keys": list(GO_EXPLORE_CELL_INFO_KEYS),
    }


def test_go_explore_route_phase_is_compact_and_caps_counters() -> None:
    phase = go_explore_route_phase(
        loop_command_active=True,
        loop_correct_count=99,
        loop_pass_count=2,
        player_task=int(PlayerTask.VERTICAL_PIPE_ENTRY),
    )

    assert phase == 0b01011111


def test_go_explore_cell_keys_include_area_and_tile_position_buckets() -> None:
    class FakeNative:
        x_pos = np.asarray([100, 2_000], dtype=np.uint16)

    task = MarioTask.__new__(MarioTask)
    task.n_envs = 2
    task.native = FakeNative()
    task.previous_level_hi = np.asarray([1, 2], dtype=np.int16)
    task.previous_level_lo = np.asarray([3, 4], dtype=np.int16)
    task.cell_area_id = np.asarray([5, 6], dtype=np.int16)
    task.cell_y_pos = np.asarray([31, 32], dtype=np.int32)
    task.cell_area_pointer = np.asarray([7, 8], dtype=np.int16)
    task.cell_loop_command_active = np.asarray([False, True], dtype=np.bool_)
    task.cell_loop_correct_count = np.asarray([0, 2], dtype=np.int16)
    task.cell_loop_pass_count = np.asarray([0, 1], dtype=np.int16)
    task.cell_player_motion = np.asarray([0, 1], dtype=np.int8)
    task.cell_player_power = np.asarray([0, 2], dtype=np.int8)
    task.cell_player_task = np.asarray([8, 2], dtype=np.int8)
    observations = np.zeros((2, 1, 1, 1), dtype=np.uint8)

    keys = task.go_explore_cell_keys(observations)
    unpacked = tuple(GO_EXPLORE_CELL_KEY_STRUCT.unpack(key) for key in keys)

    assert unpacked[0] == (
        1,
        3,
        5,
        7,
        100 // GO_EXPLORE_CELL_X_BUCKET_PIXELS,
        31 // GO_EXPLORE_CELL_Y_BUCKET_PIXELS,
        0,
        1,
        0,
    )
    assert unpacked[1] == (
        2,
        4,
        6,
        8,
        2_000 // GO_EXPLORE_CELL_X_BUCKET_PIXELS,
        32 // GO_EXPLORE_CELL_Y_BUCKET_PIXELS,
        go_explore_route_phase(
            loop_command_active=True,
            loop_correct_count=2,
            loop_pass_count=1,
            player_task=2,
        ),
        0,
        2,
    )
    assert all(len(key) == GO_EXPLORE_CELL_KEY_BYTES for key in keys)


def test_go_explore_cell_keys_ignore_visual_but_preserve_pipe_entry_state() -> None:
    class FakeNative:
        x_pos = np.asarray([100], dtype=np.uint16)

    task = MarioTask.__new__(MarioTask)
    task.n_envs = 1
    task.native = FakeNative()
    task.previous_level_hi = np.asarray([1], dtype=np.int16)
    task.previous_level_lo = np.asarray([4], dtype=np.int16)
    task.cell_area_id = np.asarray([2], dtype=np.int16)
    task.cell_y_pos = np.asarray([48], dtype=np.int32)
    task.cell_area_pointer = np.asarray([9], dtype=np.int16)
    task.cell_loop_command_active = np.asarray([False], dtype=np.bool_)
    task.cell_loop_correct_count = np.asarray([0], dtype=np.int16)
    task.cell_loop_pass_count = np.asarray([0], dtype=np.int16)
    task.cell_player_motion = np.asarray([0], dtype=np.int8)
    task.cell_player_power = np.asarray([1], dtype=np.int8)
    task.cell_player_task = np.asarray([8], dtype=np.int8)

    dark = np.zeros((1, 1, 1, 1), dtype=np.uint8)
    bright = np.full((1, 1, 1, 1), 255, dtype=np.uint8)

    grounded = task.go_explore_cell_keys(dark)
    assert grounded == task.go_explore_cell_keys(bright)

    task.native.x_pos[0] = 108
    assert grounded != task.go_explore_cell_keys(dark)

    task.native.x_pos[0] = 100
    task.cell_player_motion[0] = int(PlayerMotion.JUMPING_OR_SWIMMING)
    assert grounded != task.go_explore_cell_keys(dark)


def _runs(*values: tuple[int, int]) -> tuple[ActionRun, ...]:
    return tuple(ActionRun(action, duration) for action, duration in values)


def test_task_snapshots_restore_emulator_and_reward_accounting() -> None:
    class FakeNative:
        def __init__(self) -> None:
            self.reset_options = None

        def capture_snapshots(self, mask):
            return tuple(
                f"native-{lane}" if selected else None
                for lane, selected in enumerate(mask)
            )

        def reset(self, *, options):
            self.reset_options = options
            return None, {
                "area_id": np.asarray([7, 0], dtype=np.int16),
                "y_pos": np.asarray([48, 0], dtype=np.int32),
                "area_pointer": np.asarray([9, 0], dtype=np.int16),
                "loop_command_active": np.asarray([True, False]),
                "loop_correct_count": np.asarray([2, 0], dtype=np.int16),
                "loop_pass_count": np.asarray([1, 0], dtype=np.int16),
                "player_motion": np.asarray([0, 1], dtype=np.int8),
                "player_power": np.asarray([2, 0], dtype=np.int8),
                "player_task": np.asarray([8, 0], dtype=np.int8),
            }

    task = MarioTask.__new__(MarioTask)
    task.n_envs = 2
    task.native = FakeNative()
    task.go_explore_cells = True
    task.cell_area_id = np.asarray([1, 2], dtype=np.int16)
    task.cell_y_pos = np.asarray([16, 32], dtype=np.int32)
    task.cell_area_pointer = np.asarray([3, 4], dtype=np.int16)
    task.cell_loop_command_active = np.asarray([False, False])
    task.cell_loop_correct_count = np.asarray([0, 0], dtype=np.int16)
    task.cell_loop_pass_count = np.asarray([0, 0], dtype=np.int16)
    task.cell_player_motion = np.asarray([1, 2], dtype=np.int8)
    task.cell_player_power = np.asarray([0, 1], dtype=np.int8)
    task.cell_player_task = np.asarray([8, 8], dtype=np.int8)
    task.episode_steps = np.asarray([3, 4], dtype=np.int64)
    task.last_progress_step = np.asarray([2, 3], dtype=np.int64)
    task.episode_returns = np.asarray([10.0, 20.0])
    task.previous_lives = np.asarray([2, 1], dtype=np.int16)
    task.previous_level_hi = np.asarray([0, 1], dtype=np.int16)
    task.previous_level_lo = np.asarray([0, 2], dtype=np.int16)
    task.previous_score = np.asarray([100, 200], dtype=np.int64)
    task.level_max_x = np.asarray([80, 90], dtype=np.int64)
    task.completed_base = np.asarray([256, 512], dtype=np.int64)
    task.max_global_x = np.asarray([336, 602], dtype=np.int64)
    task.previous_x = np.asarray([75, 85], dtype=np.int64)
    task.seen_scroll_transitions = [{(1, 2)}, {(3, 4)}]
    mask = np.asarray([True, False], dtype=np.bool_)

    snapshots = task.capture_snapshots(mask)
    task.episode_steps[0] = 99
    task.episode_returns[0] = -1.0
    task.previous_x[0] = 0
    task.seen_scroll_transitions[0].clear()
    task.restore_lanes(mask, snapshots)

    assert task.native.reset_options["snapshots"] == ["native-0", None]
    assert task.episode_steps.tolist() == [3, 4]
    assert task.episode_returns.tolist() == [10.0, 20.0]
    assert task.previous_x.tolist() == [75, 85]
    assert task.cell_area_id.tolist() == [7, 2]
    assert task.cell_y_pos.tolist() == [48, 32]
    assert task.cell_area_pointer.tolist() == [9, 4]
    assert task.cell_loop_command_active.tolist() == [True, False]
    assert task.cell_loop_correct_count.tolist() == [2, 0]
    assert task.cell_loop_pass_count.tolist() == [1, 0]
    assert task.cell_player_motion.tolist() == [0, 2]
    assert task.cell_player_power.tolist() == [2, 1]
    assert task.cell_player_task.tolist() == [8, 8]
    assert task.seen_scroll_transitions == [{(1, 2)}, {(3, 4)}]


def test_progress_ignores_invalid_scroll_sentinel() -> None:
    current = np.asarray([120, 0xFFFE, 0xFFFF, 250], dtype=np.int64)
    previous = np.asarray([100, 180, 200, 240], dtype=np.int64)

    np.testing.assert_array_equal(
        sanitize_progress_x(current, previous),
        np.asarray([120, 180, 200, 250], dtype=np.int64),
    )


def test_scroll_transition_progress_counts_each_route_edge_once() -> None:
    seen: list[set[tuple[int, int]]] = [set(), set(), set()]
    blocked = np.asarray([False, False, True])

    first = mark_new_scroll_transitions(
        np.asarray([741, 1159, 500]),
        np.asarray([256, 144, 0]),
        blocked,
        seen,
    )
    repeated = mark_new_scroll_transitions(
        np.asarray([743, 1160, 500]),
        np.asarray([258, 145, 0]),
        blocked,
        seen,
    )

    np.testing.assert_array_equal(first, [True, True, False])
    np.testing.assert_array_equal(repeated, [False, False, False])


def test_step_reward_charges_time_on_every_transition() -> None:
    rewards = shape_step_rewards(
        np.asarray([0, 10, 0]),
        np.asarray([0, 100, 0]),
        np.asarray([False, False, True]),
        step_cost=0.1,
    )

    np.testing.assert_allclose(rewards, [-0.1, 10.9, -25.1])


def test_score_first_reward_makes_time_only_a_score_tiebreaker() -> None:
    max_episode_steps = 4_500
    step_cost = score_first_step_cost(max_episode_steps)
    rewards = shape_step_rewards(
        np.asarray([500, 0]),
        np.asarray([1, 0]),
        np.asarray([False, False]),
        step_cost=step_cost,
        reward_mode=REWARD_MODE_SCORE_FIRST,
    )

    np.testing.assert_allclose(rewards, [1.0 - step_cost, -step_cost])
    slow_one_point = 1.0 - max_episode_steps * step_cost
    fast_zero_points = -step_cost
    assert slow_one_point > fast_zero_points
    assert 1_000.0 - 1_000 * step_cost > 1_000.0 - 1_200 * step_cost


def test_speedrun_reward_ignores_progress_and_score_and_penalizes_time_and_death() -> (
    None
):
    rewards = shape_step_rewards(
        np.asarray([500, 0, 999]),
        np.asarray([0, 10_000, 50_000]),
        np.asarray([False, False, True]),
        step_cost=1.0,
        reward_mode=REWARD_FUNCTION_SPEEDRUN,
    )

    np.testing.assert_allclose(rewards, [-1.0, -1.0, -26.0])
    assert -100.0 > -120.0


def test_training_flags_continue_and_protect_policies_by_default() -> None:
    parser = train_module.build_parser()

    defaults = parser.parse_args(["Level1-1"])
    stopped = parser.parse_args(
        [
            "Level1-1",
            "--stop-on-completion",
            "--overwrite",
            "--noop-reset-max",
            "120",
        ]
    )

    assert defaults.lanes == 64
    assert defaults.algorithm == "go-explore"
    assert defaults.noop_reset_max == 0
    assert defaults.continue_after_completion is True
    assert defaults.overwrite is False
    assert stopped.noop_reset_max == 120
    assert stopped.continue_after_completion is False
    assert stopped.overwrite is True


def test_algorithm_defaults_make_beam_score_first_and_go_explore_speedrun() -> None:
    parser = train_module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["Level1-1", "--algorithm", "jerk"])

    beam = parser.parse_args(["Level1-1", "--algorithm", "beam"])
    go_explore = parser.parse_args(["Level1-1", "--algorithm", "go-explore"])

    train_module._apply_algorithm_defaults(parser, beam)
    train_module._apply_algorithm_defaults(parser, go_explore)

    assert beam.step_cost == pytest.approx(
        score_first_step_cost(beam.max_episode_steps)
    )
    assert go_explore.step_cost == 1.0
    assert go_explore.reward_function == REWARD_FUNCTION_SPEEDRUN


def test_speedrun_reward_id_selects_unit_time_cost() -> None:
    parser = train_module.build_parser()
    args = parser.parse_args(
        ["Level1-1", "--reward-function", REWARD_FUNCTION_SPEEDRUN]
    )

    train_module._apply_algorithm_defaults(parser, args)

    assert args.reward_function == REWARD_FUNCTION_SPEEDRUN
    assert args.step_cost == 1.0


def test_only_default_canonical_or_explicit_runs_force_policy_overwrite() -> None:
    parser = train_module.build_parser()
    default = parser.parse_args(["Level1-1"])
    custom_default = parser.parse_args(["Level1-1", "--output", "runs/custom"])
    beam = parser.parse_args(["Level1-1", "--algorithm", "beam"])
    forced_beam = parser.parse_args(["Level1-1", "--algorithm", "beam", "--overwrite"])
    score_first = parser.parse_args(
        ["Level1-1", "--reward-function", REWARD_FUNCTION_SCORE_FIRST]
    )

    assert train_module._force_policy_overwrite(default)
    assert not train_module._force_policy_overwrite(custom_default)
    assert not train_module._force_policy_overwrite(beam)
    assert train_module._force_policy_overwrite(forced_beam)
    assert not train_module._force_policy_overwrite(score_first)


def test_training_log_helpers_are_readable_and_emit_exact_play_commands() -> None:
    box = train_module._format_box(
        "Training complete",
        [("Result", "level completed"), ("Transitions", "500,000")],
    )
    progress = train_module._format_progress(
        {
            "timesteps": 250_000,
            "loop_fps": 32_500.0,
            "episodes": 123,
            "best_mean_reward": 3_049.95,
            "best_progress": 3_129.0,
            "best_program_steps": 943,
            "best_program_runs": 242,
            "retained_count": 256,
            "locked_count": 1,
        },
        1_000_000,
    )

    assert box.startswith("╭─ Training complete")
    assert "Result       level completed" in box
    wrapped_box = train_module._format_box(
        "Training complete",
        [("Saved", "/" + "very-long-policy-path/" * 10)],
    )
    assert max(map(len, wrapped_box.splitlines())) <= 92
    assert "25.00%" in progress
    assert "32,500 steps/s" in progress
    assert "policy 943 steps / 242 runs" in progress
    assert (
        train_module._play_command(
            "Level1-1",
            Path("runs/Level1-1/Level1-1.zip"),
            default_output=True,
            rom_path=None,
        )
        == "smb-turbo play Level1-1"
    )
    assert (
        train_module._play_command(
            "Level1-1",
            Path("custom run/Level1-1.zip"),
            default_output=False,
            rom_path=Path("roms/Mario Bros.nes"),
        )
        == "smb-turbo play Level1-1 --policy 'custom run/Level1-1.zip' "
        "--rom 'roms/Mario Bros.nes'"
    )


def test_policy_save_requires_force_to_replace_existing_file(tmp_path) -> None:
    policy_path = tmp_path / "Level1-1.zip"
    policy_path.write_bytes(b"existing policy")
    policy = ActionRunPolicy(
        action_names=ACTIONS,
        action_runs=_runs((1, 2)),
        fallback_action=0,
    )

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        train_module._save_policy(policy, policy_path)
    assert policy_path.read_bytes() == b"existing policy"

    train_module._save_policy(policy, policy_path, force=True)
    assert ActionRunPolicy.load(policy_path).action_runs == _runs((1, 2))


def test_level_change_does_not_end_episode() -> None:
    boundary = episode_boundary(
        previous_lives=2,
        current_lives=2,
        previous_level=(0, 0),
        current_level=(0, 1),
        episode_steps=1_000,
        max_episode_steps=4_500,
    )

    assert boundary.level_changed
    assert not boundary.life_loss
    assert not boundary.done


def test_life_loss_and_stall_end_failed_attempts() -> None:
    life_loss = episode_boundary(
        previous_lives=2,
        current_lives=1,
        previous_level=(0, 0),
        current_level=(0, 0),
        episode_steps=100,
        max_episode_steps=4_500,
    )
    stalled = episode_boundary(
        previous_lives=2,
        current_lives=2,
        previous_level=(0, 0),
        current_level=(0, 0),
        episode_steps=300,
        max_episode_steps=4_500,
        stalled=True,
    )

    assert life_loss.done and life_loss.life_loss
    assert stalled.done and stalled.stalled


def test_action_runs_are_canonical_and_truncate_inside_a_run() -> None:
    canonical = canonicalize_runs(_runs((1, 3), (1, 4), (2, 5)))

    assert canonical == _runs((1, 7), (2, 5))
    assert truncate_runs(canonical, 9) == _runs((1, 7), (2, 2))
    with pytest.raises(ValueError, match="durations must be positive"):
        ActionRun(1, 0)


def test_action_run_policy_zip_round_trip_and_lane_resets(tmp_path) -> None:
    path = tmp_path / "model.zip"
    ActionRunPolicy(
        action_names=ACTIONS,
        action_runs=_runs((2, 2), (4, 1)),
        fallback_action=0,
    ).save(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [ACTION_RUN_POLICY_MEMBER]
        assert json.loads(archive.read(ACTION_RUN_POLICY_MEMBER))["algorithm_id"] == (
            "action-run"
        )
    loaded = ActionRunPolicy.load(path)

    obs = np.zeros((2, 1), dtype=np.float32)
    assert loaded.predict(obs)[0].tolist() == [2, 2]
    assert loaded.predict(obs)[0].tolist() == [2, 2]
    assert loaded.predict(obs)[0].tolist() == [4, 4]
    loaded.reset_lanes([True, False])
    assert loaded.predict(obs)[0].tolist() == [2, 0]
    assert loaded.step_count == 3
    assert loaded.run_count == 2


def test_flat_v1_policy_is_rejected(tmp_path) -> None:
    path = tmp_path / "old.zip"
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(
            LEGACY_JERK_POLICY_MEMBER,
            json.dumps(
                {
                    "schema_version": 1,
                    "algorithm_id": "jerk",
                    "action_names": list(ACTIONS),
                    "action_sequence": [1, 3],
                    "fallback_action": 0,
                }
            ),
        )

    with pytest.raises(ValueError, match="unsupported legacy action-run"):
        load_action_run_checkpoint(path)


def test_legacy_action_run_policy_is_still_loadable(tmp_path) -> None:
    path = tmp_path / "legacy.zip"
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(
            LEGACY_JERK_POLICY_MEMBER,
            json.dumps(
                {
                    "schema_version": 2,
                    "algorithm_id": "jerk",
                    "action_names": list(ACTIONS),
                    "action_runs": [[1, 2]],
                    "fallback_action": 0,
                }
            ),
        )

    policy = load_action_run_checkpoint(path)

    assert policy.action_runs == _runs((1, 2))


def test_named_run_checkpoint_round_trip(tmp_path) -> None:
    path = save_action_run_checkpoint(
        tmp_path / "policy.zip",
        (("right", 2), ("right_a", 1)),
        timesteps=123,
        episodes=4,
        best_reward=99.0,
        metadata={"terminate_on_level_change": False},
    )

    policy = load_action_run_checkpoint(path)
    observations = np.zeros((1, 1, 1, 1), dtype=np.uint8)
    actions = [int(policy.predict(observations)[0][0]) for _ in range(4)]

    assert actions == [1, 1, 3, 0]
    assert policy.timesteps == 123
    assert policy.metadata["terminate_on_level_change"] is False
    assert policy.action_set == "standard"
