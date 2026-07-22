from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from gymnasium import spaces

from supermariobrosnes_turbo import training as train_module
from supermariobrosnes_turbo import ACTION_SETS
from supermariobrosnes_turbo.jerk import (
    ActionRun,
    JERK_POLICY_MEMBER,
    JerkPolicy,
    JerkSearch,
    RetainedProgram,
    canonicalize_runs,
    load_jerk_checkpoint,
    save_jerk_checkpoint,
    truncate_runs,
)
from supermariobrosnes_turbo.training import (
    GO_EXPLORE_CELL_FRAME_SHAPE,
    GO_EXPLORE_CELL_KEY_BYTES,
    GO_EXPLORE_CELL_X_BUCKET_PIXELS,
    GO_EXPLORE_CELL_Y_BUCKET_PIXELS,
    MarioJerkTask,
    REWARD_MODE_SCORE_FIRST,
    go_explore_frame_bytes,
    mark_new_scroll_transitions,
    episode_boundary,
    exploit_probability,
    sanitize_progress_x,
    score_first_step_cost,
    shape_step_rewards,
)


ACTIONS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left")


def test_jerk_task_uses_minimal_native_observation_and_default_action_set(
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

    task = MarioJerkTask(
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
    assert task.native.config["info_filter"] == "none"


def test_jerk_task_accepts_a_state_general_down_action_set(monkeypatch) -> None:
    class FakeNative:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.action_preset = kwargs["use_restricted_actions"]
            self.action_meanings = ACTION_SETS[self.action_preset]

    monkeypatch.setattr(train_module, "SuperMarioBrosNesTurboVecEnv", FakeNative)

    task = MarioJerkTask(
        state="Level8-4",
        state_dir=None,
        rom_path=None,
        seed=0,
        n_envs=2,
        max_episode_steps=100,
        stall_steps=10,
        step_cost=0.1,
        action_set="standard",
    )

    assert task.action_names == ACTION_SETS["standard"]


def test_go_explore_task_uses_native_masked_downscaled_observations(
    monkeypatch,
) -> None:
    class FakeNative:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.config = kwargs
            self.action_meanings = ACTION_SETS[kwargs["use_restricted_actions"]]

    monkeypatch.setattr(train_module, "SuperMarioBrosNesTurboVecEnv", FakeNative)

    task = MarioJerkTask(
        state="Level1-1",
        state_dir=None,
        rom_path=None,
        seed=0,
        n_envs=2,
        max_episode_steps=100,
        stall_steps=10,
        step_cost=0.1,
        visual_cell_observations=True,
    )

    assert task.native.config["obs_crop"] == (32, 0, 0, 0)
    assert task.native.config["obs_crop_mode"] == "mask"
    assert task.native.config["obs_resize"] == GO_EXPLORE_CELL_FRAME_SHAPE
    assert task.native.config["obs_resize_algorithm"] == "area"
    assert task.native.config["obs_grayscale"] is True
    assert task.native.config["frame_stack"] == 1
    assert task.native.config["info_filter"] == {
        "mode": "all",
        "keys": ["area_id", "y_pos"],
    }


def test_go_explore_frame_bytes_quantize_each_lane_without_hashing() -> None:
    observations = np.zeros((3, 1, *GO_EXPLORE_CELL_FRAME_SHAPE), dtype=np.uint8)
    observations[1, 0, 3, 4] = 31
    observations[2, 0, 3, 4] = 32

    encoded = go_explore_frame_bytes(observations)

    assert len(encoded) == 3
    assert all(len(frame) == GO_EXPLORE_CELL_KEY_BYTES for frame in encoded)
    assert encoded[0] == encoded[1]
    assert encoded[0] != encoded[2]
    quantized = np.frombuffer(encoded[2], dtype=np.uint8).reshape(
        GO_EXPLORE_CELL_FRAME_SHAPE
    )
    assert quantized[3, 4] == 1


def test_go_explore_cell_keys_include_area_and_tile_position_buckets() -> None:
    class FakeNative:
        x_pos = np.asarray([100, 2_000], dtype=np.uint16)

    task = MarioJerkTask.__new__(MarioJerkTask)
    task.n_envs = 2
    task.native = FakeNative()
    task.previous_level_hi = np.asarray([1, 2], dtype=np.int16)
    task.previous_level_lo = np.asarray([3, 4], dtype=np.int16)
    task.cell_area_id = np.asarray([5, 6], dtype=np.int16)
    task.cell_y_pos = np.asarray([31, 32], dtype=np.int64)
    observations = np.zeros((2, 1, *GO_EXPLORE_CELL_FRAME_SHAPE), dtype=np.uint8)

    keys = task.go_explore_cell_keys(observations)

    assert keys[0][:2] == (1, 3)
    assert keys[1][:2] == (2, 4)
    assert keys[0][2:5] == (
        5,
        100 // GO_EXPLORE_CELL_X_BUCKET_PIXELS,
        31 // GO_EXPLORE_CELL_Y_BUCKET_PIXELS,
    )
    assert keys[1][2:5] == (
        6,
        2_000 // GO_EXPLORE_CELL_X_BUCKET_PIXELS,
        32 // GO_EXPLORE_CELL_Y_BUCKET_PIXELS,
    )
    assert keys[0][5] == keys[1][5]
    assert len(keys[0]) == 6


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
            }

    task = MarioJerkTask.__new__(MarioJerkTask)
    task.n_envs = 2
    task.native = FakeNative()
    task.visual_cell_observations = True
    task.cell_area_id = np.asarray([1, 2], dtype=np.int16)
    task.cell_y_pos = np.asarray([16, 32], dtype=np.int64)
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
    assert task.seen_scroll_transitions == [{(1, 2)}, {(3, 4)}]


def _search(
    *,
    seed: int = 7,
    n_envs: int = 1,
    initial_probability: float = 0.25,
    max_probability: float = 0.9,
    protected_prefix_runs: int = 8,
    max_prefix_shorten_runs: int = 16,
    deep_mutation_probability: float = 0.25,
    run_duration_mean: float = 4.0,
    run_duration_max: int = 32,
    retained_limit: int = 8,
) -> JerkSearch:
    return JerkSearch(
        n_envs=n_envs,
        seed=seed,
        total_timesteps=10,
        action_names=ACTIONS,
        fallback_action="noop",
        archive_replay_probability_initial=initial_probability,
        archive_replay_probability_max=max_probability,
        protected_prefix_runs=protected_prefix_runs,
        max_prefix_shorten_runs=max_prefix_shorten_runs,
        deep_mutation_probability=deep_mutation_probability,
        run_duration_mean=run_duration_mean,
        run_duration_max=run_duration_max,
        retained_limit=retained_limit,
    )


def _candidate(
    runs: tuple[ActionRun, ...],
    mean_return: float,
    *,
    completed: bool = False,
    progress: float = 0.0,
) -> RetainedProgram:
    return RetainedProgram(
        runs=runs,
        return_sum=mean_return,
        return_count=1,
        completed=completed,
        progress=progress,
    )


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


def test_training_flags_continue_and_protect_policies_by_default() -> None:
    parser = train_module.build_parser()

    defaults = parser.parse_args(["Level1-1"])
    stopped = parser.parse_args(
        ["Level1-1", "--stop-on-completion", "--overwrite"]
    )

    assert defaults.lanes == 64
    assert defaults.algorithm == "go-explore"
    assert defaults.continue_after_completion is True
    assert defaults.overwrite is False
    assert stopped.continue_after_completion is False
    assert stopped.overwrite is True


def test_algorithm_defaults_make_beam_and_go_explore_score_first() -> None:
    parser = train_module.build_parser()
    jerk = parser.parse_args(["Level1-1", "--algorithm", "jerk"])
    beam = parser.parse_args(["Level1-1", "--algorithm", "beam"])
    go_explore = parser.parse_args(["Level1-1", "--algorithm", "go-explore"])

    train_module._apply_algorithm_defaults(parser, jerk)
    train_module._apply_algorithm_defaults(parser, beam)
    train_module._apply_algorithm_defaults(parser, go_explore)

    assert jerk.step_cost == train_module.STEP_COST
    assert beam.step_cost == pytest.approx(
        score_first_step_cost(beam.max_episode_steps)
    )
    assert go_explore.step_cost == pytest.approx(
        score_first_step_cost(go_explore.max_episode_steps)
    )


def test_only_default_canonical_or_explicit_runs_force_policy_overwrite() -> None:
    parser = train_module.build_parser()
    default = parser.parse_args(["Level1-1"])
    custom_default = parser.parse_args(
        ["Level1-1", "--output", "runs/custom"]
    )
    beam = parser.parse_args(["Level1-1", "--algorithm", "beam"])
    forced_beam = parser.parse_args(
        ["Level1-1", "--algorithm", "beam", "--overwrite"]
    )

    assert train_module._force_policy_overwrite(default)
    assert not train_module._force_policy_overwrite(custom_default)
    assert not train_module._force_policy_overwrite(beam)
    assert train_module._force_policy_overwrite(forced_beam)


def test_active_best_candidate_carries_observed_progress() -> None:
    search = _search()
    search.next_actions()

    search.observe([5.0], [False], progresses=[432.0])

    candidate = search.best_candidate()
    assert candidate is not None
    assert candidate.progress == 432.0

    search.next_actions()
    search.observe(
        [-10.0],
        [True],
        {0: SimpleNamespace(completed=False, progress=999.0)},
        progresses=[999.0],
    )
    retained = search.best_candidate()
    assert retained is not None
    assert retained.progress == 432.0


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


def test_training_refuses_existing_policy_without_force(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    policy_path = output / "Level1-1.zip"
    policy_path.write_bytes(b"existing policy")
    monkeypatch.setattr(
        train_module, "resolve_state_name", lambda state, **_kwargs: state
    )

    with pytest.raises(SystemExit, match="pass --overwrite"):
        train_module.main(
            ["Level1-1", "--algorithm", "jerk", "--output", str(output)]
        )

    assert policy_path.read_bytes() == b"existing policy"


def test_policy_save_requires_force_to_replace_existing_file(tmp_path) -> None:
    policy_path = tmp_path / "Level1-1.zip"
    policy_path.write_bytes(b"existing policy")
    policy = JerkPolicy(
        action_names=ACTIONS,
        action_runs=_runs((1, 2)),
        fallback_action=0,
    )

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        train_module._save_policy(policy, policy_path)
    assert policy_path.read_bytes() == b"existing policy"

    train_module._save_policy(policy, policy_path, force=True)
    assert JerkPolicy.load(policy_path).action_runs == _runs((1, 2))


def test_training_continues_on_completion_unless_disabled(tmp_path, monkeypatch) -> None:
    class FakeTask:
        instances: list["FakeTask"] = []

        def __init__(self, **_kwargs) -> None:
            self.steps = 0
            self.instances.append(self)

        def reset(self) -> np.ndarray:
            return np.zeros((1, 1), dtype=np.uint8)

        def step(self, actions):
            self.steps += 1
            count = len(actions)
            successes = np.ones(count, dtype=np.bool_)
            records = {
                lane: SimpleNamespace(completed=True, progress=3161.0)
                for lane in range(count)
            }
            return (
                np.zeros((count, 1), dtype=np.uint8),
                np.ones(count, dtype=np.float64),
                np.zeros(count, dtype=np.bool_),
                records,
                successes,
            )

        def reset_lanes(self, _mask) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(train_module, "MarioJerkTask", FakeTask)
    monkeypatch.setattr(
        train_module, "resolve_state_name", lambda state, **_kwargs: state
    )

    stopped_output = tmp_path / "stopped"
    assert (
        train_module.main(
            [
                "Level1-1",
                "--algorithm",
                "jerk",
                "--output",
                str(stopped_output),
                "--transitions",
                "4",
                "--lanes",
                "1",
                "--log-every",
                "10",
                "--stop-on-completion",
            ]
        )
        == 0
    )
    assert FakeTask.instances[-1].steps == 1
    stopped_metrics = json.loads(
        stopped_output.joinpath("episodes.jsonl").read_text().splitlines()[-1]
    )
    assert stopped_metrics["stopped_on_completion"] is True
    assert stopped_metrics["budget_exhausted"] is False

    continuous_output = tmp_path / "continuous"
    assert (
        train_module.main(
            [
                "Level1-1",
                "--algorithm",
                "jerk",
                "--output",
                str(continuous_output),
                "--transitions",
                "4",
                "--lanes",
                "1",
                "--log-every",
                "10",
            ]
        )
        == 0
    )
    assert FakeTask.instances[-1].steps == 4
    continuous_metrics = json.loads(
        continuous_output.joinpath("episodes.jsonl").read_text().splitlines()[-1]
    )
    assert continuous_metrics["stopped_on_completion"] is False
    assert continuous_metrics["budget_exhausted"] is True


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


def test_jerk_search_retains_successful_program() -> None:
    search = _search(run_duration_mean=1.0, run_duration_max=1)
    first = search.next_actions()
    search.observe([1.0], [False])
    second = search.next_actions()
    record = SimpleNamespace(completed=True, progress=3161.0)
    search.observe([2.0], [True], {0: record})

    candidate = search.best_candidate()
    assert candidate is not None
    assert candidate.completed is True
    assert candidate.progress == 3161
    assert candidate.step_count == 2
    assert [run.action for run in candidate.runs] == [int(first[0]), int(second[0])]
    assert candidate.mean_return == 3.0


def test_jerk_run_sampling_is_seeded_bounded_and_covers_action_set() -> None:
    first = _search(initial_probability=0.0, max_probability=0.0)
    second = _search(initial_probability=0.0, max_probability=0.0)

    first_actions = [int(first.next_actions()[0]) for _ in range(512)]
    second_actions = [int(second.next_actions()[0]) for _ in range(512)]

    assert first_actions == second_actions
    assert set(first_actions) == set(range(len(ACTIONS)))
    assert all(run.duration <= first.run_duration_max for run in first._lanes[0].runs)
    assert all(
        left.action != right.action
        for left, right in zip(first._lanes[0].runs, first._lanes[0].runs[1:])
    )


def test_jerk_replays_runs_then_samples_a_new_run() -> None:
    search = _search(
        initial_probability=1.0,
        max_probability=1.0,
        protected_prefix_runs=3,
    )
    prefix = _runs((1, 2), (2, 1), (3, 2))
    search._retained[prefix] = _candidate(prefix, 1.0, progress=100.0)
    search._start_lane(0)

    replayed = [int(search.next_actions()[0]) for _ in range(5)]
    sampled_suffix = int(search.next_actions()[0])

    assert replayed == [1, 1, 2, 3, 3]
    assert sampled_suffix in range(len(ACTIONS))
    assert sampled_suffix != 3
    assert search._lanes[0].mode == "explore"


def test_jerk_local_parent_cut_uses_run_boundaries() -> None:
    search = _search(
        initial_probability=1.0,
        max_probability=1.0,
        protected_prefix_runs=8,
        max_prefix_shorten_runs=16,
        deep_mutation_probability=0.0,
    )
    program = _runs(*( (index % 2, 2) for index in range(40) ))
    search._retained[program] = _candidate(program, 1.0, progress=100.0)
    search._start_lane(0)

    replay_limit = search._lanes[0].replay_limit_runs
    assert 24 <= replay_limit <= 39
    assert replay_limit >= search.protected_prefix_runs
    assert len(program) - replay_limit <= search.max_prefix_shorten_runs


def test_jerk_archive_distribution_is_return_weighted() -> None:
    search = _search()
    low = _runs((0, 1))
    middle = _runs((1, 1))
    high = _runs((2, 1))
    search._retained[high] = _candidate(high, 5.0)
    search._retained[low] = _candidate(low, -5.0)
    search._retained[middle] = _candidate(middle, 0.0)

    candidates, probabilities = search._retained_distribution()

    assert [candidate.runs for candidate in candidates] == [low, middle, high]
    assert np.all(probabilities > 0.0)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[1] == pytest.approx(1.0 / 3.0)
    assert probabilities[2] == pytest.approx(2.0 / 3.0)


def test_jerk_duplicate_updates_stats_and_unique_insert_evicts_worst() -> None:
    search = _search(retained_limit=1)
    first = _runs((0, 1))
    second = _runs((1, 1))
    third = _runs((2, 1))
    fourth = _runs((3, 1))
    search._upsert_retained(first, score_return=1.0, completed=False, progress=10.0)
    search._upsert_retained(second, score_return=2.0, completed=False, progress=20.0)
    search._upsert_retained(second, score_return=4.0, completed=True, progress=20.0)
    search._upsert_retained(third, score_return=3.0, completed=False, progress=30.0)
    search._upsert_retained(fourth, score_return=4.0, completed=True, progress=40.0)

    assert search.retained_count == 3
    assert search.locked_count == 2
    assert search.incomplete_retained_count == 1
    assert set(search._retained) == {second, third, fourth}
    assert search._retained[second].mean_return == 3.0
    assert search._retained[second].completed is True


def test_completed_candidates_rank_by_return_only() -> None:
    short = _candidate(_runs((1, 2)), 1.0, completed=True, progress=9999.0)
    long = _candidate(_runs((1, 3)), 2.0, completed=True, progress=10.0)
    same_return_more_runs = _candidate(
        _runs((1, 1), (2, 2)), 1.0, completed=True, progress=20.0
    )

    assert long.rank > short.rank
    assert short.rank == same_return_more_runs.rank


def test_incomplete_candidates_rank_by_return_only() -> None:
    farther = _candidate(_runs((1, 3)), 1.0, progress=100.0)
    richer = _candidate(_runs((1, 1)), 1000.0, progress=99.0)
    same_return_less_progress = _candidate(_runs((2, 4)), 1.0, progress=1.0)

    assert richer.rank > farther.rank
    assert farther.rank == same_return_less_progress.rank


def test_completed_candidates_rank_above_incomplete_candidates() -> None:
    completed = _candidate(_runs((1, 3)), 1.0, completed=True)
    incomplete = _candidate(_runs((1, 1)), 1000.0)

    assert completed.rank > incomplete.rank


def test_jerk_policy_zip_round_trip_and_lane_resets(tmp_path) -> None:
    path = tmp_path / "model.zip"
    JerkPolicy(
        action_names=ACTIONS,
        action_runs=_runs((2, 2), (4, 1)),
        fallback_action=0,
    ).save(path)
    loaded = JerkPolicy.load(path)

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
            JERK_POLICY_MEMBER,
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

    with pytest.raises(ValueError, match="unsupported JERK policy schema version"):
        load_jerk_checkpoint(path)


def test_named_run_checkpoint_round_trip(tmp_path) -> None:
    path = save_jerk_checkpoint(
        tmp_path / "policy.zip",
        (("right", 2), ("right_a", 1)),
        timesteps=123,
        episodes=4,
        best_reward=99.0,
        metadata={"terminate_on_level_change": False},
    )

    policy = load_jerk_checkpoint(path)
    observations = np.zeros((1, 1, 1, 1), dtype=np.uint8)
    actions = [int(policy.predict(observations)[0][0]) for _ in range(4)]

    assert actions == [1, 1, 3, 0]
    assert policy.timesteps == 123
    assert policy.metadata["terminate_on_level_change"] is False
    assert policy.action_set == "standard"


def test_exploit_probability_matches_rlab_schedule() -> None:
    assert exploit_probability(0, 1_000) == pytest.approx(0.25)
    assert exploit_probability(500, 1_000) == pytest.approx(0.75)
    assert exploit_probability(1_000, 1_000) == pytest.approx(0.9)
