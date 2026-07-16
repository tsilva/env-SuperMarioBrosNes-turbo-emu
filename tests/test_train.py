from __future__ import annotations

import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

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
from train import (
    episode_boundary,
    exploit_probability,
    sanitize_progress_x,
    shape_step_rewards,
)


ACTIONS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left")


def _runs(*values: tuple[int, int]) -> tuple[ActionRun, ...]:
    return tuple(ActionRun(action, duration) for action, duration in values)


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


def test_step_reward_charges_time_on_every_transition() -> None:
    rewards = shape_step_rewards(
        np.asarray([0, 10, 0]),
        np.asarray([0, 100, 0]),
        np.asarray([False, False, True]),
        step_cost=0.1,
    )

    np.testing.assert_allclose(rewards, [-0.1, 10.9, -25.1])


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


def test_exploit_probability_matches_rlab_schedule() -> None:
    assert exploit_probability(0, 1_000) == pytest.approx(0.25)
    assert exploit_probability(500, 1_000) == pytest.approx(0.75)
    assert exploit_probability(1_000, 1_000) == pytest.approx(0.9)
