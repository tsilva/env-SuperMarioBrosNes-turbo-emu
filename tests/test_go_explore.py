from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from supermariobrosnes_turbo import go_explore as go_explore_module
from supermariobrosnes_turbo.go_explore import GoExploreCandidate, GoExploreSearch
from supermariobrosnes_turbo.jerk import ActionRun, JerkPolicy


ACTIONS = ("noop", "right", "a")


def _search(*, n_envs: int = 1, explore_steps: int = 4) -> GoExploreSearch:
    return GoExploreSearch(
        n_envs=n_envs,
        seed=7,
        action_names=ACTIONS,
        fallback_action="noop",
        explore_steps=explore_steps,
        run_duration_mean=1.0,
        run_duration_max=1,
    )


def test_go_explore_candidate_canonicalizes_external_trajectories_once() -> None:
    candidate = GoExploreCandidate(
        runs=(ActionRun(1, 2), ActionRun(1, 3), ActionRun(2, 4)),
        episode_return=1.0,
        progress=2.0,
    )

    assert candidate.runs == (ActionRun(1, 5), ActionRun(2, 4))
    assert candidate.step_count == 9


def test_go_explore_observe_does_not_recanonicalize_lane_trajectories(
    monkeypatch,
) -> None:
    search = _search()
    search.initialize(("root",), ("root-snapshot",))

    def unexpected_canonicalization(_runs):
        raise AssertionError("canonical lane trajectory was rebuilt")

    monkeypatch.setattr(
        go_explore_module, "canonicalize_runs", unexpected_canonicalization
    )
    search.next_actions()
    observation = search.observe([1.0], [False], ["new"], progresses=[16.0])
    search.commit_archive(("new-snapshot",))

    assert observation.archive_mask.tolist() == [True]
    assert search.archive["new"].step_count == 1
    assert search.best_candidate().step_count == 1


def test_go_explore_archives_best_batch_trajectory_for_each_cell() -> None:
    search = _search(n_envs=2)
    search.initialize(("root", "root"), ("root-0", "root-1"))
    search.next_actions()

    observation = search.observe(
        [1.0, 2.0],
        [False, False],
        ["cell", "cell"],
        progresses=[10.0, 10.0],
    )

    np.testing.assert_array_equal(observation.archive_mask, [False, True])
    search.commit_archive((None, "best-cell-snapshot"))
    cell = search.archive["cell"]
    assert cell.snapshot == "best-cell-snapshot"
    assert cell.episode_return == 2.0
    assert cell.step_count == 1
    assert cell.visits == 2


def test_go_explore_replaces_a_shorter_cell_path_with_higher_return() -> None:
    search = _search(explore_steps=4)
    search.initialize(("root",), ("root-snapshot",))

    search.next_actions()
    first = search.observe([1.0], [False], ["cell"], progresses=[10.0])
    search.commit_archive(("shorter-snapshot",))
    assert first.archive_mask.tolist() == [True]
    assert search.archive["cell"].step_count == 1

    search.next_actions()
    second = search.observe([2.0], [False], ["cell"], progresses=[10.0])
    search.commit_archive(("higher-return-snapshot",))

    assert second.archive_mask.tolist() == [True]
    cell = search.archive["cell"]
    assert cell.snapshot == "higher-return-snapshot"
    assert cell.episode_return == 3.0
    assert cell.step_count == 2
    assert search.archive_update_count == 1


def test_go_explore_restores_archived_cells_after_exploration_horizon() -> None:
    search = _search(explore_steps=1)
    search.initialize(("root",), ("root-snapshot",))
    search.next_actions()
    observation = search.observe(
        [1.0],
        [False],
        ["new"],
        progresses=[16.0],
    )
    search.commit_archive(("new-snapshot",))

    assert observation.restart_mask.tolist() == [True]
    assert search.restart(observation.restart_mask)[0] in {
        "root-snapshot",
        "new-snapshot",
    }
    assert search.archive_count == 2
    assert search.archive_selection_count == 1
    assert search.archive_visit_count == 2
    assert search.archive_update_count == 0


def test_go_explore_reports_archive_memory_including_native_snapshots() -> None:
    search = _search(explore_steps=1)
    search.initialize(
        ("root",),
        (SimpleNamespace(native=SimpleNamespace(nbytes=4_096)),),
    )
    initial_bytes = search.archive_memory_bytes

    assert initial_bytes >= 4_096

    search.next_actions()
    observation = search.observe([1.0], [False], ["new"], progresses=[16.0])
    search.commit_archive(
        (SimpleNamespace(native=SimpleNamespace(nbytes=8_192)),)
    )

    assert observation.archive_mask.tolist() == [True]
    assert search.archive_memory_bytes >= initial_bytes + 8_192


def test_go_explore_updates_archive_memory_without_rescanning_archive(
    monkeypatch,
) -> None:
    search = _search(explore_steps=1)
    search.initialize(("root",), ("root-snapshot",))
    original_deep_sizeof = go_explore_module._deep_sizeof
    scanned_archive = False

    def tracked_deep_sizeof(value, seen=None):
        nonlocal scanned_archive
        scanned_archive |= value is search.archive
        return original_deep_sizeof(value, seen)

    monkeypatch.setattr(go_explore_module, "_deep_sizeof", tracked_deep_sizeof)
    search.next_actions()
    search.observe([1.0], [False], ["new"], progresses=[16.0])
    search.commit_archive(("new-snapshot",))

    assert not scanned_archive


def test_go_explore_tracks_recent_new_cells_per_visit() -> None:
    search = _search()
    search.initialize(("root",), ("root-snapshot",))

    search.next_actions()
    search.observe([1.0], [False], ["new"], progresses=[16.0])
    search.commit_archive(("new-snapshot",))

    assert search.archive_recent_visit_window == 1
    assert search.archive_recent_new_cell_rate == 1.0
    assert search.archive_visits_per_cell == 1.0

    search.next_actions()
    search.observe([0.0], [False], ["new"], progresses=[16.0])

    assert search.archive_recent_visit_window == 2
    assert search.archive_recent_new_cell_rate == 0.5
    assert search.archive_visits_per_cell == 1.5


def test_go_explore_credits_and_samples_the_best_success_lineage(
    monkeypatch,
) -> None:
    search = _search(explore_steps=8)
    search.initialize(("root",), ("root-snapshot",))

    search.next_actions()
    search.observe([1.0], [False], ["prefix"], progresses=[100.0])
    search.commit_archive(("prefix-snapshot",))
    assert search.archive["prefix"].parent_key == "root"

    with monkeypatch.context() as patch:
        patch.setattr(search, "_select_cell", lambda _lane: search.archive["prefix"])
        search.restart([True])

    search.next_actions()
    search.observe([2.0], [False], ["winning-suffix"], progresses=[200.0])
    search.commit_archive(("winning-suffix-snapshot",))
    assert search.archive["winning-suffix"].parent_key == "prefix"

    search.next_actions()
    search.observe(
        [5.0],
        [True],
        ["terminal"],
        {0: SimpleNamespace(completed=True, progress=300.0)},
        progresses=[300.0],
    )

    assert search.best_success_return == 8.0
    assert search.success_guided_cell_count == 3
    for key in ("root", "prefix", "winning-suffix"):
        assert search.archive[key].best_success_return == 8.0

    monkeypatch.setattr(
        go_explore_module, "SUCCESS_GUIDED_RESTORE_PROBABILITY", 1.0
    )
    selected = {
        search.restart([True])[0]
        for _ in range(30)
    }

    assert selected <= {
        "root-snapshot",
        "prefix-snapshot",
        "winning-suffix-snapshot",
    }
    assert selected
    assert search.success_guided_selection_count == 30


def test_go_explore_locks_successes_and_selects_them_only_by_return() -> None:
    search = _search(explore_steps=1)
    search.initialize(("root",), ("root-snapshot",))
    for reward, progress in ((5.0, 100.0), (7.0, 50.0), (6.0, 500.0)):
        search.next_actions()
        result = search.observe(
            [reward],
            [True],
            ["terminal"],
            {0: SimpleNamespace(completed=True, progress=progress)},
            progresses=[progress],
        )
        search.restart(result.restart_mask)

    assert search.successful_episodes == 3
    assert search.best_success_return == 7.0
    assert search.policy().best_reward == 7.0
    assert search.improvement_count == 1


def test_go_explore_policy_uses_beam_compatible_action_run_format(
    tmp_path: Path,
) -> None:
    search = _search()
    search.initialize(("root",), ("root-snapshot",))
    search.next_actions()
    search.observe(
        [3.0],
        [False],
        ["cell"],
        progresses=[20.0],
    )
    search.commit_archive(("cell-snapshot",))
    path = tmp_path / "go-explore.zip"

    search.policy().save(path)
    policy = JerkPolicy.load(path)

    assert policy.action_runs == search.best_candidate().runs
    assert policy.metadata["search_algorithm"] == "go-explore"
    assert policy.metadata["go_explore_phase"] == "trajectory_finding"
    assert policy.metadata["robustification"] is False
