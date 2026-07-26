from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from supermariobrosnes_turbo import go_explore as go_explore_module
from supermariobrosnes_turbo.go_explore import GoExploreCandidate, GoExploreSearch
from supermariobrosnes_turbo.action_run import ActionRun, ActionRunPolicy


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


def test_go_explore_selection_weight_tree_matches_linear_cdf() -> None:
    weights = [2.0, 1.0, 3.0, 4.0]
    tree = go_explore_module._SelectionWeightTree()
    for weight in weights:
        tree.append(weight)

    for unit in (0.0, 0.11, 0.29, 0.31, 0.67, 0.999):
        expected = int(
            np.searchsorted(
                np.cumsum(weights),
                unit * sum(weights),
                side="right",
            )
        )
        assert tree.sample(unit) == expected

    tree.add(1, 4.0)
    weights[1] += 4.0
    assert tree.total == sum(weights)
    for unit in (0.07, 0.25, 0.51, 0.83):
        expected = int(
            np.searchsorted(
                np.cumsum(weights),
                unit * sum(weights),
                side="right",
            )
        )
        assert tree.sample(unit) == expected


def test_go_explore_selection_tree_matches_previous_seeded_choices() -> None:
    count = 257
    visits = np.arange(1, count + 1, dtype=np.int64) % 19 + 1
    reference_selections = np.zeros(count, dtype=np.int64)
    tree_selections = np.zeros(count, dtype=np.int64)
    weights = 1.0 / np.sqrt(1.0 + reference_selections) + 1.0 / np.sqrt(1.0 + visits)
    tree = go_explore_module._SelectionWeightTree()
    for weight in weights:
        tree.append(float(weight))
    seed = np.random.SeedSequence([7, 0, 0x474F4558])
    reference_rng = np.random.default_rng(seed)
    tree_rng = np.random.default_rng(seed)

    for _ in range(1_000):
        weights = 1.0 / np.sqrt(1.0 + reference_selections) + 1.0 / np.sqrt(
            1.0 + visits
        )
        expected = int(reference_rng.choice(count, p=weights / weights.sum()))
        actual = tree.sample(tree_rng.random())
        assert actual == expected
        reference_selections[expected] += 1
        old_weight = 1.0 / np.sqrt(1.0 + tree_selections[actual]) + 1.0 / np.sqrt(
            1.0 + visits[actual]
        )
        tree_selections[actual] += 1
        new_weight = 1.0 / np.sqrt(1.0 + tree_selections[actual]) + 1.0 / np.sqrt(
            1.0 + visits[actual]
        )
        tree.add(actual, float(new_weight - old_weight))


def test_go_explore_restart_updates_only_selected_cell_weight(monkeypatch) -> None:
    search = _search(n_envs=8)
    keys = tuple(f"cell-{index}" for index in range(search.n_envs))
    snapshots = tuple(f"snapshot-{index}" for index in range(search.n_envs))
    search.initialize(keys, snapshots)
    original_weight = go_explore_module._archive_cell_selection_weight
    evaluated_keys: list[str] = []

    def tracked_weight(cell):
        evaluated_keys.append(cell.key)
        return original_weight(cell)

    monkeypatch.setattr(
        go_explore_module,
        "_archive_cell_selection_weight",
        tracked_weight,
    )
    restored = search.restart([True, False, False, False, False, False, False, False])

    assert restored[0] in snapshots
    assert evaluated_keys == [
        next(cell.key for cell in search.archive.values() if cell.selections == 1)
    ]


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
    search.commit_archive((SimpleNamespace(native=SimpleNamespace(nbytes=8_192)),))

    assert observation.archive_mask.tolist() == [True]
    assert search.archive_memory_bytes >= initial_bytes + 8_192


def test_go_explore_updates_archive_memory_from_cached_cell_estimates(
    monkeypatch,
) -> None:
    search = _search(explore_steps=1)
    search.initialize(
        ("root",),
        (SimpleNamespace(native=SimpleNamespace(nbytes=4_096)),),
    )
    original_estimator = go_explore_module._archive_cell_memory_bytes
    estimated_keys: list[str] = []

    def tracked_estimator(cell):
        estimated_keys.append(cell.key)
        return original_estimator(cell)

    monkeypatch.setattr(
        go_explore_module,
        "_archive_cell_memory_bytes",
        tracked_estimator,
    )
    initial_bytes = search.archive_memory_bytes
    search.next_actions()
    search.observe([1.0], [False], ["root"], progresses=[16.0])
    search.commit_archive((SimpleNamespace(native=SimpleNamespace(nbytes=8_192)),))

    assert estimated_keys == ["root"]
    assert search.archive_memory_bytes >= initial_bytes + 4_096


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

    monkeypatch.setattr(go_explore_module, "SUCCESS_GUIDED_RESTORE_PROBABILITY", 1.0)
    selected = {search.restart([True])[0] for _ in range(30)}

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
    policy = ActionRunPolicy.load(path)

    assert policy.action_runs == search.best_candidate().runs
    assert policy.metadata["search_algorithm"] == "go-explore"
    assert policy.metadata["go_explore_phase"] == "trajectory_finding"
    assert policy.metadata["robustification"] is False
