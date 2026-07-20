from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from supermariobrosnes_turbo.go_explore import GoExploreSearch
from supermariobrosnes_turbo.jerk import JerkPolicy


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
