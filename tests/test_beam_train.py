from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import train_beam
from supermariobrosnes_turbo.beam import BeamSearch
from supermariobrosnes_turbo.jerk import ActionRun, JerkPolicy, RetainedProgram


ACTIONS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left")


def _runs(*values: tuple[int, int]) -> tuple[ActionRun, ...]:
    return tuple(ActionRun(action, duration) for action, duration in values)


def _search(*, n_envs: int = 1, beam_width: int = 2) -> BeamSearch:
    return BeamSearch(
        n_envs=n_envs,
        seed=7,
        action_names=ACTIONS,
        fallback_action="noop",
        beam_width=beam_width,
        refresh_episodes=1,
        protected_prefix_runs=1,
        mutation_runs=1,
        branch_durations=(1, 2),
        run_duration_mean=1.0,
        run_duration_max=1,
    )


def test_beam_prunes_to_width_and_keeps_success_above_higher_failure() -> None:
    search = _search(beam_width=2)
    search._upsert_candidate(
        _runs((1, 1)), score_return=100.0, completed=False, progress=100.0
    )
    search._upsert_candidate(
        _runs((2, 1)), score_return=2.0, completed=True, progress=200.0
    )
    search._upsert_candidate(
        _runs((3, 1)), score_return=1.0, completed=False, progress=300.0
    )
    search._refresh_beam()

    assert search.beam_count == 2
    assert _runs((2, 1)) in search._beam
    assert _runs((1, 1)) in search._beam


def test_beam_expands_parent_prefix_with_systematic_action_run_branch() -> None:
    search = _search()
    parent_runs = _runs((1, 2), (2, 2))
    search._beam[parent_runs] = RetainedProgram(
        runs=parent_runs,
        return_sum=10.0,
        return_count=1,
    )
    search._parents = tuple(search._beam.values())
    search._start_lane(0)

    actions = [int(search.next_actions()[0]) for _ in range(3)]

    assert actions == [1, 1, 0]
    assert search._lanes[0].runs == list(_runs((1, 2), (0, 1)))


def test_beam_observation_refreshes_parents_and_emits_compatible_policy() -> None:
    search = _search()
    first = int(search.next_actions()[0])
    search.observe(
        [3.0],
        [True],
        {0: SimpleNamespace(completed=True, progress=3161.0)},
    )

    policy = search.policy()

    assert search.generation == 1
    assert search.successful_episodes == 1
    assert isinstance(policy, JerkPolicy)
    assert policy.action_runs == _runs((first, 1))
    assert policy.metadata["search_algorithm"] == "beam"


def test_beam_cli_defaults_to_separate_output_and_shared_action_contract() -> None:
    args = train_beam.build_parser().parse_args(["Level1-1"])

    assert args.n_envs == 64
    assert args.beam_width == 16
    assert args.stop_on_completion is True
    assert train_beam.run_directory_for_level("Level1-1") == Path(
        "runs/Level1-1-beam"
    )
