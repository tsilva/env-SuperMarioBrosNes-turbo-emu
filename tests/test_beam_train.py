from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from supermariobrosnes_turbo import beam_training, training
from supermariobrosnes_turbo.beam import BeamCandidate, BeamSearch
from supermariobrosnes_turbo.jerk import (
    ActionRun,
    JerkPolicy,
    RetainedProgram,
    run_directory_for_state,
)


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


def test_beam_prunes_to_width_and_keeps_success_plus_furthest_failure() -> None:
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
    assert _runs((3, 1)) in search._beam


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


def test_beam_active_best_candidate_carries_observed_progress() -> None:
    search = _search()
    search.next_actions()

    search.observe([5.0], [False], progresses=[654.0])

    candidate = search.best_candidate()
    assert candidate is not None
    assert candidate.progress == 654.0

    search.next_actions()
    search.observe(
        [-10.0],
        [True],
        {0: SimpleNamespace(completed=False, progress=999.0)},
        progresses=[999.0],
    )
    retained = search.best_candidate()
    assert retained is not None
    assert retained.progress == 999.0


def test_score_first_beam_retains_furthest_prefix_not_earliest_return() -> None:
    search = _search()

    for progress, done in ((10.0, False), (20.0, False), (30.0, True)):
        search.next_actions()
        search.observe(
            [-0.1],
            [done],
            (
                {0: SimpleNamespace(completed=False, progress=progress)}
                if done
                else None
            ),
            progresses=[progress],
        )

    candidate = search.best_candidate()
    assert candidate is not None
    assert candidate.progress == 30.0
    assert candidate.step_count == 3
    assert candidate.incomplete_return == pytest.approx(-0.3)


def test_beam_cli_defaults_to_canonical_output_and_shared_action_contract() -> None:
    parser = training.build_parser()
    args = parser.parse_args(["Level1-1", "--algorithm", "beam"])
    training._apply_algorithm_defaults(parser, args)

    assert args.lanes == 64
    assert args.beam_width == 16
    assert args.protected_prefix_runs == 8
    assert args.improvement_protected_prefix_runs == 0
    assert args.action_set == "standard"
    assert args.initial_policy is None
    assert args.continue_after_completion is False
    assert run_directory_for_state("Level1-1") == Path("runs/Level1-1")
    assert beam_training._overwrite_existing(args)


def test_beam_custom_output_requires_explicit_overwrite() -> None:
    parser = training.build_parser()
    custom = parser.parse_args(
        ["Level1-1", "--algorithm", "beam", "--output", "runs/comparison"]
    )
    forced = parser.parse_args(
        [
            "Level1-1",
            "--algorithm",
            "beam",
            "--output",
            "runs/comparison",
            "--overwrite",
        ]
    )

    assert not beam_training._overwrite_existing(custom)
    assert beam_training._overwrite_existing(forced)


def test_beam_keeps_completed_return_separate_from_failed_prefix_score() -> None:
    search = _search()
    runs = _runs((1, 2))

    search._upsert_candidate(
        runs, score_return=100.0, completed=False, progress=200.0
    )
    candidate = search._upsert_candidate(
        runs, score_return=10.0, completed=True, progress=300.0
    )

    assert candidate.incomplete_return == 100.0
    assert candidate.completed_return == 10.0
    assert candidate.mean_return == 10.0
    assert search.best_success_return == 10.0


def test_improvement_mode_promotes_success_and_mutates_from_root() -> None:
    search = BeamSearch(
        n_envs=1,
        seed=7,
        action_names=("noop",),
        fallback_action="noop",
        beam_width=1,
        refresh_episodes=1,
        protected_prefix_runs=1,
        mutation_runs=1,
        branch_durations=(1,),
        run_duration_mean=1.0,
        run_duration_max=1,
        improve_after_completion=True,
    )
    search.next_actions()

    search.observe(
        [1.0],
        [True],
        {0: SimpleNamespace(completed=True, progress=10.0)},
    )

    assert search.improvement_mode
    assert search.best_success_return == 1.0
    assert search.coverage_total == 1
    assert search._lanes[0].parent is search.best_candidate()
    assert {
        job.replay_limit_runs for job in search._coverage_templates
    } == {0}
    assert {
        job.resume_parent_run_index for job in search._coverage_templates
    } == {1}

    search.next_actions()
    search.observe(
        [2.0],
        [True],
        {0: SimpleNamespace(completed=True, progress=20.0)},
    )

    assert search.best_success_return == 2.0
    assert search.improvement_count == 1
    assert search.policy().best_reward == 2.0


def test_beam_reserves_incomplete_parent_capacity_after_success() -> None:
    search = _search(beam_width=4)
    for action, score in ((1, 10.0), (2, 9.0), (3, 8.0), (4, 7.0)):
        search._upsert_candidate(
            _runs((action, 1)),
            score_return=score,
            completed=True,
            progress=100.0,
        )
    for action, score in ((5, 100.0), (6, 90.0)):
        search._upsert_candidate(
            _runs((action, 1)),
            score_return=score,
            completed=False,
            progress=200.0,
        )

    search._refresh_beam()

    retained = tuple(search._beam.values())
    assert sum(candidate.completed for candidate in retained) == 2
    assert sum(not candidate.completed for candidate in retained) == 2
    assert isinstance(search.best_candidate(), BeamCandidate)


def test_unsolved_beam_switches_to_systematic_deepening() -> None:
    search = BeamSearch(
        n_envs=1,
        seed=7,
        action_names=("noop", "right"),
        fallback_action="noop",
        beam_width=1,
        refresh_episodes=1,
        protected_prefix_runs=0,
        mutation_runs=1,
        branch_durations=(1,),
        run_duration_mean=1.0,
        run_duration_max=1,
        deepening_after_generations=1,
    )
    search.next_actions()

    search.observe(
        [1.0],
        [True],
        {0: SimpleNamespace(completed=False, progress=10.0)},
    )

    assert search.improvement_mode
    assert search.best_success_return is None
    assert search.cut_depth == 1
    assert search.coverage_total == 4


def test_deepening_alternates_earlier_cuts_with_local_refinement() -> None:
    search = _search(beam_width=1)
    parent = BeamCandidate(
        runs=_runs(*((index % 2 + 1, 1) for index in range(8))),
        incomplete_return=1.0,
    )
    search._beam = {parent.runs: parent}
    search._parents = (parent,)
    search._improvement_mode = True
    search._cut_depth = 1
    search._deepening_frontier = 1

    search._advance_improvement_generation()
    assert search.cut_depth == 2
    search._advance_improvement_generation()
    assert search.cut_depth == 1
    search._advance_improvement_generation()
    assert search.cut_depth == 4


def test_completed_parent_mutation_replays_its_proven_suffix() -> None:
    search = _search(beam_width=1)
    parent = BeamCandidate(
        runs=_runs((1, 1), (2, 1), (3, 1)),
        completed_return=10.0,
        progress=100.0,
    )
    search._beam = {parent.runs: parent}
    search._parents = (parent,)
    search._best_success = parent
    search._improvement_mode = True
    search._cut_depth = 2
    search._prepare_coverage()
    job = next(
        job
        for job in search._coverage_templates
        if job.branch == ActionRun(4, 1)
    )
    search._lanes[0] = search._state_for_job(job)

    actions = [int(search.next_actions()[0]) for _ in range(3)]

    assert actions == [1, 4, 3]
    assert job.replay_limit_runs == 1
    assert job.resume_parent_run_index == 2
    assert all(
        template.replay_limit_runs < len(parent.runs)
        for template in search._coverage_templates
    )


def test_beam_warm_start_replays_the_seed_program_exactly_on_one_lane() -> None:
    search = _search(n_envs=2)
    runs = _runs((1, 2), (2, 1))

    search.seed_program(runs)
    actions = [int(search.next_actions()[0]) for _ in range(3)]

    assert actions == [1, 1, 2]


def test_beam_remaps_basic_warm_start_into_standard_by_action_name() -> None:
    policy = JerkPolicy(
        action_names=ACTIONS,
        action_runs=_runs((1, 2), (6, 1)),
        fallback_action=0,
    )

    remapped = beam_training._remap_policy_runs(
        policy,
        tuple(training.ACTION_SETS["standard"]),
    )

    assert remapped == _runs((1, 2), (6, 1))


def test_beam_rejects_warm_start_action_absent_from_selected_table() -> None:
    policy = JerkPolicy(
        action_names=("noop", "left"),
        action_runs=_runs((1, 1)),
        fallback_action=0,
    )

    with pytest.raises(ValueError, match="'left'"):
        beam_training._remap_policy_runs(policy, ("noop", "right"))


def test_strictly_better_unsolved_frontier_is_queued_for_immediate_extension() -> None:
    search = _search(beam_width=1)
    parent = BeamCandidate(
        runs=_runs((1, 1)),
        incomplete_return=1.0,
    )
    search._beam = {parent.runs: parent}
    search._parents = (parent,)
    search._improvement_mode = True
    search._cut_depth = 1
    search._prepare_coverage()

    better = search._upsert_candidate(
        _runs((1, 1), (2, 1)),
        score_return=2.0,
        completed=False,
        progress=20.0,
    )

    assert search.frontier_pending == len(search._branches)
    assert all(job.parent is better for job in search._frontier_queue)
    assert not any(job.required for job in search._frontier_queue)
