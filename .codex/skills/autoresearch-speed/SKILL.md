---
name: autoresearch-speed
description: Lightweight Super Mario Bros NES throughput optimization loop for this repo. Use when optimizing, profiling, benchmarking, or iterating on emulator speed around env_steps_per_sec without feature or accuracy regressions, including repeated +10% cumulative improvement rounds resumed from AUTORESEARCH_ROOT_PATH state.
---

# Autoresearch Speed

## Guardrails

- Read this skill's `SPECS.md` and repo-root `SPECS.md` first.
- Work on the live current branch. Do not create/switch branches, fork workers, use SSH/tailnet/cloud/Modal, merge, push, or delete branches without explicit current-turn approval.
- Count only valid `env_steps_per_sec` gains. Each candidate needs a concrete speed mechanism, cheap falsification path, and no risk to API shape, deterministic lanes, observation bytes, rewards, resets, terminations, infos, state loading, or benchmark workload.
- Get the ROM from an explicit flag or the Stable Retro-compatible `RETRO_DATA_PATH` tree. Continue resolving `AUTORESEARCH_ROOT_PATH` from flags, env, or `.env`. Keep mutable artifacts out of the repo; the controller owns root files such as `results.tsv`, `current.json`, `ideas.md`, and `scratchpad.md`.

## Round Contract

Open-ended throughput launches run one improvement round. Target `>= 1.10` cumulative `env_steps_per_sec` versus the fixed round baseline unless the user requests more.

Every newly created autoresearch goal starts a fresh improvement round from the
live `HEAD`. Completed rounds and accepted results in controller state are
history only: do not resume their baseline or treat their gains as progress in
the new goal. Only a continuation of the same active goal resumes that goal's
already-fixed round baseline.

At orientation: run `status`; read `results.tsv`, `current.json`, `scratchpad.md`,
and `ideas.md`. For a new goal, resolve and calibrate live `HEAD`, then fix that
exact commit as the round baseline. For a continuation of the same goal, recover
its fixed baseline and latest accepted `keep`, `keep_stack`, or
`keep_small_gain` result from the goal's recorded state.

Keep the baseline fixed for the invocation. Stack small accepted gains, but complete the round only with accepted controller evidence against that baseline at ratio `>= 1.10`. Unaccepted live `HEAD` speedups do not count. Do not finish by rediscovering an old accepted fast commit.

On completion: record aggregate evidence, update `scratchpad.md`, leave work committed on the current branch, report artifacts and next baseline, and stop.

## Control Plane

Use `scripts/autoresearch.py` as the routine interface. The controller owns command shapes, benchmark sizes, exact-ref tiers, checks, recording, and next-step hints.

```bash
.venv/bin/python scripts/autoresearch.py init
.venv/bin/python scripts/autoresearch.py status
.venv/bin/python scripts/autoresearch.py next
.venv/bin/python scripts/autoresearch.py probe [--dry-run]
.venv/bin/python scripts/autoresearch.py diagnose [--quick] [--profile] [--dry-run]
.venv/bin/python scripts/autoresearch.py screen <baseline_ref> <candidate_ref> [--dry-run] [--no-record] [-- <extra runner args>]
.venv/bin/python scripts/autoresearch.py checks [--quick] [--dry-run] [--surface auto|native|python|benchmark]
.venv/bin/python scripts/autoresearch.py accept-stack <baseline_ref> <candidate_ref> [--dry-run] [--no-record] [-- <extra runner args>]
.venv/bin/python scripts/autoresearch.py accept <baseline_ref> <candidate_ref> [--full] [--dry-run] [--no-record] [-- <extra runner args>]
.venv/bin/python scripts/autoresearch.py calibrate <ref> [--full] [--dry-run] [--no-record] [-- <extra runner args>]
.venv/bin/python scripts/autoresearch.py record <aggregate.json> [--status <status>] [--description "..."] [--artifact <path>]
```

Use `scripts/run_git_ref_benchmark.py`, `RESULTS_TSV_COLUMNS`, and `scripts/benchmark_sps.py` only as implementation truth. Do not duplicate, rename, split, or alias benchmark fields. If prose and code disagree, trust code and fix prose.

## Loop

1. `orient`: read specs/controller state, lock baseline/target, run `next`.
2. `falsify`: use source inspection, prior rejects, `probe`, `diagnose --quick`, `diagnose --profile`, or narrow equivalence checks. These justify discard or continued work, never keep.
3. `prototype`: make one small edit. After native edits run `.venv/bin/python -m maturin develop --release` before Python timing.
4. `check`: before exact-ref triage run `scripts/autoresearch.py checks --quick --surface <surface>`. Add targeted tests first for observations, rewards, terminations, resets, actions, infos, preprocessing, state loading, or public Python contracts.
5. `screen`: commit only plausible candidates, then `scripts/autoresearch.py screen <baseline_ref> <candidate_ref>`. One default screen per candidate is enough unless asked; cheap screens may promote low-risk winners into a bundle.
6. `accept-stack`: use for screened bundles with likely `>= 5%` upside, one large candidate, about three good candidates, or about eight cheap screens. Run correctness checks once, then `scripts/autoresearch.py accept-stack <baseline_ref> <candidate_ref>`. For round completion, baseline must be the fixed round baseline and accepted median ratio `>= 1.10`. The stack is the proof unit.
7. `accept`: use for strongest evidence, contract-sensitive changes, public claims, baseline resets, or direct user requests: `scripts/autoresearch.py accept <baseline_ref> <candidate_ref>`. Use `--full` only when needed.
8. `record`: finalized `screen`, `accept-stack`, `accept`, and `calibrate` runs auto-record by default. Use manual `record` only for imports or overrides.
9. `retrospect`: write compact lessons to `scratchpad.md`; do not update docs or skills from scratchpad notes unless asked.

## Evidence Rules

- `probe`, `diagnose`, and `make benchmark` are learning evidence only.
- `local_triage` is screening only.
- `stack_acceptance` can justify `keep_stack` for a stack, not individual attribution.
- `local_acceptance` is strongest for `keep`, `keep_small_gain`, accepted speedups, or gold-standard baseline updates.
- Never complete a round from `inconclusive`, load-failed, busy-machine, or non-recorded evidence, even if raw ratio is above `1.10`.
- Discard when likely gain is smaller than proof cost. Treat failures as regressions unless proven unrelated.
- Do not accept busy-machine or load-failed official measurements unless the user explicitly forces or waives that gate.
- Use `calibrate <ref>` periodically for accepted-baseline variance; calibration is context, not paired acceptance.

## Candidate Shape

Prefer Rust changes in `src/emulator.rs`, `src/vec_env.rs`, and `src/py_api.rs` that improve real PPO rollout/training throughput under stochastic policy actions. Reject benchmark-only shortcuts that depend on repeated identical states, uniform/no-op/deterministic actions, disabled PPO-relevant termination/info semantics, or conditions unlikely during PPO rollout collection. Preserve deterministic lanes, public Python API, benchmark workload identity, observation/reward/termination semantics, state handling, and cropped grayscale tile/sprite rendering semantics.

## Reporting

Report compactly:

```text
phase: orient | diagnose | prototype | screen | acceptance | record | pause
inputs: refs/artifacts read
outputs: commits/results/artifacts written
decision: keep | keep_small_gain | keep_stack | discard | discard_stack | inconclusive | defer | pause
next: one concrete next action
```

Pause when access fails, user limits are exhausted, regressions persist, refs are missing, metadata or timing is untrustworthy, unrelated branch changes would be included, two candidates die for the same reason, or the user asks to stop.

At session end, report branch, baseline/candidate refs, latest aggregate, accepted commits, rejects, checks, changed files, cleanup status, benchmark limits, artifacts, round baseline, round target, cumulative ratio versus the round baseline, and the next experiment.
