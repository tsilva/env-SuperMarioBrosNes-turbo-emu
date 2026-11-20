---
name: mario-emu_benchmark-latest-release
description: Benchmark the newest stable env-supermariobrosnes-turbo-emu PyPI release with the canonical TurboBench Super Mario workload, publish only verified official evidence to the project's Hugging Face dataset, verify the uploaded revision, and update local benchmark documentation. Use when the user says /mario-emu_benchmark-latest-release, asks to benchmark the latest/current/released version, asks to refresh official performance numbers, or asks to publish new env-SuperMarioBrosNes-turbo-emu benchmark evidence to Hugging Face.
---

# Benchmark Latest Release

Run the public package—not a checkout—against `stable-retro==1.0.1`, preserve
the current official workload, and publish the self-verifying evidence only
after every gate passes.

## Fixed contract

- Candidate: newest stable version shown by PyPI for
  `env-supermariobrosnes-turbo-emu`. This project is explicitly exempt from the
  seven-day package quarantine; all other TurboBench package quarantine policy
  remains unchanged.
- Baseline: `stable-retro@1.0.1` only. Do not add `stable-retro-turbo` to the
  public comparison unless the user explicitly changes that policy.
- Benchmark harness: `turbobench-cli==1.0.1`, profile
  `supermario/canonical-v1`, provider
  Python `3.14`, with no `--quick`, `--force-busy`, `--allow-dirty`, `--steps`,
  or `--shapes` overrides.
- Publication verifier: `turbobench-cli==1.0.2`. Bundles retain their recorded
  1.0.1 harness identity; use 1.0.2 only after the run because it verifies
  Git-hosted bundles whose optional empty `media/` directory is omitted.
- Host: the documented canonical x86-64 Linux host with AMD Ryzen 5 7600X
  (`beast-3` when that SSH alias is available). Never silently replace the
  official time series with another host.
- HF dataset: `tsilva/env-supermariobrosnes-turbo-emu-benchmarks`.
- Publication: exact PyPI artifacts only, no checkout, ROM, gameplay media,
  local paths, secrets, diagnostic bundles, or inconclusive/invalid claims.
- History: add one immutable `v<VERSION>` HF tag per package version; never
  delete or replace an existing version with a different bundle ID.

Treat TurboBench as the authority for eligibility. The explicit
`env-supermariobrosnes-turbo-emu` exemption permits its newest stable release to run
immediately, but no other eligibility, exact-artifact, or validity gate is
waived. Do not benchmark an older release while calling it latest and do not
upload a diagnostic run.

## 1. Resolve and preflight

From the env-SuperMarioBrosNes-turbo-emu repository root, resolve the public release:

```bash
python3 .codex/skills/mario-emu_benchmark-latest-release/scripts/benchmark_release.py \
  latest --require-eligible
```

Record `version`, `uploaded_at`, `eligible_at`, and `quarantine_exempt` from the
JSON. Require `quarantine_exempt: true`, and confirm that the version is visible
on PyPI and has the Linux x86-64 `cp39-abi3` wheel.

On the canonical benchmark host:

1. Ensure the host is otherwise idle. Do not terminate unrelated work without
   the user's authorization.
2. Ensure `TURBOBENCH_ASSET_ROOT` points to the private Stable
   Retro-compatible data root containing the canonical ROM and Level1-1 through
   Level1-4 states. Never print, copy, or upload the ROM.
   On the provisioned `beast-3` host, set it explicitly in every SSH command or
   session; do not rely on non-interactive shells loading a profile:

```bash
export TURBOBENCH_ASSET_ROOT=/home/tsilva/.local/share/turbobench/assets/stable
```

3. Install the pinned harness and run the doctor:

```bash
uv tool install --force \
  --exclude-newer-package turbobench-cli=2026-08-13T14:50:17Z \
  turbobench-cli==1.0.1
turbobench --version
turbobench doctor supermario/canonical-v1
```

Require `turbobench 1.0.1` and a passing doctor. Stop on any prerequisite,
asset, platform, package-resolution, or load failure. Never use an override to
turn a failed preflight into publishable evidence.

## 2. Run the official comparison

Choose a new external result directory. Keep bundles out of this Git working
tree. Substitute the resolved version exactly:

```bash
turbobench compare supermario/canonical-v1 \
  --left env-supermariobrosnes-turbo-emu@<VERSION> \
  --right stable-retro@1.0.1 \
  --python 3.14 \
  --output <EXTERNAL_RESULTS>/smb-<VERSION>-vs-stable-retro-1.0.1
```

Do not add `--promo`; the public evidence dataset intentionally excludes
gameplay media. Let a busy-host gate wait for its normal timeout. If the command
fails, preserve any `.partial` directory for diagnosis and stop before HF.

Verify the completed bundle independently with the same pinned CLI:

```bash
turbobench verify <BUNDLE>
```

Require `passed: true`, `claim.status: official`, `validity.passed: true`, the
exact candidate and baseline versions, all three shapes `1,16,32`, and PyPI
source identities. Treat an inconclusive but valid result as evidence that may
be retained locally, but do not publish it as the new official package claim.

When the benchmark ran over SSH, copy only the completed portable bundle into a
fresh local external directory with `rsync -a` or `scp -r`, then run the local
verification above again. Never transfer `TURBOBENCH_ASSET_ROOT`, the ROM, the
provider-runtime cache, or a `.partial` directory into the publication stage.

## 3. Prepare one HF commit

Replace the benchmark harness with the pinned publication verifier before
staging or checking a downloaded publication:

```bash
uv tool install --refresh --force \
  --exclude-newer-package turbobench-cli=2026-08-13T15:41:56Z \
  turbobench-cli==1.0.2
turbobench --version
```

Require `turbobench 1.0.2`. This must not change the recorded harness version
inside the already completed result bundle.

Read the repository's local Runbox configuration. Require the approved
`mario-benchmark-parent` profile to be ready before querying Hugging Face. If it
is unavailable or a check fails, keep the verified bundle and report the failed
gate.

Use a new owner-only directory under the repository's ignored
`.internal/benchmark-publication/`. Set `WORK` to that relative directory,
`VERSION` to the verified package version, and `BUNDLE` to the absolute path of
the verified local bundle. Keep the commit receipt and plan in `WORK`.

```bash
runbox --profile mario-benchmark-parent check
runbox --profile mario-benchmark-parent run -- mario-benchmark-parent \
  > "$WORK/parent.json"
```

Require `repository` in the receipt to match the fixed dataset. Use the returned
commit as `PARENT`. A verified `null` parent becomes `absent` for initial
publication. Use the shared helper from the release pinned by the parent
approval. Set `BENCHMARK_HELPER`, `BENCHMARK_PYTHON`, and `BENCHMARK_SDK` to
the reviewed helper, interpreter and locked SDK paths. Run its local and
anonymous commands outside a service session.

```bash
benchmark_helper=(
  "$BENCHMARK_PYTHON" -I -S -B -c
  'import runpy, sys; sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name="__main__")'
  "$BENCHMARK_SDK" "$BENCHMARK_HELPER"
)
DATASET=tsilva/env-supermariobrosnes-turbo-emu-benchmarks
PROJECT=env-supermariobrosnes-turbo-emu
index_args=()
if [ "$PARENT" != absent ]; then
  "${benchmark_helper[@]}" fetch-index --parent-receipt "$PWD/$WORK/parent.json" \
    --repository "$DATASET" --project "$PROJECT" > "$WORK/current-index.json"
  index_args=(--existing-index "$PWD/$WORK/current-index.json")
fi
python3 .codex/skills/mario-emu_benchmark-latest-release/scripts/benchmark_release.py \
  stage --bundle "$BUNDLE" --version "$VERSION" --output "$PWD/$WORK/stage" \
  "${index_args[@]}"
"${benchmark_helper[@]}" prepare --stage "$PWD/$WORK/stage" --repository "$DATASET" \
  --project "$PROJECT" --version "$VERSION" --parent "$PARENT" > "$WORK/plan.json"
```

Stop if the parent, index, or staging command fails. The staging helper
re-verifies the bundle and produces the benchmark index and README. Record the
plan's SHA-256 as `PLAN_SHA256` and preserve the staged bytes.

## 4. Upload, tag, and verify remotely

Obtain approval of the exact publication plan and confirm the
`mario-benchmark-publish` profile is ready for this stage, plan and digest. Its
gateway admits one commit against the verified parent and one immutable
version tag for the acknowledged commit. A stale parent or conflicting tag
requires review before any further attempt.

```bash
runbox --profile mario-benchmark-publish check
runbox --profile mario-benchmark-publish run -- mario-benchmark-publish \
  "$WORK/stage" "$WORK/plan.json" "$PLAN_SHA256" > "$WORK/receipt.json"
```

Select a new empty absolute `DOWNLOADED` directory. Verify the same tag and
exact published bytes anonymously, then run the benchmark's own verifier.

```bash
"${benchmark_helper[@]}" download-back --plan "$PWD/$WORK/plan.json" \
  --receipt "$PWD/$WORK/receipt.json" --repository "$DATASET" --destination "$DOWNLOADED"
python3 .codex/skills/mario-emu_benchmark-latest-release/scripts/benchmark_release.py \
  verify-snapshot --root "$DOWNLOADED" --version "$VERSION"
```

Publication is complete only after this download-back verification passes.
Do not generate or regenerate an HF DOI automatically; report the exact tag and
commit, and tell the user that DOI generation remains an optional HF Settings
action.

## 5. Refresh project documentation

After remote verification, update `BENCHMARKS.md`, the README benchmark summary,
and their existing community-package tests to describe the new version and
numbers. Link to the exact HF `v<VERSION>` tag and bundle directory, preserve
the bundle ID, package hashes, machine profile, and reproduction command, and
state that this is first-party self-verified evidence—not independent
reproduction. Run the focused documentation tests. Do not commit or push these
repository changes unless the user separately asks.

## Final report

Lead with the HF tagged dataset URL. Report the package version, baseline,
bundle ID, HF commit SHA and tag, shape-local medians/ratios/CIs, benchmark host,
download-back verification result, and local documentation/test status. If the
workflow stops, report the exact failed gate and whether any bundle or partial
bundle was preserved; never imply that HF publication completed.
