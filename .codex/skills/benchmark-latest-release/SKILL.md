---
name: benchmark-latest-release
description: Benchmark the newest stable supermariobrosnes-turbo PyPI release with the canonical TurboBench Super Mario workload, publish only verified official evidence to the project's Hugging Face dataset, verify the uploaded revision, and update local benchmark documentation. Use when the user says /benchmark-latest-release, asks to benchmark the latest/current/released version, asks to refresh official performance numbers, or asks to publish new SuperMarioBros-Nes-turbo benchmark evidence to Hugging Face.
---

# Benchmark Latest Release

Run the public package—not a checkout—against `stable-retro==1.0.1`, preserve
the current official workload, and publish the self-verifying evidence only
after every gate passes.

## Fixed contract

- Candidate: newest stable version shown by PyPI for
  `supermariobrosnes-turbo`. This project is explicitly exempt from the
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
- HF dataset: `tsilva/supermariobros-nes-turbo-benchmarks`.
- Publication: exact PyPI artifacts only, no checkout, ROM, gameplay media,
  local paths, secrets, diagnostic bundles, or inconclusive/invalid claims.
- History: add one immutable `v<VERSION>` HF tag per package version; never
  delete or replace an existing version with a different bundle ID.

Treat TurboBench as the authority for eligibility. The explicit
`supermariobrosnes-turbo` exemption permits its newest stable release to run
immediately, but no other eligibility, exact-artifact, or validity gate is
waived. Do not benchmark an older release while calling it latest and do not
upload a diagnostic run.

## 1. Resolve and preflight

From the SuperMarioBros-Nes-turbo repository root, resolve the public release:

```bash
python3 .codex/skills/benchmark-latest-release/scripts/benchmark_release.py \
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
  --left supermariobrosnes-turbo@<VERSION> \
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

Authenticate without exposing tokens:

```bash
hf auth whoami --format json
```

If authentication is absent or the active identity cannot write to `tsilva`,
stop and ask the user to run `hf auth login`. Never print `hf auth token`.

Create a temporary staging directory with `mktemp -d`. First query the dataset.
If it exists, download its current index into the staging parent:

```bash
hf datasets info tsilva/supermariobros-nes-turbo-benchmarks --format json
hf download tsilva/supermariobros-nes-turbo-benchmarks \
  benchmark-index.json --type dataset --local-dir <CURRENT>
```

Treat absence as a first publication only when the dataset-info command
explicitly reports that the repository was not found. Stop on authentication,
permission, network, download, or malformed-index failures.

Prepare the upload. Pass `--existing-index` only when the downloaded file
exists:

```bash
python3 .codex/skills/benchmark-latest-release/scripts/benchmark_release.py \
  stage \
  --bundle <BUNDLE> \
  --version <VERSION> \
  --output <UPLOAD_STAGE> \
  --existing-index <CURRENT>/benchmark-index.json
```

The helper re-verifies the bundle, enforces the provider/host/media contract,
copies it into the stable HF layout, and generates `README.md` plus
`benchmark-index.json`. Do not edit the staged bundle after this succeeds.

## 4. Upload, tag, and verify remotely

Create the public dataset if necessary, then upload the complete staging
directory as one commit. Do not use `--delete`:

```bash
hf repos create tsilva/supermariobros-nes-turbo-benchmarks \
  --type dataset --public --exist-ok
hf upload tsilva/supermariobros-nes-turbo-benchmarks \
  <UPLOAD_STAGE> . --type dataset \
  --commit-message "Add TurboBench evidence for v<VERSION>" \
  --format json
```

Capture the returned commit SHA. Create the immutable version tag at that exact
SHA; if the tag already exists, require it to resolve to the same commit:

```bash
hf repos tag create tsilva/supermariobros-nes-turbo-benchmarks v<VERSION> \
  --type dataset --revision <HF_COMMIT_SHA> \
  --message "Official TurboBench evidence for supermariobrosnes-turbo <VERSION>"
```

Download the tag into a fresh directory and verify the published bundle:

```bash
hf download tsilva/supermariobros-nes-turbo-benchmarks \
  --type dataset --revision v<VERSION> --local-dir <DOWNLOADED>
python3 .codex/skills/benchmark-latest-release/scripts/benchmark_release.py \
  verify-snapshot --root <DOWNLOADED> --version <VERSION>
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
