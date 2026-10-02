---
name: build-release
description: Launch and monitor an env-supermariobrosnes-turbo-emu PyPI release. Use when the user says /build-release, asks to cut a release, asks to tag/publish a version, asks whether a release made it to PyPI, or asks for env-supermariobrosnes-turbo-emu release artifacts.
---

# Build Release

Read and apply the shared `$release-workflow` skill at
`/Users/tsilva/.codex/skills/release-workflow/SKILL.md` before execution.
It owns common preflight, publication safeguards, `$push` integration,
workflow monitoring, verification, and reporting. The rules below are this
project's adapter; they retain its invocation default and required gates.
If the shared skill is unavailable, stop and report the missing dependency.

A bare `$build-release` or `/build-release` invocation requests the full
publication flow. Explicitly local, dry-run, or inspection requests must not
launch `make release` or its publishing script.

Use this skill to launch the repo-owned `env-supermariobrosnes-turbo-emu` release flow
and monitor it until the package is visible on PyPI. The release implementation
lives in `scripts/release.py` and the `Makefile` `release` target. Prefer that
path over manually replaying version bumps, tags, wheel builds, validation, or
uploads.

This repo is not a fork, so versioning is owned here: use normal project
versions from `pyproject.toml` and `Cargo.toml`, not upstream-aligned `.postN`
versions unless the user explicitly asks for one.

`make release` runs `uv sync --extra dev --group dev` and then
`scripts/release.py`. The script enforces a clean tree, configured upstream,
synced remote state, unused PyPI version, version consistency, locked dependency
resolution, local checks, release commit, tag creation, and atomic push. It
leaves release history in GitHub Releases, where the workflow generates notes
from committed changes since the previous release. It stages only version and lock
metadata; no checked-in changelog is required.
An untagged project version is treated as a pending release; otherwise the
default is the next patch version. Failed preparation restores the release
files it changed. The pushed tag triggers
`.github/workflows/release.yml`, which builds and audits Apple-silicon macOS
and Linux x86-64 wheels plus a source distribution. It publishes through PyPI
trusted publishing and then creates a GitHub Release with the audited
artifacts.

Do not upload to PyPI manually unless the user explicitly asks for a manual
recovery path after the GitHub Actions publish path fails. Never print or commit
PyPI tokens. Do not create or switch branches unless the user explicitly asks.

## Required certification

Before launching a publishing command, confirm the following existing
specification requirements in the repository-owned release path.

The root specification additionally requires the exact final canonical-host
wheel to pass immutable TurboBench parity for the canonical ROM and public
Level1-1 through Level1-4 corpus. Confirm that evidence and provider-owned
cross-platform consistency checks before publication; the CPython 3.9 feature
smoke and ROM-free ABI smoke do not replace parity certification.

## Flow

1. Launch the release command from the repo root:

```bash
make release
```

If the user explicitly requested a non-default bump or exact version, run the
script directly because the Make target does not pass arguments through. Choose
exactly one `scripts/release.py` invocation:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev --group dev
scripts/release.py --to <version>
```

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev --group dev
scripts/release.py --part minor
```

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev --group dev
scripts/release.py --part major
```

For "next version" or no version preference, use `make release`; it defaults to
the next patch version. If the user typed `make releaes`, treat it as a typo and
use the actual `release` target.

2. Let the release script own the release gates.

Do not manually duplicate the old local wheel-building checklist. If
`make release` fails, report the failing stage and exact relevant error, then
stop. Common failures include a dirty worktree, unsynced upstream, an existing
PyPI version, formatting/test failures, tag collisions, or push failures.
No release-note preparation is required from the user. Follow the shared
release-note policy; the GitHub Release job generates the initial notes.

Releases containing the processed research-info catalog also require a
fail-closed installed-wheel feature smoke on CPython 3.9 in a maintainer
environment that has the canonical ROM. Run the helper against the exact wheel
being released and keep its JSON evidence outside the repository:

```bash
uv run python .codex/skills/build-release/scripts/release_build.py \
  smoke-feature-wheel <wheel> \
  --python <python3.9> \
  --rom <canonical-rom.nes> \
  --evidence <external-artifact-dir>/research-info-smoke.json
```

The command must fail when Python is not 3.9, the ROM is absent or has the
wrong canonical hash, or the installed wheel does not exercise mixed legacy
and extra infos. Do not describe a ROM-free public-CI smoke as feature-level
validation; it validates only the stable ABI surface.

3. Capture the released tag and version.

The command should end with output like:

```bash
Released v<version>: pushed <branch> and tag to <remote>.
GitHub Actions will build, validate, and publish the release distributions from the pushed tag.
```

If needed, confirm the tag after the command succeeds:

```bash
git describe --tags --exact-match HEAD
```

4. Follow the shared monitoring and verification procedure for the `release.yml`
tag-push run at the full `v<version>` commit SHA. A `workflow_dispatch` run
validates artifacts but never publishes. Verify PyPI project `env-supermariobrosnes-turbo-emu` and
the GitHub Release for the same tag.

Require the macOS arm64 and Linux x86_64 wheels plus one source distribution.
Allow 30 attempts at 20-second intervals for PyPI visibility, with a
20-second request timeout; report unresolved visibility before any bounded retry.

Keep `.codex/skills/build-release/scripts/release_build.py` as the workflow's
release helper and for narrow diagnostics. Use it directly only when inspecting
versions, PyPI presence, or workflow build failures; do not re-create the
release locally unless the user asks for manual recovery.

## Update GradLab after successful publication

After the release succeeds and the exact PyPI version and required GitHub
Release artifacts pass external verification, update GradLab to consume the
latest successfully published `env-supermariobrosnes-turbo-emu` version. Complete
this step as part of the full publication flow; local builds, dry runs, and
inspection-only requests do not trigger it.

Read `/Users/tsilva/repos/tsilva/gradlab/AGENTS.md` and its required
specifications before editing. Synchronize GradLab's current branch with its
configured upstream and preserve existing work. Update every matching exact
pin in `pyproject.toml`, including platform-specific project dependencies and
the `train-runtime` dependency group. Use the just-verified release version;
if GradLab already consumes a newer verified publication, do not downgrade it.
Regenerate `uv.lock` with `uv lock --upgrade-package env-supermariobrosnes-turbo-emu`,
preserving unrelated pins, supply-chain constraints, and existing per-package
release-age exceptions. Review the dependency diff, validate lock consistency,
and run GradLab's relevant provider compatibility checks.

Report the GradLab version/pin and lockfile update separately from release
success. If synchronization, resolution, or validation fails, preserve the
published release and report the downstream update as incomplete with its
blocker; do not repeat publication.
