---
name: build-release
description: Launch and monitor a SuperMarioBros-Nes-turbo PyPI release. Use when the user says /build-release, asks to cut a release, asks to tag/publish a version, asks whether a release made it to PyPI, or asks for supermariobrosnes-turbo release artifacts.
---

# Build Release

Use this skill to launch the repo-owned `supermariobrosnes-turbo` release flow
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
uses any checked-in `CHANGES.md` `Unreleased` prose when present and otherwise
generates concise notes from commit subjects since the previous release tag.
It promotes those notes to the target version and release date, creates a fresh
`Unreleased` section, and stages the changelog with the version and lock files.
An untagged project version is treated as a pending release; otherwise the
default is the next patch version. Failed preparation restores the release
files it changed. The pushed tag triggers
`.github/workflows/release.yml`, which builds and audits macOS ARM64, macOS
Intel, Linux x86-64, Linux ARM64, and Windows x86-64 wheels plus a source
distribution. It publishes through PyPI trusted publishing and then creates a
GitHub Release with the audited artifacts.

Do not upload to PyPI manually unless the user explicitly asks for a manual
recovery path after the GitHub Actions publish path fails. Never print or commit
PyPI tokens. Do not create or switch branches unless the user explicitly asks.

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
No release-note preparation is required from the user. When `Unreleased` is
empty, let the script generate notes from the commits since the previous tag.

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

4. Monitor the GitHub Actions release workflow for the pushed tag.

Use `gh` if it is available:

```bash
release_sha="$(git rev-list -n 1 v<version>)"
gh run list --workflow release.yml --commit "$release_sha" --limit 5 \
  --json databaseId,status,conclusion,event,headBranch,headSha,displayTitle,url
gh run watch <run-id> --exit-status
```

If the commit-filtered query does not find the run, list recent release runs and
pick the run whose event/ref corresponds to the pushed tag:

```bash
gh run list --workflow release.yml --limit 10 \
  --json databaseId,status,conclusion,event,headBranch,headSha,displayTitle,url
```

The workflow publishes only for tag-push events. `workflow_dispatch` builds are
validation builds and do not publish.

5. After the workflow succeeds, poll PyPI until the released version appears.

```bash
python - <<'PY'
import json
import time
import urllib.request

package = "supermariobrosnes-turbo"
version = "<version>"
url = f"https://pypi.org/pypi/{package}/json"

for attempt in range(30):
    with urllib.request.urlopen(url, timeout=20) as response:
        data = json.load(response)
    files = data.get("releases", {}).get(version, [])
    if files:
        print(f"https://pypi.org/project/{package}/{version}/")
        print(f"https://pypi.org/project/{package}/")
        for file in files:
            print(file["filename"])
        break
    print(f"waiting for PyPI to show {package} {version} ({attempt + 1}/30)")
    time.sleep(20)
else:
    raise SystemExit(f"{package} {version} did not appear on PyPI yet")
PY
```

6. If PyPI still does not show the version after a successful workflow, wait a
little longer and retry before declaring failure. PyPI indexing can lag briefly.
If the publish job failed, report the job URL and the failing step; do not try a
manual Twine upload unless the user explicitly asks.

## Useful Inspection Commands

```bash
gh run view <run-id> --web
gh run view <run-id> --log-failed
gh run view <run-id> --json url,status,conclusion,event,headBranch,headSha,displayTitle
```

The final PyPI package URLs are:

```
https://pypi.org/project/supermariobrosnes-turbo/<version>/
https://pypi.org/project/supermariobrosnes-turbo/
```

The GitHub Actions workflow environment URL is:

```
https://pypi.org/p/supermariobrosnes-turbo
```

Keep `.codex/skills/build-release/scripts/release_build.py` as the workflow's
release helper and for narrow diagnostics. Use it directly only when inspecting
versions, PyPI presence, or workflow build failures; do not re-create the
release locally unless the user asks for manual recovery.

## Final Response

When the release reaches PyPI, lead with the PyPI version URL. Also report the
tag, GitHub Actions run URL, workflow conclusion, GitHub Release URL, and all
published wheel and source-distribution filenames.
If the release did not reach PyPI, report the exact failed command/job/step and
the next recovery action.
