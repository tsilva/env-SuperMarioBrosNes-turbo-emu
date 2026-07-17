# Changelog

## Unreleased

- Replace provider-side saved-state sampling and per-lane constructor states
  with immutable `state_catalog` entries selected explicitly through
  `options["state_indices"]`, including active-index and reset-info reporting.
- Add the package-owned `smb-turbo import`, `train`, and `play` command tree,
  replacing the fragmented installed commands while retaining thin checkout
  launchers.
- Make training and playback use exact discoverable state identifiers, expose
  beam search through `train --algorithm beam`, default state-less playback to
  `Level1-1`, and simplify the public options.
- Add contribution and conduct policies, expand oldest/latest Python CI, and
  make strict Clippy validation part of the contributor workflow.
- Clarify the licensing boundary for promotional gameplay media and remove
  repository-local editor color customization.

## 0.3.1 - 2026-07-16

- Add Stable Retro-compatible ROM importing and `RETRO_DATA_PATH` discovery,
  replacing the project-specific `ROM_PATH` and `.env` lookup.
- Add an explicit MIT license, legal notices, governance, support and security
  policies, and issue and pull-request templates.
- Add pull-request CI, typed-package/version metadata, expanded project
  metadata, platform documentation, source distributions, and macOS Intel,
  Linux ARM64, and Windows wheels.
- Consolidate the upstream Stable Retro benchmark into the primary benchmark
  command and remove the obsolete turbo-oracle scripts.

## 0.3.0 - 2026-07-14

- Fix Gymnasium autoreset to `AutoresetMode.DISABLED` and remove the constructor option.
- Remove provider-owned life-loss/level-change rules, terminal payload synthesis,
  and dynamic reset-policy mutation APIs.
- Add lane-local masked reset through `options["reset_mask"]`, including active
  lanes selected by an external task kernel.
- Add deterministic scalar/per-lane reset seeds and explicit catalog selection
  through `options["start_indices"]` without mutating unselected lanes.
- Return terminal observations and raw infos directly, and reject stepping
  pending-reset lanes until they are reset explicitly.

## 0.2.25 - 2026-07-13

- Validate releases on Python 3.14 while preserving the CPython 3.9 stable ABI.
- Add platform-scoped Cargo caches to release builds.
- Align native PPO playback with its training observation and action contract.
- Add training profiles, completion-rate logging, emulator hot-path work, and
  release-code cleanup.

## 0.2.23 - 2026-07-10

- Add and validate Super Mario Bros-specific score, timer, rendering, and
  offscreen fast-forward optimizations.
- Publish audited macOS ARM64 and Linux x86-64 stable-ABI wheels through the
  repository release workflow.

Earlier release details are available in the repository's Git history and tags.
