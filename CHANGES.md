# Changelog

## Unreleased

- Nothing yet.

## 0.4.2 - 2026-07-21

- Add an opt-in catalog of processed player, area, engine, and six-slot enemy
  state to Gymnasium infos, including public enum types and exact requested-key
  filtering without changing the legacy default infos.
- Add immutable per-lane CPU RAM snapshots for researchers who explicitly need
  unprocessed state.

## 0.4.1 - 2026-07-21

- Restore Python 3.9 compatibility for the public `ActionTable` alias and the
  release distribution smoke test.

## 0.4.0 - 2026-07-20

- Unify built-in, game-owned preset, and inline exact action tables under
  `use_restricted_actions`, with Mario presets loaded from packaged
  `metadata.json` and `MULTI_DISCRETE` parity with Stable Retro Turbo
  `1.0.1.post34`.
- Add Go-Explore trajectory discovery with exact archived-state restoration,
  without robustification, and save discovered trajectories in the same
  action-run policy format used by beam search.
- Allow training without a state to process all 32 canonical levels in order,
  with isolated per-level outputs and an overall level-progress bar in the TUI.

## 0.3.5 - 2026-07-20

- Use `stable-retro-turbo==1.0.1.post33` for optional oracle and benchmark
  compatibility tooling on Python 3.14.

## 0.3.4 - 2026-07-20

- Add reusable, per-lane live snapshot handles through
  `capture_snapshots(mask)` and mixed snapshot/catalog restoration through
  masked `reset()`, including exact cross-lane fan-out without advancing
  emulation.

## 0.3.3 - 2026-07-17

- Keep the local release checks aligned with CI and refresh the deliberate Rust
  dependency-closure baseline after lockfile regeneration.

## 0.3.2 - 2026-07-17

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
