<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/SuperMarioBros-Nes-turbo/main/logo.png" alt="SuperMarioBros-Nes-turbo logo" width="320" />

  **🚀 Blazing fast SuperMarioBros-Nes environment for Reinforcement Learning 🍄**

  [![CI](https://github.com/tsilva/SuperMarioBros-Nes-turbo/actions/workflows/ci.yml/badge.svg)](https://github.com/tsilva/SuperMarioBros-Nes-turbo/actions/workflows/ci.yml)
  [![PyPI](https://img.shields.io/pypi/v/supermariobrosnes-turbo.svg)](https://pypi.org/project/supermariobrosnes-turbo/)
  [![Python](https://img.shields.io/pypi/pyversions/supermariobrosnes-turbo.svg)](https://pypi.org/project/supermariobrosnes-turbo/)
  [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/LICENSE)
</div>

High-throughput Rust-backed Gymnasium `VectorEnv` for Super Mario Bros NES
(mapper 0/NROM).

Purpose-built for this ROM and recursively auto-tuned with agents, it leverages
ROM-specific quirks while performing vectorization and all preprocessing
natively in Rust. That delivers environment stepping up to 17× faster than
[Stable Retro](https://github.com/Farama-Foundation/stable-retro).

<div align="center">
  <img src="media/mario-promo/mario-throughput-comparison.gif" alt="Speed Comparison" width="640" />
</div>

## Why it is fast

Throughput comes from the complete normal step path, not from disabling
preprocessing, infos, or resets:

- **Fixed workload.** It verifies the canonical ROM hash and accepts only
  mapper 0/NROM, avoiding general mapper and emulator-core dispatch.
- **One native vector engine.** All lanes live in one Rust process rather than
  separate environment workers, avoiding per-lane process and IPC overhead.
- **Direct NumPy buffers.** A single Python-to-Rust `step_into` call reads
  contiguous actions and writes observations, rewards, terminations, and info
  fields in place; the GIL is released while the batch runs.
- **Parallel batched lanes.** At four or more environments, independent lanes
  step in parallel with Rayon; smaller batches avoid parallel scheduling cost.
- **Reused memory.** Each lane owns persistent emulator state, action state, and
  scratch buffers; frame stacks shift in place instead of allocating new frames.
- **Native rollout bookkeeping.** Frame skip, sticky actions, reward,
  termination, selective-reset state, and info extraction stay in the Rust loop.
- **SMB routine summaries.** At known program counters it fast-forwards
  interpreter-equivalent work for idle jumps, sprite-0 polling, timer control,
  OAM clearing, controller reads, scroll updates, digit math, collision and
  off-screen helpers, relative-position math, and sprite-object drawing.
- **Event-bounded PPU stepping.** CPU cycles accumulate until the next relevant
  PPU boundary—vblank, pre-render, sprite-0 hit, or frame end—rather than
  paying a PPU update on every instruction.
- **Direct observation rendering.** It renders the needed background tiles and
  OAM sprites from PPU memory directly to grayscale, then applies the canonical
  crop/mask and integer area resize to `84×84`; it does not first materialize a
  generic RGB frame.
- **Canonical resize kernels.** The common grayscale `84×84` area-resize path
  uses fixed geometry and specialized integer kernels instead of a general
  resampling pipeline.

ROM- and timing-specific fast paths are guarded by their preconditions;
unsupported work falls back to the instruction interpreter.

## Install and run

Use `uv` from a local checkout:

```bash
git clone https://github.com/tsilva/SuperMarioBros-Nes-turbo.git
cd SuperMarioBros-Nes-turbo
uv sync --frozen
uv run maturin develop --release
```

For authenticated Hugging Face policy downloads, add the optional playback
extra:

```bash
uv sync --frozen --extra playback
```

ROM files are not included. Pass `--rom-path` to scripts, set `ROM_PATH` in the environment or a repo-root `.env`, or provide `rom_path=` to the constructor. The supported ROM has SHA-256:

```text
f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de
```

## Use

```python
from supermariobrosnes_turbo import (
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    action_batch,
)

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    rom_path="/path/to/SuperMarioBros.nes",
    state="Level1-1",
    num_envs=16,
    use_restricted_actions=Actions.ALL,
    frame_skip=4,
    obs_grayscale=True,
    obs_crop=(32, 0, 0, 0),
    obs_resize=(84, 84),
    obs_layout="chw",
    frame_stack=4,
)

observations, infos = env.reset(seed=123)
actions = action_batch("right", env.num_envs)
observations, rewards, terminated, truncated, infos = env.step(actions)

done = terminated | truncated
if done.any():
    observations, reset_infos = env.reset(
        options={"reset_mask": done.copy()},
    )

env.close()
```

`reset()` returns `(observations, infos)`. `step()` returns `(observations, rewards, terminations, truncations, infos)`. Autoreset is disabled, so terminal lanes must be selectively reset before the next step.

## Train and play

Train the included observation-free JERK action-sequence policy for a named level:

```bash
uv run python train.py Level1-1 --rom /path/to/SuperMarioBros.nes
```

The discovery phase matches rlab's vectorized JERK search: 16 lanes uniformly explore the simple action set, retain the best-return prefixes, and increasingly replay shortened archive prefixes. Failed attempts end on life loss, 300 steps without progress, or the 4,500-step limit. A level change accepts the first successful sequence without resetting the environment. JERK then generically removes action chunks, retaining a mutation only when it still completes in fewer steps; across retrains, a longer policy cannot replace a shorter verified incumbent. The level name deterministically selects both the run directory and policy file, so `Level1-1` writes `runs/Level1-1-jerk/Level1-1.zip`. `play.py <Level>` always starts from that level and automatically replays its matching policy when available; as gameplay advances, it switches to each new level's matching policy. Without a starting policy, it opens the requested level for manual play:

```bash
uv run python play.py Level1-1 --rom-path /path/to/SuperMarioBros.nes
```

## Commands

```bash
uv sync --frozen --extra dev                 # install the development environment
uv run maturin develop --release             # build the optimized Rust extension
make test                                    # run Rust and Python regression tests
make test-retro-oracle                       # run ROM-backed parity and policy tests

uv run python scripts/smoke_smb.py --rom-path /path/to/SuperMarioBros.nes
uv run python scripts/play.py --rom-path /path/to/SuperMarioBros.nes --mode external
uv run python scripts/benchmark_sps.py --rom-path /path/to/SuperMarioBros.nes --num-envs 16 --steps 500 --repeats 3
uv run python scripts/benchmark_sps.py --stable-retro-baseline --rom-path /path/to/SuperMarioBros.nes --num-envs 16 --steps 500 --repeats 3
make benchmark-report                         # paired Turbo vs upstream Stable Retro report
```

## Benchmark

Turbo `0.3.0` vs upstream `stable-retro==1.0.1`, using seven alternating
paired runs per shape. Results are host-specific; reproduce with
`make benchmark-report`.

| Machine ID | Commit | Envs | Median SPS | Baseline median SPS | Median speedup | 95% bootstrap CI | Measured pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `amd-ryzen-5-3600-6c` | `545131bf` | 1 | 5,458.4 | 411.4 | 13.27x | 12.85x–13.33x | 7 |
| `amd-ryzen-5-3600-6c` | `545131bf` | 16 | 28,591.6 | 1,847.7 | 15.49x | 15.30x–15.83x | 7 |
| `amd-ryzen-5-3600-6c` | `545131bf` | 32 | 35,778.4 | 1,958.0 | 18.27x | 18.20x–18.30x | 7 |
| `apple-m1-pro-8c` | `ae1171e` | 1 | 8,574.5 | 584.3 | 14.68x | 14.61x–14.78x | 7 |
| `apple-m1-pro-8c` | `ae1171e` | 16 | 36,675.3 | 2,608.5 | 13.79x | 13.45x–14.55x | 7 |
| `apple-m1-pro-8c` | `ae1171e` | 32 | 43,443.0 | 2,555.0 | 17.23x | 16.38x–17.86x | 7 |

### `amd-ryzen-5-3600-6c` machine specifications

| Component | Specification |
| --- | --- |
| System | ASUS desktop; ROG STRIX B550-F GAMING (WI-FI) motherboard, BIOS 2803 |
| CPU | AMD Ryzen 5 3600 (Zen 2), 6 physical cores / 12 threads, boost enabled, 4.208 GHz reported maximum |
| CPU cache | 384 KiB L1 (192 KiB data + 192 KiB instruction), 3 MiB L2, 32 MiB L3 |
| Memory | 32 GiB system RAM |
| Storage | 1 TB nominal WDC WDS100T2B0C-00PXH0 NVMe SSD (931.5 GiB reported) |
| OS | Ubuntu 26.04, Linux 7.0.0-27-generic, glibc 2.43, x86_64 |
| CPU frequency policy | `amd_pstate` active, `powersave` scaling governor |
| Runtime | CPython 3.13.14; `supermariobrosnes-turbo==0.3.0`, `stable-retro==1.0.1`, `numpy==2.5.0`, `gymnasium==1.3.0` |

The Ryzen results were measured from clean commit `545131bf` after the
ROM-backed parity checks passed. The session-start one-minute load was 0.23,
below the protocol limit of 4.0. The Apple results used Python 3.14.4 from clean
commit `ae1171e`.

## Notes

- Python `>=3.9`, `uv`, and a Rust toolchain are required for a source build.
- The emulator supports only `SuperMarioBros-Nes-v0` on mapper 0/NROM. It is not a general NES or Stable Retro replacement.
- Named saved states are packaged from `Level1-1` through `Level8-4`, with additional variants. `state=` also accepts a path, bytes, one state per lane, or a weighted mapping.
- `Actions.ALL` and `Actions.FILTERED` accept per-button masks. `Actions.DISCRETE` accepts Stable Retro-compatible 36-way discrete actions.
- `train.py` implements JERK (Just Enough Retained Knowledge): it uniformly explores action sequences, retains the best reward-reaching prefixes, replays shortened archive prefixes, and minimizes completed sequences by deletion. It does not use observations, PyTorch, or Stable Baselines3.
- `scripts/benchmark_sps.py` benchmarks this package by default or upstream `stable-retro==1.0.1` with `--stable-retro-baseline`. Both use frame skip 4, four grayscale frames, a zeroed 32-row HUD, integer area resize to `84x84`, CHW output, deterministic sampled actions, and manual terminal-lane resets. Stable Retro mode requires Python `>=3.10`.
- The play scripts require a discoverable native SDL2 library and open local gameplay windows.
- This is an unofficial research project and is not affiliated with or endorsed by Nintendo. See [NOTICE.md](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/NOTICE.md).

## Forking

This project is maintained for its current scope. If you need a different
direction or behavior, please fork it.

## Architecture

![SuperMarioBros-Nes-turbo architecture diagram](https://raw.githubusercontent.com/tsilva/SuperMarioBros-Nes-turbo/main/architecture.png)

## License

The project code is licensed under the [MIT License](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/LICENSE). Third-party game names, marks, and user-supplied content are not covered by that license; see [NOTICE.md](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/NOTICE.md).
