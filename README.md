<div align="center">
  <img src="logo.png" alt="SuperMarioBros-Nes-turbo logo" width="320" />

  **🚀 Blazing fast SuperMarioBros-Nes environment for Reinforcement Learning 🍄**
</div>

**SuperMarioBros-Nes-turbo** is a Rust-backed Gymnasium vector environment for
reinforcement-learning researchers working with Super Mario Bros NES. On the
canonical mapper 0/NROM benchmark, it delivers **up to roughly 17×** the
end-to-end step and preprocessing throughput of
[Stable Retro](https://github.com/Farama-Foundation/stable-retro).

<div align="center">
  <img src="media/mario-promo/mario-throughput-comparison.gif" alt="Stable Retro and SuperMarioBros-Nes-turbo throughput comparison" width="640" />
</div>

## ⚡ Why it is fast

- **Focused scope.** It specializes in the canonical Super Mario Bros mapper
  0/NROM workload.
- **Native vector engine.** One Rust engine owns all lanes, releases the GIL,
  and parallelizes batches of four or more environments with Rayon.
- **One efficient call.** Actions, emulation, preprocessing, frame stacks,
  rewards, termination, and infos share reused buffers across one
  Python-to-Rust call.
- **Optimized rendering.** Guarded game-routine fast paths, event-bounded PPU
  stepping, and direct grayscale rendering avoid unnecessary interpreter and
  image work.

*Unsupported fast-path cases fall back to the instruction interpreter.*

## 📦 Install

From a local checkout:

```bash
git clone https://github.com/tsilva/SuperMarioBros-Nes-turbo.git
cd SuperMarioBros-Nes-turbo
uv sync --frozen
uv run maturin develop --release
```

**Requirements:** Python `>=3.9`, [uv](https://docs.astral.sh/uv/), and a Rust
toolchain. Add `--extra playback` to `uv sync` for authenticated Hugging Face
policy downloads.

**ROM setup:** ROM files are not included. Import the supported ROM from a file,
directory, or ZIP archive:

```bash
uv run python -m supermariobrosnes_turbo.import /path/to/roms
```

The importer uses the Stable Retro-compatible `RETRO_DATA_PATH` layout, or the
equivalent data tree inside the installed package when the variable is unset.
`rom_path=` and `--rom-path` remain available as overrides. The canonical ROM
SHA-256 is:

```text
f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de
```

## 🎮 Use

```python
from supermariobrosnes_turbo import (
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    action_batch,
)

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
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
observations, rewards, terminated, truncated, infos = env.step(
    action_batch("right", env.num_envs)
)

done = terminated | truncated
if done.any():
    observations, reset_infos = env.reset(
        options={"reset_mask": done.copy()},
    )

env.close()
```

**Important:** Autoreset is disabled. Selectively reset terminal lanes before
stepping again.

## 🏁 Train and play

```bash
uv run python train.py Level1-1
uv run python play.py Level1-1
```

**Training** searches observation-free `(action, duration)` programs and retains
useful prefixes. It stops on the first level completion by default; pass
`--no-stop-on-completion` to continue through the transition budget. Existing
policies are protected unless `--force` is passed. `Level1-1` writes
`runs/Level1-1-jerk/Level1-1.zip`; playback uses the matching trained policy when
available and switches policies as levels change. Run either command with
`--help` for configuration options.

## 🧰 Commands

```bash
uv sync --frozen --extra dev          # install development dependencies
uv run maturin develop --release      # build the optimized Rust extension
make test                             # run Rust and Python tests
make test-retro-oracle                # run ROM-backed parity and policy tests
make benchmark                        # benchmark SuperMarioBros-Nes-turbo locally
make benchmark-report                 # compare SuperMarioBros-Nes-turbo with Stable Retro
```

## 📈 Benchmark

[![SuperMarioBros-Nes-turbo versus Stable Retro median environment throughput](media/benchmark-throughput.svg)](BENCHMARKS.md)

See [BENCHMARKS.md](BENCHMARKS.md) for results, protocol, and machine details.

## Notes

- **Scope:** This emulator supports only `SuperMarioBros-Nes-v0` on mapper
  0/NROM; it is not a general NES or Stable Retro replacement.
- **States:** Packaged states cover `Level1-1` through `Level8-4`, with
  additional variants. `state=` also accepts paths, bytes, per-lane states, and
  weighted mappings.
- **Actions:** `Actions.ALL` and `Actions.FILTERED` accept per-button masks;
  `Actions.DISCRETE` provides Stable Retro-compatible 36-way actions.
- **Playback:** Play commands require a discoverable native SDL2 library and
  open local gameplay windows.
- **Affiliation:** This unofficial research project is not affiliated with or
  endorsed by Nintendo. See [NOTICE.md](NOTICE.md).

## Architecture

![SuperMarioBros-Nes-turbo architecture diagram](architecture.png)

## License

Code is licensed under the [MIT License](LICENSE). Third-party names, marks, and
user-supplied content are excluded; see [NOTICE.md](NOTICE.md).
