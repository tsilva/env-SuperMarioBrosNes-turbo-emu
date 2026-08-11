<div align="center">
  <img src="image-assets/icon/icon-1024.png" alt="SuperMarioBros-Nes-turbo icon" width="240" />

  **🚀 Blazing fast SuperMarioBros-Nes environment for Reinforcement Learning 🍄**
</div>

**SuperMarioBros-Nes-turbo** is a Python library for reinforcement-learning
researchers who need fast, deterministic Super Mario Bros NES rollouts. It
provides independent Gymnasium vector lanes, selective reset, saved-state
catalogs, snapshots, and configurable observations in one optimized environment.
Supply your own supported ROM, then play immediately or use the vector API from
Python.

In the [verified `0.6.2` benchmarks](BENCHMARKS.md), it measured **15.63× to
17.94×** the throughput of original
[Stable Retro](https://github.com/Farama-Foundation/stable-retro) and **7.46× to
8.05×** Stable Retro Turbo across the matched vector shapes.

## Quick start

Prebuilt wheels support Python `>=3.9` on Apple-silicon macOS and x86-64 Linux.
Install the package and launch Level 1-1 with your local ROM:

```bash
python -m pip install supermariobrosnes-turbo
smb-turbo play --rom /absolute/path/to/SuperMarioBros.nes
```

Playback opens a local window and requires a discoverable SDL2 runtime. Use the
arrow keys or `A`/`D` to move, `X`/`J`/Space to jump, `Z`/`K`/Shift to run, and
Escape to quit.

To register the ROM once for later commands, use a Stable Retro-compatible data
directory:

```bash
export RETRO_DATA_PATH="${XDG_DATA_HOME:-$HOME/.local/share}/retro"
smb-turbo import /absolute/path/to/SuperMarioBros.nes
smb-turbo play
```

ROM files are never included in this repository or its distributions.

## Why researchers use it

- **High throughput.** Rust owns batched emulation, preprocessing, rewards,
  termination, and infos while one native call advances every lane.
- **Deterministic lanes.** Each seeded lane has independent emulator state,
  random state, observation history, sticky action, and counters.
- **Explicit episode control.** Autoreset is permanently disabled; callers reset
  only selected lanes through `options["reset_mask"]`.
- **Training-ready data.** Configure action tables, grayscale or RGB frames,
  crop or masking, resize, frame skip, max-pooling, frame stacking, and CHW or
  HWC layouts.
- **Reusable starts.** Packaged states cover `Level1-1` through `Level8-4`, and
  live snapshots can be restored without advancing emulation.

## Use from Python

```python
from supermariobrosnes_turbo import (
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    action_batch,
)

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    state="Level1-1",
    rom_path="/absolute/path/to/SuperMarioBros.nes",
    num_envs=16,
    use_restricted_actions=Actions.ALL,
)

try:
    observations, infos = env.reset(seed=123)
    observations, rewards, terminated, truncated, infos = env.step(
        action_batch("right", env.num_envs)
    )

    done = terminated | truncated
    if done.any():
        observations, reset_infos = env.reset(
            options={"reset_mask": done.copy()},
        )
finally:
    env.close()
```

See [API.md](API.md) for the complete action, observation, state, snapshot,
rendering, playback, and research-info contracts.

## Train with GradLab

Training implementations and recipes live in
[GradLab](https://github.com/tsilva/gradlab), outside this environment
repository. Run either published, version-pinned recipe from any directory with
your local ROM.

Short PPO demonstration:

```bash
uvx gradlab@0.1.1 train SuperMarioBros-Nes-v0/Level1-1/turbo-demo --rom /absolute/path/to/SuperMarioBros.nes
```

Go-Explore trajectory discovery capped at 20 million transitions:

```bash
uvx gradlab@0.1.1 train SuperMarioBros-Nes-v0/Level1-1/go-explore-jerk-20m --rom /absolute/path/to/SuperMarioBros.nes
```

GradLab downloads the pinned runtime on first use, verifies the ROM in place,
shows live progress, and writes a playable `final_model.zip` below `./runs`.
The PPO demonstration runs 98,304 steps across 16 environments and takes roughly
two minutes on the calibrated M1 Pro; timing varies by hardware. When a run
finishes or stops safely, GradLab prints its version-pinned playback command.

## Commands

```bash
smb-turbo import /path/to/roms       # register the supported ROM
smb-turbo play                       # play Level1-1 manually or with its policy
smb-turbo play Level2-1 --fps max    # choose an exact state and run uncapped
```

For source builds, tests, and contribution commands, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Benchmarks

[BENCHMARKS.md](BENCHMARKS.md) contains the exact workloads, results, evidence,
machine profile, and reproduction commands. Install the recorded
[TurboBench 1.0.0](https://pypi.org/project/turbobench-cli/1.0.0/) runner with:

```bash
uv tool install \
  --exclude-newer-package turbobench-cli=2026-08-12T00:00:00Z \
  turbobench-cli==1.0.0
```

## Notes

- **Scope:** This is specialized for `SuperMarioBros-Nes-v0` on mapper 0/NROM;
  it is not a general NES emulator or Stable Retro replacement.
- **ROM identity:** The canonical ROM SHA-256 is
  `f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de`.
- **ROM discovery:** `RETRO_DATA_PATH`, `rom_path=`, and `smb-turbo play --rom`
  are supported. Imported ROMs use
  `<RETRO_DATA_PATH>/stable/SuperMarioBros-Nes-v0/rom.nes`.
- **Playback:** `smb-turbo play` and `play.py` use exact state identifiers,
  default to `Level1-1`, and automatically select matching action-run policies
  as canonical levels change.
- **Affiliation:** This unofficial research project is not affiliated with or
  endorsed by Nintendo. See [NOTICE.md](NOTICE.md).

## Architecture

![SuperMarioBros-Nes-turbo architecture diagram](architecture.png)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the native component boundaries and
verification hooks.

## License

Code is licensed under the [MIT License](LICENSE). Third-party names, marks, and
user-supplied content are excluded; see [NOTICE.md](NOTICE.md).
