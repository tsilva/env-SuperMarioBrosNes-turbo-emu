<p align="center">
  <img src="https://raw.githubusercontent.com/tsilva/env-SuperMarioBrosNes-turbo-emu/main/image-assets/logo/logo-1024.png" alt="env-SuperMarioBrosNes-turbo-emu logo" width="240" />
  <br />
  <strong>🚀 Blazing fast SuperMarioBros-Nes environment for Reinforcement Learning 🍄</strong>
</p>

**env-SuperMarioBrosNes-turbo-emu** is a specialized Python environment for
reinforcement-learning researchers who need fast, reproducible Super Mario Bros
NES rollouts. Supply your own supported ROM, then play immediately or use its
deterministic Gymnasium vector API from Python.

In the [verified `0.6.4` benchmarks](BENCHMARKS.md), it measured **15.42× to
17.27×** the throughput of original
[Stable Retro](https://github.com/Farama-Foundation/stable-retro) across matched
vector shapes.

## Quick start

Prebuilt wheels support Python `>=3.9` on Apple-silicon macOS and x86-64 Linux.
Install the CLI with [uv](https://docs.astral.sh/uv/) and launch Level 1-1 with
your local ROM:

```bash
uv tool install env-supermariobrosnes-turbo-emu
smb-turbo play --rom /absolute/path/to/SuperMarioBros.nes
```

Playback requires a discoverable SDL2 runtime. Use the arrow keys or `A`/`D` to
move, `X`/`J`/Space to jump, `Z`/`K`/Shift to run, and Escape to quit.

Register the ROM once to use later commands without its path:

```bash
export RETRO_DATA_PATH="${XDG_DATA_HOME:-$HOME/.local/share}/retro"
smb-turbo import /absolute/path/to/SuperMarioBros.nes
smb-turbo play
smb-turbo play Level2-1 --fps max  # choose a state and run uncapped
```

ROM files are never included in this repository or its distributions.

## What it provides

- **Fast deterministic vectors.** Native batched emulation, preprocessing,
  rewards, termination, and infos run through one `step()`. Seeded lanes remain
  independent, autoreset stays disabled, and reset masks affect only selected
  lanes.
- **Reusable exact starts.** Select saved states per lane or capture and restore
  live snapshots without advancing emulation or disturbing other lanes.
- **Configurable inputs and observations.** Choose button masks or discrete
  action tables, grayscale or RGB, frame skip, max-pooling, crop behavior,
  resizing, frame stacking, and CHW or HWC layouts.
- **Research and playback tools.** Request raw or semantic game signals, inspect
  immutable API metadata, and use manual or framework-free action-run playback
  that follows matching policies as levels change.

See [API.md](API.md) for the complete environment, action, observation, state,
snapshot, rendering, playback, and research-info contracts.

## Compared with Stable Retro

[Stable Retro](https://stable-retro.farama.org/python/) is the broader choice
for multiple games and emulators, multiplayer, RAM observations, and BK2 movie
recording. Turbo specializes in one-player Super Mario Bros NES on mapper
0/NROM, with native vector execution, selective resets, image observations,
state catalogs, snapshots, and separately available batched RAM. It is not a
drop-in Stable Retro replacement.

## Use from Python

Add the package to a uv-managed project:

```bash
uv add env-supermariobrosnes-turbo-emu
```

```python
import gymnasium as gym

from env_supermariobrosnes_turbo_emu import Actions, action_batch

env = gym.make_vec(
    "env_supermariobrosnes_turbo_emu:EnvSuperMarioBrosNesTurboEmu-v0",
    game="SuperMarioBros-Nes-v0",
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

The module-qualified ID registers the vector-only factory; `game` is required.
The native `EnvSuperMarioBrosNesTurboEmuVecEnv` constructor is also public.

## Train with GradLab

Training implementations and recipes live in
[GradLab](https://github.com/tsilva/gradlab), outside this repository. Run either
published, version-pinned recipe from any directory with your local ROM.

```bash
# Short PPO demonstration
uvx gradlab@0.2.1 train SuperMarioBros-Nes-v0/Level1-1/turbo-demo --rom /absolute/path/to/SuperMarioBros.nes

# Go-Explore trajectory discovery capped at 20 million transitions
uvx gradlab@0.2.1 train SuperMarioBros-Nes-v0/Level1-1/go-explore-jerk-20m --rom /absolute/path/to/SuperMarioBros.nes
```

GradLab verifies the ROM, shows live progress, writes a playable
`final_model.zip` below `./runs`, and prints a version-pinned playback command.

## Notes

- Imported ROMs use
  `<RETRO_DATA_PATH>/stable/SuperMarioBros-Nes-v0/rom.nes`.
- The canonical ROM SHA-256 is
  `f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de`.
- Source builds, tests, parity checks, and contribution instructions are in
  [CONTRIBUTING.md](CONTRIBUTING.md).
- This unofficial research project is not affiliated with or endorsed by
  Nintendo. See [NOTICE.md](NOTICE.md).

## Architecture

![env-SuperMarioBrosNes-turbo-emu architecture diagram](https://raw.githubusercontent.com/tsilva/env-SuperMarioBrosNes-turbo-emu/main/architecture.png)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the native component boundaries and
verification hooks.

## License

Code is licensed under the [MIT License](LICENSE). Third-party names, marks, and
user-supplied content are excluded; see [NOTICE.md](NOTICE.md).
