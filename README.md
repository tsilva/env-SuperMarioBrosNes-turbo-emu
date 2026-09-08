<p align="center">
  <img src="https://raw.githubusercontent.com/tsilva/env-SuperMarioBrosNes-turbo-emu/main/image-assets/logo/logo-1024.png" alt="SuperMarioBrosNes-turbo logo" width="240" />
  <br />
  <strong>🍄 More Mario rollouts. Less waiting. ⚡</strong>
</p>

<p align="center">
  <a href="https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/actions/workflows/ci.yml"><img src="https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status" /></a>
  <a href="https://pypi.org/project/env-supermariobrosnes-turbo-emu/"><img src="https://img.shields.io/pypi/v/env-supermariobrosnes-turbo-emu" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/env-supermariobrosnes-turbo-emu/"><img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue" alt="Python 3.9 or newer" /></a>
  <a href="https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/env-supermariobrosnes-turbo-emu" alt="MIT license" /></a>
</p>

**SuperMarioBrosNes-turbo** is a Python environment for reinforcement-learning
researchers running Super Mario Bros NES experiments. Collect rollouts through
a deterministic Gymnasium vector API, with independent environments and reusable
state snapshots.

**15.4–17.3× the environment-step throughput of Stable Retro** in matched
benchmarks of v0.6.4 against Stable Retro v1.0.1 on a Ryzen 5 7600X.
See [workloads, methodology, and results](BENCHMARKS.md).

Bring your own supported ROM. Start playing in two commands, or use the
[Python API](#use-from-python) below.

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

## What it provides

- Collect experience across independent, seeded environments with native batched
  emulation and observation processing.
- Repeat exact starts using saved states or live snapshots.
- Reset or restore selected environments without disturbing the others.
- Choose actions, observations, and game signals to suit your experiment.

See [API.md](API.md) for action tables, preprocessing options, state and snapshot
controls, playback, and research-info contracts.

## Compared with Stable Retro

[Stable Retro](https://stable-retro.farama.org/python/) is the broader choice
for multiple games and emulators, multiplayer, RAM observations, and BK2 movie
recording. Turbo specializes in one-player Super Mario Bros NES on mapper
0/NROM, with native vector execution, selective resets, image observations,
state catalogs, snapshots, and separately available batched RAM. It is not a
drop-in Stable Retro replacement.

## Train with GradLab

Training implementations and recipes live in
[GradLab](https://github.com/tsilva/gradlab), outside this repository. Run either
published, version-pinned recipe from any directory with your local ROM.

```bash
# Short PPO demonstration
uvx --python 3.14 gradlab@0.2.2 train SuperMarioBros-Nes-v0/Level1-1/turbo-demo --rom-path /absolute/path/to/SuperMarioBros.nes

# Go-Explore trajectory discovery capped at 20 million transitions
uvx --python 3.14 gradlab@0.2.2 train SuperMarioBros-Nes-v0/Level1-1/go-explore-20m --rom-path /absolute/path/to/SuperMarioBros.nes
```

GradLab verifies the ROM, shows live progress, writes a playable
`final_model.zip` below `./runs`, and prints a version-pinned playback command.

## Register your ROM

Register the ROM once to use later commands without its path:

```bash
export RETRO_DATA_PATH="${XDG_DATA_HOME:-$HOME/.local/share}/retro"
smb-turbo import /absolute/path/to/SuperMarioBros.nes
smb-turbo play
smb-turbo play Level2-1 --fps max  # choose a state and run uncapped
```

ROM files are never included in this repository or its distributions.

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

![SuperMarioBrosNes-turbo architecture diagram](https://raw.githubusercontent.com/tsilva/env-SuperMarioBrosNes-turbo-emu/main/architecture.png)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the native component boundaries and
verification hooks.

## License

Code is licensed under the [MIT License](LICENSE). Third-party names, marks, and
user-supplied content are excluded; see [NOTICE.md](NOTICE.md).
