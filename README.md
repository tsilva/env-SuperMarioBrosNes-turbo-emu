<p align="center">
  <img src="https://raw.githubusercontent.com/tsilva/env-SuperMarioBrosNes-turbo-emu/main/logo.png" alt="SuperMarioBrosNes-turbo logo" width="240" />
  <br />
  <!-- repo-tagline:start -->
  <strong>🍄 More Mario rollouts. Less waiting ⚡</strong>
  <!-- repo-tagline:end -->
</p>

<p align="center">
  <a href="https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/actions/workflows/ci.yml"><img src="https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status" /></a>
  <a href="https://pypi.org/project/env-supermariobrosnes-turbo-emu/"><img src="https://img.shields.io/pypi/v/env-supermariobrosnes-turbo-emu" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/env-supermariobrosnes-turbo-emu/"><img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue" alt="Python 3.9 or newer" /></a>
  <a href="https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/env-supermariobrosnes-turbo-emu" alt="MIT license" /></a>
</p>

SuperMarioBrosNes-turbo is a Python environment for parallel reinforcement
learning on Super Mario Bros NES. Run independent lanes through a deterministic
Gymnasium vector API and save state snapshots for repeatable experiments.

**15.4–17.3× the environment-step throughput of Stable Retro** in matched
benchmarks of historical v0.6.4 against Stable Retro v1.0.1 on a Ryzen 5 7600X.
See [workloads, methodology, and results](https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/BENCHMARKS.md).

## Quick start

Prebuilt wheels support Python `>=3.9` on Apple-silicon macOS 14 or newer
and x86-64 Linux with glibc 2.28 or newer.
Install the CLI with [uv](https://docs.astral.sh/uv/) and launch Level 1-1 with
your local ROM:

```bash
uv tool install env-supermariobrosnes-turbo-emu
smb-turbo play --rom /absolute/path/to/SuperMarioBros.nes
```

Playback opens a window and requires a discoverable SDL2 runtime; see
[playback setup](https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/API.md#playback) if SDL2 or `smb-turbo` is unavailable.
Use the arrow keys or `A`/`D` to move, `X`/`J`/Space to jump,
`Z`/`K`/Shift to run, and Escape to quit.

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

See [API.md](https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/API.md) for action tables, preprocessing options, state and snapshot
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
uvx --python 3.14 gradlab@0.2.4 train SuperMarioBros-Nes-v0/Level1-1/turbo-demo --rom-path /absolute/path/to/SuperMarioBros.nes --no-wandb --set train.backend.config.device=auto

# Go-Explore trajectory discovery capped at 20 million transitions
uvx --python 3.14 gradlab@0.2.4 train SuperMarioBros-Nes-v0/Level1-1/go-explore-20m --rom-path /absolute/path/to/SuperMarioBros.nes --no-wandb
```

GradLab verifies the ROM, shows live progress, and exports
`final_model.zip` below `~/.config/gradlab/runs` by default, and prints a
version-pinned playback command. These commands keep training outputs local;
the PPO demo automatically selects an available Torch device. GradLab 0.2.4
installs Mario environment 0.7.4 in an isolated environment and includes its
local web player. See [training outputs and bounded runs](https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/API.md#gradlab-training).

- [Register your ROM](https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/API.md#playback)
- [Native architecture](https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/ARCHITECTURE.md)
- [Build, test, and contribute](https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/CONTRIBUTING.md)

## License

[MIT License](https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/LICENSE).
See [NOTICE.md](https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu/blob/main/NOTICE.md)
for third-party names, marks, and user-supplied content.
