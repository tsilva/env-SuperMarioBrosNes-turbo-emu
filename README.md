<div align="center">
  <img src="logo.png" alt="SuperMarioBros-Nes-turbo logo" width="320" />

  **🚀 Blazing fast SuperMarioBros-Nes environment for Reinforcement Learning 🍄**
</div>

<div align="center">
  <img src="media/mario-promo/mario-throughput-comparison.gif" alt="Stable Retro and SuperMarioBros-Nes-turbo throughput comparison" width="640" />
</div>

**SuperMarioBros-Nes-turbo** is a Rust-backed Gymnasium vector environment for
reinforcement-learning researchers working with Super Mario Bros NES. In the
published `0.3.0` mapper 0/NROM benchmark, it measured **13.27× to 18.27×** the
end-to-end step and preprocessing throughput of
[Stable Retro](https://github.com/Farama-Foundation/stable-retro), depending on
the host and number of environments.

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

Install the prebuilt package from PyPI:

```bash
python -m pip install supermariobrosnes-turbo
```

Prebuilt wheels support Python `>=3.9` on macOS, Linux, and Windows without a
Rust toolchain. See [CONTRIBUTING.md](CONTRIBUTING.md) for the source checkout
and development setup.

**ROM setup:** ROM files are not included. Set `RETRO_DATA_PATH` to a
user-writable data directory, then import the supported ROM from a file,
directory, or ZIP archive.

On macOS or Linux:

```bash
export RETRO_DATA_PATH="${XDG_DATA_HOME:-$HOME/.local/share}/retro"
smb-turbo import /path/to/roms
```

On Windows PowerShell:

```powershell
$env:RETRO_DATA_PATH = "$env:LOCALAPPDATA\retro"
smb-turbo import C:\path\to\roms
```

The importer writes
`<RETRO_DATA_PATH>/stable/SuperMarioBros-Nes-v0/rom.nes`. If the variable is
unset, it uses the equivalent data tree inside the installed package instead.
`rom_path=` and the CLI's `--rom` remain available as overrides. The canonical
ROM SHA-256 is:

```text
f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de
```

## 🎮 Use

```python
import numpy as np

from supermariobrosnes_turbo import (
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    action_batch,
)

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    state="Level1-1",
    num_envs=16,
    use_restricted_actions="basic",
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
    state_indices = np.full(env.num_envs, -1, dtype=np.int32)
    state_indices[done] = 0
    observations, reset_infos = env.reset(
        options={"reset_mask": done.copy(), "state_indices": state_indices},
    )
```

**Important:** Autoreset is disabled. Selectively reset terminal lanes before
stepping again.

Live positions can be captured without advancing emulation and restored into
any lane of the same environment:

```python
capture_mask = np.zeros(env.num_envs, dtype=np.bool_)
capture_mask[0] = True
captured = env.capture_snapshots(capture_mask)

restore_mask = np.zeros(env.num_envs, dtype=np.bool_)
restore_mask[3] = True
starts = [None] * env.num_envs
starts[3] = captured[0]
observations, infos = env.reset(
    options={"reset_mask": restore_mask, "snapshots": starts},
)
env.close()
```

Handles are reusable, session-local, and intentionally not pickleable. A
single masked reset can mix snapshot starts with ordinary `state_indices`;
`infos["start_source"]` distinguishes `"snapshot"` from `"environment"`.

## 🔬 Processed research infos

The original `INFO_KEYS` remain the default. Additional semantic game state is
opt-in, so the environment only decodes and returns the extra keys a caller
requests:

```python
from supermariobrosnes_turbo import AreaType, PlayerMotion

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    state="Level1-1",
    info_filter={
        "mode": "all",
        "keys": [
            "x_pos",
            "y_pos",
            "area_type",
            "player_motion",
            "enemy_active",
            "enemy_x_pos",
        ],
    },
)
observations, infos = env.reset()
in_water = infos["area_type"] == AreaType.WATER
climbing = infos["player_motion"] == PlayerMotion.CLIMBING
```

`EXTRA_INFO_KEYS` lists the opt-in catalog and `AVAILABLE_INFO_KEYS` combines it
with the legacy keys. Explicit selections reject unknown names, remove
duplicates, and return only selected game-state keys in catalog order. The
`terminal` and `none` modes retain their existing meaning; reset lifecycle
metadata and Gymnasium `_key` masks are not game-state selections.

All selectable game-state variables are listed below in their canonical
`AVAILABLE_INFO_KEYS` order. Legacy variables are returned by default; extra
variables are returned only when explicitly named in `info_filter["keys"]`.

| Key | Set | Shape | NumPy dtype | Meaning |
| --- | --- | --- | --- | --- |
| `x_pos` | Legacy/default | `(num_envs,)` | `np.int_` | Combined horizontal world position. |
| `coins` | Legacy/default | `(num_envs,)` | `np.int_` | Legacy coin counter. |
| `levelHi` | Legacy/default | `(num_envs,)` | `np.int_` | Legacy world-number component. |
| `levelLo` | Legacy/default | `(num_envs,)` | `np.int_` | Legacy level-number component. |
| `lives` | Legacy/default | `(num_envs,)` | `np.int_` | Signed legacy life counter; `-1` signals game over. |
| `score` | Legacy/default | `(num_envs,)` | `np.int_` | Decoded decimal game score. |
| `scrolling` | Legacy/default | `(num_envs,)` | `np.int_` | Legacy horizontal-scrolling signal. |
| `time` | Legacy/default | `(num_envs,)` | `np.int_` | Decoded decimal level timer. |
| `xscrollHi` | Legacy/default | `(num_envs,)` | `np.int_` | High/page component of horizontal scroll. |
| `xscrollLo` | Legacy/default | `(num_envs,)` | `np.int_` | Low component of horizontal scroll. |
| `area_id` | Extra/opt-in | `(num_envs,)` | `np.int16` | Stable internal subarea identifier. |
| `area_type` | Extra/opt-in | `(num_envs,)` | `np.int8` | `AreaType`: `UNKNOWN=-1`, `WATER=0`, `GROUND=1`, `UNDERGROUND=2`, `CASTLE=3`. |
| `y_pos` | Extra/opt-in | `(num_envs,)` | `np.int32` | Combined world-space vertical position. |
| `y_screen_pos` | Extra/opt-in | `(num_envs,)` | `np.int16` | Screen-relative vertical position. |
| `player_motion` | Extra/opt-in | `(num_envs,)` | `np.int8` | `PlayerMotion`: `UNKNOWN=-1`, `GROUND=0`, `JUMPING_OR_SWIMMING=1`, `FALLING=2`, `CLIMBING=3`. |
| `player_power` | Extra/opt-in | `(num_envs,)` | `np.int8` | `PlayerPower`: `UNKNOWN=-1`, `SMALL=0`, `BIG=1`, `FIRE=2`. |
| `is_large` | Extra/opt-in | `(num_envs,)` | `np.bool_` | Normalized large-player hitbox/size state. |
| `x_velocity` | Extra/opt-in | `(num_envs,)` | `np.int16` | Sign-extended horizontal velocity in SMB velocity units. |
| `y_velocity` | Extra/opt-in | `(num_envs,)` | `np.int16` | Sign-extended vertical velocity in SMB velocity units. |
| `facing` | Extra/opt-in | `(num_envs,)` | `np.int8` | `Direction`: `LEFT=-1`, `NONE=0`, `RIGHT=1`. |
| `is_crouching` | Extra/opt-in | `(num_envs,)` | `np.bool_` | Normalized crouching state. |
| `is_swimming` | Extra/opt-in | `(num_envs,)` | `np.bool_` | Normalized swimming state. |
| `injury_timer` | Extra/opt-in | `(num_envs,)` | `np.int16` | Injury/invulnerability countdown in game-timer ticks. |
| `star_timer` | Extra/opt-in | `(num_envs,)` | `np.int16` | Star-power countdown in game-timer ticks. |
| `game_mode` | Extra/opt-in | `(num_envs,)` | `np.int8` | `GameMode`: `UNKNOWN=-1`, `TITLE=0`, `GAMEPLAY=1`, `VICTORY=2`, `GAME_OVER=3`. |
| `player_task` | Extra/opt-in | `(num_envs,)` | `np.int8` | `PlayerTask`: `UNKNOWN=-1`, `ENTRANCE_TIMER_SETUP=0`, `VINE_AUTO_CLIMB=1`, `VERTICAL_PIPE_ENTRY=2`, `SIDE_PIPE_ENTRY=3`, `FLAGPOLE_SLIDE=4`, `LEVEL_END=5`, `LOSE_LIFE=6`, `PLAYER_ENTRANCE=7`, `PLAYER_CONTROL=8`, `CHANGE_SIZE=9`, `INJURY_BLINK=10`, `PLAYER_DEATH=11`, `FIRE_FLOWER_TRANSFORM=12`. |
| `enemy_active` | Extra/opt-in | `(num_envs, 6)` | `np.bool_` | Normalized active mask for the six enemy/object slots. |
| `enemy_type_id` | Extra/opt-in | `(num_envs, 6)` | `np.int16` | Stable SMB object-category ID; inactive slots are `-1`. |
| `enemy_x_pos` | Extra/opt-in | `(num_envs, 6)` | `np.int32` | Combined horizontal world positions; inactive slots are `-1`. |
| `enemy_y_pos` | Extra/opt-in | `(num_envs, 6)` | `np.int32` | Combined vertical positions; inactive slots are `-1`. |
| `enemy_x_velocity` | Extra/opt-in | `(num_envs, 6)` | `np.int16` | Signed horizontal velocities; inactive slots are `0`. |
| `enemy_y_velocity` | Extra/opt-in | `(num_envs, 6)` | `np.int16` | Signed vertical velocities; inactive slots are `0`. |
| `enemy_facing` | Extra/opt-in | `(num_envs, 6)` | `np.int8` | Normalized `Direction`; inactive slots are `0` (`NONE`). |

The environment may also add lifecycle metadata independently of the selected
game-state variables:

| Key | When present | Meaning |
| --- | --- | --- |
| `state_index` | Reset lanes | Active state-catalog index. |
| `start_source` | Reset lanes | Whether the lane started from the environment or a snapshot. |
| `terminated` | Terminated lanes | The lane reached a game terminal state. |
| `truncated` | Truncated lanes | The lane reached an external episode limit. |
| `_<key>` | With each emitted key | Gymnasium boolean mask identifying lanes for which that key is valid. |

Returned game-state arrays are owned copies and cannot be changed by later
environment steps. Unknown categorical engine values become `UNKNOWN = -1`.

Researchers who intentionally need unprocessed state can call `env.ram()` for
an immutable owned `(num_envs, 2048)` `uint8` snapshot. RAM addresses and byte
decoding are not part of the semantic `info` contract.

## 🏁 Train and play

```bash
smb-turbo train Level1-1
smb-turbo play
```

**Training** searches observation-free `(action, duration)` programs with
Go-Explore and the `standard` action set by default. It consumes the transition
budget as an anytime improvement search, keeping the best completed trajectory
locked and publishing only higher-return completions; pass `--stop-on-completion`
to stop after the first completed path. Go-Explore ranks raw game-score gains
first and charges each step `1 / (max_episode_steps + 1)`, so higher score always
wins while fewer steps break equal-score ties. Every completion is appended to
`successes.jsonl`.

A new default Go-Explore run replaces the existing canonical run; custom outputs
and explicit Beam or JERK runs remain protected unless `--overwrite` is passed.
`Level1-1` writes `runs/Level1-1/Level1-1.zip`; playback uses the matching trained
policy when available and switches policies as levels change. Running
`smb-turbo play` without a state starts from `Level1-1`; pass an exact state
identifier to start elsewhere. Playback defaults to 30 FPS; pass `--fps max` (or
its `--fpx max` alias) to run without an explicit delay or renderer-vsync cap.
Run either command with `--help` for configuration options. Policy playback
defaults to `--view raw`, which displays RGB directly from its sole emulator
without grayscale conversion, cropping, resizing, max-pooling, or frame stacking;
`--view preprocessed` instead shows the transformed policy observation.

State names are exact identifiers from the configured state catalog. This
includes canonical names such as `Level1-1`, packaged variants such as
`Level2-1-clouds-easy`, and imported names such as `Custom`; shorthand and case
normalization are intentionally unsupported.

In an interactive terminal, all trainers automatically open a full-screen
dashboard with live transition, throughput, search, best-path, and event stats.
Redirected output and CI use the existing plain logs; pass `--ui plain` to
select them explicitly. Press `q` or `Ctrl-C` in the dashboard for a safe stop:
the current vector step finishes, final metrics are written, and the best
policy is saved when a candidate exists.

The checkout-compatible `uv run python train.py Level1-1` and
`uv run python play.py` entry points remain available.

To use JERK instead of the default Go-Explore search while keeping the same
action-run representation, episode boundary, and playback format, run:

```bash
smb-turbo train Level1-1 --algorithm jerk --overwrite
smb-turbo play Level1-1
```

To use fixed-width Beam search instead, run:

```bash
smb-turbo train Level1-1 --algorithm beam --overwrite
smb-turbo play Level1-1
```

Beam ranks completed trajectories with the same score-first return as Go-Explore,
retains incomplete alternatives by furthest progress, and systematically moves
splice mutations from the tail toward the root while replaying the proven suffix.
A compatible `--initial-policy` with a smaller action table is remapped by action
name into the selected table, so historical `basic` policies can seed the
`standard` search without restricting it.

Go-Explore uses the same canonical `(action, duration)` ZIP format as Beam, so
the regular playback command needs no algorithm-specific mode. It performs
trajectory finding with exact archived-state restoration and no robustification.
Go-Explore cells are keyed by level, sublevel, and the raw bytes of the native
8x8 grayscale frame after HUD masking and 3-bit quantization; horizontal position
is not part of the cell key. Keeping the 64-byte visual value directly avoids
application-level hash collisions while remaining negligible beside a snapshot.
Go-Explore ranks paths with raw game-score gains first and charges each step
`1 / (max_episode_steps + 1)` by default, so the entire episode's time charge is
less than one score point. Higher score therefore always wins, while fewer steps
break ties; an explicit `--step-cost` overrides that default.
After the first completion, half of archived restores continue novelty-weighted
exploration and half sample underused cells across the best successful
trajectory. Success return is propagated through parent-linked archived cells,
so score improvement can mutate the whole proven route instead of only states
near the flag.

Omit the training state to process all 32 canonical levels in game order:

```bash
smb-turbo train
```

The transition budget applies independently to each level. Every policy and its
artifacts are written under `runs/<State>/`; with `--output <Directory>`, each
level instead uses `<Directory>/<State>/`. Interactive campaigns keep one TUI
open and show separate progress bars for the current level's transitions and the
overall 32-level campaign. A level that exhausts its budget is reported and the
campaign continues to the next level; a safe stop ends the current level and
does not start another. The default `standard` action set keeps pipe-dependent
levels searchable; an explicit `--action-set` is respected.

New default runs use `runs/<State>/` regardless of algorithm. For compatibility,
playback still discovers historical algorithm-specific directories, preferring
`runs/<State>-beam/` over `runs/<State>-jerk/`.

## 🧰 Commands

```bash
smb-turbo import /path/to/roms        # import the supported ROM
smb-turbo train Level1-1              # train one state-keyed beam policy
smb-turbo train                       # train all 32 canonical levels in order
smb-turbo play                        # play Level1-1 manually or with its policy
uv sync --frozen --extra dev --group dev  # install development dependencies
uv run maturin develop --release      # build the optimized Rust extension
make test                             # run Rust and Python tests
make test-retro-oracle                # run ROM-backed parity and policy tests
make benchmark                        # benchmark SuperMarioBros-Nes-turbo locally
make benchmark-report                 # compare SuperMarioBros-Nes-turbo with Stable Retro
uv run python scripts/benchmark_info_filter.py --rom /path/to/rom.nes  # diagnostic infos overhead
```

## 📈 Benchmark

[![SuperMarioBros-Nes-turbo versus Stable Retro median environment throughput](media/benchmark-throughput.svg)](BENCHMARKS.md)

The chart records the published `0.3.0` comparison. See
[BENCHMARKS.md](BENCHMARKS.md) for exact results, protocol, and machine details.
`benchmark_info_filter.py` is a paired diagnostic for the optional research-info
path only; its output is never eligible for autoresearch acceptance records.

## Notes

- **Scope:** This emulator supports only `SuperMarioBros-Nes-v0` on mapper
  0/NROM; it is not a general NES or Stable Retro replacement.
- **States:** Packaged states cover `Level1-1` through `Level8-4`, with
  additional variants. `state=` accepts one name, path, or byte payload;
  `state_catalog=` preloads an ordered selection for explicit per-lane resets.
- **Actions:** `Actions.ALL` and `Actions.FILTERED` accept per-button masks;
  `Actions.DISCRETE` provides Stable Retro-compatible 36-way actions and
  `Actions.MULTI_DISCRETE` exposes the three restricted button groups. Named
  metadata presets (`basic`, `standard`, `right-jump`, `basic-start`) and inline button
  tables such as `[[], ["RIGHT"], ["RIGHT", "A"]]` produce exact discrete
  spaces through `use_restricted_actions`.
- **Playback:** Play commands require a discoverable native SDL2 library and
  open local gameplay windows.
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) and follow the
  [Code of Conduct](CODE_OF_CONDUCT.md).
- **Affiliation:** This unofficial research project is not affiliated with or
  endorsed by Nintendo. See [NOTICE.md](NOTICE.md).

## Architecture

![SuperMarioBros-Nes-turbo architecture diagram](architecture.png)

## License

Code is licensed under the [MIT License](LICENSE). Third-party names, marks, and
user-supplied content are excluded; see [NOTICE.md](NOTICE.md).
