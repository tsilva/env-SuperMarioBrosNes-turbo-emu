# API reference

`SuperMarioBrosNesTurboVecEnv` is the public Gymnasium vector environment for
the supported `SuperMarioBros-Nes-v0` mapper 0/NROM workload. This reference
collects the contracts that are intentionally summarized in the README.

## Turbo Vector API v1

The environment advertises the strict Turbo Vector API v1:

- `metadata["turbo_api_version"]` is `1`, and `metadata["render_modes"]`
  advertises `rgb_array`.
- Immutable `capabilities` and `signal_schema` declarations describe supported
  features and the dtype, shape, and reset/step availability of every signal.
- `buttons`, `action_mode`, `action_preset`, `action_table`, `action_meanings`,
  and `action_table_hash` expose resolved action semantics.
- `state_catalog` is an immutable ordered tuple. Callers select reset states
  with an `int32` `state_indices` array and inspect the read-only active indices
  with `active_state_indices()`.
- `observation_ownership` and `observation_buffer_depth` declare the lifetime of
  returned observations.
- `render_lane(index)` renders one lane, `get_images()` renders every lane, and
  `render()` renders lane zero.

The environment conforms to Gymnasium's vector reset and step returns. Autoreset
is permanently disabled: stepping a terminal lane again is an error until that
lane is explicitly reset through `options["reset_mask"]`.

## Actions and observations

`use_restricted_actions` accepts these built-in modes:

- `Actions.ALL` and `Actions.FILTERED` use per-button masks.
- `Actions.DISCRETE` provides Stable Retro-compatible 36-way actions.
- `Actions.MULTI_DISCRETE` exposes the three restricted button groups.
- Named metadata presets include `basic`, `standard`, `right-jump`, and
  `basic-start`.
- Caller-supplied button tables such as `[[], ["RIGHT"], ["RIGHT", "A"]]`
  create exact discrete action spaces.

Observation options include grayscale or RGB, frame skip, optional max-pooling,
crop removal or masking, resize, frame stacking, and CHW or HWC layouts.

## States and selective reset

Construction accepts one packaged or named state, state path, or byte payload.
Use `state_catalog=` to preload an ordered set for explicit per-lane selection.
Index zero is the deterministic default.

```python
import numpy as np

from supermariobrosnes_turbo import SuperMarioBrosNesTurboVecEnv

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    rom_path="/absolute/path/to/SuperMarioBros.nes",
    num_envs=4,
    state_catalog=("Level1-1", "Level1-2"),
)

try:
    observations, infos = env.reset(seed=123)

    reset_mask = np.array([False, True, False, True], dtype=np.bool_)
    state_indices = np.array([-1, 1, -1, 0], dtype=np.int32)
    observations, reset_infos = env.reset(
        options={
            "reset_mask": reset_mask,
            "state_indices": state_indices,
        },
    )
finally:
    env.close()
```

Resetting selected lanes leaves every unselected lane's emulator state, random
stream, observation history, sticky action, and counters unchanged. Seeded
random reset-NOOP advancement is lane-local, opt-in, and disabled by default.

## Live snapshots

Live positions can be captured without advancing emulation and restored into
another lane of the same environment or encoded for a compatible destination:

```python
import numpy as np

from supermariobrosnes_turbo import SuperMarioBrosNesTurboVecEnv


def make_env():
    return SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        rom_path="/absolute/path/to/SuperMarioBros.nes",
        num_envs=4,
    )


source = make_env()
destination = make_env()
try:
    source.reset(seed=123)
    destination.reset(seed=123)

    capture_mask = np.array([True, False, False, False], dtype=np.bool_)
    captured = source.capture_snapshots(capture_mask)

    # Restore source lane 0 into source lane 3 without advancing other lanes.
    source_restore_mask = np.array([False, False, False, True], dtype=np.bool_)
    source_starts = [None, None, None, captured[0]]
    source.reset(
        options={
            "reset_mask": source_restore_mask,
            "snapshots": source_starts,
        },
    )

    # Encode before closing source, then bind the bytes to destination lane 3.
    payloads = source.encode_snapshots(captured)
    destination_payloads = [None, None, None, payloads[0]]
    destination_handles = destination.decode_snapshots(destination_payloads)
    destination_restore_mask = np.array(
        [False, False, False, True],
        dtype=np.bool_,
    )
    destination.reset(
        options={
            "reset_mask": destination_restore_mask,
            "snapshots": destination_handles,
        },
    )
finally:
    source.close()
    destination.close()
```

Live handles are reusable, same-instance, session-local, and intentionally not
pickleable. Portable bytes bind decoded handles to a compatible destination.
The immutable `snapshot_codec_metadata` declaration identifies
`supermariobrosnes-turbo.portable-v1`. One masked reset can mix snapshot starts
with ordinary `state_indices`; `infos["start_source"]` distinguishes
`"snapshot"` from `"environment"`.

## Research infos

The original `INFO_KEYS` remain the default. Additional semantic game state is
opt-in, so only explicitly requested extra keys are decoded and returned:

```python
from supermariobrosnes_turbo import AreaType, PlayerMotion

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    state="Level1-1",
    rom_path="/absolute/path/to/SuperMarioBros.nes",
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
env.close()
```

`EXTRA_INFO_KEYS` lists the opt-in catalog and `AVAILABLE_INFO_KEYS` combines it
with the legacy keys. Explicit selections reject unknown names, remove
duplicates, and return selected game-state keys in catalog order. The
`terminal` and `none` modes retain their documented behavior.

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
| `area_pointer` | Extra/opt-in | `(num_envs,)` | `np.int16` | Current SMB area-data pointer for routes that reuse coordinates. |
| `loop_command_active` | Extra/opt-in | `(num_envs,)` | `np.bool_` | Whether SMB's castle loop command is active. |
| `loop_correct_count` | Extra/opt-in | `(num_envs,)` | `np.int16` | Accepted steps in the active castle-loop route. |
| `loop_pass_count` | Extra/opt-in | `(num_envs,)` | `np.int16` | Completed castle-loop passes. |

The environment may also add lifecycle metadata independently of selected
game-state variables:

| Key | When present | Meaning |
| --- | --- | --- |
| `state_index` | Reset lanes | Active state-catalog index. |
| `start_source` | Reset lanes | Whether the lane started from the environment or a snapshot. |
| `terminated` | Terminated lanes | The lane reached a native game terminal state. |
| `truncated` | Truncated lanes | The lane reached an external episode limit. |
| `_<key>` | With each emitted key | Gymnasium mask identifying lanes for which that key is valid. |

Returned game-state arrays are owned copies and cannot be changed by later
steps. Unknown categorical values become `UNKNOWN = -1`. Call `env.ram()` for
an immutable owned `(num_envs, 2048)` `uint8` CPU RAM snapshot when unprocessed
state is required; RAM addresses are not part of the semantic info contract.

## Playback

`smb-turbo play` without a state starts from `Level1-1`; pass an exact state
identifier to start elsewhere. Packaged identifiers include canonical names
such as `Level1-1` and variants such as `Level2-1-clouds-easy`. Shorthand and
case normalization are intentionally unsupported.

Playback is manual unless a compatible state-keyed action-run policy exists
below `runs/`. As gameplay enters another canonical level, playback selects
that level's matching policy when available. Historical discovery prefers
`runs/<State>-beam/` over `runs/<State>-jerk/`; these action-run policies require
no neural-network framework.

Playback defaults to 30 FPS. Pass `--fps max` (or the legacy `--fpx max` alias)
for uncapped playback. Policy playback defaults to `--view raw`; use
`--view preprocessed` to display the transformed policy observation. The
checkout-compatible entry point is `uv run python play.py`.

GradLab models use GradLab's interactive player. Each documented GradLab train
command prints the matching version-pinned playback command for its model.
