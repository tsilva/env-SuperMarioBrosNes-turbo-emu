from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum, Flag
import gzip
from importlib import resources
import json
import os
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from ._supermariobrosnes_turbo import _RetroVecEnv as _CoreRetroVecEnv
from .roms import default_rom_path, game_data_path, resolve_required_rom_path


VISIBLE_WIDTH = 240
VISIBLE_HEIGHT = 224
NES_BUTTONS = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
BUTTON_TO_INDEX = {name: index for index, name in enumerate(NES_BUTTONS) if name is not None}
CORE_ACTION_MEANINGS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left", "start")
ACTION_SETS = {
    "simple": ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left"),
    "right": ("right", "right_b", "right_a", "right_a_b"),
    "full": CORE_ACTION_MEANINGS,
}
ACTION_MEANINGS = ACTION_SETS["simple"]
ACTION_BUTTONS = {
    "noop": (),
    "right": ("RIGHT",),
    "right_b": ("RIGHT", "B"),
    "right_a": ("RIGHT", "A"),
    "right_a_b": ("RIGHT", "A", "B"),
    "a": ("A",),
    "left": ("LEFT",),
    "start": ("START",),
}
MASK_BIT_WEIGHTS = (1 << np.arange(len(NES_BUTTONS), dtype=np.uint16)).astype(np.uint16)
DEFAULT_STABLE_RETRO_GAME = "SuperMarioBros-Nes-v0"
GZIP_MAGIC = b"\x1f\x8b"
INFO_KEYS = (
    "x_pos",
    "coins",
    "levelHi",
    "levelLo",
    "lives",
    "score",
    "scrolling",
    "time",
    "xscrollHi",
    "xscrollLo",
)
_BASE_INFO_ARRAYS = (
    ("x_pos", "_x_pos"),
    ("coins", "_coins"),
    ("levelHi", "_level_hi"),
    ("levelLo", "_level_lo"),
    ("lives", "_lives"),
    ("score", "_score"),
    ("scrolling", "_scrolling"),
    ("time", "_time"),
    ("xscrollHi", "_xscroll_hi"),
    ("xscrollLo", "_xscroll_lo"),
)
StateSpec = Union[str, Path, bytes, bytearray, memoryview]


class Actions(Enum):
    """Small Stable Retro action-mode stand-in for SMB-only compatibility."""

    ALL = 0
    FILTERED = 1
    DISCRETE = 2
    MULTI_DISCRETE = 3


class State(Enum):
    DEFAULT = -1
    NONE = 0


class Observations(Enum):
    IMAGE = 0
    RAM = 1


class Integrations(Flag):
    STABLE = 1
    EXPERIMENTAL_ONLY = 2
    CONTRIB_ONLY = 4
    CUSTOM_ONLY = 8
    EXPERIMENTAL = STABLE | EXPERIMENTAL_ONLY
    CONTRIB = STABLE | CONTRIB_ONLY
    CUSTOM = STABLE | CUSTOM_ONLY
    ALL = STABLE | EXPERIMENTAL_ONLY | CONTRIB_ONLY | CUSTOM_ONLY


def _expand_rom_path(path: str | Path | None) -> str:
    return str(resolve_required_rom_path(path))


def _stable_retro_state_dir() -> Path | None:
    try:
        import stable_retro.data  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        state_path = stable_retro.data.get_file_path(
            DEFAULT_STABLE_RETRO_GAME,
            "Level1-1.state",
            stable_retro.data.Integrations.ALL,
        )
    except Exception:
        return None
    if not state_path:
        return None
    return Path(state_path).parent


def _packaged_state_dir() -> Path | None:
    state_dir = resources.files(__package__).joinpath("data", DEFAULT_STABLE_RETRO_GAME)
    if not state_dir.is_dir():
        return None
    return Path(str(state_dir))


def _candidate_state_dirs(state_dir: str | Path | None = None) -> list[Path]:
    candidates: list[Path | None] = []
    if state_dir is not None:
        candidates.append(Path(state_dir).expanduser())
    env_dir = os.environ.get("SUPERMARIOBROSNES_FASTENV_STATE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(game_data_path())
    candidates.append(_packaged_state_dir())
    candidates.append(_stable_retro_state_dir())
    dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved not in seen and resolved.exists():
            dirs.append(resolved)
            seen.add(resolved)
    return dirs


def list_available_states(state_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return available stable-retro Super Mario Bros state names."""
    states: set[str] = set()
    for directory in _candidate_state_dirs(state_dir):
        states.update(
            path.stem for path in directory.glob("*.state") if not path.name.startswith("_")
        )
    return tuple(sorted(states))


def _resolve_state_path(state: str | Path, state_dir: str | Path | None = None) -> Path:
    raw_path = Path(state).expanduser()
    if raw_path.exists():
        return raw_path

    state_name = str(state)
    state_file = state_name if state_name.endswith(".state") else f"{state_name}.state"
    for directory in _candidate_state_dirs(state_dir):
        candidate = directory / state_file
        if candidate.exists():
            return candidate

    dirs = ", ".join(str(path) for path in _candidate_state_dirs(state_dir)) or "<none>"
    raise FileNotFoundError(
        f"could not resolve state {state_name!r}; checked direct path and state dirs: {dirs}"
    )


def _load_initial_state(
    state: StateSpec,
    state_dir: str | Path | None = None,
) -> bytes:
    if isinstance(state, (bytes, bytearray, memoryview)):
        raw = bytes(state)
    else:
        raw = _resolve_state_path(state, state_dir).read_bytes()
    if raw.startswith(GZIP_MAGIC):
        return gzip.decompress(raw)
    return raw


def _state_label(state: StateSpec, fallback: str) -> str:
    if isinstance(state, (bytes, bytearray, memoryview)):
        return fallback
    return str(state)


def _normalize_initial_state_config(
    state: StateSpec | Sequence[StateSpec] | Mapping[StateSpec, float] | None,
    state_dir: str | Path | None,
    num_envs: int,
) -> tuple[list[bytes], tuple[str, ...], list[float] | None]:
    if state is None:
        return [], (), None

    if isinstance(state, Mapping):
        if not state:
            raise ValueError("weighted state mapping must contain at least one state")
        initial_states: list[bytes] = []
        state_names: list[str] = []
        state_weights: list[float] = []
        for index, (state_value, weight_value) in enumerate(state.items()):
            weight = float(weight_value)
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError("weighted state values must be non-negative finite numbers")
            initial_states.append(_load_initial_state(state_value, state_dir))
            state_names.append(_state_label(state_value, f"state-{index}"))
            state_weights.append(weight)
        total_weight = float(np.sum(state_weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("weighted state values must sum to a positive finite number")
        return initial_states, tuple(state_names), [weight / total_weight for weight in state_weights]

    if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray, memoryview)):
        states = list(state)
        if len(states) != num_envs:
            raise ValueError(
                "state sequences must provide exactly one state per env slot: "
                f"got {len(states)} states for num_envs={num_envs}"
            )
        if not states:
            raise ValueError("state sequence must contain at least one state")
        return (
            [_load_initial_state(state_value, state_dir) for state_value in states],
            tuple(_state_label(state_value, f"state-{index}") for index, state_value in enumerate(states)),
            None,
        )

    return [_load_initial_state(state, state_dir)], (_state_label(state, "state-0"),), None


def _resolve_action_set(action_set: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(action_set, str):
        try:
            return ACTION_SETS[action_set]
        except KeyError as exc:
            valid = ", ".join(sorted(ACTION_SETS))
            raise ValueError(f"unknown action_set {action_set!r}; valid values: {valid}") from exc

    actions = tuple(str(action) for action in action_set)
    if not actions:
        raise ValueError("action_set must contain at least one action")
    unknown = [action for action in actions if action not in CORE_ACTION_MEANINGS]
    if unknown:
        valid = ", ".join(CORE_ACTION_MEANINGS)
        raise ValueError(f"unknown action(s) {unknown!r}; valid actions: {valid}")
    return actions


def _core_action_ids(action_meanings: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [CORE_ACTION_MEANINGS.index(action) for action in action_meanings],
        dtype=np.uint8,
    )


def _action_masks(action_meanings: tuple[str, ...]) -> np.ndarray:
    masks = np.zeros((len(action_meanings), len(NES_BUTTONS)), dtype=np.uint8)
    for action_index, action_name in enumerate(action_meanings):
        for button in ACTION_BUTTONS[action_name]:
            masks[action_index, BUTTON_TO_INDEX[button]] = 1
    return masks


def action_mask(action_name: str) -> np.ndarray:
    """Return a single SMB button mask for a named core action."""
    if action_name not in ACTION_BUTTONS:
        valid = ", ".join(CORE_ACTION_MEANINGS)
        raise ValueError(f"unknown action {action_name!r}; valid actions: {valid}")
    mask = np.zeros((len(NES_BUTTONS),), dtype=np.uint8)
    for button in ACTION_BUTTONS[action_name]:
        mask[BUTTON_TO_INDEX[button]] = 1
    return mask


def action_batch(actions: str | Sequence[str], num_envs: int) -> np.ndarray:
    """Return an `(num_envs, num_buttons)` SMB action mask batch."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if isinstance(actions, str):
        names = [actions] * num_envs
    else:
        names = [str(action) for action in actions]
        if len(names) != num_envs:
            raise ValueError(
                "action sequences must provide exactly one action per env slot: "
                f"got {len(names)} actions for num_envs={num_envs}"
            )
    masks = np.zeros((num_envs, len(NES_BUTTONS)), dtype=np.uint8)
    for env_idx, name in enumerate(names):
        masks[env_idx] = action_mask(name)
    return masks


def _normalize_action_mode(value: Any) -> str:
    name = getattr(value, "name", value)
    text = str(name).split(".")[-1].upper()
    if text not in {"ALL", "FILTERED", "DISCRETE"}:
        raise ValueError("use_restricted_actions must be Actions.ALL, Actions.FILTERED, or Actions.DISCRETE")
    return text


def _normalize_obs_copy(obs_copy: str) -> tuple[str, bool, bool]:
    if isinstance(obs_copy, bool):
        raise ValueError("obs_copy must be 'copy', 'safe_view', or 'unsafe_view'")
    mode = str(obs_copy).lower()
    if mode == "copy":
        return mode, True, False
    if mode == "safe_view":
        return mode, False, False
    if mode == "unsafe_view":
        return mode, False, True
    raise ValueError("obs_copy must be 'copy', 'safe_view', or 'unsafe_view'")


def _normalize_info_keys(info_keys: Any) -> list[str] | None:
    if info_keys is None:
        return None
    if isinstance(info_keys, str):
        raise ValueError("info_filter keys must be a sequence of strings, not a string")
    return [str(key) for key in info_keys]


def _normalize_info_filter(info_filter: Any) -> tuple[str, list[str] | None]:
    if info_filter is None:
        return "all", None
    if isinstance(info_filter, str):
        return _validate_info_mode(info_filter), None
    if not isinstance(info_filter, Mapping):
        raise ValueError("info_filter must be a mode string or a mapping with mode/keys")
    unknown = set(info_filter) - {"mode", "keys"}
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"unknown info_filter keys: {names}")
    return _validate_info_mode(str(info_filter.get("mode", "all"))), _normalize_info_keys(
        info_filter.get("keys", None),
    )


def _validate_info_mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized not in {"all", "terminal", "none"}:
        raise ValueError("info_filter mode must be 'all', 'terminal', or 'none'")
    return normalized


def _normalize_reward_clip(reward_clip: bool | Sequence[float]) -> tuple[bool, float, float]:
    if not reward_clip:
        return False, -1.0, 1.0
    if reward_clip is True:
        return True, -1.0, 1.0
    if not isinstance(reward_clip, Sequence) or isinstance(reward_clip, (str, bytes, bytearray)):
        raise ValueError("reward_clip must be False, True, or a (low, high) pair")
    if len(reward_clip) != 2:
        raise ValueError("reward_clip must be False, True, or a (low, high) pair")
    low, high = float(reward_clip[0]), float(reward_clip[1])
    if low > high:
        raise ValueError("reward_clip low must be <= high")
    return True, low, high


def _normalize_obs_layout(obs_layout: str) -> str:
    layout = str(obs_layout).lower()
    if layout not in {"chw", "hwc"}:
        raise ValueError("obs_layout must be 'chw' or 'hwc'")
    return layout


def _normalize_resize_algorithm(obs_resize_algorithm: str) -> str:
    algorithm = str(obs_resize_algorithm).lower()
    if algorithm not in {"area", "nearest", "bilinear"}:
        raise ValueError("obs_resize_algorithm must be one of: nearest, bilinear, area")
    return algorithm


def _normalize_obs_crop_mode(obs_crop_mode: str) -> Literal["remove", "mask"]:
    mode = str(obs_crop_mode).lower()
    if mode not in {"remove", "mask"}:
        raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
    return mode  # type: ignore[return-value]


def _normalize_obs_crop_fill(obs_crop_fill: int) -> int:
    try:
        fill = int(obs_crop_fill)
    except (TypeError, ValueError) as exc:
        raise ValueError("obs_crop_fill must be in [0, 255]") from exc
    if fill < 0 or fill > 255:
        raise ValueError("obs_crop_fill must be in [0, 255]")
    return fill


def _normalize_retro_crop(obs_crop: Sequence[int] | None) -> tuple[int, int, int, int]:
    if obs_crop is None:
        return 0, 0, 0, 0
    if len(obs_crop) != 4:
        raise ValueError("obs_crop must be a (top, bottom, left, right) tuple")
    top, bottom, left, right = (int(value) for value in obs_crop)
    if min(top, right, bottom, left) < 0:
        raise ValueError("obs_crop values must be non-negative")
    if top + bottom >= VISIBLE_HEIGHT or left + right >= VISIBLE_WIDTH:
        raise ValueError("obs_crop removes the whole visible frame")
    return top, bottom, left, right


def _normalize_retro_resize(
    obs_resize: Sequence[int] | None,
    source_width: int,
    source_height: int,
) -> tuple[int, int]:
    if obs_resize is None:
        return source_width, source_height
    if len(obs_resize) != 2:
        raise ValueError("obs_resize must be a (height, width) tuple")
    height, width = (int(value) for value in obs_resize)
    if width <= 0 or height <= 0:
        raise ValueError("obs_resize dimensions must be positive")
    return width, height


class SuperMarioBrosNesTurboVecEnv(VectorEnv):
    """Gymnasium VectorEnv for the native Super Mario Bros NES fast path."""

    metadata = {"render_modes": ["rgb_array"], "autoreset_mode": AutoresetMode.DISABLED}
    _BUTTON_COMBOS = [[0, 16, 32], [0, 64, 128], [0, 1, 256, 257]]

    def __init__(
        self,
        game: str,
        state: Any = State.DEFAULT,
        scenario: str | Path | None = None,
        info: str | Path | None = None,
        use_restricted_actions: Any = Actions.FILTERED,
        record: bool = False,
        players: int = 1,
        inttype: Any = Integrations.STABLE,
        obs_type: Any = Observations.IMAGE,
        render_mode: str = "human",
        *,
        num_envs: int = 1,
        num_threads: int | None = None,
        rom_path: str | Path | None = None,
        obs_copy: str = "copy",
        obs_resize: Sequence[int] | None = None,
        obs_crop: Sequence[int] | None = None,
        obs_crop_mode: Literal["remove", "mask"] = "remove",
        obs_crop_fill: int = 0,
        obs_grayscale: bool = False,
        obs_resize_algorithm: str = "nearest",
        obs_layout: str = "hwc",
        frame_skip: int = 1,
        frame_stack: int = 1,
        maxpool_last_two: bool = False,
        noop_reset_max: int = 0,
        sticky_action_prob: float = 0.0,
        reward_clip: bool | Sequence[float] = False,
        info_filter: Any = "all",
    ) -> None:
        if str(game) != DEFAULT_STABLE_RETRO_GAME:
            raise ValueError(f"SuperMarioBrosNesTurboVecEnv only supports {DEFAULT_STABLE_RETRO_GAME!r}")
        if players != 1:
            raise ValueError("SuperMarioBrosNesTurboVecEnv currently supports players=1")
        if record:
            raise ValueError("SuperMarioBrosNesTurboVecEnv does not support movie recording")
        obs_type_name = getattr(obs_type, "name", obs_type)
        if obs_type is not None and str(obs_type_name).split(".")[-1].upper() != "IMAGE":
            raise ValueError("SuperMarioBrosNesTurboVecEnv currently supports image observations only")
        if info not in (None, "data") and not str(info).endswith(".json"):
            raise ValueError("SuperMarioBrosNesTurboVecEnv only supports the SMB data info file")
        if scenario not in (None, "scenario") and not str(scenario).endswith(".json"):
            raise ValueError("SuperMarioBrosNesTurboVecEnv only supports the SMB scenario file")

        noop_reset_max = int(noop_reset_max)
        sticky_action_prob = float(sticky_action_prob)
        if noop_reset_max < 0:
            raise ValueError("noop_reset_max must be non-negative")
        if not 0.0 <= sticky_action_prob <= 1.0:
            raise ValueError("sticky_action_prob must be between 0.0 and 1.0")
        crop_top, crop_bottom, crop_left, crop_right = _normalize_retro_crop(obs_crop)
        normalized_crop_mode = _normalize_obs_crop_mode(obs_crop_mode)
        normalized_crop_fill = _normalize_obs_crop_fill(obs_crop_fill)
        has_crop = any((crop_top, crop_bottom, crop_left, crop_right))
        mask_crop = normalized_crop_mode == "mask" and has_crop
        source_width = VISIBLE_WIDTH if mask_crop else VISIBLE_WIDTH - crop_left - crop_right
        source_height = VISIBLE_HEIGHT if mask_crop else VISIBLE_HEIGHT - crop_top - crop_bottom
        resize_width, resize_height = _normalize_retro_resize(obs_resize, source_width, source_height)
        action_mode = _normalize_action_mode(use_restricted_actions)
        self.autoreset_mode = AutoresetMode.DISABLED
        self.game = str(game)
        self.action_meanings = ACTION_SETS["full"]
        self.action_set = "full"
        self._core_action_ids = _core_action_ids(self.action_meanings)
        self._action_masks = _action_masks(self.action_meanings)
        state = _normalize_retro_state(state)
        self._state_collection = isinstance(state, Mapping) or (
            isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray, memoryview))
        )
        initial_states, initial_state_names, initial_state_weights = _normalize_initial_state_config(
            state,
            None,
            num_envs,
        )
        self.obs_layout = _normalize_obs_layout(obs_layout)
        self.obs_copy, self._copy_obs, self._unsafe_view = _normalize_obs_copy(obs_copy)
        self._info_mode, self._info_keys = _normalize_info_filter(info_filter)
        self.reward_clip, self.reward_clip_low, self.reward_clip_high = _normalize_reward_clip(reward_clip)
        self.obs_resize_algorithm = _normalize_resize_algorithm(obs_resize_algorithm)
        self.obs_crop_mode = normalized_crop_mode
        self.obs_crop_fill = normalized_crop_fill
        self.crop_left = crop_left
        self.crop_right = crop_right
        self._output_resize_width = int(resize_width)
        self._output_resize_height = int(resize_height)
        if self._output_resize_width <= 0 or self._output_resize_height <= 0:
            raise ValueError("resize_width and resize_height must be > 0")
        self._needs_python_postprocess = False
        self._core = _CoreRetroVecEnv(
            _expand_rom_path(_resolve_rom_path(str(game), rom_path)),
            num_envs,
            frame_skip,
            bool(obs_grayscale),
            frame_stack,
            False,
            crop_top,
            crop_bottom,
            self._output_resize_width,
            self._output_resize_height,
            initial_states,
            list(initial_state_names),
            initial_state_weights,
            0,
            bool(maxpool_last_two),
            noop_reset_max,
            sticky_action_prob,
            crop_left,
            crop_right,
            normalized_crop_mode,
            normalized_crop_fill,
            self.obs_resize_algorithm,
        )
        self._state_policy_names = tuple(self._core.initial_state_policy_names())
        self.initial_state_names = self._state_policy_names
        self.num_envs = self._core.num_envs
        self.num_threads = self.num_envs if num_threads is None else int(num_threads)
        self.num_buttons = len(NES_BUTTONS)
        self._mask_to_core_action_ids = self._build_mask_to_core_action_ids()
        self.button_combos = [list(combo) for combo in self._BUTTON_COMBOS]
        self.use_restricted_actions = use_restricted_actions
        self._action_mode = action_mode
        self.frame_skip = self._core.frame_skip
        self.obs_grayscale = self._core.grayscale
        self.frame_stack = self._core.frame_stack
        self.maxpool_last_two = self._core.frame_maxpool
        self.noop_reset_max = self._core.noop_reset_max
        self.sticky_action_prob = self._core.sticky_action_prob
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.resize_width = self._output_resize_width
        self.resize_height = self._output_resize_height
        self.single_action_space = (
            spaces.Discrete(36)
            if action_mode == "DISCRETE"
            else spaces.MultiBinary(self.num_buttons)
        )
        self.action_space = (
            spaces.MultiDiscrete([36] * self.num_envs)
            if action_mode == "DISCRETE"
            else spaces.MultiBinary((self.num_envs, self.num_buttons))
        )
        self._public_channels = self._core.obs_shape()[1]
        if self.obs_layout == "chw":
            self._single_obs_shape = (
                self._public_channels,
                self._output_resize_height,
                self._output_resize_width,
            )
        else:
            self._single_obs_shape = (
                self._output_resize_height,
                self._output_resize_width,
                self._public_channels,
            )
        self.single_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self._single_obs_shape,
            dtype=np.uint8,
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.render_mode = render_mode
        self.viewer = None
        self.closed = False
        self._pending_seed: int | Sequence[int | None] | None = None
        self._pending_options: dict[str, Any] | list[dict[str, Any]] | None = None

        self._actions = np.zeros((self.num_envs,), dtype=np.uint8)
        self._last_action_masks: np.ndarray | None = None
        self._last_action_ids = np.empty((self.num_envs,), dtype=np.uint8)
        self._obs = np.empty(self._core.obs_shape(), dtype=np.uint8)
        self._unsafe_public_obs: np.ndarray | None = None
        self._safe_public_obs = [
            np.empty((self.num_envs, *self._single_obs_shape), dtype=np.uint8),
            np.empty((self.num_envs, *self._single_obs_shape), dtype=np.uint8),
        ]
        self._safe_public_obs_index = 0
        self._rewards = np.empty((self.num_envs,), dtype=np.float32)
        self._reward_return = np.empty((self.num_envs,), dtype=np.float32)
        self._terminated = np.empty((self.num_envs,), dtype=np.bool_)
        self._truncated = np.empty((self.num_envs,), dtype=np.bool_)
        self._x_pos = np.empty((self.num_envs,), dtype=np.uint16)
        self._coins = np.empty((self.num_envs,), dtype=np.uint8)
        self._level_hi = np.empty((self.num_envs,), dtype=np.int16)
        self._level_lo = np.empty((self.num_envs,), dtype=np.int16)
        self._lives = np.empty((self.num_envs,), dtype=np.int16)
        self._score = np.empty((self.num_envs,), dtype=np.uint32)
        self._scrolling = np.empty((self.num_envs,), dtype=np.int16)
        self._time = np.empty((self.num_envs,), dtype=np.uint16)
        self._xscroll_hi = np.empty((self.num_envs,), dtype=np.uint8)
        self._xscroll_lo = np.empty((self.num_envs,), dtype=np.uint8)
        self._active_state_indices = np.empty((self.num_envs,), dtype=np.int32)
        self._active_state_labels: tuple[str | None, ...] = tuple(None for _ in range(self.num_envs))
        self._info_all_lanes_mask = np.ones((self.num_envs,), dtype=np.bool_)
        self._rgb_frames: np.ndarray | None = (
            np.empty(self._core.rgb_frame_shape(), dtype=np.uint8)
            if render_mode == "rgb_array"
            else None
        )
        self._write_active_state_indices()

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if options is None and isinstance(self._pending_options, dict):
            reset_options = dict(self._pending_options)
        else:
            reset_options = {} if options is None else dict(options)
        reset_mask = reset_options.pop("reset_mask", None)
        if reset_mask is None:
            reset_mask = np.ones(self.num_envs, dtype=np.bool_)
        elif not isinstance(reset_mask, np.ndarray):
            raise TypeError("options['reset_mask'] must be a NumPy array")
        elif reset_mask.shape != (self.num_envs,):
            raise ValueError(f"options['reset_mask'] must have shape {(self.num_envs,)}")
        elif reset_mask.dtype != np.bool_:
            raise TypeError("options['reset_mask'] must have dtype np.bool_")
        elif not np.any(reset_mask):
            raise ValueError("options['reset_mask'] must select at least one lane")

        start_indices = reset_options.pop("start_indices", None)
        if reset_options:
            names = ", ".join(sorted(reset_options))
            raise ValueError(f"unsupported reset option(s): {names}")
        if start_indices is None:
            start_indices = np.full(self.num_envs, -1, dtype=np.int32)
        elif not isinstance(start_indices, np.ndarray):
            raise TypeError("options['start_indices'] must be a NumPy array")
        elif start_indices.shape != (self.num_envs,):
            raise ValueError(f"options['start_indices'] must have shape {(self.num_envs,)}")
        elif start_indices.dtype != np.int32:
            raise TypeError("options['start_indices'] must have dtype np.int32")

        seeds = self._normalize_reset_seed(seed)
        if seed is not None and not isinstance(seed, Sequence):
            super().reset(seed=int(seed))
        self._core.reset_masked_into(self._obs, reset_mask, start_indices, seeds)
        self._rewards[reset_mask] = 0
        self._terminated[reset_mask] = False
        self._truncated[reset_mask] = False
        self._write_active_state_indices()
        self._write_info_arrays()
        lane_infos: list[dict[str, Any]] = []
        for index in range(self.num_envs):
            if not bool(reset_mask[index]):
                lane_infos.append({})
                continue
            info = self._reset_info_dict(index)
            if self._info_mode != "none":
                raw_info = self._base_info_dict(index)
                if self._info_keys is not None:
                    raw_info = {
                        key: raw_info[key]
                        for key in self._info_keys
                        if key in raw_info
                    }
                raw_info.update(info)
                info = raw_info
            lane_infos.append(info)
        infos = self._vector_infos(lane_infos)
        self._pending_seed = None
        self._pending_options = None
        return self._return_obs(), infos

    def _normalize_reset_seed(
        self,
        seed: int | Sequence[int | None] | None,
    ) -> list[int | None]:
        if seed is None:
            seed = self._pending_seed
        if seed is None:
            return [None for _ in range(self.num_envs)]
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray)):
            seeds = [None if value is None else int(value) for value in seed]
            if len(seeds) != self.num_envs:
                raise ValueError("seed sequence length must match num_envs")
            return seeds
        base = int(seed)
        return [base + index for index in range(self.num_envs)]

    def seed(
        self,
        seed: int | Sequence[int | None] | None = None,
    ) -> list[int | None]:
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        seeds = self._normalize_reset_seed(seed)
        self._pending_seed = list(seeds)
        return seeds

    def set_options(self, options: dict[str, Any] | list[dict[str, Any]] | None = None) -> None:
        if options is None:
            options = {}
        self._pending_options = deepcopy(options)

    def enable_profiler(self) -> None:
        self._core.enable_profiler()

    def reset_profiler(self) -> None:
        self._core.reset_profiler()

    def disable_profiler(self) -> None:
        self._core.disable_profiler()

    def profiler_snapshot(self, top_n: int = 64) -> dict[str, Any]:
        return json.loads(self._core.profiler_snapshot(int(top_n)))

    def step_async(self, actions: np.ndarray) -> None:
        np.copyto(self._actions, self._actions_to_core_ids(actions))

    def _actions_to_core_ids(self, actions: Any) -> np.ndarray:
        if self._action_mode == "DISCRETE":
            masks = self._discrete_actions_to_masks(actions)
        else:
            masks = np.asarray(actions, dtype=np.uint8)
            if masks.shape == (self.num_buttons,):
                masks = masks.reshape(1, self.num_buttons)
            if masks.shape != (self.num_envs, self.num_buttons):
                raise ValueError(
                    f"actions must have shape {(self.num_envs, self.num_buttons)}, got {masks.shape}",
                )
        if self._last_action_masks is not None and np.array_equal(masks, self._last_action_masks):
            return self._last_action_ids
        self._last_action_ids[:] = self._mask_to_core_action_ids[
            self._button_mask_indices(masks)
        ]
        self._last_action_masks = masks.copy()
        return self._last_action_ids

    @staticmethod
    def _button_mask_indices(masks: np.ndarray) -> np.ndarray:
        return (masks.astype(np.uint16, copy=False) @ MASK_BIT_WEIGHTS).astype(np.uint16, copy=False)

    def _build_mask_to_core_action_ids(self) -> np.ndarray:
        lookup = np.empty(1 << self.num_buttons, dtype=np.uint8)
        for mask_index in range(lookup.size):
            mask = ((mask_index & MASK_BIT_WEIGHTS) != 0).astype(np.uint8)
            lookup[mask_index] = self._mask_to_core_action(mask)
        return lookup

    def _discrete_actions_to_masks(self, actions: Any) -> np.ndarray:
        values = np.asarray(actions, dtype=np.int64).reshape(-1)
        if values.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape {(self.num_envs,)}, got {values.shape}")
        masks = np.zeros((self.num_envs, self.num_buttons), dtype=np.uint8)
        for env_idx, raw_value in enumerate(values):
            value = int(raw_value)
            if value < 0 or value >= 36:
                raise ValueError("DISCRETE actions must be in [0, 35]")
            action_bits = 0
            for combo in self._BUTTON_COMBOS:
                current = value % len(combo)
                value //= len(combo)
                action_bits |= combo[current]
            for button_idx in range(self.num_buttons):
                masks[env_idx, button_idx] = (action_bits >> button_idx) & 1
        return masks

    def _mask_to_core_action(self, mask: np.ndarray) -> int:
        pressed = {button for button, index in BUTTON_TO_INDEX.items() if int(mask[index])}
        if "START" in pressed and not (pressed & {"LEFT", "RIGHT", "A", "B"}):
            return CORE_ACTION_MEANINGS.index("start")
        if "RIGHT" in pressed:
            if "A" in pressed and "B" in pressed:
                return CORE_ACTION_MEANINGS.index("right_a_b")
            if "A" in pressed:
                return CORE_ACTION_MEANINGS.index("right_a")
            if "B" in pressed:
                return CORE_ACTION_MEANINGS.index("right_b")
            return CORE_ACTION_MEANINGS.index("right")
        if "LEFT" in pressed:
            return CORE_ACTION_MEANINGS.index("left")
        if "A" in pressed:
            return CORE_ACTION_MEANINGS.index("a")
        return CORE_ACTION_MEANINGS.index("noop")

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self.step_async(actions)
        return self.step_wait_gymnasium()

    def step_wait_gymnasium(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self._core.step_into(
            self._actions,
            self._obs,
            self._rewards,
            self._terminated,
            self._truncated,
            self._x_pos,
            self._coins,
            self._level_hi,
            self._level_lo,
            self._lives,
            self._score,
            self._scrolling,
            self._time,
            self._xscroll_hi,
            self._xscroll_lo,
        )
        has_terminal = bool(np.any(self._terminated) or np.any(self._truncated))
        obs, rewards, terminated, truncated = (
            self._return_obs(),
            self._return_rewards(),
            self._terminated,
            self._truncated,
        )
        infos = self._step_infos(has_terminal)
        return obs, rewards, terminated, truncated, infos

    def step_gymnasium(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        return self.step(actions)

    def _write_info_arrays(self) -> None:
        self._core.info_into(
            self._x_pos,
            self._coins,
            self._level_hi,
            self._level_lo,
            self._lives,
            self._score,
            self._scrolling,
            self._time,
            self._xscroll_hi,
            self._xscroll_lo,
        )

    def _write_active_state_indices(self) -> None:
        raw_indices = np.asarray(self._core.active_state_indices(), dtype=np.int32)
        core_names = tuple(self._core.initial_state_names)
        policy_indices = {name: index for index, name in enumerate(self._state_policy_names)}
        active_labels: list[str | None] = []
        public_indices = raw_indices.copy()
        for lane, raw_index in enumerate(raw_indices):
            if int(raw_index) < 0:
                active_labels.append(None)
                public_indices[lane] = -1
                continue
            label = core_names[int(raw_index)]
            active_labels.append(label)
            if label in policy_indices:
                public_indices[lane] = policy_indices[label]
        self._active_state_labels = tuple(active_labels)
        self._active_state_indices[:] = public_indices

    def _return_obs(self) -> np.ndarray:
        public = self._public_obs_view()
        if self.obs_copy == "copy":
            return public.copy()
        if self.obs_copy == "safe_view":
            self._safe_public_obs_index = 1 - self._safe_public_obs_index
            out = self._safe_public_obs[self._safe_public_obs_index]
            np.copyto(out, public)
            return out
        return public

    def _public_obs_view(self) -> np.ndarray:
        chw = self._obs
        if self.obs_layout == "chw":
            return chw
        return np.transpose(chw, (0, 2, 3, 1))

    def _public_single_obs(self, obs: np.ndarray) -> np.ndarray:
        batch = obs.reshape((1, *obs.shape))
        original = self._obs
        try:
            self._obs = batch
            return self._public_obs_view()[0].copy()
        finally:
            self._obs = original

    def _return_rewards(self) -> np.ndarray:
        if not self.reward_clip:
            return self._rewards
        np.clip(self._rewards, self.reward_clip_low, self.reward_clip_high, out=self._reward_return)
        return self._reward_return

    def _step_infos(self, has_terminal: bool) -> dict[str, Any]:
        if self._info_mode == "none":
            return {}
        if self._info_mode == "terminal" and not has_terminal:
            return {}
        if has_terminal:
            return self._vector_infos([self._info_dict(index) for index in range(self.num_envs)])
        return self._base_vector_infos()

    def _info_dict(self, index: int) -> dict[str, Any]:
        terminal = bool(self._terminated[index]) or bool(self._truncated[index])
        if self._info_mode == "none":
            return {}
        if self._info_mode == "terminal" and not terminal:
            return {}

        info = self._base_info_dict(index)
        if self._info_keys is not None:
            info = {key: info[key] for key in self._info_keys if key in info}
        if terminal:
            if bool(self._terminated[index]):
                info["terminated"] = True
            if bool(self._truncated[index]):
                info["truncated"] = True
        return info

    def _vector_infos(self, lane_infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
        infos: dict[str, Any] = {}
        for index, lane_info in enumerate(lane_infos):
            infos = self._add_info(infos, lane_info, index)
        return infos

    def _base_vector_infos(self) -> dict[str, Any]:
        infos: dict[str, Any] = {}
        for key, attr_name in _BASE_INFO_ARRAYS:
            if self._info_keys is not None and key not in self._info_keys:
                continue
            infos[key] = getattr(self, attr_name).astype(np.int_, copy=True)
            infos[f"_{key}"] = self._info_all_lanes_mask.copy()
        return infos

    def _base_info_dict(self, index: int) -> dict[str, Any]:
        return {
            "x_pos": int(self._x_pos[index]),
            "coins": int(self._coins[index]),
            "levelHi": int(self._level_hi[index]),
            "levelLo": int(self._level_lo[index]),
            "lives": int(self._lives[index]),
            "score": int(self._score[index]),
            "scrolling": int(self._scrolling[index]),
            "time": int(self._time[index]),
            "xscrollHi": int(self._xscroll_hi[index]),
            "xscrollLo": int(self._xscroll_lo[index]),
        }

    def _reset_info_dict(self, index: int) -> dict[str, Any]:
        if not self._state_collection:
            return {}
        state = self._active_state_labels[index]
        if state is None:
            return {}
        return {"state": state, "start_state": state}

    @property
    def x_pos(self) -> np.ndarray:
        return self._x_pos

    @property
    def coins(self) -> np.ndarray:
        return self._coins

    @property
    def level_hi(self) -> np.ndarray:
        return self._level_hi

    @property
    def level_lo(self) -> np.ndarray:
        return self._level_lo

    @property
    def lives(self) -> np.ndarray:
        return self._lives

    @property
    def score(self) -> np.ndarray:
        return self._score

    @property
    def scrolling(self) -> np.ndarray:
        return self._scrolling

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def xscroll_hi(self) -> np.ndarray:
        return self._xscroll_hi

    @property
    def xscroll_lo(self) -> np.ndarray:
        return self._xscroll_lo

    def active_state_indices(self) -> np.ndarray:
        view = self._active_state_indices.view()
        view.setflags(write=False)
        return view

    def active_states(self) -> tuple[str | None, ...]:
        return self._active_state_labels

    def close(self) -> None:
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            return [None for _ in range(self.num_envs)]
        if self._rgb_frames is None:
            self._rgb_frames = np.empty(self._core.rgb_frame_shape(), dtype=np.uint8)
        self._core.rgb_frames_into(self._rgb_frames)
        return [self._rgb_frames[index].copy() for index in range(self.num_envs)]

    def render(self) -> np.ndarray | None:
        frames = [frame for frame in self.get_images() if frame is not None]
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return np.concatenate(frames, axis=0)

def _normalize_state_value(state: Any) -> Any:
    name = getattr(state, "name", None)
    if name == "DEFAULT":
        return "Level1-1"
    if name == "NONE":
        return None
    return state


def _normalize_retro_state(state: Any) -> Any:
    if isinstance(state, Mapping):
        return {_normalize_state_value(key): value for key, value in state.items()}
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray, memoryview)):
        return [_normalize_state_value(value) for value in state]
    return _normalize_state_value(state)


def _resolve_rom_path(game: str, rom_path: str | Path | None) -> str | Path:
    return resolve_required_rom_path(rom_path, game)
