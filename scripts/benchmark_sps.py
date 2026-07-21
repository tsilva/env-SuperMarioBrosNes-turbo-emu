from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Sequence

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
import numpy as np

try:
    from benchmark_rom import validate_rom_hash
    from benchmark_workload import (
        CANONICAL_ACTION_NAMES,
        CANONICAL_ACTION_SEED,
        CANONICAL_CROP_BOTTOM,
        CANONICAL_CROP_TOP,
        CANONICAL_FRAME_SKIP,
        CANONICAL_FRAME_STACK,
        CANONICAL_NUM_ENVS,
        CANONICAL_OBS_CROP_MODE,
        CANONICAL_RESIZE_HEIGHT,
        CANONICAL_RESIZE_WIDTH,
        CANONICAL_STATE_NAMES,
    )
except ModuleNotFoundError:
    from scripts.benchmark_rom import validate_rom_hash
    from scripts.benchmark_workload import (
        CANONICAL_ACTION_NAMES,
        CANONICAL_ACTION_SEED,
        CANONICAL_CROP_BOTTOM,
        CANONICAL_CROP_TOP,
        CANONICAL_FRAME_SKIP,
        CANONICAL_FRAME_STACK,
        CANONICAL_NUM_ENVS,
        CANONICAL_OBS_CROP_MODE,
        CANONICAL_RESIZE_HEIGHT,
        CANONICAL_RESIZE_WIDTH,
        CANONICAL_STATE_NAMES,
    )


GAME = "SuperMarioBros-Nes-v0"
DEFAULT_ROM = None
DEFAULT_STATES = CANONICAL_STATE_NAMES
DEFAULT_BENCHMARK_ACTIONS = CANONICAL_ACTION_NAMES
DEFAULT_ACTION_SEED = CANONICAL_ACTION_SEED
DEFAULT_OBS_CROP_MODE = CANONICAL_OBS_CROP_MODE
DEFAULT_MIN_START_LOAD_LIMIT = 4.0
DEFAULT_START_LOAD_CPU_FRACTION = 0.5
TURBO_PACKAGE = "supermariobrosnes-turbo"
TURBO_IMPORT = "supermariobrosnes_turbo"
STABLE_RETRO_PACKAGE = "stable-retro-turbo"
STABLE_RETRO_IMPORT = "stable_retro"
CORE_ACTION_MEANINGS = (
    "noop",
    "right",
    "right_b",
    "right_a",
    "right_a_b",
    "a",
    "left",
    "start",
)
ACTION_SETS = {
    "basic": ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left"),
    "right-jump": ("right", "right_b", "right_a", "right_a_b"),
    "basic-start": CORE_ACTION_MEANINGS,
}
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


@dataclass(frozen=True)
class PreprocessingConfig:
    frame_skip: int
    frame_stack: int
    grayscale: bool
    crop_top: int
    crop_bottom: int
    crop_mode: str
    resize_width: int
    resize_height: int
    maxpool_last_two: bool = False


@dataclass(frozen=True)
class StableRetroWorkerConfig:
    integration_path: str
    state: str | None
    preprocessing: PreprocessingConfig
    game: str = GAME


class StableRetroPreprocessingEnv(gym.Wrapper):
    """Apply the benchmark preprocessing and frame skip inside one scalar worker."""

    def __init__(self, env: gym.Env, config: PreprocessingConfig) -> None:
        super().__init__(env)
        self.config = config
        channels = 1 if config.grayscale else 3
        self._channels = channels
        self._previous_raw_frame: np.ndarray | None = None
        self._stack = np.empty(
            (channels * config.frame_stack, config.resize_height, config.resize_width),
            dtype=np.uint8,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self._stack.shape,
            dtype=np.uint8,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._previous_raw_frame = np.asarray(observation, dtype=np.uint8).copy()
        frame = preprocess_frame(observation, self.config)
        for offset in range(0, self._stack.shape[0], self._channels):
            self._stack[offset : offset + self._channels] = frame
        return self._stack.copy(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        observation: np.ndarray | None = None
        penultimate: np.ndarray | None = None
        for _ in range(self.config.frame_skip):
            if observation is not None:
                penultimate = observation
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        if observation is None:
            raise RuntimeError("Stable Retro produced no observation during frame skip")
        raw_frame = np.asarray(observation, dtype=np.uint8)
        if self.config.maxpool_last_two:
            previous = penultimate if penultimate is not None else self._previous_raw_frame
            if previous is not None:
                raw_frame = np.maximum(np.asarray(previous, dtype=np.uint8), raw_frame)
        self._previous_raw_frame = np.asarray(observation, dtype=np.uint8).copy()
        frame = preprocess_frame(raw_frame, self.config)
        if self._stack.shape[0] > self._channels:
            self._stack[: -self._channels] = self._stack[self._channels :]
        self._stack[-self._channels :] = frame
        return self._stack.copy(), total_reward, terminated, truncated, info


class StableRetroVectorEnv:
    """Own a vector env and its temporary ROM integration overlay."""

    def __init__(
        self,
        env: AsyncVectorEnv | SyncVectorEnv,
        overlay: tempfile.TemporaryDirectory[str],
        active_states: Sequence[str | None],
        buttons: Sequence[str | None],
    ) -> None:
        self.env = env
        self.overlay = overlay
        self.num_envs = env.num_envs
        self._active_states = tuple(active_states)
        self.buttons = tuple(buttons)
        self._closed = False
        self._pending_seed: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if self._pending_seed is not None and "seed" not in kwargs:
            kwargs["seed"] = self._pending_seed
            self._pending_seed = None
        return self.env.reset(*args, **kwargs)

    def step(self, actions: np.ndarray) -> Any:
        return self.env.step(actions)

    def active_states(self) -> tuple[str | None, ...]:
        return self._active_states

    def seed(self, seed: int | None = None) -> None:
        self._pending_seed = seed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.env.close()
        finally:
            self.overlay.cleanup()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark raw Super Mario Bros vector-environment steps per second."
    )
    parser.add_argument(
        "--stable-retro-baseline",
        action="store_true",
        help="Benchmark stable-retro-turbo RetroEnv workers instead of this repository.",
    )
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=DEFAULT_ROM,
        help="Path to the SMB NES ROM. Defaults to Stable Retro-compatible discovery.",
    )
    parser.add_argument("--num-envs", type=int, default=CANONICAL_NUM_ENVS)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--frame-skip", type=int, default=CANONICAL_FRAME_SKIP)
    parser.add_argument("--frame-stack", type=int, default=CANONICAL_FRAME_STACK)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--crop-top", type=int, default=CANONICAL_CROP_TOP)
    parser.add_argument("--crop-bottom", type=int, default=CANONICAL_CROP_BOTTOM)
    parser.add_argument("--obs-crop-mode", choices=("remove", "mask"), default=DEFAULT_OBS_CROP_MODE)
    parser.add_argument("--resize-width", type=int, default=CANONICAL_RESIZE_WIDTH)
    parser.add_argument("--resize-height", type=int, default=CANONICAL_RESIZE_HEIGHT)
    parser.add_argument("--action-set", choices=sorted(ACTION_SETS), default="basic")
    parser.add_argument(
        "--actions",
        default=None,
        help=(
            "Comma-separated actions sampled per vector step. Defaults to "
            f"{','.join(DEFAULT_BENCHMARK_ACTIONS)}."
        ),
    )
    parser.add_argument(
        "--action",
        choices=CORE_ACTION_MEANINGS,
        default=None,
        help="Legacy single-action override. If set, --actions is ignored.",
    )
    parser.add_argument("--action-seed", type=int, default=DEFAULT_ACTION_SEED)
    parser.add_argument("--state", default=None)
    parser.add_argument(
        "--states",
        default=None,
        help=(
            "Comma-separated stable-retro states assigned round-robin across lanes, "
            f"default: {','.join(DEFAULT_STATES)} unless --state is provided."
        ),
    )
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--include-info", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--terminate-on-flag", action="store_true")
    parser.add_argument("--no-start-game", action="store_true")
    parser.add_argument("--pre-start-steps", type=int, default=30)
    parser.add_argument("--start-steps", type=int, default=8)
    parser.add_argument("--post-start-steps", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--max-start-load",
        type=float,
        default=None,
        help=(
            "Fail before benchmarking if 1-minute load exceeds this value. "
            "Default: max(4.0, logical_cpus * 0.5)."
        ),
    )
    parser.add_argument("--skip-load-preflight", action="store_true")
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="Enable candidate-only Rust hot-path profiling and write benchmark+profile JSON.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for field in (
        "num_envs",
        "steps",
        "repeats",
        "frame_skip",
        "frame_stack",
        "resize_width",
        "resize_height",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    for field in (
        "warmup",
        "crop_top",
        "crop_bottom",
        "pre_start_steps",
        "start_steps",
        "post_start_steps",
    ):
        if getattr(args, field) < 0:
            raise ValueError(f"--{field.replace('_', '-')} must be non-negative")
    if args.crop_top + args.crop_bottom >= 224:
        raise ValueError("--crop-top plus --crop-bottom must be less than 224")
    if args.max_start_load is not None and args.max_start_load <= 0:
        raise ValueError("--max-start-load must be positive")
    for action in selected_actions_for_args(args):
        if action not in ACTION_SETS[args.action_set]:
            raise ValueError(
                f"action {action!r} is not in action_set={args.action_set!r}; "
                f"valid actions: {', '.join(ACTION_SETS[args.action_set])}"
            )
    if args.state is not None and args.states is not None:
        raise ValueError("--state and --states are mutually exclusive")
    if args.stable_retro_baseline and args.profile_output is not None:
        raise ValueError("--profile-output is only available for the turbo backend")
    if args.terminate_on_flag:
        raise ValueError("--terminate-on-flag is disabled for matched provider-native benchmarks")


def parse_states(states: str | None) -> tuple[str, ...] | None:
    if states is None:
        return None
    parsed = tuple(state.strip() for state in states.split(","))
    if not parsed or not all(parsed):
        raise ValueError("--states must be a comma-separated list without empty entries")
    return parsed


def parse_actions(actions: str | None) -> tuple[str, ...]:
    if actions is None:
        return DEFAULT_BENCHMARK_ACTIONS
    parsed = tuple(action.strip() for action in actions.split(","))
    if not parsed or not all(parsed):
        raise ValueError("--actions must be a comma-separated list without empty entries")
    return parsed


def selected_actions_for_args(args: argparse.Namespace) -> tuple[str, ...]:
    return (args.action,) if args.action is not None else parse_actions(args.actions)


def initial_states_for_args(args: argparse.Namespace) -> tuple[str, ...] | None:
    if args.state is not None:
        return None
    return parse_states(args.states) if args.states is not None else DEFAULT_STATES


def lane_states(num_envs: int, states: tuple[str, ...] | None) -> list[str] | None:
    if states is None:
        return None
    return [states[index % len(states)] for index in range(num_envs)]


def assigned_lane_states(args: argparse.Namespace) -> list[str | None]:
    if args.state is not None:
        return [args.state] * args.num_envs
    states = lane_states(args.num_envs, args.parsed_states)
    return states if states is not None else [None] * args.num_envs


def has_initial_state(args: argparse.Namespace) -> bool:
    return args.state is not None or args.parsed_states is not None


def backend_for_args(args: argparse.Namespace) -> str:
    return "stable-retro" if args.stable_retro_baseline else "turbo"


def preprocessing_for_args(args: argparse.Namespace) -> PreprocessingConfig:
    return PreprocessingConfig(
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        grayscale=not args.rgb,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        crop_mode=args.obs_crop_mode,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
    )


def default_max_start_load() -> float:
    return max(DEFAULT_MIN_START_LOAD_LIMIT, (os.cpu_count() or 1) * DEFAULT_START_LOAD_CPU_FRACTION)


def load_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_load_preflight:
        return {"enabled": False, "start_1min": None, "max_start_load": None, "load_ok": True}
    max_start_load = args.max_start_load or default_max_start_load()
    try:
        start_1min = os.getloadavg()[0]
    except (AttributeError, OSError) as exc:
        return {
            "enabled": True,
            "start_1min": None,
            "max_start_load": max_start_load,
            "load_ok": True,
            "unavailable_reason": str(exc),
        }
    load_ok = start_1min < max_start_load
    result = {
        "enabled": True,
        "start_1min": start_1min,
        "max_start_load": max_start_load,
        "load_ok": load_ok,
    }
    if not load_ok:
        raise SystemExit(
            f"Refusing to benchmark: 1-minute load {start_1min:.2f} meets or exceeds "
            f"--max-start-load {max_start_load:.2f}. Use --skip-load-preflight to override."
        )
    return result


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def package_metadata(backend: str = "turbo") -> dict[str, str | None]:
    name, import_name = (
        (STABLE_RETRO_PACKAGE, STABLE_RETRO_IMPORT)
        if backend == "stable-retro"
        else (TURBO_PACKAGE, TURBO_IMPORT)
    )
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"name": name, "version": version, "import": import_name}


def resolve_verified_rom_path(path: str | Path | None = None) -> Path:
    from supermariobrosnes_turbo import resolve_required_rom_path

    try:
        resolved = resolve_required_rom_path(path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not resolved.exists():
        raise SystemExit(f"ROM path does not exist: {resolved}")
    if not resolved.is_file():
        raise SystemExit(f"ROM path is not a file: {resolved}")
    validate_rom_hash(resolved)
    return resolved.resolve()


def rayon_num_threads() -> int | str | None:
    raw = os.environ.get("RAYON_NUM_THREADS")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def stable_retro_module() -> Any:
    if sys.version_info[:2] != (3, 14):
        raise SystemExit("Stable Retro baseline mode requires Python 3.14")
    try:
        return importlib.import_module(STABLE_RETRO_IMPORT)
    except ImportError as exc:
        raise SystemExit(
            "Stable Retro baseline mode requires the optional development dependency "
            "stable-retro-turbo==1.0.1.post33; run `uv sync --extra dev`."
        ) from exc


def stable_retro_buttons() -> tuple[str | None, ...]:
    stable_retro = stable_retro_module()
    return tuple(stable_retro.get_system_info("Nes")["buttons"])


def named_action_mask(action_name: str, buttons: Sequence[str | None]) -> np.ndarray:
    try:
        pressed = ACTION_BUTTONS[action_name]
    except KeyError as exc:
        raise ValueError(f"unknown action {action_name!r}") from exc
    index_by_button = {button: index for index, button in enumerate(buttons) if button is not None}
    missing = [button for button in pressed if button not in index_by_button]
    if missing:
        raise ValueError(f"backend button ordering does not contain {missing!r}")
    mask = np.zeros(len(buttons), dtype=np.uint8)
    for button in pressed:
        mask[index_by_button[button]] = 1
    return mask


def fill_action(
    backend: str,
    num_envs: int,
    action_name: str,
    buttons: Sequence[str | None] | None = None,
) -> np.ndarray:
    if backend == "turbo":
        candidate = importlib.import_module(TURBO_IMPORT)
        return candidate.action_batch(action_name, num_envs)
    if buttons is None:
        buttons = stable_retro_buttons()
    return np.repeat(named_action_mask(action_name, buttons)[None, :], num_envs, axis=0)


def action_templates(
    backend: str,
    num_envs: int,
    action_names: Sequence[str],
    buttons: Sequence[str | None] | None = None,
) -> tuple[np.ndarray, ...]:
    return tuple(fill_action(backend, num_envs, name, buttons) for name in action_names)


def sampled_action_sequence(
    templates: Sequence[np.ndarray], count: int, seed: int
) -> tuple[np.ndarray, ...]:
    if count <= 0:
        return ()
    if len(templates) == 1:
        return tuple(templates[0] for _ in range(count))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(templates), size=count)
    return tuple(templates[int(index)] for index in indices)


def integer_area_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize HW or HWC uint8 images with the candidate's integer area bins."""
    if image.ndim not in (2, 3):
        raise ValueError(f"expected HW or HWC image, got shape {image.shape}")
    source_height, source_width = image.shape[:2]
    y0 = np.arange(height, dtype=np.int64) * source_height // height
    y1 = np.maximum(
        np.arange(1, height + 1, dtype=np.int64) * source_height // height,
        y0 + 1,
    ).clip(max=source_height)
    x0 = np.arange(width, dtype=np.int64) * source_width // width
    x1 = np.maximum(
        np.arange(1, width + 1, dtype=np.int64) * source_width // width,
        x0 + 1,
    ).clip(max=source_width)
    integral = np.asarray(image, dtype=np.uint64).cumsum(axis=0).cumsum(axis=1)
    pad_width = ((1, 0), (1, 0)) + (((0, 0),) if image.ndim == 3 else ())
    integral = np.pad(integral, pad_width, mode="constant")
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    counts = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    if image.ndim == 3:
        counts = counts[:, :, None]
    return (sums // counts).astype(np.uint8)


def preprocess_frame(observation: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    frame = np.asarray(observation, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an HWC RGB observation, got shape {frame.shape}")
    if config.crop_mode == "mask":
        if config.crop_top or config.crop_bottom:
            frame = frame.copy()
            if config.crop_top:
                frame[: config.crop_top] = 0
            if config.crop_bottom:
                frame[-config.crop_bottom :] = 0
    else:
        end = frame.shape[0] - config.crop_bottom if config.crop_bottom else frame.shape[0]
        frame = frame[config.crop_top : end]
    if config.grayscale:
        rgb = frame.astype(np.uint32)
        frame = ((77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2] + 128) >> 8).astype(
            np.uint8
        )
    resized = integer_area_resize(frame, config.resize_height, config.resize_width)
    if resized.ndim == 2:
        return resized[None, :, :]
    return np.moveaxis(resized, -1, 0)


def make_stable_retro_worker(config: StableRetroWorkerConfig) -> gym.Env:
    """Top-level, picklable worker factory used by AsyncVectorEnv spawn."""
    stable_retro = importlib.import_module(STABLE_RETRO_IMPORT)
    stable_retro.data.add_custom_integration(config.integration_path)
    kwargs: dict[str, Any] = {
        "game": config.game,
        "use_restricted_actions": stable_retro.Actions.ALL,
        "inttype": stable_retro.data.Integrations.ALL,
        "render_mode": "rgb_array",
    }
    if config.state is not None:
        kwargs["state"] = config.state
    env = stable_retro.RetroEnv(**kwargs)
    return StableRetroPreprocessingEnv(env, config.preprocessing)


def create_stable_retro_overlay(
    rom_path: Path,
    lane_state_names: Sequence[str | None],
    state_dir: Path | None = None,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    overlay = tempfile.TemporaryDirectory(prefix="stable-retro-smb-")
    game_dir = Path(overlay.name) / GAME
    game_dir.mkdir(parents=True)
    (game_dir / "rom.nes").symlink_to(rom_path.resolve())
    upstream = stable_retro_module()
    packaged_state_dir = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "supermariobrosnes_turbo"
        / "data"
        / GAME
    )
    explicit_state_dir = Path(state_dir).expanduser() if state_dir is not None else None
    for state in dict.fromkeys(lane_state_names):
        if state is None:
            continue
        filename = state if state.endswith(".state") else f"{state}.state"
        explicit = explicit_state_dir / filename if explicit_state_dir is not None else None
        if explicit is not None and explicit.is_file():
            (game_dir / filename).symlink_to(explicit.resolve())
            continue
        upstream_path = upstream.data.get_file_path(
            GAME, filename, upstream.data.Integrations.STABLE
        )
        if upstream_path is not None:
            continue
        packaged = packaged_state_dir / filename
        if not packaged.is_file():
            raise FileNotFoundError(
                f"Stable Retro state {state!r} is unavailable upstream and was not found at {packaged}"
            )
        (game_dir / filename).symlink_to(packaged.resolve())
    return overlay, Path(overlay.name)


def create_stable_retro_vector_env(
    *,
    rom_path: Path,
    lane_state_names: Sequence[str | None],
    preprocessing: PreprocessingConfig,
    state_dir: Path | None = None,
    asynchronous: bool = True,
    context: str | None = "spawn",
) -> StableRetroVectorEnv:
    stable_retro_module()
    buttons = stable_retro_buttons()
    overlay, integration_path = create_stable_retro_overlay(
        rom_path, lane_state_names, state_dir
    )
    factories = [
        partial(
            make_stable_retro_worker,
            StableRetroWorkerConfig(str(integration_path), state, preprocessing),
        )
        for state in lane_state_names
    ]
    try:
        if asynchronous:
            env = AsyncVectorEnv(
                factories,
                shared_memory=True,
                copy=True,
                context=context,
                autoreset_mode=AutoresetMode.DISABLED,
            )
        else:
            env = SyncVectorEnv(factories, copy=True, autoreset_mode=AutoresetMode.DISABLED)
    except BaseException:
        overlay.cleanup()
        raise
    return StableRetroVectorEnv(env, overlay, lane_state_names, buttons)


def create_turbo_env(args: argparse.Namespace, rom_path: Path) -> Any:
    candidate = importlib.import_module(TURBO_IMPORT)
    if args.state_dir is not None:
        os.environ["SUPERMARIOBROSNES_FASTENV_STATE_DIR"] = str(args.state_dir)
    state_config = (
        {"state": args.state}
        if args.parsed_states is None
        else {"state_catalog": tuple(dict.fromkeys(args.parsed_states))}
    )
    return candidate.SuperMarioBrosNesTurboVecEnv(
        GAME,
        **state_config,
        rom_path=rom_path,
        num_envs=args.num_envs,
        use_restricted_actions=candidate.Actions.ALL,
        frame_skip=args.frame_skip,
        obs_grayscale=not args.rgb,
        frame_stack=args.frame_stack,
        obs_crop=(args.crop_top, args.crop_bottom, 0, 0),
        obs_crop_mode=args.obs_crop_mode,
        obs_resize=(args.resize_height, args.resize_width),
        obs_resize_algorithm="area",
        obs_layout="chw",
        maxpool_last_two=False,
    )


def step_env(env: Any, actions: np.ndarray) -> None:
    _obs, _rewards, terminated, truncated, _infos = env.step(actions)
    reset_mask = np.asarray(terminated) | np.asarray(truncated)
    if np.any(reset_mask):
        options = {"reset_mask": reset_mask}
        if getattr(env, "state_catalog", ()):
            state_indices = np.full(env.num_envs, -1, dtype=np.int32)
            state_indices[reset_mask] = env.active_state_indices()[reset_mask]
            options["state_indices"] = state_indices
        env.reset(options=options)


def step_action_sequence(env: Any, actions: Sequence[np.ndarray]) -> None:
    for action in actions:
        step_env(env, action)


def prepare_game(
    env: Any,
    args: argparse.Namespace,
    backend: str,
    buttons: Sequence[str | None] | None,
) -> np.ndarray:
    reset_options = None
    if backend == "turbo" and args.parsed_states is not None:
        catalog_indices = {name: index for index, name in enumerate(env.state_catalog)}
        reset_options = {
            "state_indices": np.asarray(
                [catalog_indices[state] for state in assigned_lane_states(args)],
                dtype=np.int32,
            )
        }
    obs, _infos = env.reset(options=reset_options)
    if args.no_start_game or has_initial_state(args) or "start" not in ACTION_SETS[args.action_set]:
        return obs
    noop = fill_action(backend, args.num_envs, "noop", buttons)
    start = fill_action(backend, args.num_envs, "start", buttons)
    step_action_sequence(env, (noop,) * args.pre_start_steps)
    step_action_sequence(env, (start,) * args.start_steps)
    step_action_sequence(env, (noop,) * args.post_start_steps)
    return obs


def run_once(env: Any, actions: Sequence[np.ndarray], args: argparse.Namespace) -> dict[str, float]:
    start = time.perf_counter()
    step_action_sequence(env, actions)
    elapsed = time.perf_counter() - start
    batch_sps = args.steps / elapsed
    env_sps = batch_sps * args.num_envs
    return {
        "elapsed_s": elapsed,
        "batch_steps_per_sec": batch_sps,
        "env_steps_per_sec": env_sps,
        "emulated_frames_per_sec": env_sps * args.frame_skip,
    }


def summarize(values: list[float]) -> dict[str, float]:
    result = {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}
    result["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def build_result(
    args: argparse.Namespace,
    obs: np.ndarray,
    runs: list[dict[str, float]],
    active_states: tuple[str | None, ...],
    load: dict[str, Any],
    rom_path: Path,
    backend: str | None = None,
) -> dict[str, Any]:
    if backend is None:
        backend = "stable-retro" if getattr(args, "stable_retro_baseline", False) else "turbo"
    batch_sps = [run["batch_steps_per_sec"] for run in runs]
    env_sps = [run["env_steps_per_sec"] for run in runs]
    frame_sps = [run["emulated_frames_per_sec"] for run in runs]
    elapsed = [run["elapsed_s"] for run in runs]
    mean_batch_sps = statistics.fmean(batch_sps)
    return {
        "backend": backend,
        "package": package_metadata(backend),
        "config": {
            "rom_path": str(rom_path),
            "rom_sha256": sha256_path(rom_path),
            "rayon_num_threads": rayon_num_threads() if backend == "turbo" else None,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack,
            "frame_maxpool": False,
            "grayscale": not args.rgb,
            "crop_top": args.crop_top,
            "crop_bottom": args.crop_bottom,
            "obs_crop_mode": args.obs_crop_mode,
            "resize_width": args.resize_width,
            "resize_height": args.resize_height,
            "obs_resize_algorithm": "area",
            "obs_layout": "chw",
            "action_set": args.action_set,
            "action": args.action,
            "actions": list(args.parsed_actions),
            "action_seed": args.action_seed,
            "state": args.state,
            "states": list(args.parsed_states) if args.parsed_states is not None else None,
            "lane_states": list(active_states) if has_initial_state(args) else None,
            "state_dir": str(args.state_dir) if args.state_dir is not None else None,
            "include_info": True,
            "terminate_on_flag": False,
            "termination": "provider_native",
            "start_game": (
                not args.no_start_game
                and not has_initial_state(args)
                and "start" in ACTION_SETS[args.action_set]
            ),
            "vectorization": (
                "gymnasium.AsyncVectorEnv" if backend == "stable-retro" else "native"
            ),
        },
        "observation": {
            "shape": list(obs.shape),
            "dtype": str(obs.dtype),
            "bytes": int(obs.nbytes),
            "mib": obs.nbytes / (1024**2),
        },
        "load": load,
        "runs": runs,
        "summary": {
            "elapsed_s": summarize(elapsed),
            "batch_steps_per_sec": summarize(batch_sps),
            "env_steps_per_sec": summarize(env_sps),
            "emulated_frames_per_sec": summarize(frame_sps),
            "obs_buffer_gib_per_sec": (obs.nbytes * mean_batch_sps) / (1024**3),
        },
    }


def print_human(result: dict[str, Any]) -> None:
    config = result["config"]
    obs = result["observation"]
    summary = result["summary"]
    print(
        f"backend={result['backend']} package={result['package']['name']}@{result['package']['version']}"
    )
    print(
        "config="
        f"num_envs={config['num_envs']} steps={config['steps']} repeats={config['repeats']} "
        f"frame_skip={config['frame_skip']} frame_stack={config['frame_stack']} "
        f"grayscale={config['grayscale']} crop=({config['crop_top']},{config['crop_bottom']}) "
        f"obs_crop_mode={config['obs_crop_mode']} "
        f"resize={config['resize_width']}x{config['resize_height']} "
        f"actions={config['actions']} state={config['state']} states={config['states']} "
        f"termination={config['termination']} vectorization={config['vectorization']}"
    )
    if config["lane_states"] is not None:
        print(f"lane_states={config['lane_states']}")
    load = result["load"]
    if load["enabled"] and load["start_1min"] is not None:
        print(
            f"load_preflight=start_1min={load['start_1min']:.2f} "
            f"max_start_load={load['max_start_load']:.2f} load_ok={load['load_ok']}"
        )
    elif load["enabled"]:
        print(f"load_preflight=unavailable load_ok={load['load_ok']}")
    else:
        print("load_preflight=disabled")
    print(f"obs_shape={tuple(obs['shape'])} obs_dtype={obs['dtype']} obs_mib={obs['mib']:.2f}")
    for index, run in enumerate(result["runs"], start=1):
        print(
            f"run={index} elapsed_s={run['elapsed_s']:.6f} "
            f"batch_steps_per_sec={run['batch_steps_per_sec']:.1f} "
            f"env_steps_per_sec={run['env_steps_per_sec']:.1f} "
            f"emulated_frames_per_sec={run['emulated_frames_per_sec']:.1f}"
        )
    print(
        "summary="
        f"env_steps_per_sec_mean={summary['env_steps_per_sec']['mean']:.1f} "
        f"env_steps_per_sec_stdev={summary['env_steps_per_sec']['stdev']:.1f} "
        f"best_env_steps_per_sec={summary['env_steps_per_sec']['max']:.1f} "
        f"emulated_frames_per_sec_mean={summary['emulated_frames_per_sec']['mean']:.1f} "
        f"obs_buffer_gib_per_sec={summary['obs_buffer_gib_per_sec']:.2f}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    args.parsed_states = initial_states_for_args(args)
    args.parsed_actions = selected_actions_for_args(args)
    backend = backend_for_args(args)
    rom_path = resolve_verified_rom_path(args.rom_path)
    load = load_preflight(args)
    if backend == "stable-retro":
        env: Any = create_stable_retro_vector_env(
            rom_path=rom_path,
            lane_state_names=assigned_lane_states(args),
            preprocessing=preprocessing_for_args(args),
            state_dir=args.state_dir,
        )
        buttons: Sequence[str | None] | None = env.buttons
    else:
        env = create_turbo_env(args, rom_path)
        buttons = None
    try:
        if args.profile_output is not None:
            env.enable_profiler()
        obs = prepare_game(env, args, backend, buttons)
        if backend == "turbo":
            catalog = env.state_catalog
            active_states = tuple(
                catalog[int(index)] if int(index) >= 0 else None
                for index in env.active_state_indices()
            )
        else:
            active_states = env.active_states()
        templates = action_templates(backend, args.num_envs, args.parsed_actions, buttons)
        warmup_actions = sampled_action_sequence(templates, args.warmup, args.action_seed + 1)
        measured_actions = sampled_action_sequence(templates, args.steps, args.action_seed)
        step_action_sequence(env, warmup_actions)
        if args.profile_output is not None:
            env.reset_profiler()
        runs = [run_once(env, measured_actions, args) for _ in range(args.repeats)]
        result = build_result(args, obs, runs, active_states, load, rom_path, backend)
        if args.profile_output is not None:
            result["profiler"] = env.profiler_snapshot()
    finally:
        env.close()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.profile_output is not None:
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
