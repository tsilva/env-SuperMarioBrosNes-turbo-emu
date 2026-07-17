from __future__ import annotations

import argparse
import ctypes
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

try:
    from .manual_playback import (
        DEFAULT_ROM,
        NES_HEIGHT,
        NES_WIDTH,
        SDL_INIT_VIDEO,
        SDL_PIXELFORMAT_RGB24,
        SDL_QUIT,
        SDL_RENDERER_ACCELERATED,
        SDL_RENDERER_PRESENTVSYNC,
        SDL_TEXTUREACCESS_STREAMING,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOW_SHOWN,
        SdlUnavailableError,
        configure_sdl,
        display_frame_from_obs,
        load_sdl2,
    )
except ImportError:
    from .manual_playback import (
        DEFAULT_ROM,
        NES_HEIGHT,
        NES_WIDTH,
        SDL_INIT_VIDEO,
        SDL_PIXELFORMAT_RGB24,
        SDL_QUIT,
        SDL_RENDERER_ACCELERATED,
        SDL_RENDERER_PRESENTVSYNC,
        SDL_TEXTUREACCESS_STREAMING,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOW_SHOWN,
        SdlUnavailableError,
        configure_sdl,
        display_frame_from_obs,
        load_sdl2,
    )
from . import (
    ACTION_SETS,
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    action_mask,
    resolve_required_rom_path,
)
from .jerk import load_jerk_checkpoint
from .jerk import find_policy_path_for_state

from .benchmark_sps import (
    PreprocessingConfig,
    create_stable_retro_vector_env,
    named_action_mask,
    stable_retro_buttons,
)


DEFAULT_HF_FILENAME = "final_model.zip"
DEFAULT_GAME = "SuperMarioBros-Nes-v0"
HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)(?:/(?P<rest>.*))?$"
)


class ModelResolutionError(RuntimeError):
    pass


def parse_hf_source(source: str) -> tuple[str, str | None, str | None] | None:
    match = HF_URL_RE.match(source)
    if match:
        repo_id = match.group("repo")
        rest = match.group("rest") or ""
        parts = rest.split("/") if rest else []
        if len(parts) >= 3 and parts[0] in {"blob", "resolve"}:
            revision = parts[1]
            filename = "/".join(parts[2:])
            return repo_id, filename, revision
        return repo_id, None, None
    if (
        "/" in source
        and not source.endswith((".json", ".zip"))
        and not Path(source).expanduser().exists()
    ):
        return source, None, None
    return None


def resolve_model_path(source: str, filename: str | None, cache_dir: Path) -> Path:
    local_path = Path(source).expanduser()
    if local_path.exists():
        return local_path

    hf_source = parse_hf_source(source)
    if hf_source is None:
        raise ModelResolutionError(
            f"model source does not exist and is not a Hugging Face repo/url: {source}"
        )

    repo_id, source_filename, revision = hf_source
    target_filename = filename or source_filename
    if target_filename is None:
        cached_filename = find_cached_hf_checkpoint_filename(
            repo_id, revision=revision, cache_dir=cache_dir
        )
        target_filename = (
            cached_filename
            or find_hf_checkpoint_filename(repo_id, revision=revision)
            or DEFAULT_HF_FILENAME
        )

    cached_path = cached_hf_file(
        repo_id, filename=target_filename, revision=revision, cache_dir=cache_dir
    )
    if cached_path is not None:
        return cached_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return download_direct_hf_file(
            repo_id,
            filename=target_filename,
            revision=revision or "main",
            cache_dir=cache_dir,
        )

    path = hf_hub_download(
        repo_id=repo_id,
        filename=target_filename,
        revision=revision,
        cache_dir=cache_dir,
    )
    return Path(path)


def cached_hf_file(
    repo_id: str, filename: str, revision: str | None, cache_dir: Path
) -> Path | None:
    repo_cache = cache_dir.expanduser() / f"models--{repo_id.replace('/', '--')}"
    snapshot_dirs: list[Path] = []
    if revision is not None:
        snapshot_dirs.append(repo_cache / "snapshots" / revision)
    else:
        ref = repo_cache / "refs" / "main"
        try:
            main_revision = ref.read_text().strip()
        except FileNotFoundError:
            main_revision = ""
        if main_revision:
            snapshot_dirs.append(repo_cache / "snapshots" / main_revision)
        snapshot_root = repo_cache / "snapshots"
        if snapshot_root.exists():
            snapshot_dirs.extend(
                sorted(path for path in snapshot_root.iterdir() if path.is_dir())
            )

    seen: set[Path] = set()
    for snapshot_dir in snapshot_dirs:
        path = snapshot_dir / filename
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def find_cached_hf_checkpoint_filename(
    repo_id: str, revision: str | None, cache_dir: Path
) -> str | None:
    repo_cache = cache_dir.expanduser() / f"models--{repo_id.replace('/', '--')}"
    snapshot_dirs: list[Path] = []
    if revision is not None:
        snapshot_dirs.append(repo_cache / "snapshots" / revision)
    else:
        ref = repo_cache / "refs" / "main"
        try:
            main_revision = ref.read_text().strip()
        except FileNotFoundError:
            main_revision = ""
        if main_revision:
            snapshot_dirs.append(repo_cache / "snapshots" / main_revision)
        snapshot_root = repo_cache / "snapshots"
        if snapshot_root.exists():
            snapshot_dirs.extend(
                sorted(path for path in snapshot_root.iterdir() if path.is_dir())
            )

    checkpoint_files: list[str] = []
    seen_dirs: set[Path] = set()
    for snapshot_dir in snapshot_dirs:
        if snapshot_dir in seen_dirs or not snapshot_dir.exists():
            continue
        seen_dirs.add(snapshot_dir)
        checkpoint_files.extend(
            str(path.relative_to(snapshot_dir))
            for suffix in ("*.zip", "*.json")
            for path in snapshot_dir.rglob(suffix)
        )

    unique_checkpoint_files = sorted(set(checkpoint_files))
    if len(unique_checkpoint_files) == 1:
        return unique_checkpoint_files[0]
    if DEFAULT_HF_FILENAME in unique_checkpoint_files:
        return DEFAULT_HF_FILENAME
    return None


def find_hf_checkpoint_filename(repo_id: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        return None
    files = list_repo_files(repo_id, revision=revision)
    checkpoint_files = sorted(
        path for path in files if path.endswith((".zip", ".json"))
    )
    if len(checkpoint_files) == 1:
        return checkpoint_files[0]
    if DEFAULT_HF_FILENAME in checkpoint_files:
        return DEFAULT_HF_FILENAME
    return None


def download_direct_hf_file(
    repo_id: str, filename: str, revision: str, cache_dir: Path
) -> Path:
    safe_name = urllib.parse.quote(f"{repo_id}/{revision}/{filename}", safe="")
    target = cache_dir.expanduser() / "direct" / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    quoted_filename = "/".join(urllib.parse.quote(part) for part in filename.split("/"))
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{quoted_filename}"
    urllib.request.urlretrieve(url, target)
    return target


def stable_action_masks(action_names: tuple[str, ...], rom_path: Path) -> np.ndarray:
    del rom_path
    buttons = stable_retro_buttons()
    return np.stack([named_action_mask(name, buttons) for name in action_names])


def json_default(value):
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def lane_info(infos: dict[str, object], lane: int = 0) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in infos.items():
        if key.startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask, dtype=np.bool_)[lane]):
            continue
        if isinstance(value, dict):
            result[key] = lane_info(value, lane)
        else:
            result[key] = value[lane]  # type: ignore[index]
    return result


def apply_checkpoint_defaults(args: argparse.Namespace, model_path: Path) -> None:
    """Select the preprocessing contract used to create the checkpoint."""
    del model_path
    if args.backend == "auto":
        args.backend = "native"
    if args.max_pool_frames is None:
        args.max_pool_frames = False
    if args.crop_mode is None:
        args.crop_mode = "remove"


def level_name_from_counters(level: tuple[int, int]) -> str:
    level_hi, level_lo = (int(value) for value in level)
    if level_hi < 0 or level_lo < 0:
        raise ValueError(f"invalid in-game level counters: {level}")
    return f"Level{level_hi + 1}-{level_lo + 1}"


class SdlPolicyPlayer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.action_names = ACTION_SETS[args.action_set]
        self.model_path = resolve_model_path(args.model, args.filename, args.cache_dir)
        apply_checkpoint_defaults(args, self.model_path)
        self.model = load_jerk_checkpoint(self.model_path)
        self._validate_model(self.model)
        self.initial_model = self.model
        self.initial_model_path = self.model_path
        self.current_policy_level = args.state
        self.current_state = args.state

        self.rom_path = resolve_required_rom_path(args.rom_path)
        if args.backend == "native":
            self.action_masks = np.stack(
                [action_mask(name) for name in self.action_names]
            ).astype(np.uint8)
        else:
            self.action_masks = stable_action_masks(self.action_names, self.rom_path)
        self.env = self.make_env()
        self.obs, infos = self.env.reset()
        self.model.reset()
        self.display_env = self.make_display_env() if args.view == "raw" else None
        if self.display_env is not None:
            self.display_obs, _display_infos = self.display_env.reset()
        else:
            self.display_obs = self.obs
        self.display_info: dict[str, object] = {}
        initial_frame = self.current_display_frame()
        self.display_height, self.display_width = initial_frame.shape[:2]
        self.scale = args.scale
        self.frame_delay_s = 1.0 / max(1, args.fps)
        self.episode = 1
        self.step = 0
        self.reward = 0.0
        self.max_x = 0
        self.info = lane_info(infos, 0)
        self.last_lives = int(self.info.get("lives", 0))
        self.last_level = (
            int(self.info.get("levelHi", 0)),
            int(self.info.get("levelLo", 0)),
        )
        self.level_changes = 0
        self.action = 0
        self.frames_rendered = 0
        self.running = True
        self.next_tick = time.perf_counter() + self.frame_delay_s
        self.fps_window_start = time.perf_counter()
        self.fps_window_frames = 0
        self.display_fps = 0.0
        self.sdl = load_sdl2()
        configure_sdl(self.sdl)
        if self.sdl.SDL_Init(SDL_INIT_VIDEO) != 0:
            raise SdlUnavailableError(self.sdl_error())
        self.sdl.SDL_SetHint(b"SDL_RENDER_SCALE_QUALITY", b"nearest")
        self.window = self.sdl.SDL_CreateWindow(
            b"SuperMarioBros-Nes-turbo player",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            self.display_width * self.scale,
            self.display_height * self.scale,
            SDL_WINDOW_SHOWN,
        )
        if not self.window:
            error = self.sdl_error()
            self.sdl.SDL_Quit()
            raise SdlUnavailableError(error)
        self.renderer = self.sdl.SDL_CreateRenderer(
            self.window,
            -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC,
        )
        if not self.renderer:
            error = self.sdl_error()
            self.sdl.SDL_DestroyWindow(self.window)
            self.sdl.SDL_Quit()
            raise SdlUnavailableError(error)
        self.texture = self.sdl.SDL_CreateTexture(
            self.renderer,
            SDL_PIXELFORMAT_RGB24,
            SDL_TEXTUREACCESS_STREAMING,
            self.display_width,
            self.display_height,
        )
        if not self.texture:
            error = self.sdl_error()
            self.sdl.SDL_DestroyRenderer(self.renderer)
            self.sdl.SDL_DestroyWindow(self.window)
            self.sdl.SDL_Quit()
            raise SdlUnavailableError(error)

    def _validate_model(self, model) -> None:
        if model.action_set != self.args.action_set:
            raise ValueError(
                f"checkpoint action_set={model.action_set!r} does not match "
                f"--action-set={self.args.action_set!r}"
            )
        if model.action_count != len(self.action_names):
            raise ValueError(
                f"model action count {model.action_count} does not match "
                f"action_set={self.args.action_set!r} with "
                f"{len(self.action_names)} actions",
            )

    def activate_named_level_policy(self, level_name: str) -> bool:
        if self.args.level_policy_root is None:
            return False
        model_path = find_policy_path_for_state(
            level_name, runs_root=self.args.level_policy_root
        )
        if model_path is None:
            return False
        model = load_jerk_checkpoint(model_path)
        self._validate_model(model)
        model.reset()
        self.model = model
        self.model_path = model_path
        self.current_policy_level = level_name
        return True

    def activate_level_policy(self, level: tuple[int, int]) -> bool:
        return self.activate_named_level_policy(level_name_from_counters(level))

    def advance_to_level_policy(self, level: tuple[int, int]) -> bool:
        level_name = level_name_from_counters(level)
        if not self.activate_named_level_policy(level_name):
            return False

        new_env = self.make_env(level_name)
        new_display_env = None
        try:
            new_obs, new_infos = new_env.reset()
            new_display_env = (
                self.make_display_env(level_name) if self.args.view == "raw" else None
            )
            if new_display_env is not None:
                new_display_obs, _display_infos = new_display_env.reset()
            else:
                new_display_obs = new_obs
        except Exception:
            if new_display_env is not None:
                new_display_env.close()
            new_env.close()
            raise

        old_env = self.env
        old_display_env = self.display_env
        self.env = new_env
        self.display_env = new_display_env
        self.obs = new_obs
        self.display_obs = new_display_obs
        self.display_info = {}
        self.info = lane_info(new_infos, 0)
        self.last_lives = int(self.info.get("lives", 0))
        self.last_level = level
        self.current_state = level_name
        old_env.close()
        if old_display_env is not None:
            old_display_env.close()
        return True

    def reset_current_policy(self) -> None:
        if self.activate_named_level_policy(self.current_state):
            return
        self.model = self.initial_model
        self.model_path = self.initial_model_path
        self.current_policy_level = self.args.state
        self.model.reset()

    def make_env(self, state: str | None = None):
        state = state or self.current_state
        if self.args.backend == "native":
            env = SuperMarioBrosNesTurboVecEnv(
                self.args.game,
                state=state,
                rom_path=self.rom_path,
                num_envs=1,
                num_threads=1,
                render_mode="rgb_array",
                use_restricted_actions=Actions.ALL,
                frame_skip=self.args.frame_skip,
                obs_grayscale=True,
                frame_stack=self.args.frame_stack,
                maxpool_last_two=self.args.max_pool_frames,
                obs_crop=(self.args.crop_top, self.args.crop_bottom, 0, 0),
                obs_crop_mode=self.args.crop_mode,
                obs_crop_fill=0,
                obs_resize=(self.args.resize_height, self.args.resize_width),
                obs_resize_algorithm="area",
                obs_layout="chw",
                obs_copy="safe_view",
                reward_clip=False,
                info_filter="all",
            )
            env.seed(self.args.seed)
            return env

        env = create_stable_retro_vector_env(
            rom_path=self.rom_path,
            lane_state_names=[state],
            state_dir=self.args.state_dir,
            preprocessing=PreprocessingConfig(
                frame_skip=self.args.frame_skip,
                frame_stack=self.args.frame_stack,
                grayscale=True,
                crop_top=self.args.crop_top,
                crop_bottom=self.args.crop_bottom,
                crop_mode=self.args.crop_mode,
                resize_width=self.args.resize_width,
                resize_height=self.args.resize_height,
                maxpool_last_two=self.args.max_pool_frames,
            ),
            asynchronous=False,
        )
        env.seed(self.args.seed)
        return env

    def make_display_env(self, state: str | None = None):
        state = state or self.current_state
        if self.args.backend == "native":
            env = SuperMarioBrosNesTurboVecEnv(
                self.args.game,
                state=state,
                rom_path=self.rom_path,
                num_envs=1,
                num_threads=1,
                render_mode="rgb_array",
                use_restricted_actions=Actions.ALL,
                frame_skip=self.args.frame_skip,
                obs_grayscale=False,
                frame_stack=1,
                maxpool_last_two=False,
                obs_crop=None,
                obs_resize=(NES_HEIGHT, NES_WIDTH),
                obs_resize_algorithm="area",
                obs_layout="chw",
                obs_copy="safe_view",
                reward_clip=False,
                info_filter="all",
            )
            env.seed(self.args.seed)
            return env

        env = create_stable_retro_vector_env(
            rom_path=self.rom_path,
            lane_state_names=[state],
            state_dir=self.args.state_dir,
            preprocessing=PreprocessingConfig(
                frame_skip=self.args.frame_skip,
                frame_stack=1,
                grayscale=False,
                crop_top=0,
                crop_bottom=0,
                crop_mode="remove",
                resize_width=NES_WIDTH,
                resize_height=NES_HEIGHT,
                maxpool_last_two=False,
            ),
            asynchronous=False,
        )
        env.seed(self.args.seed)
        return env

    def run(self) -> None:
        try:
            self.render()
            while self.running:
                self.poll_events()
                self.policy_step()
                self.render()
                self.frames_rendered += 1
                self.fps_window_frames += 1
                now = time.perf_counter()
                elapsed = now - self.fps_window_start
                if elapsed >= 0.5:
                    self.display_fps = self.fps_window_frames / elapsed
                    self.fps_window_frames = 0
                    self.fps_window_start = now
                if (
                    self.args.auto_close_frames is not None
                    and self.frames_rendered >= self.args.auto_close_frames
                ):
                    break
                self.sleep_until_next_frame()
        finally:
            self.close()

    def policy_step(self) -> None:
        action, _ = self.model.predict(self.obs)
        self.action = int(np.asarray(action).reshape(-1)[0])
        obs, rewards, terminations, truncations, infos = self.env.step(
            self.action_masks[[self.action]]
        )
        terminated_value = bool(terminations[0])
        truncated_value = bool(truncations[0])
        self.obs = obs
        self.step_display_env()
        self.reward += float(rewards[0])
        step_info = lane_info(infos, 0)
        self.info = dict(step_info)
        self.step += 1
        self.max_x = max(self.max_x, int(self.info.get("x_pos", 0)))
        current_lives = int(self.info.get("lives", self.last_lives))
        life_loss = current_lives < self.last_lives
        self.last_lives = current_lives
        current_level = (
            int(self.info.get("levelHi", self.last_level[0])),
            int(self.info.get("levelLo", self.last_level[1])),
        )
        if current_level != self.last_level:
            self.level_changes += 1
            if self.advance_to_level_policy(current_level):
                return
        self.last_level = current_level
        if terminated_value or truncated_value or life_loss:
            self.hold_terminal_frame()
            self.print_episode_summary(life_loss, terminated_value, truncated_value)
            if self.args.episodes > 0 and self.episode >= self.args.episodes:
                self.running = False
                return
            self.episode += 1
            self.step = 0
            self.reward = 0.0
            self.max_x = 0
            self.obs, infos = self.env.reset()
            self.reset_current_policy()
            self.info = lane_info(infos, 0)
            self.last_lives = int(self.info.get("lives", 0))
            self.last_level = (
                int(self.info.get("levelHi", 0)),
                int(self.info.get("levelLo", 0)),
            )
            self.level_changes = 0
            if self.display_env is not None:
                self.display_obs, _display_infos = self.display_env.reset()
            else:
                self.display_obs = self.obs
            self.display_info = {}

    def step_display_env(self) -> None:
        if self.display_env is None:
            self.display_obs = self.obs
            self.display_info = self.info
            return
        (
            display_obs,
            _rewards,
            display_terminations,
            display_truncations,
            display_infos,
        ) = self.display_env.step(
            self.action_masks[[self.action]],
        )
        self.display_obs = display_obs
        step_info = lane_info(display_infos, 0)
        if bool(display_terminations[0] or display_truncations[0]):
            self.display_info = dict(step_info)
        else:
            self.display_info = step_info

    def hold_terminal_frame(self) -> None:
        hold_frames = self.args.hold_done_frames
        for _ in range(max(0, hold_frames)):
            self.poll_events()
            if not self.running:
                break
            self.render()
            self.sleep_until_next_frame()

    def print_episode_summary(
        self, life_loss: bool, terminated: bool, truncated: bool
    ) -> None:
        summary = {
            "episode": self.episode,
            "steps": self.step,
            "reward": self.reward,
            "max_x": self.max_x,
            "life_loss": life_loss,
            "level_changes": self.level_changes,
            "terminated": terminated,
            "truncated": truncated,
            "final_info": self.info,
        }
        print(json.dumps(summary, default=json_default, sort_keys=True), flush=True)

    def poll_events(self) -> None:
        event = ctypes.create_string_buffer(64)
        while self.sdl.SDL_PollEvent(ctypes.byref(event)):
            event_type = ctypes.c_uint32.from_buffer(event).value
            if event_type == SDL_QUIT:
                self.running = False

    def render(self) -> None:
        frame = self.current_display_frame()
        if frame.ndim == 2:
            height, width = frame.shape
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            rgb[:, :, 0] = frame
            rgb[:, :, 1] = frame
            rgb[:, :, 2] = frame
            frame = rgb
        else:
            frame = np.ascontiguousarray(frame)

        if (
            self.sdl.SDL_UpdateTexture(
                self.texture,
                None,
                frame.ctypes.data_as(ctypes.c_void_p),
                frame.strides[0],
            )
            != 0
        ):
            raise RuntimeError(self.sdl_error())
        self.sdl.SDL_RenderClear(self.renderer)
        self.sdl.SDL_RenderCopy(self.renderer, self.texture, None, None)
        self.sdl.SDL_RenderPresent(self.renderer)

    def current_display_frame(self) -> np.ndarray:
        obs = self.display_obs if self.args.view == "raw" else self.obs
        return display_frame_from_obs(obs[0], grayscale=self.args.view != "raw")

    def sleep_until_next_frame(self) -> None:
        self.next_tick += self.frame_delay_s
        delay_s = self.next_tick - time.perf_counter()
        if delay_s < -self.frame_delay_s:
            self.next_tick = time.perf_counter() + self.frame_delay_s
            delay_s = self.frame_delay_s
        if delay_s > 0:
            self.sdl.SDL_Delay(max(1, round(delay_s * 1000)))

    def close(self) -> None:
        self.env.close()
        if self.display_env is not None:
            self.display_env.close()
        if getattr(self, "texture", None):
            self.sdl.SDL_DestroyTexture(self.texture)
            self.texture = None
        if getattr(self, "renderer", None):
            self.sdl.SDL_DestroyRenderer(self.renderer)
            self.renderer = None
        if getattr(self, "window", None):
            self.sdl.SDL_DestroyWindow(self.window)
            self.window = None
        if getattr(self, "sdl", None):
            self.sdl.SDL_Quit()

    def sdl_error(self) -> str:
        raw = self.sdl.SDL_GetError()
        return raw.decode("utf-8", errors="replace") if raw else "unknown SDL error"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a JERK Mario action sequence from disk or Hugging Face.",
    )
    parser.add_argument(
        "model", help="Local JERK .zip/.json, HF repo id, or Hugging Face URL"
    )
    parser.add_argument(
        "--filename", default=None, help="Checkpoint filename inside an HF repo"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf_cache"))
    parser.add_argument(
        "--backend",
        choices=("auto", "stable-retro", "native"),
        default="auto",
        help="auto selects the native backend",
    )
    parser.add_argument("--game", default=DEFAULT_GAME)
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=DEFAULT_ROM,
        help="Path to the SMB NES ROM. Defaults to Stable Retro-compatible discovery.",
    )
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument(
        "--level-policy-root",
        type=Path,
        default=None,
        help="Load a matching runs/<Level>/<Level>.zip after level changes",
    )
    parser.add_argument("--view", choices=("raw", "preprocessed"), default="raw")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=0, help="0 means play forever")
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument(
        "--max-pool-frames", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--crop-top", type=int, default=32)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--crop-mode", choices=("remove", "mask"), default=None)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--action-set", choices=tuple(ACTION_SETS), default="simple")
    parser.add_argument("--hold-done-frames", type=int, default=0)
    parser.add_argument("--auto-close-frames", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        SdlPolicyPlayer(args).run()
    except SdlUnavailableError as exc:
        raise SystemExit(f"SDL backend unavailable: {exc}") from exc


if __name__ == "__main__":
    main()
