from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import struct
import time
import zlib
from pathlib import Path

import numpy as np

from . import (
    ACTION_BUTTONS,
    BUTTON_TO_INDEX,
    NES_BUTTONS,
    Actions,
    CORE_ACTION_MEANINGS as ACTION_MEANINGS,
)
from . import (
    SuperMarioBrosNesTurboVecEnv,
    default_rom_path,
    resolve_required_rom_path,
)


DEFAULT_ROM = default_rom_path()
NES_WIDTH = 256
NES_HEIGHT = 240


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


SDL_INIT_VIDEO = 0x00000020
SDL_WINDOWPOS_CENTERED = 0x2FFF0000
SDL_WINDOW_SHOWN = 0x00000004
SDL_RENDERER_ACCELERATED = 0x00000002
SDL_RENDERER_PRESENTVSYNC = 0x00000004
SDL_TEXTUREACCESS_STREAMING = 1
SDL_PIXELFORMAT_RGB24 = 0x17101803
SDL_QUIT = 0x100
SDL_KEYDOWN = 0x300
SDL_KEYUP = 0x301
SDLK_RETURN = 13
SDLK_ESCAPE = 27
SDLK_SPACE = 32
SDLK_RIGHT = 1073741903
SDLK_LEFT = 1073741904
SDL_SCANCODE_LSHIFT = 225
SDL_SCANCODE_RSHIFT = 229


def parse_fps(value: str) -> int | None:
    """Parse a positive FPS limit, or ``max`` for uncapped playback."""
    if value.casefold() == "max":
        return None
    try:
        fps = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "FPS must be a positive integer or 'max'"
        ) from exc
    if fps <= 0:
        raise argparse.ArgumentTypeError("FPS must be a positive integer or 'max'")
    return fps


def frame_delay_for_fps(fps: int | None) -> float | None:
    return None if fps is None else 1.0 / fps


def renderer_flags_for_fps(fps: int | None) -> int:
    flags = SDL_RENDERER_ACCELERATED
    if fps is not None:
        flags |= SDL_RENDERER_PRESENTVSYNC
    return flags


class SdlUnavailableError(RuntimeError):
    pass


class SdlTextureWindow:
    def __init__(
        self,
        owner: "SdlExternalVecPlayer",
        title: str,
        initial_frame: np.ndarray,
        scale: int,
        x: int,
        y: int,
    ) -> None:
        if scale <= 0:
            raise ValueError("window scale must be positive")
        self.owner = owner
        self.sdl = owner.sdl
        self.title = title
        self.scale = scale
        self.texture = None
        self.renderer = None
        self.window = None

        frame = rgb_frame(initial_frame)
        self.height, self.width = frame.shape[:2]
        self.window = self.sdl.SDL_CreateWindow(
            title.encode("utf-8"),
            x,
            y,
            self.width * scale,
            self.height * scale,
            SDL_WINDOW_SHOWN,
        )
        if not self.window:
            raise SdlUnavailableError(owner.sdl_error())
        self.renderer = self.sdl.SDL_CreateRenderer(
            self.window,
            -1,
            renderer_flags_for_fps(owner.fps),
        )
        if not self.renderer:
            error = owner.sdl_error()
            self.close()
            raise SdlUnavailableError(error)
        self.texture = self.sdl.SDL_CreateTexture(
            self.renderer,
            SDL_PIXELFORMAT_RGB24,
            SDL_TEXTUREACCESS_STREAMING,
            self.width,
            self.height,
        )
        if not self.texture:
            error = owner.sdl_error()
            self.close()
            raise SdlUnavailableError(error)
        self.render(frame)

    @property
    def pixel_width(self) -> int:
        return self.width * self.scale

    def render(self, frame: np.ndarray) -> None:
        frame = rgb_frame(frame)
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"{self.title} frame size changed from {(self.height, self.width)} to {frame.shape[:2]}"
            )
        if (
            self.sdl.SDL_UpdateTexture(
                self.texture,
                None,
                frame.ctypes.data_as(ctypes.c_void_p),
                frame.strides[0],
            )
            != 0
        ):
            raise RuntimeError(self.owner.sdl_error())
        self.sdl.SDL_RenderClear(self.renderer)
        self.sdl.SDL_RenderCopy(self.renderer, self.texture, None, None)
        self.sdl.SDL_RenderPresent(self.renderer)

    def set_title(self, title: str) -> None:
        if self.window:
            self.sdl.SDL_SetWindowTitle(self.window, title.encode("utf-8"))

    def close(self) -> None:
        if self.texture:
            self.sdl.SDL_DestroyTexture(self.texture)
            self.texture = None
        if self.renderer:
            self.sdl.SDL_DestroyRenderer(self.renderer)
            self.renderer = None
        if self.window:
            self.sdl.SDL_DestroyWindow(self.window)
            self.window = None


class SdlExternalVecPlayer:
    """Keyboard player that feeds actions through a one-lane vector env."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.frame_skip <= 0:
            raise ValueError("--frame-skip must be positive")
        if args.frame_stack <= 0:
            raise ValueError("--frame-stack must be positive")
        if args.crop_top < 0 or args.crop_bottom < 0:
            raise ValueError("--crop-top and --crop-bottom must be non-negative")

        self.env = SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            state=args.state,
            rom_path=resolve_required_rom_path(args.rom_path),
            num_envs=1,
            use_restricted_actions=Actions.ALL,
            render_mode="rgb_array",
            frame_skip=args.frame_skip,
            obs_grayscale=True,
            frame_stack=args.frame_stack,
            obs_crop=(args.crop_top, args.crop_bottom, 0, 0),
            obs_resize=(args.resize_height, args.resize_width),
            obs_resize_algorithm="area",
            obs_layout="chw",
        )
        self.scale = args.scale
        self.stack_scale = args.stack_scale
        self.fps = args.fps
        self.frame_delay_s = frame_delay_for_fps(self.fps)
        self.stack_obs = self.reset_one()
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.info: dict[str, object] = {}
        self.frames_rendered = 0
        self.auto_close_frames = args.auto_close_frames

        try:
            self.sdl = load_sdl2()
        except Exception:
            self.env.close()
            raise
        configure_sdl(self.sdl)
        self.windows: list[SdlTextureWindow] = []
        if self.sdl.SDL_Init(SDL_INIT_VIDEO) != 0:
            self.env.close()
            raise SdlUnavailableError(self.sdl_error())
        self.sdl.SDL_SetHint(b"SDL_RENDER_SCALE_QUALITY", b"nearest")
        try:
            self.rgb_window = SdlTextureWindow(
                self,
                "SuperMarioBros-Nes-turbo RGB",
                self.raw_rgb_frame(),
                self.scale,
                64,
                64,
            )
            self.windows.append(self.rgb_window)
            self.stack_window = SdlTextureWindow(
                self,
                "SuperMarioBros-Nes-turbo frame stack",
                display_frame_from_obs(self.stack_obs, grayscale=True),
                self.stack_scale,
                64 + self.rgb_window.pixel_width + 24,
                64,
            )
            self.windows.append(self.stack_window)
        except Exception:
            self.close()
            raise

        self.pressed_keys: set[int] = set()
        self.pressed_scancodes: set[int] = set()
        self.running = True
        self.next_tick = (
            None
            if self.frame_delay_s is None
            else time.perf_counter() + self.frame_delay_s
        )
        self.fps_window_start = time.perf_counter()
        self.fps_window_frames = 0
        self.display_fps = 0.0
        self.next_status_update = 0.0

    def run(self) -> None:
        try:
            self.render()
            while self.running:
                self.poll_events()
                action = self.current_action()
                self.stack_obs, reward, self.terminated, self.truncated, self.info = (
                    self.step_one(action)
                )
                self.reward += reward
                if self.terminated or self.truncated:
                    self.stack_obs = self.reset_one()
                    self.reward = 0.0

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
                    self.auto_close_frames is not None
                    and self.frames_rendered >= self.auto_close_frames
                ):
                    break

                self.sleep_until_next_frame()
        finally:
            self.close()

    def sleep_until_next_frame(self) -> None:
        if self.frame_delay_s is None:
            return
        assert self.next_tick is not None
        self.next_tick += self.frame_delay_s
        delay_s = self.next_tick - time.perf_counter()
        if delay_s < -self.frame_delay_s:
            self.next_tick = time.perf_counter() + self.frame_delay_s
            delay_s = self.frame_delay_s
        if delay_s > 0:
            self.sdl.SDL_Delay(max(1, round(delay_s * 1000)))

    def poll_events(self) -> None:
        event = ctypes.create_string_buffer(64)
        while self.sdl.SDL_PollEvent(ctypes.byref(event)):
            event_type = ctypes.c_uint32.from_buffer(event).value
            if event_type == SDL_QUIT:
                self.running = False
            elif event_type in (SDL_KEYDOWN, SDL_KEYUP):
                scancode = ctypes.c_int32.from_buffer(event, 16).value
                keycode = ctypes.c_int32.from_buffer(event, 20).value
                if event_type == SDL_KEYDOWN:
                    self.pressed_scancodes.add(scancode)
                    self.pressed_keys.add(keycode)
                    if keycode == SDLK_ESCAPE:
                        self.running = False
                else:
                    self.pressed_scancodes.discard(scancode)
                    self.pressed_keys.discard(keycode)

    def render(self) -> None:
        self.rgb_window.render(self.raw_rgb_frame())
        self.stack_window.render(display_frame_from_obs(self.stack_obs, grayscale=True))

        now = time.perf_counter()
        if now >= self.next_status_update:
            self.next_status_update = now + 0.1
            gameplay_title = (
                "SuperMarioBros-Nes-turbo RGB  "
                f"action={ACTION_MEANINGS[self.current_action()]} "
                f"x={self.info.get('x_pos', 0)} lives={self.info.get('lives', 0)} "
                f"reward={self.reward:.1f} fps={self.display_fps:.0f}"
            )
            stack_title = f"SuperMarioBros-Nes-turbo frame stack  obs={tuple(self.stack_obs.shape)}"
            self.rgb_window.set_title(gameplay_title)
            self.stack_window.set_title(stack_title)

    def step_one(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_name = ACTION_MEANINGS[action]
        action_mask = np.zeros((1, len(NES_BUTTONS)), dtype=np.uint8)
        for button in ACTION_BUTTONS[action_name]:
            action_mask[0, BUTTON_TO_INDEX[button]] = 1
        obs, rewards, terminated, truncated, infos = self.env.step_gymnasium(
            action_mask
        )
        return (
            obs[0],
            float(rewards[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            lane_info(infos, 0),
        )

    def reset_one(self) -> np.ndarray:
        obs, _infos = self.env.reset()
        return obs[0]

    def raw_rgb_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("render_mode='rgb_array' did not return a frame")
        return np.ascontiguousarray(frame)

    def current_action(self) -> int:
        if SDLK_RETURN in self.pressed_keys:
            return action_id("start")

        right = SDLK_RIGHT in self.pressed_keys or ord("d") in self.pressed_keys
        left = SDLK_LEFT in self.pressed_keys or ord("a") in self.pressed_keys
        jump = any(key in self.pressed_keys for key in (ord("x"), ord("j"), SDLK_SPACE))
        run = (
            ord("z") in self.pressed_keys
            or ord("k") in self.pressed_keys
            or SDL_SCANCODE_LSHIFT in self.pressed_scancodes
            or SDL_SCANCODE_RSHIFT in self.pressed_scancodes
        )

        if left and not right:
            return action_id("left")
        if right and jump and run:
            return action_id("right_a_b")
        if right and jump:
            return action_id("right_a")
        if right and run:
            return action_id("right_b")
        if right:
            return action_id("right")
        if jump:
            return action_id("a")
        return action_id("noop")

    def close(self) -> None:
        for window in reversed(getattr(self, "windows", [])):
            window.close()
        self.windows = []
        self.env.close()
        if getattr(self, "sdl", None):
            self.sdl.SDL_Quit()

    def sdl_error(self) -> str:
        raw = self.sdl.SDL_GetError()
        return raw.decode("utf-8", errors="replace") if raw else "unknown SDL error"


def load_sdl2() -> ctypes.CDLL:
    candidates = [
        ctypes.util.find_library("SDL2"),
        "/opt/homebrew/lib/libSDL2-2.0.0.dylib",
        "/opt/homebrew/lib/libSDL2.dylib",
        "/usr/local/lib/libSDL2-2.0.0.dylib",
        "/usr/local/lib/libSDL2.dylib",
    ]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    details = "; ".join(errors) if errors else "no SDL2 library candidates found"
    raise SdlUnavailableError(details)


def configure_sdl(sdl: ctypes.CDLL) -> None:
    if hasattr(sdl, "SDL_SetMainReady"):
        sdl.SDL_SetMainReady.argtypes = []
        sdl.SDL_SetMainReady.restype = None
        sdl.SDL_SetMainReady()

    sdl.SDL_Init.argtypes = [ctypes.c_uint32]
    sdl.SDL_Init.restype = ctypes.c_int
    sdl.SDL_Quit.argtypes = []
    sdl.SDL_Quit.restype = None
    sdl.SDL_GetError.argtypes = []
    sdl.SDL_GetError.restype = ctypes.c_char_p
    sdl.SDL_SetHint.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    sdl.SDL_SetHint.restype = ctypes.c_int
    sdl.SDL_CreateWindow.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    sdl.SDL_CreateWindow.restype = ctypes.c_void_p
    sdl.SDL_DestroyWindow.argtypes = [ctypes.c_void_p]
    sdl.SDL_DestroyWindow.restype = None
    sdl.SDL_SetWindowTitle.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    sdl.SDL_SetWindowTitle.restype = None
    sdl.SDL_CreateRenderer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    sdl.SDL_CreateRenderer.restype = ctypes.c_void_p
    sdl.SDL_DestroyRenderer.argtypes = [ctypes.c_void_p]
    sdl.SDL_DestroyRenderer.restype = None
    sdl.SDL_CreateTexture.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    sdl.SDL_CreateTexture.restype = ctypes.c_void_p
    sdl.SDL_DestroyTexture.argtypes = [ctypes.c_void_p]
    sdl.SDL_DestroyTexture.restype = None
    sdl.SDL_UpdateTexture.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    sdl.SDL_UpdateTexture.restype = ctypes.c_int
    sdl.SDL_RenderClear.argtypes = [ctypes.c_void_p]
    sdl.SDL_RenderClear.restype = ctypes.c_int
    sdl.SDL_RenderCopy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    sdl.SDL_RenderCopy.restype = ctypes.c_int
    sdl.SDL_RenderPresent.argtypes = [ctypes.c_void_p]
    sdl.SDL_RenderPresent.restype = None
    sdl.SDL_PollEvent.argtypes = [ctypes.c_void_p]
    sdl.SDL_PollEvent.restype = ctypes.c_int
    sdl.SDL_Delay.argtypes = [ctypes.c_uint32]
    sdl.SDL_Delay.restype = None


def action_id(name: str) -> int:
    return ACTION_MEANINGS.index(name)


def latest_frame(obs: np.ndarray) -> np.ndarray:
    if obs.ndim != 3:
        raise ValueError(f"expected CHW observation, got shape {obs.shape}")
    if obs.shape[0] == 1:
        return np.ascontiguousarray(obs[0])
    if obs.shape[0] == 3:
        return np.ascontiguousarray(np.moveaxis(obs, 0, -1))
    raise ValueError(
        f"play mode expects unstacked grayscale or RGB observation, got shape {obs.shape}"
    )


def display_frame_from_obs(obs: np.ndarray, grayscale: bool) -> np.ndarray:
    if obs.ndim != 3:
        raise ValueError(f"expected CHW observation, got shape {obs.shape}")
    if grayscale:
        return tile_grayscale_channels(obs)
    if obs.shape[0] == 1:
        return np.ascontiguousarray(obs[0])
    if obs.shape[0] == 3:
        return np.ascontiguousarray(np.moveaxis(obs, 0, -1))
    if obs.shape[0] % 3 == 0:
        return tile_rgb_frames(obs)
    return tile_grayscale_channels(obs)


def rgb_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        height, width = frame.shape
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = frame
        rgb[:, :, 1] = frame
        rgb[:, :, 2] = frame
        return np.ascontiguousarray(rgb)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return np.ascontiguousarray(frame)
    raise ValueError(
        f"expected HxW grayscale or HxWx3 RGB frame, got shape {frame.shape}"
    )


def grid_size(n: int) -> tuple[int, int]:
    cols = 1
    while cols * cols < n:
        cols += 1
    rows = (n + cols - 1) // cols
    return rows, cols


def tile_grayscale_channels(obs: np.ndarray) -> np.ndarray:
    channels, height, width = obs.shape
    rows, cols = grid_size(channels)
    grid = np.zeros((rows * height, cols * width), dtype=np.uint8)
    for channel in range(channels):
        row = channel // cols
        col = channel % cols
        y0 = row * height
        x0 = col * width
        grid[y0 : y0 + height, x0 : x0 + width] = obs[channel]
    return np.ascontiguousarray(grid)


def tile_rgb_frames(obs: np.ndarray) -> np.ndarray:
    frame_count = obs.shape[0] // 3
    height = obs.shape[1]
    width = obs.shape[2]
    rows, cols = grid_size(frame_count)
    grid = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for frame_idx in range(frame_count):
        row = frame_idx // cols
        col = frame_idx % cols
        y0 = row * height
        x0 = col * width
        frame = np.moveaxis(obs[frame_idx * 3 : (frame_idx + 1) * 3], 0, -1)
        grid[y0 : y0 + height, x0 : x0 + width] = frame
    return np.ascontiguousarray(grid)


def png_from_frame(frame: np.ndarray) -> bytes:
    if frame.ndim == 2:
        height, width = frame.shape
        color_type = 0
        row_iter = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        height, width, _ = frame.shape
        color_type = 2
        row_iter = frame
    else:
        raise ValueError(
            f"expected HxW grayscale or HxWx3 RGB frame, got shape {frame.shape}"
        )

    rows = bytearray()
    for row in row_iter:
        rows.append(0)
        rows.extend(row.tobytes())

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(data, checksum)
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", checksum & 0xFFFFFFFF)
        )

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    )
    png.extend(chunk(b"IDAT", zlib.compress(bytes(rows), level=1)))
    png.extend(chunk(b"IEND", b""))
    return bytes(png)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("external",), default="external")
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=DEFAULT_ROM,
        help="Path to the SMB NES ROM. Defaults to Stable Retro-compatible discovery.",
    )
    parser.add_argument(
        "--fps",
        "--fpx",
        type=parse_fps,
        default=60,
        metavar="FPS|max",
        help="playback frame-rate limit, or 'max' for uncapped playback",
    )
    parser.add_argument(
        "--scale", type=int, default=2, help="Scale for the main RGB gameplay window."
    )
    parser.add_argument(
        "--stack-scale",
        type=int,
        default=2,
        help="Scale for the side frame-stack window.",
    )
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--crop-top", type=int, default=32)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--state", default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--auto-close-frames", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode != "external":
        raise ValueError(f"unsupported play mode: {args.mode}")
    try:
        SdlExternalVecPlayer(args).run()
    except SdlUnavailableError as exc:
        raise SystemExit(f"SDL backend unavailable: {exc}") from exc


if __name__ == "__main__":
    main()
