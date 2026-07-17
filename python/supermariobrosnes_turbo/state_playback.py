"""Play an exact Mario state manually or with a trained action-run policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import ACTION_SETS
from .jerk import policy_path_for_state, resolve_state_name
from .manual_playback import DEFAULT_ROM, SdlExternalVecPlayer, SdlUnavailableError
from .policy_playback import DEFAULT_GAME, SdlPolicyPlayer


DEFAULT_STATE = "Level1-1"


def resolve_state_policy(
    state: str,
    *,
    runs_dir: str | Path = "runs",
) -> Path | None:
    path = policy_path_for_state(state, runs_root=runs_dir)
    return path if path.is_file() else None


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "state",
        nargs="?",
        default=DEFAULT_STATE,
        help=f"exact state identifier (default: {DEFAULT_STATE})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manual", action="store_true", help="force manual playback")
    mode.add_argument(
        "--policy",
        metavar="SOURCE",
        help="local JERK policy, Hugging Face repository, or Hugging Face URL",
    )
    parser.add_argument(
        "--rom",
        type=Path,
        default=DEFAULT_ROM,
        help="ROM path; defaults to Stable Retro-compatible discovery",
    )
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--backend", choices=("auto", "native", "stable-retro"), default="auto"
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
    parser.add_argument(
        "--stack-scale",
        type=int,
        default=2,
        help="scale for the manual frame-stack window",
    )
    parser.add_argument("--filename", default=None, help="policy filename in an HF repo")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/hf_cache")
    )
    return parser


def _manual_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        mode="external",
        rom_path=args.rom,
        fps=args.fps,
        scale=args.scale,
        stack_scale=args.stack_scale,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        state=args.state,
        state_dir=args.state_dir,
        auto_close_frames=args.auto_close_frames,
    )


def _policy_args(args: argparse.Namespace, source: str | Path) -> argparse.Namespace:
    return argparse.Namespace(
        model=str(source),
        filename=args.filename,
        cache_dir=args.cache_dir,
        backend=args.backend,
        game=DEFAULT_GAME,
        rom_path=args.rom,
        state=args.state,
        state_dir=args.state_dir,
        level_policy_root=args.runs_dir,
        view=args.view,
        fps=args.fps,
        scale=args.scale,
        episodes=args.episodes,
        seed=args.seed,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        max_pool_frames=args.max_pool_frames,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        crop_mode=args.crop_mode,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        action_set=args.action_set,
        hold_done_frames=args.hold_done_frames,
        auto_close_frames=args.auto_close_frames,
    )


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        args.state = resolve_state_name(args.state, state_dir=args.state_dir)
    except ValueError as exc:
        parser.error(str(exc))

    source: str | Path | None = args.policy
    if not args.manual and source is None:
        source = resolve_state_policy(args.state, runs_dir=args.runs_dir)

    try:
        if args.manual or source is None:
            if source is None and not args.manual:
                expected = policy_path_for_state(args.state, runs_root=args.runs_dir)
                print(f"No trained policy at {expected}; starting manual playback.")
            else:
                print(f"Starting manual playback from {args.state}.")
            SdlExternalVecPlayer(_manual_args(args)).run()
        else:
            print(f"Playing {args.state} with policy {source}.")
            SdlPolicyPlayer(_policy_args(args, source)).run()
    except SdlUnavailableError as exc:
        raise SystemExit(f"SDL backend unavailable: {exc}") from exc
    return 0
