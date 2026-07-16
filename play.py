#!/usr/bin/env python3
"""Play a named Mario level, automatically using its trained JERK policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from supermariobrosnes_turbo.jerk import normalize_level_name, policy_path_for_level


def resolve_level_policy(
    level: str,
    *,
    runs_root: str | Path = "runs",
) -> Path | None:
    path = policy_path_for_level(level, runs_root=runs_root)
    return path if path.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", help="Level state to start, for example Level1-1")
    parser.add_argument(
        "playback_args",
        nargs=argparse.REMAINDER,
        help="Additional options forwarded to the selected player",
    )
    return parser


def _validate_playback_args(playback_args: list[str]) -> None:
    if "--state" in playback_args:
        raise ValueError("play.py derives --state from the level; do not pass --state")
    if "--level-policy-root" in playback_args:
        raise ValueError(
            "play.py derives --level-policy-root; do not pass --level-policy-root"
        )


def policy_playback_argv(
    level: str,
    playback_args: list[str],
    *,
    runs_root: str | Path = "runs",
) -> list[str] | None:
    name = normalize_level_name(level)
    _validate_playback_args(playback_args)
    policy_path = resolve_level_policy(name, runs_root=runs_root)
    if policy_path is None:
        return None
    return [
        str(policy_path),
        "--state",
        name,
        "--level-policy-root",
        str(runs_root),
        *playback_args,
    ]


def manual_playback_argv(level: str, playback_args: list[str]) -> list[str]:
    name = normalize_level_name(level)
    _validate_playback_args(playback_args)
    return ["--state", name, *playback_args]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy_args = policy_playback_argv(args.level, args.playback_args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if policy_args is not None:
        from scripts import play_policy

        parsed = play_policy.parse_args(policy_args)
        try:
            play_policy.SdlPolicyPlayer(parsed).run()
        except play_policy.SdlUnavailableError as exc:
            raise SystemExit(f"SDL backend unavailable: {exc}") from exc
    else:
        from scripts import play as manual_play

        try:
            manual_args = manual_playback_argv(args.level, args.playback_args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        parsed = manual_play.parse_args(manual_args)
        try:
            manual_play.SdlExternalVecPlayer(parsed).run()
        except manual_play.SdlUnavailableError as exc:
            raise SystemExit(f"SDL backend unavailable: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
