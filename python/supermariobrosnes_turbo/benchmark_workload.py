"""Packaged benchmark defaults used by playback adapters."""

from __future__ import annotations

import argparse
import shlex

CANONICAL_NUM_ENVS = 16
CANONICAL_FRAME_SKIP = 4
CANONICAL_FRAME_STACK = 4
CANONICAL_CROP_TOP = 32
CANONICAL_CROP_BOTTOM = 0
CANONICAL_RESIZE_WIDTH = 84
CANONICAL_RESIZE_HEIGHT = 84
CANONICAL_OBS_CROP_MODE = "mask"
CANONICAL_ACTION_SET = "basic"
CANONICAL_ACTION_NAMES = ("noop", "right", "right_b", "right_a")
CANONICAL_ACTION_SEED = 0
CANONICAL_STATE_NAMES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")
CANONICAL_TERMINATE_ON_FLAG = False
CANONICAL_START_GAME = False


def joined_states() -> str:
    return ",".join(CANONICAL_STATE_NAMES)


def joined_actions(actions: tuple[str, ...] = CANONICAL_ACTION_NAMES) -> str:
    return ",".join(actions)


def canonical_env_args(*, actions: tuple[str, ...] = CANONICAL_ACTION_NAMES) -> list[str]:
    args = [
        "--num-envs",
        str(CANONICAL_NUM_ENVS),
        "--frame-skip",
        str(CANONICAL_FRAME_SKIP),
        "--frame-stack",
        str(CANONICAL_FRAME_STACK),
        "--crop-top",
        str(CANONICAL_CROP_TOP),
        "--crop-bottom",
        str(CANONICAL_CROP_BOTTOM),
        "--obs-crop-mode",
        CANONICAL_OBS_CROP_MODE,
        "--resize-width",
        str(CANONICAL_RESIZE_WIDTH),
        "--resize-height",
        str(CANONICAL_RESIZE_HEIGHT),
        "--states",
        joined_states(),
        "--action-set",
        CANONICAL_ACTION_SET,
        "--actions",
        joined_actions(actions),
        "--action-seed",
        str(CANONICAL_ACTION_SEED),
    ]
    if not CANONICAL_START_GAME:
        args.append("--no-start-game")
    return args


def canonical_noop_env_args() -> list[str]:
    args = canonical_env_args(actions=("noop",))
    index = args.index("--actions")
    args[index : index + 2] = ["--action", "noop"]
    return args


def shell_args(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print canonical benchmark workload arguments.")
    parser.add_argument("--noop", action="store_true", help="Print the legacy single-noop variant.")
    parsed = parser.parse_args()
    args = canonical_noop_env_args() if parsed.noop else canonical_env_args()
    print(shell_args(args))


if __name__ == "__main__":
    main()
