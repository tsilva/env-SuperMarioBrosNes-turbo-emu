from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from supermariobrosnes_turbo import (
    ACTION_BUTTONS,
    BUTTON_TO_INDEX,
    CORE_ACTION_MEANINGS as ACTION_MEANINGS,
    NES_BUTTONS,
)
from supermariobrosnes_turbo import Actions
from supermariobrosnes_turbo import SuperMarioBrosNesTurboVecEnv, default_rom_path, resolve_required_rom_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=default_rom_path(),
        help="Path to the SMB NES ROM. Defaults to Stable Retro-compatible discovery.",
    )
    return parser.parse_args()


def info_summary(env: SuperMarioBrosNesTurboVecEnv) -> str:
    return (
        f"x_pos={env.x_pos.tolist()} lives={env.lives.tolist()} "
        f"coins={env.coins.tolist()} score={env.score.tolist()} "
        f"time={env.time.tolist()} "
        f"level={list(zip(env.level_hi.tolist(), env.level_lo.tolist()))} "
        f"scrolling={env.scrolling.tolist()} "
        f"xscroll={list(zip(env.xscroll_hi.tolist(), env.xscroll_lo.tolist()))}"
    )


def actions(env: SuperMarioBrosNesTurboVecEnv, name: str) -> np.ndarray:
    batch = np.zeros((env.num_envs, len(NES_BUTTONS)), dtype=np.uint8)
    for button in ACTION_BUTTONS[name]:
        batch[:, BUTTON_TO_INDEX[button]] = 1
    return batch


def main() -> None:
    args = parse_args()
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        rom_path=resolve_required_rom_path(args.rom_path),
        num_envs=4,
        use_restricted_actions=Actions.ALL,
        frame_skip=1,
        obs_grayscale=True,
        frame_stack=1,
    )
    obs, _infos = env.reset()
    print(f"actions={ACTION_MEANINGS}")
    print(f"reset_sum={int(obs.sum())} {info_summary(env)}")

    for _ in range(20):
        obs, _rewards, _terminated, _truncated, _infos = env.step(actions(env, "noop"))
    print(
        f"after_noop_sum={int(obs.sum())} "
        f"{info_summary(env)} "
        f"unique_pixels={len(np.unique(obs[0, 0]))}"
    )

    for _ in range(10):
        obs, _rewards, _terminated, _truncated, _infos = env.step(actions(env, "start"))
    for _ in range(60):
        obs, _rewards, _terminated, _truncated, _infos = env.step(actions(env, "noop"))
    print(
        f"after_start_sum={int(obs.sum())} "
        f"{info_summary(env)} "
        f"unique_pixels={len(np.unique(obs[0, 0]))}"
    )


if __name__ == "__main__":
    main()
