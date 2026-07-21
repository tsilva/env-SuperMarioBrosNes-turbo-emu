#!/usr/bin/env python3
"""Diagnostic-only paired benchmark for opt-in research infos.

This helper is deliberately outside autoresearch acceptance. It compares two
filters on one installed candidate build and must never be recorded as a
canonical throughput result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from supermariobrosnes_turbo import (
    EXTRA_INFO_KEYS,
    INFO_KEYS,
    Actions,
    NES_BUTTONS,
    SuperMarioBrosNesTurboVecEnv,
)


def make_env(args: argparse.Namespace, keys: tuple[str, ...]):
    return SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state=args.state,
        rom_path=args.rom,
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        use_restricted_actions=Actions.ALL,
        frame_skip=args.frame_skip,
        frame_stack=4,
        obs_grayscale=True,
        obs_crop=(32, 0, 0, 0),
        obs_crop_mode="mask",
        obs_resize=(84, 84),
        obs_resize_algorithm="area",
        obs_layout="chw",
        info_filter={"mode": "all", "keys": keys},
    )


def reset_terminal_lanes(env, terminated: np.ndarray, truncated: np.ndarray) -> None:
    reset_mask = terminated | truncated
    if not bool(np.any(reset_mask)):
        return
    state_indices = np.full(env.num_envs, -1, dtype=np.int32)
    state_indices[reset_mask] = 0
    env.reset(options={"reset_mask": reset_mask.copy(), "state_indices": state_indices})


def validate_equal_trajectories(legacy, extras, actions: np.ndarray, seed: int) -> None:
    legacy_obs, _ = legacy.reset(seed=seed)
    extras_obs, _ = extras.reset(seed=seed)
    np.testing.assert_array_equal(legacy_obs, extras_obs)
    for action in actions:
        legacy_step = legacy.step(action)
        extras_step = extras.step(action)
        for expected, actual in zip(legacy_step[:4], extras_step[:4]):
            np.testing.assert_array_equal(expected, actual)
        reset_terminal_lanes(legacy, legacy_step[2], legacy_step[3])
        reset_terminal_lanes(extras, extras_step[2], extras_step[3])


def measure(env, actions: np.ndarray, seed: int) -> float:
    env.reset(seed=seed)
    started = time.perf_counter()
    for action in actions:
        _obs, _rewards, terminated, truncated, _infos = env.step(action)
        reset_terminal_lanes(env, terminated, truncated)
    elapsed = time.perf_counter() - started
    return actions.shape[0] * env.num_envs / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.trials <= 0:
        parser.error("--num-envs, --steps, and --trials must be positive")

    rng = np.random.default_rng(args.seed)
    actions = rng.integers(
        0,
        2,
        size=(args.steps, args.num_envs, len(NES_BUTTONS)),
        dtype=np.uint8,
    )
    legacy = make_env(args, INFO_KEYS)
    extras = make_env(args, EXTRA_INFO_KEYS)
    try:
        validation_steps = min(args.steps, 100)
        validate_equal_trajectories(
            legacy,
            extras,
            actions[:validation_steps],
            args.seed,
        )
        pairs: list[dict[str, float]] = []
        for trial in range(args.trials):
            trial_seed = args.seed + trial
            if trial % 2 == 0:
                legacy_sps = measure(legacy, actions, trial_seed)
                extras_sps = measure(extras, actions, trial_seed)
            else:
                extras_sps = measure(extras, actions, trial_seed)
                legacy_sps = measure(legacy, actions, trial_seed)
            pairs.append(
                {
                    "legacy_sps": legacy_sps,
                    "extras_sps": extras_sps,
                    "extras_to_legacy_ratio": extras_sps / legacy_sps,
                }
            )
        payload = {
            "benchmark": "diagnostic_info_filter_only",
            "autoresearch_eligible": False,
            "config": {
                "state": args.state,
                "num_envs": args.num_envs,
                "num_threads": args.num_threads,
                "frame_skip": args.frame_skip,
                "steps": args.steps,
                "trials": args.trials,
                "seed": args.seed,
            },
            "trajectory_validation_steps": validation_steps,
            "pairs": pairs,
            "median_extras_to_legacy_ratio": statistics.median(
                pair["extras_to_legacy_ratio"] for pair in pairs
            ),
        }
        print(json.dumps(payload, indent=2))
    finally:
        legacy.close()
        extras.close()


if __name__ == "__main__":
    main()
