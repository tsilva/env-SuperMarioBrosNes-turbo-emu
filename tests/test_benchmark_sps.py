from __future__ import annotations

from functools import partial
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import numpy as np
import pytest

from scripts import benchmark_sps
from scripts.benchmark_workload import canonical_env_args


class FakeScalarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, terminal_after: int = 2) -> None:
        self.action_space = spaces.MultiBinary(1)
        self.observation_space = spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8)
        self.terminal_after = terminal_after
        self.counter = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 0
        return np.full((4, 4, 3), 10, dtype=np.uint8), {"reset": True}

    def step(self, action):
        del action
        self.counter += 1
        observation = np.full((4, 4, 3), 10 + self.counter, dtype=np.uint8)
        return observation, float(self.counter), self.counter >= self.terminal_after, False, {
            "counter": self.counter
        }

    def snapshot(self) -> int:
        return self.counter


def make_fake_scalar_env(terminal_after: int) -> FakeScalarEnv:
    return FakeScalarEnv(terminal_after)


def canonical_preprocessing(**overrides: object) -> benchmark_sps.PreprocessingConfig:
    values = {
        "frame_skip": 4,
        "frame_stack": 4,
        "grayscale": True,
        "crop_top": 32,
        "crop_bottom": 0,
        "crop_mode": "mask",
        "resize_width": 84,
        "resize_height": 84,
    }
    values.update(overrides)
    return benchmark_sps.PreprocessingConfig(**values)


def test_default_backend_and_preprocessing_contract() -> None:
    args = benchmark_sps.parse_args([])

    assert benchmark_sps.backend_for_args(args) == "turbo"
    assert benchmark_sps.preprocessing_for_args(args) == canonical_preprocessing()

    baseline = benchmark_sps.parse_args(["--stable-retro-baseline"])
    assert benchmark_sps.backend_for_args(baseline) == "stable-retro"


def test_stable_retro_profile_is_rejected() -> None:
    args = benchmark_sps.parse_args(
        ["--stable-retro-baseline", "--profile-output", "profile.json"]
    )

    with pytest.raises(ValueError, match="only available for the turbo backend"):
        benchmark_sps.validate_args(args)


def test_canonical_cli_contains_only_supported_termination_flags() -> None:
    args = canonical_env_args()

    assert "--terminate-on-life-loss" not in args
    assert "--terminate-on-level-change" not in args
    benchmark_sps.validate_args(benchmark_sps.parse_args(args))


def test_package_metadata_tracks_selected_backend() -> None:
    assert benchmark_sps.package_metadata("turbo")["name"] == "supermariobrosnes-turbo"
    baseline = benchmark_sps.package_metadata("stable-retro")
    assert baseline["name"] == "stable-retro-turbo"
    assert baseline["import"] == "stable_retro"
    if sys.version_info[:2] == (3, 14):
        assert baseline["version"] == "1.0.1.post34"


def test_benchmark_module_does_not_eagerly_import_candidate_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scripts.benchmark_sps; "
                "assert 'supermariobrosnes_turbo' not in sys.modules"
            ),
        ],
        cwd=root,
        check=True,
    )


def test_dependency_uses_stable_retro_turbo_release() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    lockfile = (root / "uv.lock").read_text()

    dependency = "stable-retro-turbo==1.0.1.post34; python_version == '3.14'"
    assert f'"{dependency}"' in pyproject
    assert '\nname = "stable-retro-turbo"\n' in lockfile
    assert '\nname = "stable-retro"\n' not in lockfile


def test_mask_grayscale_integer_area_resize_and_chw() -> None:
    y, x = np.indices((4, 4), dtype=np.uint8)
    rgb = np.stack((x * 10, y * 20, (x + y) * 5), axis=-1)
    config = benchmark_sps.PreprocessingConfig(1, 1, True, 1, 0, "mask", 2, 2)

    actual = benchmark_sps.preprocess_frame(rgb, config)

    masked = rgb.copy()
    masked[0] = 0
    wide = masked.astype(np.uint32)
    gray = ((77 * wide[..., 0] + 150 * wide[..., 1] + 29 * wide[..., 2] + 128) >> 8).astype(
        np.uint8
    )
    expected = np.array(
        [
            [gray[:2, :2].sum() // 4, gray[:2, 2:].sum() // 4],
            [gray[2:, :2].sum() // 4, gray[2:, 2:].sum() // 4],
        ],
        dtype=np.uint8,
    )[None]
    np.testing.assert_array_equal(actual, expected)


def test_reset_padding_and_frame_stack_shift() -> None:
    wrapped = benchmark_sps.StableRetroPreprocessingEnv(
        FakeScalarEnv(terminal_after=10),
        benchmark_sps.PreprocessingConfig(1, 4, True, 0, 0, "remove", 4, 4),
    )

    reset_obs, _ = wrapped.reset()
    assert reset_obs.shape == (4, 4, 4)
    for channel in reset_obs:
        np.testing.assert_array_equal(channel, reset_obs[0])

    next_obs, *_ = wrapped.step(np.zeros(1, dtype=np.uint8))
    np.testing.assert_array_equal(next_obs[:3], reset_obs[1:])
    assert np.all(next_obs[3] == 11)


def test_async_vector_spawn_disabled_autoreset_and_masked_reset() -> None:
    env = AsyncVectorEnv(
        [partial(make_fake_scalar_env, 1), partial(make_fake_scalar_env, 3)],
        shared_memory=True,
        copy=True,
        context="spawn",
        autoreset_mode=AutoresetMode.DISABLED,
    )
    actions = np.zeros((2, 1), dtype=np.uint8)
    try:
        env.reset()
        benchmark_sps.step_env(env, actions)
        assert env.call("snapshot") == (0, 1)

        run = benchmark_sps.run_once(
            env,
            (actions, actions, actions),
            SimpleNamespace(steps=3, num_envs=2, frame_skip=4),
        )
        assert run["env_steps_per_sec"] == pytest.approx(run["batch_steps_per_sec"] * 2)
        assert run["emulated_frames_per_sec"] == pytest.approx(run["env_steps_per_sec"] * 4)
    finally:
        env.close()
