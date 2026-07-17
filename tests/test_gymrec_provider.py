from __future__ import annotations

import gymnasium as gym
import numpy as np

from supermariobrosnes_turbo.gymrec_provider import (
    PROVIDER_ID,
    _SingleLaneEnv,
    provider,
)
from supermariobrosnes_turbo.task_contract import DeclarativeTaskEnv


class _ScriptedMarioEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(0, 255, (1, 1, 3), np.uint8)

    def __init__(self, states):
        self.states = list(states)
        self.index = 0

    def _transition(self):
        state = self.states[self.index]
        return np.zeros((1, 1, 3), np.uint8), dict(state)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self._transition()

    def step(self, action):
        self.index += 1
        observation, info = self._transition()
        return observation, 2.0, False, False, info


class _FakeVectorEnv:
    num_envs = 1
    single_action_space = gym.spaces.Discrete(3)
    single_observation_space = gym.spaces.Box(0, 255, (1, 1, 3), np.uint8)

    def reset(self, *, seed=None, options=None):
        return np.zeros((1, 1, 1, 3), np.uint8), {
            "x": np.array([1]),
            "_x": np.array([True]),
        }

    def step(self, actions):
        assert actions.tolist() == [2]
        return (
            np.ones((1, 1, 1, 3), np.uint8),
            np.array([3.5]),
            np.array([True]),
            np.array([False]),
            {"x": np.array([2]), "_x": np.array([True])},
        )

    def render(self):
        return np.ones((1, 1, 3), np.uint8)

    def close(self):
        return None


def _state(x, *, score=0, lives=2, level=(0, 0)):
    return {
        "xscrollHi": x // 256,
        "xscrollLo": x % 256,
        "score": score,
        "lives": lives,
        "levelHi": level[0],
        "levelLo": level[1],
    }


def _task(**overrides):
    task = {
        "events": {
            "life_loss": {"signal": "lives", "operation": "decrease"},
            "level_change": {"signal": "level", "operation": "change"},
            "stalled": {"signal": "x", "operation": "unchanged_for", "steps": 3},
        },
        "termination": {
            "failure": ["life_loss", "stalled"],
            "success": ["level_change"],
            "max_episode_steps": 20,
        },
        "reward": {"reward_mode": "additive", "progress_reward_scale": 1.0},
    }
    task.update(overrides)
    return task


def test_entry_point_provider_identity_is_canonical():
    assert provider.provider_id == PROVIDER_ID == "supermariobrosnes-turbo"
    assert provider.contract_version == 1
    assert provider.catalog() == ("SuperMarioBros-Nes-v0",)


def test_single_lane_adapter_preserves_final_transition_values():
    env = _SingleLaneEnv(_FakeVectorEnv())
    observation, info = env.reset(seed=7)
    assert observation.shape == (1, 1, 3)
    assert info == {"x": 1}

    observation, reward, terminated, truncated, info = env.step(2)
    assert observation.tolist() == [[[1, 1, 1]]]
    assert reward == 3.5
    assert terminated is True
    assert truncated is False
    assert info == {"x": 2}


def test_task_contract_tracks_progress_across_level_completion():
    env = DeclarativeTaskEnv(
        _ScriptedMarioEnv([_state(10), _state(100), _state(0, level=(0, 1))]),
        _task(),
    )
    env.reset(seed=3)
    _obs, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (90.0, False, False)
    assert info["global_max_x_pos"] == 100

    _obs, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["level_complete"] is True
    assert info["task_outcome"] == "success"
    assert info["global_max_x_pos"] == 100


def test_task_contract_owns_life_loss_and_stall_failures():
    life_env = DeclarativeTaskEnv(
        _ScriptedMarioEnv([_state(10), _state(12, lives=1)]),
        _task(),
    )
    life_env.reset()
    _obs, reward, terminated, truncated, info = life_env.step(0)
    assert (reward, terminated, truncated) == (-23.0, True, False)
    assert info["task_outcome"] == "failure"

    stall_env = DeclarativeTaskEnv(
        _ScriptedMarioEnv([_state(10), _state(10), _state(10), _state(10)]),
        _task(),
    )
    stall_env.reset()
    stall_env.step(0)
    stall_env.step(0)
    _obs, _reward, terminated, truncated, info = stall_env.step(0)
    assert (terminated, truncated) == (True, False)
    assert info["task_events"] == ["stalled"]
    assert info["task_outcome"] == "failure"
