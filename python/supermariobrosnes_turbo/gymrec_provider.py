"""gymrec provider integration owned by supermariobrosnes-turbo."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
from collections.abc import Mapping, Sequence

import gymnasium as gym
import numpy as np

from .env import (
    NES_BUTTONS,
    SuperMarioBrosNesTurboVecEnv,
    _resolve_state_path,
    action_mask,
)
from .roms import resolve_required_rom_path
from .task_contract import DeclarativeTaskEnv


PROVIDER_ID = "supermariobrosnes-turbo"
CONTRACT_VERSION = 1


def _lane_info(infos):
    result = {}
    for key, value in (infos or {}).items():
        if str(key).startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask).reshape(-1)[0]):
            continue
        if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
            value = value[0]
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and len(value) == 1
        ):
            value = value[0]
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


class _SingleLaneEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, vector_env):
        if vector_env.num_envs != 1:
            raise ValueError("gymrec providers require exactly one environment lane")
        self.vector_env = vector_env
        self.action_space = vector_env.single_action_space
        self.observation_space = vector_env.single_observation_space
        self.render_mode = "rgb_array"
        self.system = "Nes"
        self.buttons = tuple(NES_BUTTONS)
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, infos = self.vector_env.reset(seed=seed, options=options)
        self._needs_reset = False
        return observations[0], _lane_info(infos)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError(
                "reset() must be called before step() or after a terminal step"
            )
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = int(np.asarray(action).reshape(-1)[0])
            if not self.action_space.contains(action):
                raise ValueError(f"Action {action!r} is not in {self.action_space}")
            batched = np.asarray([action], dtype=self.action_space.dtype)
        else:
            action = np.asarray(action, dtype=self.action_space.dtype)
            if not self.action_space.contains(action):
                raise ValueError(f"Action {action!r} is not in {self.action_space}")
            batched = action[np.newaxis, ...]
        observations, rewards, terminated, truncated, infos = self.vector_env.step(
            batched
        )
        is_terminated = bool(terminated[0])
        is_truncated = bool(truncated[0])
        self._needs_reset = is_terminated or is_truncated
        return (
            observations[0],
            float(rewards[0]),
            is_terminated,
            is_truncated,
            _lane_info(infos),
        )

    def render(self):
        return self.vector_env.render()

    def close(self):
        self.vector_env.close()


def _file_sha256(path):
    try:
        with open(path, "rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except (FileNotFoundError, OSError, TypeError):
        return None


def _state_asset_hashes(state, state_dir):
    if state is None:
        return {}
    if isinstance(state, Mapping):
        values = state.keys()
    elif isinstance(state, Sequence) and not isinstance(
        state, (str, bytes, bytearray, memoryview)
    ):
        values = state
    else:
        values = (state,)
    hashes = {}
    for value in values:
        if isinstance(value, (bytes, bytearray, memoryview)):
            digest = hashlib.sha256(bytes(value)).hexdigest()
            label = f"inline-{digest[:12]}"
        else:
            path = _resolve_state_path(value, state_dir)
            digest = _file_sha256(path)
            label = str(value)
        hashes[label] = digest
    return hashes


class SuperMarioBrosNesGymrecSession:
    provider_id = PROVIDER_ID

    def __init__(self, environment_id, config, render_mode):
        self.environment_id = environment_id
        self.declared_config = copy.deepcopy(dict(config))
        kwargs = copy.deepcopy(dict(config))
        task = kwargs.pop("task", None)
        if task is not None and not isinstance(task, Mapping):
            raise ValueError("config.task must be an object")
        for reserved in ("game", "num_envs", "num_threads", "render_mode"):
            if reserved in kwargs:
                raise ValueError(f"config.{reserved} is managed by the provider")
        state = kwargs.pop("state", "Level1-1")
        state_dir = kwargs.get("state_dir")
        action = (task or {}).get("action") or {}
        if "action_set" in kwargs and action.get("set") not in (
            None,
            kwargs["action_set"],
        ):
            raise ValueError("config.action_set conflicts with config.task.action.set")
        if "action_set" not in kwargs and action.get("set") not in (None, "native"):
            kwargs["action_set"] = action["set"]
        rom_path = resolve_required_rom_path(kwargs.get("rom_path"), environment_id)
        vector_env = SuperMarioBrosNesTurboVecEnv(
            environment_id,
            state=state,
            render_mode="rgb_array",
            num_envs=1,
            num_threads=1,
            **kwargs,
        )
        base_env = _SingleLaneEnv(vector_env)
        try:
            self.env = DeclarativeTaskEnv(base_env, task) if task else base_env
        except Exception:
            base_env.close()
            raise
        self.control_profile = "stable_retro.Nes"
        self.fps = max(60.0 / max(int(kwargs.get("frame_skip", 1)), 1), 1.0)
        self.effective_config = copy.deepcopy(dict(config))
        self.provenance = {
            "distribution": PROVIDER_ID,
            "version": importlib.metadata.version(PROVIDER_ID),
            "assets": {
                "rom_sha256": _file_sha256(rom_path),
                "state_sha256": _state_asset_hashes(state, state_dir),
            },
        }

    def policy_observation(self, observation):
        return observation

    def recording_observation(self, observation):
        frame = self.env.render()
        return observation if frame is None else frame

    def adapt_policy_action(self, action):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return int(np.asarray(action).reshape(-1)[0])
        return action

    def validate_policy(self, policy):
        policy_action = getattr(policy, "action_space", None)
        if policy_action is not None and getattr(policy_action, "n", None) != getattr(
            self.env.action_space, "n", None
        ):
            raise ValueError(
                "Policy action space does not match the provider action set"
            )
        policy_observation = getattr(policy, "observation_space", None)
        if policy_observation is not None and getattr(
            policy_observation, "shape", None
        ) != getattr(self.env.observation_space, "shape", None):
            raise ValueError("Policy observation space does not match the provider")

    def action_from_labels(self, labels):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            requested = {str(label).upper() for label in labels}
            meanings = getattr(self.env.unwrapped.vector_env, "action_meanings", ())
            for index, meaning in enumerate(meanings):
                mask = action_mask(meaning)
                actual = {
                    str(NES_BUTTONS[pos]).upper()
                    for pos in np.flatnonzero(mask)
                    if NES_BUTTONS[pos]
                }
                if actual == requested:
                    return index
            raise ValueError(
                f"No configured action matches controls {sorted(requested)!r}"
            )
        action = np.zeros(self.env.action_space.n, dtype=self.env.action_space.dtype)
        buttons = tuple(str(button).upper() if button else "" for button in NES_BUTTONS)
        for label in labels:
            try:
                action[buttons.index(str(label).upper())] = 1
            except ValueError as exc:
                raise ValueError(f"Control label {label!r} is unavailable") from exc
        return action


class SuperMarioBrosNesGymrecProvider:
    provider_id = PROVIDER_ID
    contract_version = CONTRACT_VERSION

    def create(self, *, environment_id, config, render_mode):
        return SuperMarioBrosNesGymrecSession(environment_id, config, render_mode)

    def catalog(self):
        return ("SuperMarioBros-Nes-v0",)


provider = SuperMarioBrosNesGymrecProvider()
