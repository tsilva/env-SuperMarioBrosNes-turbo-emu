from importlib.metadata import PackageNotFoundError, version
from typing import Any

import gymnasium as gym

from .action_tables import ActionTable

from .env import (
    ACTION_TABLES,
    ACTION_SETS,
    ACTION_MEANINGS,
    ACTION_BUTTONS,
    AVAILABLE_INFO_KEYS,
    BUTTON_TO_INDEX,
    CORE_ACTION_MEANINGS,
    EXTRA_INFO_KEYS,
    INFO_KEYS,
    NES_BUTTONS,
    Actions,
    AreaType,
    Direction,
    GameMode,
    Integrations,
    Observations,
    PlayerMotion,
    PlayerPower,
    PlayerTask,
    State,
    SuperMarioBrosNesTurboVecEnv,
    action_batch,
    action_mask,
    list_available_states,
    resolve_required_rom_path,
)
from .roms import RETRO_DATA_PATH_ENV_VAR, default_rom_path

GYMNASIUM_ENV_ID = "SuperMarioBros-Nes-Turbo-v0"
_GYMNASIUM_VECTOR_ENTRY_POINT = "supermariobrosnes_turbo:_make_gymnasium_vec_env"

try:
    __version__ = version("env-supermariobrosnes-turbo-emu")
except PackageNotFoundError:  # Source tree imported without an installed distribution.
    __version__ = "0+unknown"


def _make_gymnasium_vec_env(
    *, game: str, num_envs: int = 1, **kwargs: Any
) -> SuperMarioBrosNesTurboVecEnv:
    return SuperMarioBrosNesTurboVecEnv(
        game=game,
        num_envs=num_envs,
        **kwargs,
    )


def _register_gymnasium_env() -> None:
    existing = gym.registry.get(GYMNASIUM_ENV_ID)
    if existing is None:
        gym.register(
            id=GYMNASIUM_ENV_ID,
            entry_point=None,
            vector_entry_point=_GYMNASIUM_VECTOR_ENTRY_POINT,
        )
        return
    if (
        existing.entry_point is None
        and existing.vector_entry_point == _GYMNASIUM_VECTOR_ENTRY_POINT
        and existing.kwargs == {}
        and existing.max_episode_steps is None
        and existing.additional_wrappers == ()
    ):
        return
    raise gym.error.Error(
        f"Gymnasium environment ID {GYMNASIUM_ENV_ID!r} is already registered "
        "with a conflicting specification"
    )


_register_gymnasium_env()

__all__ = [
    "__version__",
    "ACTION_TABLES",
    "ACTION_SETS",
    "ACTION_MEANINGS",
    "ACTION_BUTTONS",
    "AVAILABLE_INFO_KEYS",
    "BUTTON_TO_INDEX",
    "CORE_ACTION_MEANINGS",
    "EXTRA_INFO_KEYS",
    "INFO_KEYS",
    "GYMNASIUM_ENV_ID",
    "NES_BUTTONS",
    "Actions",
    "AreaType",
    "ActionTable",
    "Direction",
    "GameMode",
    "Integrations",
    "Observations",
    "PlayerMotion",
    "PlayerPower",
    "PlayerTask",
    "RETRO_DATA_PATH_ENV_VAR",
    "State",
    "SuperMarioBrosNesTurboVecEnv",
    "action_batch",
    "action_mask",
    "default_rom_path",
    "list_available_states",
    "resolve_required_rom_path",
]
