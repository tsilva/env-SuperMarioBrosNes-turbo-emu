from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("supermariobrosnes-turbo")
except PackageNotFoundError:  # Source tree imported without an installed distribution.
    __version__ = "0+unknown"

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
