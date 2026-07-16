from importlib.metadata import PackageNotFoundError, version

from .env import (
    ACTION_SETS,
    ACTION_MEANINGS,
    ACTION_BUTTONS,
    BUTTON_TO_INDEX,
    CORE_ACTION_MEANINGS,
    INFO_KEYS,
    NES_BUTTONS,
    Actions,
    Integrations,
    Observations,
    State,
    SuperMarioBrosNesTurboVecEnv,
    action_batch,
    action_mask,
    default_rom_path,
    list_available_states,
    resolve_required_rom_path,
)
from .roms import RETRO_DATA_PATH_ENV_VAR

try:
    __version__ = version("supermariobrosnes-turbo")
except PackageNotFoundError:  # Source tree imported without an installed distribution.
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "ACTION_SETS",
    "ACTION_MEANINGS",
    "ACTION_BUTTONS",
    "BUTTON_TO_INDEX",
    "CORE_ACTION_MEANINGS",
    "INFO_KEYS",
    "NES_BUTTONS",
    "Actions",
    "Integrations",
    "Observations",
    "RETRO_DATA_PATH_ENV_VAR",
    "State",
    "SuperMarioBrosNesTurboVecEnv",
    "action_batch",
    "action_mask",
    "default_rom_path",
    "list_available_states",
    "resolve_required_rom_path",
]
