"""Shared runtime configuration.

Paths can be redirected without editing source files:

    CONTROLNET_MODEL_DIR           external directory containing model weights
    CONTROLNET_ANNOTATOR_CKPT_DIR  external directory containing detector weights
    CONTROLNET_TRAINING_DIR        training dataset root
    CONTROLNET_SAVE_MEMORY         1/true/yes/on enables sliced attention
"""

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ROOT_DIR / "configs"


def _env_path(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MODEL_DIR = _env_path("CONTROLNET_MODEL_DIR", ROOT_DIR / "models")
ANNOTATOR_CKPT_DIR = _env_path(
    "CONTROLNET_ANNOTATOR_CKPT_DIR",
    ROOT_DIR / "src" / "annotator" / "ckpts",
)
TRAINING_DIR = _env_path("CONTROLNET_TRAINING_DIR", ROOT_DIR / "training")

save_memory = _env_flag("CONTROLNET_SAVE_MEMORY", False)


def config_path(name):
    return str(CONFIG_DIR / name)


def model_path(name):
    return str(MODEL_DIR / name)


def annotator_ckpt_path(name):
    return str(ANNOTATOR_CKPT_DIR / name)


def training_path(*parts):
    return str(TRAINING_DIR.joinpath(*parts))
