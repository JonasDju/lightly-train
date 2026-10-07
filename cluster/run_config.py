"""Loading and validating a cluster run config (see ``pretrain_dinov2_kneeno.py``).

Imported by the scripts as ``from run_config import ...``: running ``python cluster/<script>.py`` puts this folder
first on ``sys.path``. Importing this module imports numpy (through ``kneeno``), so a script that has to set
``OPENBLAS_NUM_THREADS`` must do so before importing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from kneeno import expand_env_vars

RUN_CONFIG_KEYS = ("model", "data", "train", "method", "transform", "eval")
BLOCKS = ("data", "train", "method", "transform", "eval")  # the keys whose value is a mapping
# Named exactly like lightly_train.pretrain's arguments, which they are passed to as-is: the output directory and
# the image folder lightly-train searches for images itself (the JPEG slices of the unlabeled KneeNo dataset).
DATA_KEYS = ("out", "data")
# The optional train block: the other lightly_train.pretrain arguments a run may set, also passed as-is. Everything
# the script derives itself (strategy, num_nodes, resume_interrupted, callbacks, params_file) is deliberately absent.
TRAIN_KEYS = (
    "epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "num_workers",
    "precision",
    "float32_matmul_precision",
    "seed",
    "checkpoint",
    "optim",
    "optim_args",
    "loader_args",
    "trainer_args",
    "model_args",
    "activation_checkpoint_args",
)


def load_run_config(path: Path) -> dict[str, Any]:
    """A run config with env vars expanded; missing or null optional blocks are left out.

    Unknown top-level, ``data`` or ``train`` keys are an error, so a typo such as ``methods:`` cannot silently fall
    back to the defaults. So is a missing ``model``, ``data`` block or ``data`` key.
    """
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Run config '{path}' must be a mapping, got {type(raw).__name__}.")
    unknown = set(raw) - set(RUN_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Run config '{path}' has unknown top-level keys {sorted(unknown)}; valid: {RUN_CONFIG_KEYS}.")
    config = expand_env_vars({key: raw[key] for key in RUN_CONFIG_KEYS if raw.get(key) is not None})

    if not isinstance(config.get("model"), str):
        raise ValueError(f"Run config '{path}' needs 'model' as a string, e.g. 'model: dinov2/vitb14-notpretrained'.")
    for name in BLOCKS:
        if name in config and not isinstance(config[name], dict):
            raise ValueError(f"'{name}' in run config '{path}' must be a mapping, got {type(config[name]).__name__}.")

    data = config.get("data")
    if data is None:
        raise ValueError(f"Run config '{path}' has no 'data' block; it needs {DATA_KEYS}.")
    unknown = set(data) - set(DATA_KEYS)
    if unknown:
        raise ValueError(f"'data' in run config '{path}' has unknown keys {sorted(unknown)}; valid: {DATA_KEYS}.")
    missing = [key for key in DATA_KEYS if data.get(key) is None]
    if missing:
        raise ValueError(f"'data' in run config '{path}' is missing required keys {missing}.")

    unknown = set(config.get("train", {})) - set(TRAIN_KEYS)
    if unknown:
        raise ValueError(f"'train' in run config '{path}' has unknown keys {sorted(unknown)}; valid: {TRAIN_KEYS}.")
    return config
