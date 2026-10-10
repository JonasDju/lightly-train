#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Standalone post-pretraining KneeNo classification evaluation.

Loads a frozen encoder from a LightlyTrain pretraining checkpoint and hands it to
``kneeno.evaluation.ClassificationEvaluator``, with ``log_every_head_epoch=True`` so the
*whole* head fine-tuning curve is produced -- unlike the in-training callback
(``_callbacks/kneeno_eval.py``), which logs one point per pretraining epoch.

The counterpart of ``vjepa2/app/vjepa_2_1/eval_classification.py``, but much shorter: a
LightlyTrain checkpoint stores the model objects themselves, so there is no architecture
to rebuild from a config. Ported from branch ``3d_dinov2``; on this branch the 2D adapter
needs the input channels and the 2D crop size instead of 3D resampling settings.

Everything about the evaluation comes from one YAML file (``cluster/configs/eval-*.yaml``)
with exactly three top-level keys:

    checkpoint:  the LightlyTrain checkpoint to evaluate
    image_size:  [H, W], the crop size the model was pretrained with (not in the checkpoint)
    eval:        KneeNo's evaluation config, deep-merged over its DEFAULT_EVAL_CONFIG

The tasks are the ones whose ``eval.freq`` is set (not null and > 0, as for
``kneeno.tasks_due``); the encoder is ``eval.encoder``. Environment variables are expanded
in every string value. The outputs go next to ``eval.logging.per_label_dir``: the final
metrics as ``results.json`` and the expanded config as ``params.yaml``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from kneeno import expand_env_vars
from kneeno.evaluation import ClassificationEvaluator, load_eval_config
from kneeno.evaluation.config import ALL_TASKS
from omegaconf import DictConfig
from pydantic import ValidationError
from torch.nn import Module

from lightly_train import _logging
from lightly_train._checkpoint import Checkpoint
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs import omegaconf_utils, validate
from lightly_train._configs.config import PydanticConfig
from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train.types import ImageSizeTuple, PathLike

logger = logging.getLogger(__name__)

_STUDENT_PREFIX = "student_embedding_model.wrapped_model."
# What os.path.expandvars expands when the variable is set, i.e. what is left of an unset one.
_ENV_VAR_PATTERN = re.compile(r"\$(\{\w+\}|[A-Za-z_]\w*)")


def eval_classification(
    *,
    eval_config: PathLike,
    accelerator: str = "auto",
    overwrite: bool = False,
) -> None:
    """Evaluate a pretrained model with KneeNo's frozen-encoder classification tasks.

    Args:
        eval_config:
            Path to a YAML file with exactly the top-level keys ``checkpoint`` (the
            LightlyTrain checkpoint to evaluate), ``image_size`` (``[H, W]``, the crop
            size the model was pretrained with; every slice is resized to it) and
            ``eval`` (KneeNo's evaluation config). The tasks are those with a non-null
            ``eval.freq``, the encoder is ``eval.encoder`` ('target' or 'online').
            ``results.json`` and ``params.yaml`` are written next to
            ``eval.logging.per_label_dir``.
        accelerator:
            Hardware accelerator, e.g. 'cpu', 'gpu' or 'auto'.
        overwrite:
            Run even if results of an earlier evaluation exist: ``results.json`` and
            ``params.yaml`` are replaced, the per-label CSVs are appended to and
            TensorBoard gets a second event file.
    """
    config = EvalClassificationConfig(**locals())
    eval_classification_from_config(config=config)


def eval_classification_from_config(config: EvalClassificationConfig) -> None:
    _warnings.filter_eval_classification_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")

    raw_config, run_config = _load_eval_run_config(Path(config.eval_config))
    eval_cfg = load_eval_config(run_config.eval)
    tasks = _enabled_tasks(eval_cfg["freq"])
    encoder_choice = eval_cfg.get("encoder", "target")
    if encoder_choice not in ("target", "online"):
        raise ValueError(
            f"Invalid eval.encoder: '{encoder_choice}'. Valid encoders are: "
            "['target', 'online']"
        )
    ckpt_path = common_helpers.get_checkpoint_path(checkpoint=run_config.checkpoint)

    # Everything that can fail before the (possibly hours long) evaluation does so now.
    log_cfg = eval_cfg["logging"]
    results_path, params_path = _get_output_paths(
        per_label_dir=log_cfg["per_label_dir"], ckpt_path=ckpt_path
    )
    results_path = common_helpers.get_out_path(
        out=results_path, overwrite=config.overwrite
    )
    _check_no_previous_results(
        paths=[params_path, log_cfg["per_label_dir"], log_cfg["tensorboard_dir"]],
        overwrite=config.overwrite,
    )
    if params_path is not None:
        # Written before evaluating, so a run that crashes still records its settings.
        params_path.parent.mkdir(parents=True, exist_ok=True)
        params_path.write_text(yaml.safe_dump(raw_config, sort_keys=False))
        logger.info(f"Wrote the evaluation config to '{params_path}'")

    logger.info(f"Loading checkpoint from '{ckpt_path}'")
    ckpt = Checkpoint.from_path(checkpoint=ckpt_path)
    wrapped_model = ckpt.lightly_train.models.wrapped_model
    if encoder_choice == "online":
        _load_student_weights(wrapped_model=wrapped_model, ckpt=ckpt)
    logger.info(f"Evaluating the '{encoder_choice}' encoder on tasks {tasks}.")

    device = _get_device(accelerator=config.accelerator)
    # ModelWrapper is a protocol; every concrete wrapper is also an nn.Module.
    assert isinstance(wrapped_model, Module)
    wrapped_model.requires_grad_(False)
    wrapped_model.eval()
    wrapped_model.to(device)

    dataset_type = eval_cfg.get("data").get("dataset_type")
    normalize_args = ckpt.lightly_train.normalize_args
    adapter = DINOv2Adapter(
        dataset_type=dataset_type,
        embed_dim=wrapped_model.feature_dim(),
        # The checkpoint stores the resolved normalize args, whose length the transform
        # matched to the resolved num_channels (TransformArgs.resolve_incompatible), so
        # this is the channel count the model was trained with, "auto" included.
        num_channels=len(normalize_args.mean),
        image_size=run_config.image_size,
        normalize=(normalize_args.mean, normalize_args.std),
    )

    evaluator = ClassificationEvaluator(config=eval_cfg, adapter=adapter, device=device)
    try:
        # log_every_head_epoch=True: the point of the standalone run is the head
        # fine-tuning curve itself, not one summary point per pretraining epoch.
        metrics = evaluator.evaluate(
            wrapped_model, tasks=tasks, epoch=0, log_every_head_epoch=True
        )
    finally:
        # No-op unless the config sets logging.tensorboard_dir, but flushing a writer
        # that does exist matters if the process exits right after.
        evaluator.cleanup()

    logger.info(f"Final metrics: {metrics}")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info(f"Wrote metrics to '{results_path}'")


def eval_classification_from_dictconfig(config: DictConfig) -> None:
    logger.debug(f"Evaluating model with config: {config}")
    config_dict = omegaconf_utils.config_to_dict(config=config)
    eval_cfg = validate.pydantic_model_validate(EvalClassificationConfig, config_dict)
    eval_classification_from_config(config=eval_cfg)


class EvalClassificationConfig(PydanticConfig):
    eval_config: PathLike
    accelerator: str = "auto"
    overwrite: bool = False


class CLIEvalClassificationConfig(EvalClassificationConfig):
    eval_config: str


class EvalRunConfig(PydanticConfig):
    """The YAML file passed as ``eval_config``. Unknown keys are an error, so a typo or a
    leftover pretraining block (e.g. ``transform:``) cannot be silently ignored."""

    checkpoint: PathLike
    image_size: ImageSizeTuple
    eval: dict[str, Any]


def _load_eval_run_config(path: Path) -> tuple[dict[str, Any], EvalRunConfig]:
    """The eval config file with env vars expanded, as a dict and validated.

    The dict keeps the file's key order and is what ``params.yaml`` records.
    """
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("eval"), dict):
        raise ValueError(
            f"Eval config '{path}' has no top-level 'eval:' block. It needs the keys "
            f"{list(EvalRunConfig.model_fields)}."
        )
    raw = expand_env_vars(raw)
    # expand_env_vars leaves unset variables as they are; a path with a literal
    # '${PRETRAIN_JOB_ID}' in it would point to a nonexistent checkpoint, or create
    # such a directory for the outputs.
    unexpanded = sorted(set(_find_env_vars(raw)))
    if unexpanded:
        raise ValueError(
            f"Eval config '{path}' uses environment variables that are not set: "
            f"{unexpanded}."
        )
    try:
        run_config = EvalRunConfig.model_validate(raw)
    except ValidationError as ex:
        raise ValueError(f"Invalid eval config '{path}':\n{ex}") from ex
    return raw, run_config


def _find_env_vars(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [var for value in obj.values() for var in _find_env_vars(value)]
    if isinstance(obj, (list, tuple)):
        return [var for value in obj for var in _find_env_vars(value)]
    if isinstance(obj, str):
        return [match.group(0) for match in _ENV_VAR_PATTERN.finditer(obj)]
    return []


def _enabled_tasks(freq: dict[str, Any]) -> list[str]:
    """The tasks whose frequency is set, with ``kneeno.tasks_due``'s semantics: a missing
    key keeps KneeNo's default (``freq`` is the merged config), null or ``<= 0`` is off.
    The value itself does not matter here, there is only one evaluation."""
    tasks = []
    for task in ALL_TASKS:
        task_freq = freq.get(task)
        if task_freq is not None and task_freq > 0:
            tasks.append(task)
    if not tasks:
        raise ValueError(
            f"No task is enabled: every task in eval.freq is null or <= 0 ({freq})."
        )
    return tasks


def _get_output_paths(
    per_label_dir: str | None, ckpt_path: Path
) -> tuple[Path, Path | None]:
    """``results.json`` and ``params.yaml`` in the parent of ``per_label_dir``.

    Without a ``per_label_dir`` there is no evaluation folder: ``results.json`` goes next
    to the checkpoint and no ``params.yaml`` is written.
    """
    if per_label_dir is None:
        results_path = ckpt_path.parent / "results.json"
        logger.warning(
            "eval.logging.per_label_dir is not set in the eval config. Writing the "
            f"metrics to '{results_path}' and not saving the evaluation config."
        )
        return results_path, None
    out_dir = Path(per_label_dir).resolve().parent
    return out_dir / "results.json", out_dir / "params.yaml"


def _check_no_previous_results(paths: list[PathLike | None], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path is not None and Path(path).exists()]
    if not existing:
        return
    if not overwrite:
        raise ValueError(
            f"Results of an earlier evaluation exist: {existing}. Delete them, point "
            "eval.logging somewhere else, or set overwrite=True."
        )
    logger.warning(
        f"Overwriting results of an earlier evaluation: {existing}. The per-label CSVs "
        "are appended to, and TensorBoard gets a second event file."
    )


def _get_device(accelerator: str) -> torch.device:
    resolved = common_helpers.get_accelerator(accelerator=accelerator)
    name = resolved if isinstance(resolved, str) else type(resolved).__name__.lower()
    if "cuda" in name or name == "gpu":
        return torch.device("cuda")
    if "mps" in name:
        return torch.device("mps")
    return torch.device("cpu")


def _load_student_weights(wrapped_model: ModelWrapper, ckpt: Checkpoint) -> None:
    """Overwrite the checkpoint's (teacher) wrapper with the student's weights.

    ``Checkpoint.lightly_train.models.wrapped_model`` is the teacher: ``DINOv2.__init__``
    assigns the ``embedding_model`` it is handed to ``self.teacher_embedding_model`` and
    EMA-updates it in place, and that is the object the checkpoint callback stores. The
    student only exists inside the method's ``state_dict``, under its own prefix.
    """
    state_dict = {
        key[len(_STUDENT_PREFIX) :]: value
        for key, value in ckpt.state_dict.items()
        if key.startswith(_STUDENT_PREFIX)
    }
    if not state_dict:
        raise ValueError(
            f"encoder='online' requires student weights, but no '{_STUDENT_PREFIX}*' "
            "keys are present in the checkpoint's state_dict. Was it produced by a "
            "method with a student/teacher pair (e.g. dinov2)?"
        )
    wrapped_model.load_state_dict(state_dict)
