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
to rebuild from a config.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from kneeno.evaluation import ClassificationEvaluator, load_eval_config
from omegaconf import DictConfig
from torch.nn import Module

from lightly_train import _logging
from lightly_train._checkpoint import Checkpoint
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs import omegaconf_utils, validate
from lightly_train._configs.config import PydanticConfig
from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._methods.dinov2.dinov2_transform import DINOv2ViTTransformArgs
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train.types import ImageSizeTuple, PathLike

logger = logging.getLogger(__name__)

_STUDENT_PREFIX = "student_embedding_model.wrapped_model."


def eval_classification(
    *,
    out: PathLike,
    checkpoint: PathLike,
    eval_config: PathLike | None = None,
    image_size: ImageSizeTuple | None = None,
    encoder: str | None = None,
    tasks: list[str] | None = None,
    accelerator: str = "auto",
    overwrite: bool = False,
) -> None:
    """Evaluate a pretrained model with KneeNo's frozen-encoder classification tasks.

    Args:
        out:
            Path to a JSON file where the final metrics are written.
        checkpoint:
            Path to a LightlyTrain checkpoint, e.g.
            ``out/my_experiment/checkpoints/last.ckpt``.
        eval_config:
            Path to a KneeNo ``eval:`` YAML config. Defaults to the one shipped at
            ``lightly_train/_configs/kneeno_eval.yaml``.
        image_size:
            Global crop size ``(H, W, D)`` the model was pretrained with. Defaults to
            ``DINOv2ViTTransformArgs.image_size``. The checkpoint does not record it, so
            pass it explicitly if the pretraining run overrode
            ``transform_args.image_size``.
        encoder:
            Which encoder to evaluate: 'target' (the EMA teacher, what the checkpoint
            stores) or 'online' (the student). Defaults to the config's ``eval.encoder``.
        tasks:
            Subset of ('knn', 'linear', 'linear_pool', 'attentive_pool'). Defaults to all
            four.
        accelerator:
            Hardware accelerator, e.g. 'cpu', 'gpu' or 'auto'.
        overwrite:
            Overwrite ``out`` if it already exists.
    """
    config = EvalClassificationConfig(**locals())
    eval_classification_from_config(config=config)


def eval_classification_from_config(config: EvalClassificationConfig) -> None:
    _warnings.filter_eval_classification_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")

    out_path = common_helpers.get_out_path(out=config.out, overwrite=config.overwrite)
    ckpt_path = common_helpers.get_checkpoint_path(checkpoint=config.checkpoint)

    eval_cfg = load_eval_config(
        None if config.eval_config is None else Path(config.eval_config)
    )
    encoder_choice = config.encoder or eval_cfg.get("encoder", "target")
    if encoder_choice not in ("target", "online"):
        raise ValueError(
            f"Invalid encoder: '{encoder_choice}'. Valid encoders are: "
            "['target', 'online']"
        )

    logger.info(f"Loading checkpoint from '{ckpt_path}'")
    ckpt = Checkpoint.from_path(checkpoint=ckpt_path)
    wrapped_model = ckpt.lightly_train.models.wrapped_model
    if encoder_choice == "online":
        _load_student_weights(wrapped_model=wrapped_model, ckpt=ckpt)
    logger.info(f"Evaluating the '{encoder_choice}' encoder.")

    device = _get_device(accelerator=config.accelerator)
    # ModelWrapper is a protocol; every concrete wrapper is also an nn.Module.
    assert isinstance(wrapped_model, Module)
    wrapped_model.requires_grad_(False)
    wrapped_model.eval()
    wrapped_model.to(device)

    image_size = config.image_size or DINOv2ViTTransformArgs().image_size
    normalize_args = ckpt.lightly_train.normalize_args
    adapter = DINOv2Adapter(
        embed_dim=wrapped_model.feature_dim(),
        image_size=image_size,
        normalize=(normalize_args.mean, normalize_args.std),
    )

    evaluator = ClassificationEvaluator(config=eval_cfg, adapter=adapter, device=device)
    try:
        # log_every_head_epoch=True: the point of the standalone run is the head
        # fine-tuning curve itself, not one summary point per pretraining epoch.
        metrics = evaluator.evaluate(
            wrapped_model, tasks=config.tasks, epoch=0, log_every_head_epoch=True
        )
    finally:
        # No-op unless the config sets logging.tensorboard_dir, but flushing a writer
        # that does exist matters if the process exits right after.
        evaluator.tb.close()

    logger.info(f"Final metrics: {metrics}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info(f"Wrote metrics to '{out_path}'")


def eval_classification_from_dictconfig(config: DictConfig) -> None:
    logger.debug(f"Evaluating model with config: {config}")
    config_dict = omegaconf_utils.config_to_dict(config=config)
    eval_cfg = validate.pydantic_model_validate(EvalClassificationConfig, config_dict)
    eval_classification_from_config(config=eval_cfg)


class EvalClassificationConfig(PydanticConfig):
    out: PathLike
    checkpoint: PathLike
    eval_config: PathLike | None = None
    image_size: ImageSizeTuple | None = None
    encoder: str | None = None
    tasks: list[str] | None = None
    accelerator: str = "auto"
    overwrite: bool = False


class CLIEvalClassificationConfig(EvalClassificationConfig):
    out: str
    checkpoint: str
    eval_config: str | None = None


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
