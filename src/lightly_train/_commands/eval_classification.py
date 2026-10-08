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
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
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
    eval_config: PathLike,
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
            Path to a YAML file with a top-level ``eval:`` block (KneeNo's evaluation
            config), e.g. the ``params-pretrain.yaml`` a pretraining run wrote to its
            output directory. Its ``transform:`` block's ``image_size`` is the default
            for ``image_size``; other top-level blocks are ignored.
        image_size:
            Global crop size ``(H, W)`` the model was pretrained with; every slice is
            resized to it. The checkpoint does not record it. Defaults to the eval
            config's ``transform.image_size``, else ``DINOv2ViTTransformArgs.image_size``
            (with a warning).
        encoder:
            Which encoder to evaluate: 'target' (the EMA teacher, what the checkpoint
            stores) or 'online' (the student). Defaults to the config's ``eval.encoder``.
        tasks:
            Subset of KneeNo's tasks ('knn', 'linear', 'linear_pool', 'attentive_pool',
            'linear_slice_cls', 'attentive_slice_cls', 'attentive_slice_pool'). Defaults
            to all of them. The slice tasks need ``eval.data.series_depth > 0``.
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

    eval_cfg = load_eval_config(_read_eval_block(Path(config.eval_config)))
    image_size = config.image_size or _read_image_size(Path(config.eval_config))
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

    dataset_type = eval_cfg.get("data").get("dataset_type")
    normalize_args = ckpt.lightly_train.normalize_args
    adapter = DINOv2Adapter(
        dataset_type=dataset_type,
        embed_dim=wrapped_model.feature_dim(),
        # The checkpoint stores the resolved normalize args, whose length the transform
        # matched to the resolved num_channels (TransformArgs.resolve_incompatible), so
        # this is the channel count the model was trained with, "auto" included.
        num_channels=len(normalize_args.mean),
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
        evaluator.cleanup()

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
    eval_config: PathLike
    image_size: ImageSizeTuple | None = None
    encoder: str | None = None
    tasks: list[str] | None = None
    accelerator: str = "auto"
    overwrite: bool = False


class CLIEvalClassificationConfig(EvalClassificationConfig):
    out: str
    checkpoint: str
    eval_config: str


def _read_eval_block(path: Path) -> dict[str, Any]:
    """The ``eval:`` block of a run config such as ``params-pretrain.yaml``.

    There is no default evaluation config, so a file without one is an error rather
    than silently evaluating with KneeNo's built-in defaults.
    """
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("eval"), dict):
        raise ValueError(
            f"Eval config '{path}' has no top-level 'eval:' block. Pass a pretraining "
            "run's params-pretrain.yaml or another file with an 'eval:' block."
        )
    eval_block: dict[str, Any] = raw["eval"]
    return eval_block


def _read_image_size(path: Path) -> tuple[int, int]:
    """The training crop size, from the run config's ``transform:`` block.

    Left out (or no ``transform:`` block), it falls back to ``DINOv2ViTTransformArgs``'
    default, exactly as it did for the pretraining run, with a warning: the file may just
    not be the run config.
    """
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    transform = raw.get("transform") if isinstance(raw, dict) else None
    if isinstance(transform, dict) and "image_size" in transform:
        given = {"image_size": transform["image_size"]}
    else:
        given = {}
    image_size = DINOv2ViTTransformArgs.model_validate(given).image_size
    if not given:
        # A run trained at another crop size but evaluated from a file without its
        # transform block would otherwise be resized differently without notice.
        logger.warning(
            f"Eval config '{path}' does not set transform.image_size; using the default "
            f"{image_size}. Pass image_size or the pretraining run's params-pretrain.yaml "
            "if it used another one."
        )
    return image_size


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
