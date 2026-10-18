#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Standalone post-pretraining KneeNo classification evaluation.

Loads a frozen encoder from a LightlyTrain pretraining checkpoint or an exported model and
hands it to ``kneeno.evaluation.ClassificationEvaluator``, which logs each task's metrics once
per test set (``eval.data.test_sets``; heads early-stopped on the val split), the same protocol
as the in-training callback (``_callbacks/kneeno_eval.py``), which tests on ``internal`` only.

The counterpart of ``vjepa2/app/vjepa_2_1/eval_classification.py``, but much shorter: a
LightlyTrain checkpoint stores the model objects themselves, so there is no architecture
to rebuild from a config.

Everything about the evaluation comes from one YAML file (``cluster/configs/eval-*.yaml``)
with these top-level keys:

    weights:                       the file to evaluate, its format detected from its
                                   content: a LightlyTrain checkpoint
                                   (``<out>/checkpoints/*.ckpt``) or the state dict of the
                                   bare DINOv2 ViT, as exported by pretraining
                                   (``<out>/exported_models/exported_last.pt``) or
                                   ``lightly-train export`` (default ``part`` and ``format``)
    image_size:                    [H, W, D], the crop size the model was pretrained with
    resize_interpolation:          how volumes are resampled to it, e.g. area or area+nearest
    resize_upscale_interpolation:  the same for enlarged axes; null -> resize_interpolation
    eval:                          KneeNo's evaluation config, deep-merged over its
                                   DEFAULT_EVAL_CONFIG

and, for an exported model only (a checkpoint records them, so they are rejected with one):

    model:                         the model name the weights belong to, e.g.
                                   ``dinov2/vitb14-notpretrained``
    model_args:                    optional, the pretraining run's ``model_args``
    normalize:                     {mean: [m], std: [s]}, the pretraining run's
                                   ``transform.normalize`` (one value each: 3D models
                                   have 1 input channel)

Neither file records image_size and the two resize keys, so they must match the
pretraining run's ``transform:`` block. ``image_size`` is required; a missing resize key
falls back to ``DINOv2ViTTransformArgs``' default, as it did for training, with a warning.
An exported model holds the EMA teacher only, so ``eval.encoder: online`` needs a checkpoint.

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
from pydantic import Field, ValidationError
from torch import Tensor
from torch.nn import Module

from lightly_train import _logging
from lightly_train._checkpoint import CHECKPOINT_LIGHTLY_TRAIN_KEY, Checkpoint
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs import omegaconf_utils, validate
from lightly_train._configs.config import PydanticConfig
from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._methods.dinov2.dinov2_transform import DINOv2ViTTransformArgs
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._transforms.random_resized_crop import ResizeInterpolation
from lightly_train.types import ImageSizeTuple, PathLike

logger = logging.getLogger(__name__)

_STUDENT_PREFIX = "student_embedding_model.wrapped_model."
_PATCH_EMBED_KEY = "patch_embed.proj.weight"
# The eval config keys that describe an exported model, which a checkpoint records itself.
_EXPORTED_MODEL_KEYS = ("model", "model_args", "normalize")
# What os.path.expandvars expands when the variable is set, i.e. what is left of an unset one.
_ENV_VAR_PATTERN = re.compile(r"\$(\{\w+\}|[A-Za-z_]\w*)")
_RESIZE_KEYS = ("resize_interpolation", "resize_upscale_interpolation")
_TRANSFORM_FIELDS = DINOv2ViTTransformArgs.model_fields


def eval_classification(
    *,
    eval_config: PathLike,
    accelerator: str = "auto",
    overwrite: bool = False,
) -> None:
    """Evaluate a pretrained model with KneeNo's frozen-encoder classification tasks.

    Args:
        eval_config:
            Path to a YAML file with the top-level keys ``weights`` (a LightlyTrain
            checkpoint or an exported model, told apart by content), ``image_size``
            (``[H, W, D]``, the crop size the model was pretrained with; every volume is
            resized to it), ``resize_interpolation`` / ``resize_upscale_interpolation``
            (how, as in the pretraining run's ``transform:`` block;
            ``DINOv2ViTTransformArgs``' defaults with a warning if left out) and
            ``eval`` (KneeNo's evaluation config). An exported model also needs
            ``model`` (its model name), ``normalize`` (``mean`` and ``std``) and
            optionally ``model_args``; with a checkpoint these keys are an error. The
            tasks are those with a non-null ``eval.freq``, the encoder is
            ``eval.encoder`` ('target' or 'online'; 'online' needs a checkpoint).
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
    _warn_on_default_resize_args(run_config=run_config, path=Path(config.eval_config))
    eval_cfg = load_eval_config(run_config.eval)
    tasks = _enabled_tasks(eval_cfg["freq"])
    encoder_choice = eval_cfg.get("encoder", "target")
    if encoder_choice not in ("target", "online"):
        raise ValueError(
            f"Invalid eval.encoder: '{encoder_choice}'. Valid encoders are: "
            "['target', 'online']"
        )
    weights_path = common_helpers.get_checkpoint_path(checkpoint=run_config.weights)

    # Everything that can fail before the (possibly hours long) evaluation does so now.
    log_cfg = eval_cfg["logging"]
    results_path, params_path = _get_output_paths(
        per_label_dir=log_cfg["per_label_dir"], weights_path=weights_path
    )
    results_path = common_helpers.get_out_path(
        out=results_path, overwrite=config.overwrite
    )
    _check_no_previous_results(
        paths=[params_path, log_cfg["per_label_dir"], log_cfg["tensorboard_dir"]],
        overwrite=config.overwrite,
    )
    # Loaded before params.yaml is written: a config the weights reject (e.g. the wrong
    # model) must not leave a params.yaml behind that makes the corrected rerun refuse.
    wrapped_model, normalize = _load_encoder(
        weights_path=weights_path, run_config=run_config, encoder_choice=encoder_choice
    )
    if params_path is not None:
        # Written before evaluating, so a run that crashes still records its settings.
        params_path.parent.mkdir(parents=True, exist_ok=True)
        params_path.write_text(yaml.safe_dump(raw_config, sort_keys=False))
        logger.info(f"Wrote the evaluation config to '{params_path}'")
    logger.info(f"Evaluating the '{encoder_choice}' encoder on tasks {tasks}.")

    device = _get_device(accelerator=config.accelerator)
    # ModelWrapper is a protocol; every concrete wrapper is also an nn.Module.
    assert isinstance(wrapped_model, Module)
    wrapped_model.requires_grad_(False)
    wrapped_model.eval()
    wrapped_model.to(device)

    adapter = DINOv2Adapter(
        embed_dim=wrapped_model.feature_dim(),
        image_size=run_config.image_size,
        normalize=normalize,
        resize_interpolation=run_config.resize_interpolation,
        resize_upscale_interpolation=run_config.resize_upscale_interpolation,
    )

    evaluator = ClassificationEvaluator(config=eval_cfg, adapter=adapter, device=device)
    try:
        metrics = evaluator.evaluate(wrapped_model, tasks=tasks, epoch=0)
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


class NormalizeConfig(PydanticConfig):
    """``normalize`` of an eval config. Unlike ``NormalizeArgs`` without defaults: the 2D
    branch normalizes differently, so a forgotten value must not become this one's."""

    # Not strict, so YAML lists become tuples (as in NormalizeArgs).
    mean: tuple[float, ...] = Field(strict=False)
    std: tuple[float, ...] = Field(strict=False)


class EvalRunConfig(PydanticConfig):
    """The YAML file passed as ``eval_config``. Unknown keys are an error, so a typo or a
    leftover pretraining block (e.g. ``transform:``) cannot be silently ignored.

    ``model``, ``model_args`` and ``normalize`` describe an exported model; whether they
    are required or rejected depends on the file ``weights`` points to (``_load_encoder``).
    """

    weights: PathLike
    image_size: ImageSizeTuple
    # Defaults and validation as in the pretraining run's transform args.
    resize_interpolation: ResizeInterpolation = _TRANSFORM_FIELDS[
        "resize_interpolation"
    ].default
    resize_upscale_interpolation: ResizeInterpolation | None = _TRANSFORM_FIELDS[
        "resize_upscale_interpolation"
    ].default
    eval: dict[str, Any]
    model: str | None = None
    model_args: dict[str, Any] | None = None
    normalize: NormalizeConfig | None = None


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
    if "checkpoint" in raw:
        raise ValueError(
            f"Eval config '{path}' has a 'checkpoint' key, which is now called "
            "'weights' (a checkpoint or an exported model)."
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


def _warn_on_default_resize_args(run_config: EvalRunConfig, path: Path) -> None:
    """A run trained with e.g. "area+nearest" but evaluated from a config without the
    resize keys would otherwise be resampled differently without notice."""
    missing = [key for key in _RESIZE_KEYS if key not in run_config.model_fields_set]
    if missing:
        defaults = ", ".join(f"{key}={getattr(run_config, key)!r}" for key in missing)
        logger.warning(
            f"Eval config '{path}' does not set {', '.join(missing)}; using the default "
            f"{defaults}. Set them to the pretraining run's transform values if it used "
            "other ones."
        )


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
    per_label_dir: str | None, weights_path: Path
) -> tuple[Path, Path | None]:
    """``results.json`` and ``params.yaml`` in the parent of ``per_label_dir``.

    Without a ``per_label_dir`` there is no evaluation folder: ``results.json`` goes next
    to the weights file and no ``params.yaml`` is written.
    """
    if per_label_dir is None:
        results_path = weights_path.parent / "results.json"
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


def _load_encoder(
    weights_path: Path, run_config: EvalRunConfig, encoder_choice: str
) -> tuple[ModelWrapper, tuple[tuple[float, ...], tuple[float, ...]]]:
    """The encoder to evaluate and its ``(mean, std)``.

    ``weights_path`` is either a LightlyTrain checkpoint, which holds the model objects and
    the normalize args, or the state dict of the bare DINOv2 ViT, which holds the weights
    only: the model is then built from the config's ``model`` / ``model_args`` and the
    normalization taken from its ``normalize``. The file's content decides which it is.
    """
    logger.info(f"Loading weights from '{weights_path}'")
    # Not weights_only: a checkpoint holds pickled model objects.
    content = torch.load(weights_path, map_location="cpu", weights_only=False)
    if (
        isinstance(content, dict)
        and "state_dict" in content
        and CHECKPOINT_LIGHTLY_TRAIN_KEY in content
    ):
        logger.info("Found a LightlyTrain checkpoint.")
        return _load_encoder_from_checkpoint(
            ckpt=Checkpoint.from_dict(checkpoint=content),
            run_config=run_config,
            encoder_choice=encoder_choice,
        )
    if isinstance(content, dict) and all(
        isinstance(value, torch.Tensor) for value in content.values()
    ):
        logger.info("Found the state dict of an exported model.")
        return _load_encoder_from_state_dict(
            state_dict=content, run_config=run_config, encoder_choice=encoder_choice
        )
    raise ValueError(
        f"'{weights_path}' is neither a LightlyTrain checkpoint nor the state dict of an "
        f"exported model, but a {type(content).__name__}. Export the model with "
        "lightly-train's default format and part (format='package_default', "
        "part='model'), as pretraining does."
    )


def _load_encoder_from_checkpoint(
    ckpt: Checkpoint, run_config: EvalRunConfig, encoder_choice: str
) -> tuple[ModelWrapper, tuple[tuple[float, ...], tuple[float, ...]]]:
    given = [key for key in _EXPORTED_MODEL_KEYS if key in run_config.model_fields_set]
    if given:
        raise ValueError(
            f"The eval config sets {given}, but 'weights' is a LightlyTrain checkpoint, "
            "which records the model and its normalization itself. Remove them, or "
            "point 'weights' to an exported model."
        )
    wrapped_model = ckpt.lightly_train.models.wrapped_model
    if encoder_choice == "online":
        _load_student_weights(wrapped_model=wrapped_model, ckpt=ckpt)
    normalize_args = ckpt.lightly_train.normalize_args
    return wrapped_model, (normalize_args.mean, normalize_args.std)


def _load_encoder_from_state_dict(
    state_dict: dict[str, Tensor], run_config: EvalRunConfig, encoder_choice: str
) -> tuple[ModelWrapper, tuple[tuple[float, ...], tuple[float, ...]]]:
    if encoder_choice == "online":
        raise ValueError(
            "eval.encoder='online' requires a LightlyTrain checkpoint: an exported "
            "model holds the EMA teacher only. Use eval.encoder='target'."
        )
    missing = [
        key for key in ("model", "normalize") if getattr(run_config, key) is None
    ]
    if missing:
        raise ValueError(
            f"'weights' is an exported model, so the eval config needs {missing}: an "
            "exported model holds the weights only. Use the pretraining run's model, "
            "model_args and transform.normalize."
        )
    assert run_config.model is not None and run_config.normalize is not None
    patch_embed = state_dict.get(_PATCH_EMBED_KEY)
    if patch_embed is None:
        raise ValueError(
            f"The exported model has no '{_PATCH_EMBED_KEY}', so it is not the state "
            f"dict of a bare DINOv2 ViT (its first keys: {list(state_dict)[:3]}). Export "
            "it with part='model' and format='package_default'."
        )
    # The patch embedding's weight is (embed_dim, in_chans, *patch_size).
    num_channels = int(patch_embed.shape[1])
    normalize = run_config.normalize
    # 3D pretraining only produces 1-channel models: the volumes are grayscale, and
    # ViewTransform's NormalizeIntensity raises if normalize has more values than the
    # volume has channels. The eval adapter instead broadcasts the volume to one channel
    # per (mean, std) pair, so a mismatch would only surface in the encoder, or not at all.
    if num_channels != 1 or len(normalize.mean) != 1 or len(normalize.std) != 1:
        raise ValueError(
            "An exported 3D model must have 1 input channel and normalize exactly one "
            f"mean and one std value, but the model has {num_channels} input channels "
            f"and normalize has {len(normalize.mean)} mean / {len(normalize.std)} std "
            "values. 3D pretraining only produces 1-channel models; use the "
            "pretraining run's transform.normalize."
        )
    wrapped_model = package_helpers.get_wrapped_model(
        model=run_config.model,
        num_input_channels=num_channels,
        model_args=run_config.model_args,
        # The weights come from the file; no download (the cluster's compute nodes are
        # offline).
        load_weights=False,
    )
    if not isinstance(wrapped_model, DINOv2ViTModelWrapper):
        raise ValueError(
            f"model must be a DINOv2 ViT (dinov2/<name>), got '{run_config.model}'."
        )
    try:
        wrapped_model.get_model().load_state_dict(state_dict, strict=True)
    except RuntimeError as ex:
        raise ValueError(
            f"The exported model does not fit model='{run_config.model}' with "
            f"model_args={run_config.model_args}. Do they match the pretraining run "
            f"that produced these weights?\n{ex}"
        ) from ex
    # As DINOv2 does with its teacher, which is what the exported model is.
    wrapped_model.make_teacher()
    return wrapped_model, (normalize.mean, normalize.std)


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
