#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Epoch-end KneeNo classification evaluation of the frozen encoder.

The lightly-train counterpart of vjepa2's in-training hook in
``app/vjepa_2_1/train.py``. See ``KneeNo/README.md`` for the evaluation itself.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kneeno.evaluation import ClassificationEvaluator, load_eval_config, tasks_due
from pytorch_lightning import Callback, LightningModule, Trainer

from lightly_train._configs.config import PydanticConfig
from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.types import ImageSizeTuple

logger = logging.getLogger(__name__)

DEFAULT_EVAL_CONFIG_PATH = (
    Path(__file__).parent.parent / "_configs" / "kneeno_eval.yaml"
)


class KneeNoEvalArgs(PydanticConfig):
    config_path: str | None = None  # None -> _configs/kneeno_eval.yaml


class KneeNoEval(Callback):
    """Runs KneeNo's frozen-encoder classification tasks at the end of due epochs.

    Which tasks are due is decided by ``kneeno.evaluation.tasks_due`` from the
    ``eval.freq.<task>`` config, exactly as in vjepa2, so the two repos' curves are
    directly comparable.

    Metrics are logged through ``pl_module.log_dict``, which reaches every configured
    lightly-train logger (TensorBoard, JSONL, W&B, MLflow). The shipped config therefore
    sets ``logging.tensorboard_dir: null``, disabling KneeNo's own ``SummaryWriter``.
    """

    def __init__(
        self,
        wrapped_model: ModelWrapper,
        image_size: ImageSizeTuple,
        normalize_args: NormalizeArgs,
        config_path: str | None = None,
    ) -> None:
        self._wrapped_model = wrapped_model
        self._image_size = image_size
        self._normalize_args = normalize_args
        self._config_path = (
            Path(config_path) if config_path is not None else DEFAULT_EVAL_CONFIG_PATH
        )
        self._config = load_eval_config(self._config_path)
        self._evaluator: ClassificationEvaluator | None = None
        # Set once the evaluator has failed to build, so we warn once and stay quiet.
        self._disabled = False

    def _get_evaluator(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> ClassificationEvaluator | None:
        """Build the evaluator on first use.

        Deferred rather than done in ``__init__`` because it loads the labeled dataset
        and needs the trainer's rank/world size and the module's device -- none of which
        exist when callbacks are constructed.
        """
        if self._evaluator is not None or self._disabled:
            return self._evaluator

        feature_dim = self._wrapped_model.feature_dim()
        adapter = DINOv2Adapter(
            embed_dim=feature_dim,
            image_size=self._image_size,
            normalize=(self._normalize_args.mean, self._normalize_args.std),
        )
        try:
            self._evaluator = ClassificationEvaluator(
                config=self._config,
                adapter=adapter,
                device=pl_module.device,
                rank=trainer.global_rank,
                world_size=trainer.world_size,
            )
        except Exception as ex:
            # The labeled dataset lives on the cluster; a run without it should not die
            # at the first epoch boundary just because evaluation is on by default.
            self._disabled = True
            logger.warning(
                f"Disabling KneeNo evaluation for this run: could not load the labeled "
                f"dataset from eval.data ('{self._config['data']['data_root']}', "
                f"'{self._config['data']['label_meta']}'): {ex}. Point "
                f"callbacks.kneeno_eval.config_path at a config with valid paths, or "
                f"pass callbacks.kneeno_eval=null to silence this."
            )
        return self._evaluator

    def _resolve_encoder(self, pl_module: LightningModule) -> Any:
        """The encoder to evaluate: the EMA teacher (default) or the online student.

        ``self._wrapped_model`` *is* the teacher's wrapper -- ``DINOv2.__init__`` assigns
        the ``embedding_model`` it is given to ``self.teacher_embedding_model`` and
        EMA-updates it in place, so the same object tracks the teacher throughout.
        """
        if self._config.get("encoder", "target") != "online":
            return self._wrapped_model

        student = getattr(pl_module, "student_embedding_model", None)
        if student is None:
            logger.warning(
                "eval.encoder is 'online' but "
                f"{type(pl_module).__name__} has no 'student_embedding_model'; "
                "evaluating the target encoder instead."
            )
            return self._wrapped_model
        return student.wrapped_model

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        due = tasks_due(epoch, self._config["freq"])
        if not due:
            return

        evaluator = self._get_evaluator(trainer=trainer, pl_module=pl_module)
        if evaluator is None:
            return

        # Called on EVERY rank, deliberately and without a rank guard:
        # ClassificationEvaluator does the work on rank 0 and then hits dist.barrier() +
        # dist.broadcast_object_list(). Guarding this call would leave rank 0 waiting at
        # a barrier the other ranks never reach, deadlocking the run.
        metrics = evaluator.evaluate(
            self._resolve_encoder(pl_module),
            tasks=due,
            epoch=epoch,
            log_every_head_epoch=False,
        )
        logger.info(f"[epoch {epoch + 1}] eval ({due}): {metrics}")

        # sync_dist=False: every rank already holds the identical broadcast result, so
        # there is nothing to reduce.
        pl_module.log_dict(
            {f"eval/{name}": float(value) for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
