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
import os
from pathlib import Path
from typing import Any

from kneeno.evaluation import (
    BestModelTracker,
    ClassificationEvaluator,
    load_eval_config,
    tasks_due,
)
from pytorch_lightning import Callback, LightningModule, Trainer

from lightly_train import _distributed as distributed_helpers
from lightly_train._commands import common_helpers
from lightly_train._commands.common_helpers import ModelFormat
from lightly_train._configs.config import PydanticConfig
from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._models import package_helpers
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.types import ImageSizeTuple

logger = logging.getLogger(__name__)


class KneeNoEvalArgs(PydanticConfig):
    # The ``eval:`` block of the run config, deep-merged over KneeNo's
    # DEFAULT_EVAL_CONFIG. There is no shipped default: the callback is off unless a
    # config is passed.
    config: dict[str, Any]


class KneeNoEval(Callback):
    """Runs KneeNo's frozen-encoder classification tasks at the end of due epochs.

    Which tasks are due is decided by ``kneeno.evaluation.tasks_due`` from the
    ``eval.freq.<task>`` config, exactly as in vjepa2, so the two repos' curves are
    directly comparable.

    Metrics are logged through ``pl_module.log_dict``, which reaches every configured
    lightly-train logger (TensorBoard, JSONL, W&B, MLflow). The cluster run config
    (``cluster/configs/``) therefore sets ``logging.tensorboard_dir: null``, disabling
    KneeNo's own ``SummaryWriter``.

    With ``eval.model_selection: true``, the evaluated encoder of the best evaluation so far
    (``kneeno.BestModelTracker``: the mean of the internal k-NN k=100 and linear_pool AUROC)
    is exported to ``<out_dir>/exported_best.pt``, in the format of ``exported_last.pt``, so
    ``lightly-train eval_classification`` loads it the same way. The tracker's state is this
    callback's ``state_dict``, which Lightning stores in every checkpoint and restores on
    resume. Lightning runs ``ModelCheckpoint`` after every other callback, so the checkpoint
    of an epoch already holds that epoch's evaluation.
    """

    def __init__(
        self,
        wrapped_model: ModelWrapper,
        num_channels: int,
        image_size: ImageSizeTuple,
        normalize_args: NormalizeArgs,
        config: dict[str, Any],
        out_dir: Path,
    ) -> None:
        self._wrapped_model = wrapped_model
        self._num_channels = num_channels
        self._image_size = image_size
        self._normalize_args = normalize_args
        self._config = load_eval_config(config)
        # eval.logging.tag_prefix, as KneeNo's own writer uses it (vjepa2's tags match).
        self._tag_prefix = self._config["logging"]["tag_prefix"]
        self._best_model_path = Path(out_dir) / "exported_best.pt"
        # Only with eval.model_selection; the state is kept even without, so that turning it
        # on for a resumed run does not trip over a missing key.
        self._best_model_tracker = BestModelTracker()
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
            num_channels=self._num_channels,
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
                f"'{self._config['data']['label_meta']}'): {ex}. Fix eval.data in the "
                f"run config (callbacks.kneeno_eval.config), or drop the eval block "
                f"to silence this."
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
        # The last epoch runs every task with a frequency set, so the final model is
        # always evaluated (and seen by model selection).
        due = tasks_due(epoch, self._config["freq"], num_epochs=trainer.max_epochs)
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
            self._resolve_encoder(pl_module), tasks=due, epoch=epoch
        )
        logger.info(f"[epoch {epoch + 1}] eval ({due}): {metrics}")

        logged = {
            f"{self._tag_prefix}/{name}": float(value)
            for name, value in metrics.items()
        }
        if self._config["model_selection"]:
            logged.update(self._update_best_model(pl_module, metrics, epoch=epoch + 1))

        # sync_dist=False: every rank already holds the identical broadcast result, so
        # there is nothing to reduce.
        pl_module.log_dict(logged, on_step=False, on_epoch=True, sync_dist=False)

    def _update_best_model(
        self, pl_module: LightningModule, metrics: dict[str, Any], epoch: int
    ) -> dict[str, float]:
        """Export the evaluated encoder if ``metrics`` beat the best evaluation so far.

        Runs on every rank, which all hold the same metrics, so their trackers stay equal;
        only global rank zero writes the file. Returns the scores to log.

        :param epoch: epochs completed when the evaluation ran (``current_epoch + 1``).
        """
        tracker = self._best_model_tracker
        previous = tracker.state_dict()
        if tracker.update(metrics, epoch=epoch):
            try:
                self._export_best_model(self._resolve_encoder(pl_module))
            except Exception as ex:
                # A full disk should not kill the run; the next evaluation tries again.
                tracker.load_state_dict(previous)
                logger.warning(
                    f"[epoch {epoch}] model selection: could not export the new best "
                    f"model to '{self._best_model_path}': {ex}"
                )
            else:
                logger.info(
                    f"[epoch {epoch}] model selection: new best score "
                    f"{tracker.best_score:.4f} (previous {previous['best_score']:.4f}, "
                    f"epoch {previous['best_epoch']}), exported to "
                    f"'{self._best_model_path}'"
                )
        elif tracker.last_score is not None:
            logger.info(
                f"[epoch {epoch}] model selection: score {tracker.last_score:.4f}, best "
                f"{tracker.best_score:.4f} (epoch {tracker.best_epoch}) stays"
            )
        # The same names vjepa2 logs through KneeNo (evaluator.log_scalars).
        return {
            f"{self._tag_prefix}/model_selection/{name}": value
            for name, value in tracker.scalars().items()
        }

    def _export_best_model(self, encoder: ModelWrapper) -> None:
        """Export ``encoder`` like ``exported_last.pt``: to a temporary file, then renamed
        over ``exported_best.pt``, so a crash mid-write leaves the previous best intact."""
        if not distributed_helpers.is_global_rank_zero():
            return
        package = package_helpers.get_package_from_model(
            encoder, include_custom=True, fallback_custom=True
        )
        tmp_path = self._best_model_path.with_name(self._best_model_path.name + ".tmp")
        common_helpers.export_model(
            model=encoder,
            out=tmp_path,
            format=ModelFormat.PACKAGE_DEFAULT,
            package=package,
            log_example=False,
        )
        os.replace(tmp_path, self._best_model_path)

    def state_dict(self) -> dict[str, Any]:
        return {"best_model": self._best_model_tracker.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = state_dict.get("best_model")
        if state is None:
            logger.info(
                "The checkpoint has no model selection state (written before it "
                "existed): the best model starts over."
            )
            return
        self._best_model_tracker.load_state_dict(state)
        logger.info(f"Restored model selection state: {self._best_model_tracker}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Flushes and closes KneeNo's own TensorBoard writer (a no-op with the shipped
        # config, which disables it). The evaluator is built lazily, so there may be
        # nothing to close: no task was ever due, or evaluation disabled itself.
        # Purely local, no collectives, so it is safe on every rank.
        if self._evaluator is not None:
            self._evaluator.cleanup()
