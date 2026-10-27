#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from lightly_train._callbacks.checkpoint import ModelCheckpoint
from lightly_train._checkpoint import (
    CHECKPOINT_LIGHTLY_TRAIN_KEY,
    CheckpointLightlyTrainModels,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._transforms.transform import NormalizeArgs

from ..helpers import DummyCustomModel


class _Module(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        return self.layer(batch[0]).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        # 2 batches per epoch.
        return DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)


class _KillAtEpoch(Callback):
    """Stands in for a SLURM time limit: training stops without on_train_end."""

    def __init__(self, epoch: int) -> None:
        self._epoch = epoch

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch == self._epoch:
            raise RuntimeError("killed")


def _checkpoint_callback(dirpath: Path, **kwargs: int) -> ModelCheckpoint:
    model = DummyCustomModel()
    return ModelCheckpoint(
        models=CheckpointLightlyTrainModels(
            model=model.get_model(),
            wrapped_model=model,
            embedding_model=EmbeddingModel(wrapped_model=model),
        ),
        normalize_args=NormalizeArgs(),
        dirpath=str(dirpath),
        license_info="",
        save_last=True,
        **kwargs,
    )


def _fit(callbacks: list[Callback], max_epochs: int) -> None:
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )
    trainer.fit(_Module())


def test_model_checkpoint__last_every_epoch_before_every_n_epochs(
    tmp_path: Path,
) -> None:
    # Killed in the third epoch, long before the first periodic checkpoint: last.ckpt
    # must still hold the second epoch's end.
    checkpoint = _checkpoint_callback(tmp_path, every_n_epochs=50, save_top_k=-1)
    with pytest.raises(RuntimeError, match="killed"):
        _fit([checkpoint, _KillAtEpoch(epoch=2)], max_epochs=100)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.ckpt"]
    ckpt = torch.load(tmp_path / "last.ckpt", weights_only=False)
    assert ckpt["epoch"] == 1
    assert ckpt["global_step"] == 4
    assert CHECKPOINT_LIGHTLY_TRAIN_KEY in ckpt


def test_model_checkpoint__every_n_epochs_unchanged(tmp_path: Path) -> None:
    checkpoint = _checkpoint_callback(tmp_path, every_n_epochs=2, save_top_k=-1)
    _fit([checkpoint], max_epochs=5)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "epoch=1-step=4.ckpt",
        "epoch=3-step=8.ckpt",
        "last.ckpt",
    ]
    assert torch.load(tmp_path / "last.ckpt", weights_only=False)["epoch"] == 4
