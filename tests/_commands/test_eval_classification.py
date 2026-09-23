#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from pytest_mock import MockerFixture

from lightly_train._checkpoint import (
    Checkpoint,
    CheckpointLightlyTrain,
    CheckpointLightlyTrainModels,
)
from lightly_train._commands import eval_classification
from lightly_train._commands.eval_classification import (
    _STUDENT_PREFIX,
    EvalClassificationConfig,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._transforms.transform import NormalizeArgs

from .. import helpers

IMAGE_SIZE = (16, 8, 4)  # (H, W, D), non-cubic to catch an axis-order swap.


class _FakeLabeledDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Stands in for LabeledInternalKneeMRIDataset, which needs a dataset on the cluster."""

    num_classes = 4

    def __len__(self) -> int:
        return 12

    def effective_depth(self, index: int) -> int:
        return 5 + (index % 3)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(index)
        volume = torch.randint(
            0,
            256,
            (1, self.effective_depth(index), 20, 11),
            generator=gen,
            dtype=torch.uint8,
        )
        label = (torch.rand(self.num_classes, generator=gen) > 0.5).float()
        return volume, label


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    """A DINOv2 pretrain checkpoint, with both teacher and student weights."""
    method = helpers.get_method_dinov2()
    wrapped_model = method.teacher_embedding_model.wrapped_model
    # The student starts as a deepcopy of the teacher and only diverges once EMA
    # updates run. Perturb it so tests can tell the two encoders apart.
    with torch.no_grad():
        for param in method.student_embedding_model.parameters():
            param.add_(1.0)
    checkpoint = Checkpoint(
        state_dict=method.state_dict(),
        lightly_train=CheckpointLightlyTrain.from_now(
            models=CheckpointLightlyTrainModels(
                model=wrapped_model.get_model(),
                wrapped_model=wrapped_model,
                embedding_model=EmbeddingModel(wrapped_model=wrapped_model),
            ),
            normalize_args=NormalizeArgs(),
        ),
    )
    path = tmp_path / "last.ckpt"
    checkpoint.save(path)
    return path


@pytest.fixture
def eval_config_path(tmp_path: Path) -> Path:
    config: dict[str, Any] = {
        "data": {"num_workers": 0},
        "knn": {"batch_size": 4},
        "linear": {"epochs": 1, "batch_size": 4},
        "linear_pool": {"epochs": 1, "batch_size": 4},
        "attentive_pool": {"epochs": 1, "batch_size": 4, "num_heads": 2},
    }
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({"eval": config}))
    return path


@pytest.fixture
def fake_dataset(mocker: MockerFixture) -> None:
    mocker.patch(
        "kneeno.evaluation.classification.LabeledInternalKneeMRIDataset",
        return_value=_FakeLabeledDataset(),
    )


def test_eval_classification(
    tmp_path: Path, checkpoint_path: Path, eval_config_path: Path, fake_dataset: None
) -> None:
    out_path = tmp_path / "metrics.json"
    eval_classification.eval_classification(
        out=out_path,
        checkpoint=checkpoint_path,
        eval_config=eval_config_path,
        image_size=IMAGE_SIZE,
        tasks=["knn", "linear"],
        accelerator="cpu",
    )
    metrics = json.loads(out_path.read_text())
    assert metrics
    # "linear" needs a cls token, which DINOv2 has (unlike V-JEPA 2.1).
    assert {key.split("/")[0] for key in metrics} == {"knn", "linear"}
    assert all(isinstance(value, float) for value in metrics.values())


def test_eval_classification__online_encoder(
    tmp_path: Path, checkpoint_path: Path, eval_config_path: Path, fake_dataset: None
) -> None:
    out_path = tmp_path / "metrics.json"
    eval_classification.eval_classification(
        out=out_path,
        checkpoint=checkpoint_path,
        eval_config=eval_config_path,
        image_size=IMAGE_SIZE,
        encoder="online",
        tasks=["knn"],
        accelerator="cpu",
    )
    assert json.loads(out_path.read_text())


def test_eval_classification__invalid_encoder(
    tmp_path: Path, checkpoint_path: Path, eval_config_path: Path
) -> None:
    with pytest.raises(ValueError, match="Invalid encoder"):
        eval_classification.eval_classification(
            out=tmp_path / "metrics.json",
            checkpoint=checkpoint_path,
            eval_config=eval_config_path,
            encoder="teacher",
            accelerator="cpu",
        )


def test_load_student_weights__changes_the_teacher_weights(
    checkpoint_path: Path,
) -> None:
    """The checkpoint stores the teacher; the student lives only in the state_dict."""
    checkpoint = Checkpoint.from_path(checkpoint_path)
    wrapped_model = checkpoint.lightly_train.models.wrapped_model
    assert any(key.startswith(_STUDENT_PREFIX) for key in checkpoint.state_dict)

    before = {k: v.clone() for k, v in wrapped_model.state_dict().items()}
    eval_classification._load_student_weights(
        wrapped_model=wrapped_model, ckpt=checkpoint
    )
    after = wrapped_model.state_dict()
    assert any(not torch.equal(before[key], after[key]) for key in before)


def test_load_student_weights__without_student(checkpoint_path: Path) -> None:
    checkpoint = Checkpoint.from_path(checkpoint_path)
    stripped = Checkpoint(
        state_dict={
            k: v
            for k, v in checkpoint.state_dict.items()
            if not k.startswith(_STUDENT_PREFIX)
        },
        lightly_train=checkpoint.lightly_train,
    )
    with pytest.raises(ValueError, match="requires student weights"):
        eval_classification._load_student_weights(
            wrapped_model=checkpoint.lightly_train.models.wrapped_model, ckpt=stripped
        )


def test_eval_classification__from_dictconfig(
    tmp_path: Path,
    checkpoint_path: Path,
    eval_config_path: Path,
    fake_dataset: None,
) -> None:
    out_path = tmp_path / "metrics.json"
    config = OmegaConf.create(
        dict(
            out=str(out_path),
            checkpoint=str(checkpoint_path),
            eval_config=str(eval_config_path),
            image_size=list(IMAGE_SIZE),
            tasks=["knn"],
            accelerator="cpu",
        )
    )
    eval_classification.eval_classification_from_dictconfig(config=config)
    assert json.loads(out_path.read_text())


def test_eval_classification__parameters() -> None:
    helpers.assert_same_params(
        a=eval_classification.eval_classification, b=EvalClassificationConfig
    )
