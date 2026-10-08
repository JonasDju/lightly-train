#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from kneeno.evaluation.config import ALL_TASKS
from omegaconf import OmegaConf
from pydantic import ValidationError
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
    _read_image_size,
)
from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._transforms.transform import NormalizeArgs

from .. import helpers

# (H, W): non-square to catch an axis-order swap, and a multiple of _vittest14's patch size.
IMAGE_SIZE = (28, 14)
DEPTH = 5  # slices per volume = eval.data.series_depth


class _FakeLabeledDataset(
    torch.utils.data.Dataset[tuple[tuple[torch.Tensor, ...], torch.Tensor]]
):
    """Stands in for the labeled KneeNo datasets, which need data on the cluster.

    ``raw=True`` mimics LabeledExternalKneeMRIDataset: int16 volumes far outside 0..255 instead of
    the internal dataset's uint8 ones.
    """

    num_classes = 4
    # One item is an exam: one volume per sequence, in this order.
    sequences = ("sag", "st1", "cor", "tra")

    def __init__(self, raw: bool = False) -> None:
        self.raw = raw

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        gen = torch.Generator().manual_seed(index)
        # One depth for all volumes, as with eval.data.series_depth > 0: the 2D adapter
        # keeps the depth, so mixed depths could not be batched.
        volumes = tuple(
            torch.randint(
                0,
                4000 if self.raw else 256,
                (1, DEPTH, 20, 11),
                generator=gen,
                dtype=torch.int16 if self.raw else torch.uint8,
            )
            for s in range(len(self.sequences))
        )
        label = (torch.rand(self.num_classes, generator=gen) > 0.5).float()
        return volumes, label


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
    attentive = {"epochs": 1, "batch_size": 4, "num_heads": 2}
    config: dict[str, Any] = {
        # series_depth: the fake dataset's depth, the slice count of the slice tasks.
        "data": {"dataset_type": "internal", "num_workers": 0, "series_depth": DEPTH},
        "knn": {"batch_size": 4},
        "linear": {"epochs": 1, "batch_size": 4},
        "linear_pool": {"epochs": 1, "batch_size": 4},
        "attentive_pool": attentive,
        "linear_slice_cls": {"epochs": 1, "batch_size": 4},
        "attentive_slice_cls": attentive,
        "attentive_slice_pool": attentive,
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


def test_eval_classification__all_tasks(
    tmp_path: Path, checkpoint_path: Path, eval_config_path: Path, fake_dataset: None
) -> None:
    # tasks=None runs every task the adapter can serve: the 2D encoder serves all of
    # them, the slice tasks included.
    out_path = tmp_path / "metrics.json"
    eval_classification.eval_classification(
        out=out_path,
        checkpoint=checkpoint_path,
        eval_config=eval_config_path,
        image_size=IMAGE_SIZE,
        accelerator="cpu",
    )
    metrics = json.loads(out_path.read_text())
    assert {key.split("/")[0] for key in metrics} == set(ALL_TASKS)
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize(
    "dataset_type, dataset_class",
    [
        ("internal", "LabeledInternalKneeMRIDataset"),
        ("external", "LabeledExternalKneeMRIDataset"),
    ],
)
def test_eval_classification__dataset_type_selects_dataset_and_preprocessing(
    tmp_path: Path,
    checkpoint_path: Path,
    eval_config_path: Path,
    mocker: MockerFixture,
    dataset_type: str,
    dataset_class: str,
) -> None:
    config = yaml.safe_load(eval_config_path.read_text())
    config["eval"]["data"]["dataset_type"] = dataset_type
    eval_config_path.write_text(yaml.safe_dump(config))
    dataset = mocker.patch(
        f"kneeno.evaluation.classification.{dataset_class}",
        return_value=_FakeLabeledDataset(raw=dataset_type == "external"),
    )
    adapter = mocker.patch(
        "lightly_train._commands.eval_classification.DINOv2Adapter",
        wraps=DINOv2Adapter,
    )
    out_path = tmp_path / "metrics.json"
    eval_classification.eval_classification(
        out=out_path,
        checkpoint=checkpoint_path,
        eval_config=eval_config_path,
        image_size=IMAGE_SIZE,
        tasks=["knn"],
        accelerator="cpu",
    )
    dataset.assert_called_once()
    assert adapter.call_args.kwargs["dataset_type"] == dataset_type
    assert adapter.call_args.kwargs["image_size"] == IMAGE_SIZE
    # The checkpoint's (resolved) normalize args have one value per input channel.
    assert adapter.call_args.kwargs["num_channels"] == len(NormalizeArgs().mean) == 3
    metrics = json.loads(out_path.read_text())
    assert metrics and all(math.isfinite(v) for v in metrics.values())


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


@pytest.mark.parametrize("image_size", [None, (42, 28)])
def test_eval_classification__run_config(
    tmp_path: Path,
    checkpoint_path: Path,
    eval_config_path: Path,
    fake_dataset: None,
    mocker: MockerFixture,
    image_size: tuple[int, int] | None,
) -> None:
    """A pretraining run's params-pretrain.yaml: its eval block is used, and its
    transform.image_size unless image_size is passed explicitly."""
    run_config = {
        "method": {"center_method": "softmax"},
        "transform": {"num_channels": "auto", "image_size": list(IMAGE_SIZE)},
        **yaml.safe_load(eval_config_path.read_text()),
    }
    params_path = tmp_path / "params-pretrain.yaml"
    params_path.write_text(yaml.safe_dump(run_config))
    adapter = mocker.patch(
        "lightly_train._commands.eval_classification.DINOv2Adapter",
        wraps=DINOv2Adapter,
    )
    out_path = tmp_path / "metrics.json"
    eval_classification.eval_classification(
        out=out_path,
        checkpoint=checkpoint_path,
        eval_config=params_path,
        image_size=image_size,
        tasks=["knn"],
        accelerator="cpu",
    )
    assert json.loads(out_path.read_text())
    assert adapter.call_args.kwargs["image_size"] == (image_size or IMAGE_SIZE)


@pytest.mark.parametrize(
    "transform, expected",
    [
        (None, None),
        ({"num_channels": 1}, None),
        ({"image_size": [28, 14]}, (28, 14)),
    ],
)
def test_read_image_size__warns_on_default(
    tmp_path: Path,
    mocker: MockerFixture,
    transform: dict[str, Any] | None,
    expected: tuple[int, int] | None,
) -> None:
    """A missing image_size silently falling back to the default could evaluate a run
    at another resolution than it was trained at."""
    path = tmp_path / "params-pretrain.yaml"
    path.write_text(yaml.safe_dump({"eval": {}, "transform": transform}))
    warning = mocker.patch.object(eval_classification.logger, "warning")
    image_size = _read_image_size(path)
    if expected is not None:
        warning.assert_not_called()
        assert image_size == expected
        return
    assert image_size == (224, 224)  # DINOv2ViTTransformArgs' default
    warning.assert_called_once()
    assert "transform.image_size" in warning.call_args.args[0]


def test_eval_classification__invalid_image_size(
    tmp_path: Path, checkpoint_path: Path, eval_config_path: Path
) -> None:
    run_config = {
        "transform": {"image_size": "large"},
        **yaml.safe_load(eval_config_path.read_text()),
    }
    params_path = tmp_path / "params-pretrain.yaml"
    params_path.write_text(yaml.safe_dump(run_config))
    with pytest.raises(ValidationError, match="image_size"):
        eval_classification.eval_classification(
            out=tmp_path / "metrics.json",
            checkpoint=checkpoint_path,
            eval_config=params_path,
            accelerator="cpu",
        )


def test_eval_classification__no_eval_block(
    tmp_path: Path, checkpoint_path: Path
) -> None:
    params_path = tmp_path / "params-pretrain.yaml"
    params_path.write_text(yaml.safe_dump({"method": {"center_method": "softmax"}}))
    with pytest.raises(ValueError, match="no top-level 'eval:' block"):
        eval_classification.eval_classification(
            out=tmp_path / "metrics.json",
            checkpoint=checkpoint_path,
            eval_config=params_path,
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
