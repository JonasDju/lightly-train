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
from kneeno import LabeledExternalKneeMRIDataset
from kneeno.evaluation.config import ALL_TASKS
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
from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._transforms.transform import NormalizeArgs

from .. import helpers

IMAGE_SIZE = (16, 8, 4)  # (H, W, D), non-cubic to catch an axis-order swap.
# The tasks this volumetric encoder serves; the slice tasks need a slice-wise (2D) one.
VOLUME_TASKS = ["knn", "linear", "linear_pool", "attentive_pool"]


class _FakeLabeledDataset(
    torch.utils.data.Dataset[tuple[tuple[torch.Tensor, ...], torch.Tensor, str]]
):
    """Stands in for the labeled KneeNo datasets, which need data on the cluster.

    ``raw=True`` mimics LabeledExternalKneeMRIDataset: int16 volumes far outside 0..255 instead of
    the internal dataset's uint8 ones, and the source ``"external"``.
    """

    num_classes = 4
    # One item is an exam: one volume per sequence, in this order.
    sequences = ("sag", "st1", "cor", "tra")

    def __init__(self, raw: bool = False) -> None:
        self.raw = raw

    def __len__(self) -> int:
        return 12

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, str]:
        gen = torch.Generator().manual_seed(index)
        # Mixed native depths: DINOv2Adapter resizes depth, so they batch without resampling.
        volumes = tuple(
            torch.randint(
                0,
                4000 if self.raw else 256,
                (1, 5 + ((index + s) % 3), 20, 11),
                generator=gen,
                dtype=torch.int16 if self.raw else torch.uint8,
            )
            for s in range(len(self.sequences))
        )
        label = (torch.rand(self.num_classes, generator=gen) > 0.5).float()
        return volumes, label, "external" if self.raw else "internal"


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


def _eval_block(tmp_path: Path) -> dict[str, Any]:
    out_dir = tmp_path / "eval" / "post-training"
    return {
        "data": {"dataset_type": "internal", "num_workers": 0},
        "logging": {
            "tensorboard_dir": str(out_dir / "tb"),
            "per_label_dir": str(out_dir / "eval_per_label"),
        },
        # Every task explicit: a missing key keeps KneeNo's default frequency.
        "freq": {task: 1 if task in VOLUME_TASKS else None for task in ALL_TASKS},
        "knn": {"batch_size": 4},
        "linear": {"max_epochs": 1, "batch_size": 4},
        "linear_pool": {"max_epochs": 1, "batch_size": 4},
        "attentive_pool": {"max_epochs": 1, "batch_size": 4, "num_heads": 2},
    }


def _write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _out_dir(tmp_path: Path) -> Path:
    return tmp_path / "eval" / "post-training"


def _only(config: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    """``config`` with only ``tasks`` enabled."""
    config["eval"]["freq"] = {task: 1 if task in tasks else None for task in ALL_TASKS}
    return config


@pytest.fixture
def run_config(tmp_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint_path),
        "image_size": list(IMAGE_SIZE),
        "resize_interpolation": "area",
        "resize_upscale_interpolation": "linear",
        "eval": _eval_block(tmp_path),
    }


@pytest.fixture
def fake_dataset(mocker: MockerFixture) -> None:
    mocker.patch(
        "kneeno.evaluation.classification.LabeledInternalKneeMRIDataset",
        return_value=_FakeLabeledDataset(),
    )


def test_eval_classification(
    tmp_path: Path, run_config: dict[str, Any], fake_dataset: None
) -> None:
    # Only the tasks with a non-null freq run; "linear" needs a cls token, which DINOv2
    # has (unlike V-JEPA 2.1).
    config_path = _write_config(tmp_path, _only(run_config, ["knn", "linear"]))
    eval_classification.eval_classification(eval_config=config_path, accelerator="cpu")
    out_dir = _out_dir(tmp_path)
    metrics = json.loads((out_dir / "results.json").read_text())
    assert {key.split("/")[0] for key in metrics} == {"knn", "linear"}
    assert all(isinstance(value, float) for value in metrics.values())
    assert yaml.safe_load((out_dir / "params.yaml").read_text()) == run_config
    assert any((out_dir / "eval_per_label").iterdir())
    assert any((out_dir / "tb").iterdir())


def test_eval_classification__all_tasks(
    tmp_path: Path, run_config: dict[str, Any], fake_dataset: None
) -> None:
    # Every task this encoder serves.
    config_path = _write_config(tmp_path, run_config)
    eval_classification.eval_classification(eval_config=config_path, accelerator="cpu")
    metrics = json.loads((_out_dir(tmp_path) / "results.json").read_text())
    assert {key.split("/")[0] for key in metrics} == set(VOLUME_TASKS)
    assert all(math.isfinite(value) for value in metrics.values())


def test_eval_classification__missing_freq_keeps_default(
    tmp_path: Path, run_config: dict[str, Any], mocker: MockerFixture
) -> None:
    # The deep-merge trap: a task left out of freq keeps KneeNo's default frequency.
    run_config["eval"]["freq"] = {"knn": None}
    config_path = _write_config(tmp_path, run_config)
    evaluate = mocker.patch(
        "kneeno.evaluation.ClassificationEvaluator.evaluate", return_value={}
    )
    mocker.patch(
        "kneeno.evaluation.classification.LabeledInternalKneeMRIDataset",
        return_value=_FakeLabeledDataset(),
    )
    eval_classification.eval_classification(eval_config=config_path, accelerator="cpu")
    tasks = evaluate.call_args.kwargs["tasks"]
    assert tasks == ["linear", "linear_pool", "attentive_pool"]


@pytest.mark.parametrize("off", [None, 0])
def test_eval_classification__no_task(
    tmp_path: Path, run_config: dict[str, Any], off: int | None
) -> None:
    run_config["eval"]["freq"] = {task: off for task in ALL_TASKS}
    with pytest.raises(ValueError, match="No task is enabled"):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )
    assert not _out_dir(tmp_path).exists()


def test_eval_classification__env_vars(
    tmp_path: Path,
    run_config: dict[str, Any],
    checkpoint_path: Path,
    fake_dataset: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRETRAIN_DIR", str(tmp_path))
    run_config = _only(run_config, ["knn"])
    run_config["checkpoint"] = "${PRETRAIN_DIR}/" + checkpoint_path.name
    run_config["eval"]["logging"] = {
        "tensorboard_dir": "$PRETRAIN_DIR/eval/post-training/tb",
        "per_label_dir": "${PRETRAIN_DIR}/eval/post-training/eval_per_label",
    }
    eval_classification.eval_classification(
        eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
    )
    out_dir = _out_dir(tmp_path)
    assert json.loads((out_dir / "results.json").read_text())
    # params.yaml records the expanded paths.
    params = yaml.safe_load((out_dir / "params.yaml").read_text())
    assert params["checkpoint"] == str(checkpoint_path)
    assert params["eval"]["logging"]["per_label_dir"] == str(out_dir / "eval_per_label")


def test_eval_classification__unset_env_var(
    tmp_path: Path, run_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRETRAIN_JOB_ID", raising=False)
    run_config["eval"]["logging"]["per_label_dir"] = (
        str(tmp_path) + "/${PRETRAIN_JOB_ID}/eval_per_label"
    )
    with pytest.raises(ValueError, match=r"not set: \['\$\{PRETRAIN_JOB_ID\}'\]"):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )


def test_eval_classification__no_per_label_dir(
    tmp_path: Path,
    run_config: dict[str, Any],
    checkpoint_path: Path,
    fake_dataset: None,
    mocker: MockerFixture,
) -> None:
    run_config = _only(run_config, ["knn"])
    run_config["eval"]["logging"]["per_label_dir"] = None
    warning = mocker.patch.object(eval_classification.logger, "warning")
    eval_classification.eval_classification(
        eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
    )
    # results.json next to the checkpoint, and no params.yaml anywhere.
    assert json.loads((checkpoint_path.parent / "results.json").read_text())
    assert not list(tmp_path.rglob("params.yaml"))
    assert "per_label_dir is not set" in warning.call_args.args[0]


@pytest.mark.parametrize(
    "existing", ["results.json", "params.yaml", "eval_per_label", "tb"]
)
def test_eval_classification__previous_results(
    tmp_path: Path,
    run_config: dict[str, Any],
    fake_dataset: None,
    mocker: MockerFixture,
    existing: str,
) -> None:
    config_path = _write_config(tmp_path, _only(run_config, ["knn"]))
    path = _out_dir(tmp_path) / existing
    path.parent.mkdir(parents=True)
    if existing.endswith((".json", ".yaml")):
        path.write_text("old")
    else:
        path.mkdir()
    from_path = mocker.spy(Checkpoint, "from_path")
    with pytest.raises(ValueError, match="already exists|earlier evaluation"):
        eval_classification.eval_classification(
            eval_config=config_path, accelerator="cpu"
        )
    # Refused before the checkpoint is even loaded.
    from_path.assert_not_called()

    eval_classification.eval_classification(
        eval_config=config_path, accelerator="cpu", overwrite=True
    )
    assert json.loads((_out_dir(tmp_path) / "results.json").read_text())


@pytest.mark.parametrize(
    "change, match",
    [
        # A leftover pretraining key or block is an error, not silently ignored.
        ({"out": "metrics.json"}, r"out\n\s+Extra inputs"),
        ({"transform": {"image_size": [224, 224]}}, r"transform\n\s+Extra inputs"),
        ({"image_size": None}, r"image_size\n\s+Field required"),
        ({"image_size": [224, 224]}, r"image_size.2\n\s+Field required"),
        ({"checkpoint": None}, r"checkpoint\n\s+Field required"),
    ],
)
def test_eval_classification__invalid_config(
    tmp_path: Path, run_config: dict[str, Any], change: dict[str, Any], match: str
) -> None:
    run_config.update(change)
    run_config = {key: value for key, value in run_config.items() if value is not None}
    with pytest.raises(ValueError, match=match):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )


@pytest.mark.parametrize("eval_block", [None, "missing"])
def test_eval_classification__no_eval_block(
    tmp_path: Path, run_config: dict[str, Any], eval_block: str | None
) -> None:
    if eval_block is None:
        run_config["eval"] = None
    else:
        del run_config["eval"]
    with pytest.raises(ValueError, match="no top-level 'eval:' block"):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )


@pytest.mark.parametrize(
    "dataset_type, dataset_class",
    [
        ("internal", "LabeledInternalKneeMRIDataset"),
        ("external", "LabeledExternalKneeMRIDataset"),
    ],
)
def test_eval_classification__dataset_type_selects_dataset_and_preprocessing(
    tmp_path: Path,
    run_config: dict[str, Any],
    mocker: MockerFixture,
    dataset_type: str,
    dataset_class: str,
) -> None:
    run_config = _only(run_config, ["knn"])
    run_config["eval"]["data"]["dataset_type"] = dataset_type
    dataset = mocker.patch(
        f"kneeno.evaluation.classification.{dataset_class}",
        return_value=_FakeLabeledDataset(raw=dataset_type == "external"),
    )
    adapter = mocker.patch(
        "lightly_train._commands.eval_classification.DINOv2Adapter",
        wraps=DINOv2Adapter,
    )
    to_uint8 = mocker.spy(LabeledExternalKneeMRIDataset, "to_uint8")
    eval_classification.eval_classification(
        eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
    )
    dataset.assert_called_once()
    # The adapter learns each volume's source from the dataset's items: only external
    # volumes are quantized (num_workers is 0, so the spy sees every call).
    assert to_uint8.called == (dataset_type == "external")
    assert adapter.call_args.kwargs["image_size"] == IMAGE_SIZE
    metrics = json.loads((_out_dir(tmp_path) / "results.json").read_text())
    assert metrics and all(math.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize(
    "resize_args",
    [
        {"resize_interpolation": "area+nearest"},
        {"resize_upscale_interpolation": "linear+nearest"},
        # null is a value of its own: resize_interpolation on enlarged axes too.
        {"resize_interpolation": "area+nearest", "resize_upscale_interpolation": None},
    ],
)
def test_eval_classification__resize_args(
    tmp_path: Path,
    run_config: dict[str, Any],
    fake_dataset: None,
    mocker: MockerFixture,
    resize_args: dict[str, Any],
) -> None:
    run_config = _only(run_config, ["knn"])
    run_config.update(resize_args)
    adapter = mocker.patch(
        "lightly_train._commands.eval_classification.DINOv2Adapter",
        wraps=DINOv2Adapter,
    )
    warning = mocker.patch.object(eval_classification.logger, "warning")
    eval_classification.eval_classification(
        eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
    )
    for key in ("resize_interpolation", "resize_upscale_interpolation"):
        assert adapter.call_args.kwargs[key] == run_config[key]
    warning.assert_not_called()


@pytest.mark.parametrize(
    "missing",
    [
        ["resize_interpolation"],
        ["resize_upscale_interpolation"],
        ["resize_interpolation", "resize_upscale_interpolation"],
    ],
)
def test_eval_classification__default_resize_args_warn(
    tmp_path: Path,
    run_config: dict[str, Any],
    fake_dataset: None,
    mocker: MockerFixture,
    missing: list[str],
) -> None:
    """A missing resize key silently falling back to the default could evaluate a run
    with other resampling than it was trained with."""
    run_config = _only(run_config, ["knn"])
    run_config["resize_interpolation"] = "area+nearest"
    run_config["resize_upscale_interpolation"] = None
    for key in missing:
        del run_config[key]
    adapter = mocker.patch(
        "lightly_train._commands.eval_classification.DINOv2Adapter",
        wraps=DINOv2Adapter,
    )
    warning = mocker.patch.object(eval_classification.logger, "warning")
    eval_classification.eval_classification(
        eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
    )
    # DINOv2ViTTransformArgs' defaults, as in a training run without the keys.
    defaults = {
        "resize_interpolation": "area",
        "resize_upscale_interpolation": "linear",
    }
    for key in ("resize_interpolation", "resize_upscale_interpolation"):
        expected = defaults[key] if key in missing else run_config[key]
        assert adapter.call_args.kwargs[key] == expected
    warning.assert_called_once()
    message = warning.call_args.args[0]
    for key in ("resize_interpolation", "resize_upscale_interpolation"):
        assert (f"{key}=" in message) == (key in missing)


@pytest.mark.parametrize("spec", ["trilinear", "area+trilinear", "area+nearest+linear"])
def test_eval_classification__invalid_resize_interpolation(
    tmp_path: Path, run_config: dict[str, Any], spec: str
) -> None:
    run_config["resize_interpolation"] = spec
    with pytest.raises(ValueError, match=r"resize_interpolation\n"):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )


def test_eval_classification__online_encoder(
    tmp_path: Path,
    run_config: dict[str, Any],
    fake_dataset: None,
    mocker: MockerFixture,
) -> None:
    run_config = _only(run_config, ["knn"])
    run_config["eval"]["encoder"] = "online"
    load_student = mocker.spy(eval_classification, "_load_student_weights")
    eval_classification.eval_classification(
        eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
    )
    load_student.assert_called_once()
    assert json.loads((_out_dir(tmp_path) / "results.json").read_text())


def test_eval_classification__invalid_encoder(
    tmp_path: Path, run_config: dict[str, Any]
) -> None:
    run_config["eval"]["encoder"] = "teacher"
    with pytest.raises(ValueError, match="Invalid eval.encoder"):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
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
    tmp_path: Path, run_config: dict[str, Any], fake_dataset: None
) -> None:
    config_path = _write_config(tmp_path, _only(run_config, ["knn"]))
    config = OmegaConf.create(dict(eval_config=str(config_path), accelerator="cpu"))
    eval_classification.eval_classification_from_dictconfig(config=config)
    assert json.loads((_out_dir(tmp_path) / "results.json").read_text())


def test_eval_classification__parameters() -> None:
    helpers.assert_same_params(
        a=eval_classification.eval_classification, b=EvalClassificationConfig
    )
