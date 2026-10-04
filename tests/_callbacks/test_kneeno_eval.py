#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from pytest import LogCaptureFixture
from pytest_mock import MockerFixture

from lightly_train._callbacks.callback_args import CallbackArgs
from lightly_train._callbacks.kneeno_eval import KneeNoEval, KneeNoEvalArgs
from lightly_train._transforms.transform import NormalizeArgs

from .. import helpers

IMAGE_SIZE = (16, 8, 4)  # (H, W, D), non-cubic to catch an axis-order swap.


class _FakeLabeledDataset(
    torch.utils.data.Dataset[tuple[tuple[torch.Tensor, ...], torch.Tensor]]
):
    """Stands in for LabeledInternalKneeMRIDataset, which needs a dataset on the cluster."""

    num_classes = 4
    # One item is an exam: one volume per sequence, in this order.
    sequences = ("sag", "st1", "cor", "tra")

    def __init__(self, n: int = 16) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        gen = torch.Generator().manual_seed(index)
        # Mixed native depths: DINOv2Adapter resizes depth, so they batch without resampling.
        volumes = tuple(
            torch.randint(
                0,
                256,
                (1, 5 + ((index + s) % 3), 20, 11),
                generator=gen,
                dtype=torch.uint8,
            )
            for s in range(len(self.sequences))
        )
        label = (torch.rand(self.num_classes, generator=gen) > 0.5).float()
        return volumes, label


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "data": {"dataset_type": "internal", "num_workers": 0},
        "knn": {"batch_size": 4},
        "linear": {"epochs": 1, "batch_size": 4},
        "linear_pool": {"epochs": 1, "batch_size": 4},
        "attentive_pool": {"epochs": 1, "batch_size": 4, "num_heads": 2},
        "freq": {
            "knn": 1,
            "linear": None,
            "linear_pool": None,
            "attentive_pool": None,
        },
    }
    config.update(overrides)
    return config


def _callback(**overrides: Any) -> KneeNoEval:
    wrapper = helpers.dummy_dinov2_vit_model(patch_size=(2, 2, 2), img_size=IMAGE_SIZE)
    return KneeNoEval(
        wrapped_model=wrapper,
        image_size=IMAGE_SIZE,
        normalize_args=NormalizeArgs(),
        config=_config(**overrides),
    )


def _fake_trainer(mocker: MockerFixture, epoch: int) -> Any:
    trainer = mocker.MagicMock()
    trainer.current_epoch = epoch
    trainer.global_rank = 0
    trainer.world_size = 1
    return trainer


def _fake_module(mocker: MockerFixture) -> Any:
    module = mocker.MagicMock()
    module.device = torch.device("cpu")
    return module


def _install_dataset(callback: KneeNoEval, mocker: MockerFixture) -> None:
    """Inject a synthetic labeled dataset instead of the cluster one."""
    mocker.patch(
        "kneeno.evaluation.classification.LabeledInternalKneeMRIDataset",
        return_value=_FakeLabeledDataset(),
    )


def test_callback_args__kneeno_eval_off_by_default() -> None:
    # There is no default eval config: without one, nothing is evaluated.
    assert CallbackArgs().kneeno_eval is None


def test_kneeno_eval_args__config_required() -> None:
    with pytest.raises(ValidationError):
        KneeNoEvalArgs()  # type: ignore[call-arg]


def test_kneeno_eval__config_merged_over_kneeno_defaults() -> None:
    callback = _callback(knn={"k": [3]})
    assert callback._config["knn"]["k"] == [3]
    # Keys the passed config leaves out come from KneeNo's DEFAULT_EVAL_CONFIG.
    assert "metric" in callback._config["knn"]


def test_on_train_epoch_end__logs_metrics(mocker: MockerFixture) -> None:
    callback = _callback()
    _install_dataset(callback, mocker)
    module = _fake_module(mocker)

    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)

    module.log_dict.assert_called_once()
    logged = module.log_dict.call_args.args[0]
    assert logged, "expected metrics to be logged"
    assert all(key.startswith("eval/knn/") for key in logged), logged
    assert all(isinstance(value, float) for value in logged.values())
    # Nothing to reduce: every rank already holds the same broadcast result.
    assert module.log_dict.call_args.kwargs["sync_dist"] is False


def test_on_train_epoch_end__skips_when_no_task_is_due(mocker: MockerFixture) -> None:
    callback = _callback(
        freq={"knn": 3, "linear": None, "linear_pool": None, "attentive_pool": None},
    )
    _install_dataset(callback, mocker)
    module = _fake_module(mocker)

    # tasks_due runs a task with freq f when (epoch + 1) % f == 0.
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)
    module.log_dict.assert_not_called()

    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=2), module)
    module.log_dict.assert_called_once()


def test_on_train_epoch_end__disables_itself_when_dataset_is_missing(
    mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    """Evaluation is on by default, so a run without the labeled dataset must not die."""
    callback = _callback()
    mocker.patch(
        "kneeno.evaluation.classification.LabeledInternalKneeMRIDataset",
        side_effect=FileNotFoundError("no labels here"),
    )
    module = _fake_module(mocker)

    with caplog.at_level("WARNING"):
        callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)
    assert "Disabling KneeNo evaluation" in caplog.text
    module.log_dict.assert_not_called()

    # Second epoch: stays disabled and does not warn again.
    caplog.clear()
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=1), module)
    assert "Disabling KneeNo evaluation" not in caplog.text
    module.log_dict.assert_not_called()


def test_on_train_end__cleans_up_the_evaluator(mocker: MockerFixture) -> None:
    callback = _callback()
    _install_dataset(callback, mocker)
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), _fake_module(mocker))
    assert callback._evaluator is not None
    cleanup = mocker.spy(callback._evaluator, "cleanup")

    callback.on_train_end(_fake_trainer(mocker, epoch=0), _fake_module(mocker))

    cleanup.assert_called_once_with()


def test_on_train_end__flushes_kneeno_tensorboard_writer(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """With KneeNo's own writer switched on, everything logged is on disk after training."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    tb_dir = tmp_path / "tb"
    callback = _callback(logging={"tensorboard_dir": str(tb_dir)})
    _install_dataset(callback, mocker)
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), _fake_module(mocker))

    callback.on_train_end(_fake_trainer(mocker, epoch=0), _fake_module(mocker))

    events = EventAccumulator(str(tb_dir))
    events.Reload()
    assert any(tag.startswith("eval/knn/") for tag in events.Tags()["scalars"])


def test_on_train_end__without_evaluator_is_a_noop(mocker: MockerFixture) -> None:
    """No task was ever due: training ends without building (and loading) the evaluator."""
    callback = _callback()
    build = mocker.patch(
        "kneeno.evaluation.classification.LabeledInternalKneeMRIDataset"
    )

    callback.on_train_end(_fake_trainer(mocker, epoch=0), _fake_module(mocker))

    build.assert_not_called()
    assert callback._evaluator is None


def test_on_train_end__after_evaluation_disabled_itself(mocker: MockerFixture) -> None:
    callback = _callback()
    mocker.patch(
        "kneeno.evaluation.classification.LabeledInternalKneeMRIDataset",
        side_effect=FileNotFoundError("no labels here"),
    )
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), _fake_module(mocker))
    assert callback._disabled

    callback.on_train_end(
        _fake_trainer(mocker, epoch=0), _fake_module(mocker)
    )  # must not raise


@pytest.mark.parametrize("encoder", ["target", "online"])
def test_resolve_encoder(mocker: MockerFixture, encoder: str) -> None:
    callback = _callback(encoder=encoder)
    module = _fake_module(mocker)
    student_wrapper = helpers.dummy_dinov2_vit_model(
        patch_size=(2, 2, 2), img_size=IMAGE_SIZE
    )
    module.student_embedding_model.wrapped_model = student_wrapper

    resolved = callback._resolve_encoder(module)
    if encoder == "online":
        assert resolved is student_wrapper
    else:
        # The constructor's wrapper IS the teacher: DINOv2 EMA-updates it in place.
        assert resolved is callback._wrapped_model


def test_resolve_encoder__online_falls_back_without_student(
    mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    callback = _callback(encoder="online")
    module = mocker.MagicMock(spec=["device"])
    module.device = torch.device("cpu")

    with caplog.at_level("WARNING"):
        resolved = callback._resolve_encoder(module)
    assert resolved is callback._wrapped_model
    assert "no 'student_embedding_model'" in caplog.text
