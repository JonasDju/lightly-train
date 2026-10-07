#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError
from pytest import LogCaptureFixture
from pytest_mock import MockerFixture

from lightly_train._callbacks.callback_args import CallbackArgs
from lightly_train._callbacks.kneeno_eval import KneeNoEval, KneeNoEvalArgs
from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._transforms.transform import NormalizeArgs

from .. import helpers

IMAGE_SIZE = (16, 16)  # (H, W)


class _FakeDINOv2Adapter(DINOv2Adapter):
    """Test-only stand-in for the stub adapter, so the callback's wiring can be tested.

    Feeds the volume's centre slice to the real 2D encoder. Not a proposal for how the
    real adapter should handle volumes.
    """

    def prepare_input(
        self, volume: torch.Tensor, orientation: str | None = None
    ) -> torch.Tensor:
        image = volume[:, volume.shape[1] // 2].float()[None]  # (1, 1, H, W)
        image = F.interpolate(image, size=self.image_size, mode="bilinear")[0]
        image = image.expand(self.num_channels, -1, -1) / 255.0
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        return (image - mean) / std

    def forward_features(
        self, model: Any, batch: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        out = model.forward_features(batch)
        patches = out["features"].flatten(2).permute(0, 2, 1)  # (B, P, D)
        return {"cls": out["cls_token"], "patches": patches}


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

    def __init__(self, n: int = 16, raw: bool = False) -> None:
        self.n = n
        self.raw = raw

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        gen = torch.Generator().manual_seed(index)
        # Mixed native depths: the adapter's output does not depend on the depth.
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


def _wrapper() -> Any:
    return helpers.dummy_dinov2_vit_model(patch_size=2, img_size=IMAGE_SIZE[0])


def _callback(**overrides: Any) -> KneeNoEval:
    return KneeNoEval(
        wrapped_model=_wrapper(),
        num_channels=3,
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


def _install_dataset(
    callback: KneeNoEval, mocker: MockerFixture, fake_adapter: bool = True
) -> None:
    """Inject a synthetic labeled dataset instead of the cluster one and, unless
    ``fake_adapter`` is False, a working adapter instead of the stub."""
    mocker.patch(
        "kneeno.evaluation.classification.LabeledInternalKneeMRIDataset",
        return_value=_FakeLabeledDataset(),
    )
    if fake_adapter:
        mocker.patch(
            "lightly_train._callbacks.kneeno_eval.DINOv2Adapter", _FakeDINOv2Adapter
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


@pytest.mark.parametrize(
    "dataset_type, dataset_class",
    [
        ("internal", "LabeledInternalKneeMRIDataset"),
        ("external", "LabeledExternalKneeMRIDataset"),
    ],
)
def test_on_train_epoch_end__dataset_type_selects_dataset_and_preprocessing(
    mocker: MockerFixture, dataset_type: str, dataset_class: str
) -> None:
    callback = _callback(data={"dataset_type": dataset_type, "num_workers": 0})
    dataset = mocker.patch(
        f"kneeno.evaluation.classification.{dataset_class}",
        return_value=_FakeLabeledDataset(raw=dataset_type == "external"),
    )
    adapter = mocker.patch(
        "lightly_train._callbacks.kneeno_eval.DINOv2Adapter", wraps=_FakeDINOv2Adapter
    )
    module = _fake_module(mocker)

    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)

    dataset.assert_called_once()
    assert adapter.call_args.kwargs["dataset_type"] == dataset_type
    assert adapter.call_args.kwargs["num_channels"] == 3
    assert adapter.call_args.kwargs["image_size"] == IMAGE_SIZE
    assert adapter.call_args.kwargs["normalize"] == (
        NormalizeArgs().mean,
        NormalizeArgs().std,
    )
    assert (
        adapter.call_args.kwargs["embed_dim"] == callback._wrapped_model.feature_dim()
    )
    module.log_dict.assert_called_once()
    assert all(math.isfinite(v) for v in module.log_dict.call_args.args[0].values())


def test_on_train_epoch_end__stub_adapter_raises(mocker: MockerFixture) -> None:
    # The real adapter is a stub on this branch: a due eval must fail loudly, not
    # silently log nothing.
    callback = _callback()
    _install_dataset(callback, mocker, fake_adapter=False)

    with pytest.raises(NotImplementedError, match="stub"):
        callback.on_train_epoch_end(
            _fake_trainer(mocker, epoch=0), _fake_module(mocker)
        )
    assert isinstance(callback._evaluator.adapter, DINOv2Adapter)  # type: ignore[union-attr]


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
    student_wrapper = _wrapper()
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
