#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import functools
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from kneeno import MODEL_SELECTION_KEYS, LabeledExternalKneeMRIDataset
from kneeno.evaluation.dataset import KneeNoEvalDataset, LabeledInternalKneeMRIDataset
from kneeno.evaluation.splits import write_manifest_for
from pydantic import ValidationError
from pytest import LogCaptureFixture
from pytest_mock import MockerFixture
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from lightly_train._callbacks.callback_args import CallbackArgs
from lightly_train._callbacks.checkpoint import ModelCheckpoint
from lightly_train._callbacks.kneeno_eval import KneeNoEval, KneeNoEvalArgs
from lightly_train._checkpoint import CheckpointLightlyTrainModels
from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._torch_helpers import _torch_weights_only_false
from lightly_train._transforms.transform import NormalizeArgs

from .. import helpers

IMAGE_SIZE = (16, 8, 4)  # (H, W, D), non-cubic to catch an axis-order swap.


class _FakeLabeledDataset(
    torch.utils.data.Dataset[tuple[tuple[torch.Tensor, ...], torch.Tensor, str]]
):
    """Stands in for one labeled KneeNo dataset (one global split of it), which needs data
    on the cluster.

    ``raw=True`` mimics LabeledExternalKneeMRIDataset: int16 volumes far outside 0..255 instead of
    the internal dataset's uint8 ones, and the source ``"external"``. ``samples`` holds
    ``(case_id, depths, labels)`` like a real one's, which ``KneeNoEvalDataset`` reads (case ids
    ``<prefix>0``, ...: distinct prefixes keep its parts' case ids apart).
    """

    num_classes = 4
    label_name_map = {i: f"label_{i}" for i in range(num_classes)}
    # One item is an exam: one volume per sequence, in this order.
    sequences = LabeledInternalKneeMRIDataset.SEQUENCES
    orientations = LabeledInternalKneeMRIDataset.orientations.fget(
        LabeledInternalKneeMRIDataset
    )

    def __init__(self, n: int = 16, raw: bool = False, prefix: str = "c") -> None:
        self.raw = raw
        self.samples = [
            (f"{prefix}{i}", None, self._labels(f"{prefix}{i}")) for i in range(n)
        ]

    def _labels(self, case_id: str) -> list[int]:
        gen = torch.Generator().manual_seed(sum(map(ord, case_id)))
        return (torch.rand(self.num_classes, generator=gen) > 0.5).int().tolist()

    def __len__(self) -> int:
        return len(self.samples)

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
        label = torch.tensor(self.samples[index][2], dtype=torch.float32)
        return volumes, label, "external" if self.raw else "internal"


def _fake_eval_dataset(test_sets: tuple[str, ...] = ("internal",)) -> KneeNoEvalDataset:
    """What ``KneeNoEvalDataset.from_config`` would build, from fakes: a 16-exam internal
    train part and an 8-exam test set per name (``external`` with raw volumes)."""
    tests = {
        name: _FakeLabeledDataset(n=8, raw=name == "external", prefix=name[0])
        for name in test_sets
    }
    return KneeNoEvalDataset(_FakeLabeledDataset(prefix="t"), tests)


#: Where ``_split()`` writes its manifest; removed when the test process exits.
_MANIFEST_DIR = tempfile.TemporaryDirectory(prefix="lightly-kneeno-manifest-")


@functools.cache
def _manifest() -> Path:
    # Every test set builds the same fake train part, so one manifest serves them all.
    path = Path(_MANIFEST_DIR.name) / "split_manifest.json"
    return write_manifest_for(_fake_eval_dataset(), path)


def _split() -> dict[str, Any]:
    """``eval.split``: the one split of the fake dataset's manifest."""
    return {"manifest": str(_manifest()), "level": 1.0, "rep": 0}


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "split": _split(),
        "data": {"test_sets": ["internal"], "num_workers": 0},
        "extraction_batch_size": 4,
        "linear": {"max_epochs": 1, "batch_size": 4},
        "linear_pool": {"max_epochs": 1, "batch_size": 4},
        "attentive_pool": {"max_epochs": 1, "batch_size": 4, "num_heads": 2},
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
    return helpers.dummy_dinov2_vit_model(patch_size=(2, 2, 2), img_size=IMAGE_SIZE)


#: The default ``out_dir`` of ``_callback()``; nothing is exported there without
#: ``model_selection: true``.
_OUT_DIR = tempfile.TemporaryDirectory(prefix="lightly-kneeno-out-")


def _callback(out_dir: Path | None = None, **overrides: Any) -> KneeNoEval:
    return KneeNoEval(
        wrapped_model=_wrapper(),
        image_size=IMAGE_SIZE,
        normalize_args=NormalizeArgs(),
        # Not the defaults, so the tests can tell they reach the adapter.
        resize_interpolation="cubic",
        resize_upscale_interpolation="linear",
        config=_config(**overrides),
        out_dir=Path(_OUT_DIR.name) if out_dir is None else out_dir,
    )


def _fake_trainer(mocker: MockerFixture, epoch: int, max_epochs: int = 1000) -> Any:
    trainer = mocker.MagicMock()
    trainer.current_epoch = epoch
    trainer.max_epochs = max_epochs
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
        "kneeno.evaluation.classification.KneeNoEvalDataset.from_config",
        return_value=_fake_eval_dataset(),
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
    # The test metrics, plus each k's val metric and the temperature it picked there.
    assert all(
        key.startswith(("eval/internal/knn/", "eval/evaluation_val/knn/"))
        for key in logged
    ), logged
    assert all(isinstance(value, float) for value in logged.values())
    # Nothing to reduce: every rank already holds the same broadcast result.
    assert module.log_dict.call_args.kwargs["sync_dist"] is False


@pytest.mark.parametrize("test_sets", [("internal",), ("internal", "external")])
def test_on_train_epoch_end__test_sets_select_dataset_and_preprocessing(
    mocker: MockerFixture, test_sets: tuple[str, ...]
) -> None:
    callback = _callback(data={"test_sets": list(test_sets), "num_workers": 0})
    dataset = mocker.patch(
        "kneeno.evaluation.classification.KneeNoEvalDataset.from_config",
        return_value=_fake_eval_dataset(test_sets),
    )
    adapter = mocker.patch(
        "lightly_train._callbacks.kneeno_eval.DINOv2Adapter", wraps=DINOv2Adapter
    )
    module = _fake_module(mocker)
    to_uint8 = mocker.spy(LabeledExternalKneeMRIDataset, "to_uint8")

    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)

    dataset.assert_called_once()
    assert dataset.call_args.args[0]["test_sets"] == list(test_sets)
    # The adapter learns each volume's source from the dataset's items: only external
    # volumes are quantized (num_workers is 0, so the spy sees every call).
    assert to_uint8.called == ("external" in test_sets)
    assert adapter.call_args.kwargs["resize_interpolation"] == "cubic"
    assert adapter.call_args.kwargs["resize_upscale_interpolation"] == "linear"
    module.log_dict.assert_called_once()
    logged = module.log_dict.call_args.args[0]
    # Every test set, plus the val metrics (evaluation_val/...), which never test.
    assert {key.split("/")[1] for key in logged} == {*test_sets, "evaluation_val"}
    assert all(math.isfinite(v) for v in logged.values())


def test_on_train_epoch_end__last_epoch_runs_every_enabled_task(
    mocker: MockerFixture,
) -> None:
    callback = _callback(freq={"knn": 7, "linear_pool": 7, "linear": 3})
    evaluator = mocker.MagicMock()
    evaluator.evaluate.return_value = {}
    callback._evaluator = evaluator
    module = _fake_module(mocker)

    # Epoch 9 of 10 is the last; no frequency divides 10.
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=8, max_epochs=10), module)
    evaluator.evaluate.assert_called_once()
    assert evaluator.evaluate.call_args.kwargs["tasks"] == ["linear"]  # 9 % 3 == 0
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=9, max_epochs=10), module)
    assert evaluator.evaluate.call_args.kwargs["tasks"] == [
        "knn",
        "linear",
        "linear_pool",
    ]


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
        "kneeno.evaluation.classification.KneeNoEvalDataset.from_config",
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
    assert any(tag.startswith("eval/internal/knn/") for tag in events.Tags()["scalars"])


def test_on_train_end__without_evaluator_is_a_noop(mocker: MockerFixture) -> None:
    """No task was ever due: training ends without building (and loading) the evaluator."""
    callback = _callback()
    build = mocker.patch(
        "kneeno.evaluation.classification.KneeNoEvalDataset.from_config"
    )

    callback.on_train_end(_fake_trainer(mocker, epoch=0), _fake_module(mocker))

    build.assert_not_called()
    assert callback._evaluator is None


def test_on_train_end__after_evaluation_disabled_itself(mocker: MockerFixture) -> None:
    callback = _callback()
    mocker.patch(
        "kneeno.evaluation.classification.KneeNoEvalDataset.from_config",
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


# -- Best-model selection (eval.model_selection) ---------------------------------------

_SELECTION: dict[str, Any] = {
    "model_selection": True,
    "freq": {"knn": 1, "linear_pool": 1},
}


def _selection_metrics(score: float) -> dict[str, float]:
    return {key: score for key in MODEL_SELECTION_KEYS} | {"internal/knn/k5/auroc": 0.5}


def _mock_evaluator(
    callback: KneeNoEval, mocker: MockerFixture, scores: list[float]
) -> Any:
    """Skip the real evaluation: each due epoch returns the next score's metrics."""
    evaluator = mocker.MagicMock()
    evaluator.evaluate.side_effect = [_selection_metrics(s) for s in scores]
    callback._evaluator = evaluator
    return evaluator


def _perturb(wrapper: Any, value: float) -> None:
    with torch.no_grad():
        next(wrapper.get_model().parameters()).fill_(value)


def _first_param(path: Path) -> torch.Tensor:
    state = torch.load(path, weights_only=True)
    param: torch.Tensor = next(iter(state.values()))
    return param


def test_model_selection__exports_only_on_improvement(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    callback = _callback(out_dir=tmp_path, **_SELECTION)
    _mock_evaluator(callback, mocker, [0.6, 0.55, 0.6, 0.7])
    module = _fake_module(mocker)
    best = tmp_path / "exported_best.pt"

    for epoch, value in enumerate([1.0, 2.0, 3.0, 4.0]):
        _perturb(callback._wrapped_model, value)
        callback.on_train_epoch_end(_fake_trainer(mocker, epoch=epoch), module)
        # epoch 0 (0.6) is the best until epoch 3 (0.7); a tie at epoch 2 does not count
        expected = 4.0 if epoch == 3 else 1.0
        assert torch.all(_first_param(best) == expected), epoch

    # The same format as exported_last.pt: the bare ViT's state dict.
    assert (
        torch.load(best, weights_only=True).keys()
        == callback._wrapped_model.get_model().state_dict().keys()
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["exported_best.pt"]
    assert callback.state_dict() == {"best_model": {"best_score": 0.7, "best_epoch": 4}}
    logged = module.log_dict.call_args.args[0]
    assert logged["eval/model_selection/score"] == pytest.approx(0.7)
    assert logged["eval/model_selection/best_score"] == pytest.approx(0.7)


def test_model_selection__exports_the_online_encoder(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    callback = _callback(out_dir=tmp_path, encoder="online", **_SELECTION)
    _mock_evaluator(callback, mocker, [0.6])
    module = _fake_module(mocker)
    student = _wrapper()
    module.student_embedding_model.wrapped_model = student
    _perturb(student, 5.0)
    _perturb(callback._wrapped_model, 1.0)

    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)
    assert torch.all(_first_param(tmp_path / "exported_best.pt") == 5.0)


def test_model_selection__off_exports_nothing(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    callback = _callback(out_dir=tmp_path, freq={"knn": 1, "linear_pool": 1})
    _mock_evaluator(callback, mocker, [0.6])
    module = _fake_module(mocker)

    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)
    assert list(tmp_path.iterdir()) == []
    assert not any("model_selection" in k for k in module.log_dict.call_args.args[0])


def test_model_selection__epoch_without_score_logs_no_score(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    callback = _callback(out_dir=tmp_path, **_SELECTION)
    evaluator = _mock_evaluator(callback, mocker, [])
    evaluator.evaluate.side_effect = [{MODEL_SELECTION_KEYS[0]: 0.9}]
    module = _fake_module(mocker)

    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)
    assert list(tmp_path.iterdir()) == []
    assert not any("model_selection" in k for k in module.log_dict.call_args.args[0])


def test_model_selection__failed_export_keeps_the_previous_best(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    callback = _callback(out_dir=tmp_path, **_SELECTION)
    _mock_evaluator(callback, mocker, [0.6, 0.7, 0.7])
    module = _fake_module(mocker)
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)

    mocker.patch(
        "lightly_train._callbacks.kneeno_eval.common_helpers.export_model",
        side_effect=OSError("disk full"),
    )
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=1), module)
    assert callback.state_dict()["best_model"] == {"best_score": 0.6, "best_epoch": 1}

    mocker.stopall()
    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=2), module)
    assert callback.state_dict()["best_model"] == {"best_score": 0.7, "best_epoch": 3}


def test_on_train_epoch_end__logs_under_the_configured_tag_prefix(
    mocker: MockerFixture,
) -> None:
    callback = _callback(logging={"tag_prefix": "in_training"}, **_SELECTION)
    _mock_evaluator(callback, mocker, [0.6])
    module = _fake_module(mocker)

    callback.on_train_epoch_end(_fake_trainer(mocker, epoch=0), module)
    logged = module.log_dict.call_args.args[0]
    assert logged and all(key.startswith("in_training/") for key in logged), logged
    assert "in_training/internal/knn/k100/auroc" in logged
    assert "in_training/model_selection/best_score" in logged


def test_model_selection__state_dict_round_trip(caplog: LogCaptureFixture) -> None:
    callback = _callback(**_SELECTION)
    callback._best_model_tracker.update(_selection_metrics(0.65), epoch=5)
    restored = _callback(**_SELECTION)
    restored.load_state_dict(callback.state_dict())
    assert restored.state_dict() == callback.state_dict()
    # A checkpoint from before model selection has no state for this callback's key.
    fresh = _callback(**_SELECTION)
    caplog.clear()  # drop the restore above
    with caplog.at_level("INFO"):
        fresh.load_state_dict({})
    assert fresh.state_dict() == _callback(**_SELECTION).state_dict()
    assert "no model selection state" in caplog.text
    assert "Restored model selection state" not in caplog.text


def test_model_selection__invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="freq.linear_pool > 0"):
        _callback(model_selection=True, freq={"knn": 1})


class _Module(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss: torch.Tensor = self.layer(batch[0]).sum()
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        return DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)


class _KillAtEpoch(Callback):
    """Stands in for a SLURM time limit: training stops without on_train_end."""

    def __init__(self, epoch: int) -> None:
        self._epoch = epoch

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.current_epoch == self._epoch:
            raise RuntimeError("killed")


def _fit(callbacks: list[Callback], max_epochs: int, ckpt_path: Path | None) -> None:
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )
    # Like lightly-train's own resume (_commands/train.py): its checkpoints pickle models.
    with _torch_weights_only_false():
        trainer.fit(_Module(), ckpt_path=ckpt_path)


def test_model_selection__resume_restores_the_best(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    # Real Lightning: ModelCheckpoint is listed first but must run after the eval
    # callback, so last.ckpt holds the best score of the epoch it was written at.
    def checkpoint() -> ModelCheckpoint:
        model = helpers.DummyCustomModel()
        return ModelCheckpoint(
            models=CheckpointLightlyTrainModels(
                model=model.get_model(),
                wrapped_model=model,
                embedding_model=EmbeddingModel(wrapped_model=model),
            ),
            normalize_args=NormalizeArgs(),
            dirpath=str(tmp_path / "checkpoints"),
            license_info="",
            save_last=True,
            every_n_epochs=50,
        )

    out = tmp_path / "exported_models"
    first = _callback(out_dir=out, **_SELECTION)
    _mock_evaluator(first, mocker, [0.6, 0.7])
    with pytest.raises(RuntimeError, match="killed"):
        _fit([checkpoint(), first, _KillAtEpoch(epoch=2)], max_epochs=4, ckpt_path=None)
    exported = (out / "exported_best.pt").read_bytes()

    resumed = _callback(out_dir=out, **_SELECTION)
    evaluator = _mock_evaluator(resumed, mocker, [0.65, 0.8])
    _fit(
        [checkpoint(), resumed],
        max_epochs=3,
        ckpt_path=tmp_path / "checkpoints" / "last.ckpt",
    )

    # Epoch 3's 0.65 is below the restored 0.7 of epoch 2: nothing exported.
    assert evaluator.evaluate.call_count == 1
    assert resumed.state_dict()["best_model"] == {"best_score": 0.7, "best_epoch": 2}
    assert (out / "exported_best.pt").read_bytes() == exported
