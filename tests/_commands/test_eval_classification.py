#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import functools
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from kneeno import LabeledExternalKneeMRIDataset
from kneeno.evaluation import ClassificationEvaluator
from kneeno.evaluation.config import ALL_TASKS
from kneeno.evaluation.dataset import KneeNoEvalDataset, LabeledInternalKneeMRIDataset
from kneeno.evaluation.splits import write_manifest_for
from omegaconf import OmegaConf
from pytest_mock import MockerFixture

import lightly_train
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
from lightly_train._models import package_helpers
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._transforms.transform import NormalizeArgs

from .. import helpers

# (H, W): non-square to catch an axis-order swap, and a multiple of _vittest14's patch size.
IMAGE_SIZE = (28, 14)
DEPTH = 5  # slices per volume = eval.data.series_depth


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

    def __init__(self, n: int = 12, raw: bool = False, prefix: str = "c") -> None:
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
        label = torch.tensor(self.samples[index][2], dtype=torch.float32)
        return volumes, label, "external" if self.raw else "internal"


def _fake_eval_dataset(test_sets: tuple[str, ...] = ("internal",)) -> KneeNoEvalDataset:
    """What ``KneeNoEvalDataset.from_config`` would build, from fakes: a 12-exam internal
    train part and a 6-exam test set per name (``external`` with raw volumes)."""
    tests = {
        name: _FakeLabeledDataset(n=6, raw=name == "external", prefix=name[0])
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
def exported_path(tmp_path: Path, checkpoint_path: Path) -> Path:
    """The checkpoint's model, exported as pretraining exports it (the default part and
    format): the state dict of the bare DINOv2 ViT, i.e. of the teacher."""
    path = tmp_path / "exported_models" / "exported_last.pt"
    lightly_train.export(out=path, checkpoint=checkpoint_path)
    return path


def _exported_model_keys() -> dict[str, Any]:
    """What an eval config needs for the export of ``checkpoint_path``."""
    normalize = NormalizeArgs()
    return {
        "model": "dinov2/_vittest14",
        "model_args": None,
        "normalize": {"mean": list(normalize.mean), "std": list(normalize.std)},
    }


def _eval_block(tmp_path: Path) -> dict[str, Any]:
    attentive = {"max_epochs": 1, "batch_size": 4, "num_heads": 2}
    out_dir = tmp_path / "eval" / "post-training"
    return {
        "split": _split(),
        # series_depth: the fake dataset's depth, the slice count of the slice tasks.
        "data": {
            "test_sets": ["internal", "external"],
            "num_workers": 0,
            "series_depth": DEPTH,
        },
        "logging": {
            "tensorboard_dir": str(out_dir / "tb"),
            "per_label_dir": str(out_dir / "eval_per_label"),
        },
        # Every task explicit: a missing key keeps KneeNo's default frequency.
        "freq": {task: 1 for task in ALL_TASKS},
        "extraction_batch_size": 4,
        # Pinned rather than KneeNo's default, which the expected metric keys depend on.
        "knn": {"k": [5, 20]},
        "linear": {"max_epochs": 1, "batch_size": 4},
        "linear_pool": {"max_epochs": 1, "batch_size": 4},
        "attentive_pool": attentive,
        "linear_slice_cls": {"max_epochs": 1, "batch_size": 4},
        "attentive_slice_cls": attentive,
        "attentive_slice_pool": attentive,
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
        "weights": str(checkpoint_path),
        "image_size": list(IMAGE_SIZE),
        "eval": _eval_block(tmp_path),
    }


@pytest.fixture
def fake_dataset(mocker: MockerFixture) -> None:
    mocker.patch(
        "kneeno.evaluation.classification.KneeNoEvalDataset.from_config",
        side_effect=lambda data_cfg: _fake_eval_dataset(tuple(data_cfg["test_sets"])),
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
    # Every test set's metrics, plus the trained head's best val metric and lr, and
    # each k's val metric and temperature.
    assert {key.rsplit("/", 1)[0] for key in metrics} == {
        "internal/knn/k5",
        "internal/knn/k20",
        "external/knn/k5",
        "external/knn/k20",
        "internal/linear",
        "external/linear",
        "evaluation_val/linear",
        "evaluation_val/knn/k5",
        "evaluation_val/knn/k20",
    }
    assert all(isinstance(value, float) for value in metrics.values())
    assert yaml.safe_load((out_dir / "params.yaml").read_text()) == run_config
    assert any((out_dir / "eval_per_label").iterdir())
    assert any((out_dir / "tb").iterdir())


def test_eval_classification__all_tasks(
    tmp_path: Path, run_config: dict[str, Any], fake_dataset: None
) -> None:
    # The 2D encoder serves every task, the slice tasks included.
    config_path = _write_config(tmp_path, run_config)
    eval_classification.eval_classification(eval_config=config_path, accelerator="cpu")
    metrics = json.loads((_out_dir(tmp_path) / "results.json").read_text())
    assert {key.split("/")[1] for key in metrics} == set(ALL_TASKS)
    assert all(math.isfinite(value) for value in metrics.values())


def test_eval_classification__missing_freq_keeps_default(
    tmp_path: Path, run_config: dict[str, Any], mocker: MockerFixture
) -> None:
    # A task left out of freq keeps KneeNo's default frequency (deep merge), which is
    # null for every task: only the listed tasks run.
    run_config["eval"]["freq"] = {"linear": 1, "linear_slice_cls": 1}
    config_path = _write_config(tmp_path, run_config)
    evaluate = mocker.patch(
        "kneeno.evaluation.ClassificationEvaluator.evaluate", return_value={}
    )
    mocker.patch(
        "kneeno.evaluation.classification.KneeNoEvalDataset.from_config",
        return_value=_fake_eval_dataset(),
    )
    eval_classification.eval_classification(eval_config=config_path, accelerator="cpu")
    tasks = evaluate.call_args.kwargs["tasks"]
    assert tasks == ["linear", "linear_slice_cls"]


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
    run_config["weights"] = "${PRETRAIN_DIR}/" + checkpoint_path.name
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
    assert params["weights"] == str(checkpoint_path)
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
    load_encoder = mocker.spy(eval_classification, "_load_encoder")
    with pytest.raises(ValueError, match="already exists|earlier evaluation"):
        eval_classification.eval_classification(
            eval_config=config_path, accelerator="cpu"
        )
    # Refused before the checkpoint is even loaded.
    load_encoder.assert_not_called()

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
        ({"image_size": [224, 224, 16]}, r"image_size\n\s+Tuple should have at most 2"),
        ({"weights": None}, r"weights\n\s+Field required"),
        ({"normalize": {"mean": [0.5]}}, r"normalize.std\n\s+Field required"),
        ({"model": ["dinov2/_vittest14"]}, r"model\n\s+Input should be a valid string"),
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


@pytest.mark.parametrize("test_sets", [("internal",), ("internal", "external")])
def test_eval_classification__test_sets_select_dataset_and_preprocessing(
    tmp_path: Path,
    run_config: dict[str, Any],
    mocker: MockerFixture,
    test_sets: tuple[str, ...],
) -> None:
    run_config = _only(run_config, ["knn"])
    run_config["eval"]["data"]["test_sets"] = list(test_sets)
    dataset = mocker.patch(
        "kneeno.evaluation.classification.KneeNoEvalDataset.from_config",
        return_value=_fake_eval_dataset(test_sets),
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
    assert dataset.call_args.args[0]["test_sets"] == list(test_sets)
    # The adapter learns each volume's source from the dataset's items: only external
    # volumes are quantized (num_workers is 0, so the spy sees every call).
    assert to_uint8.called == ("external" in test_sets)
    assert adapter.call_args.kwargs["image_size"] == IMAGE_SIZE
    # The checkpoint's (resolved) normalize args have one value per input channel.
    assert adapter.call_args.kwargs["num_channels"] == len(NormalizeArgs().mean) == 3
    metrics = json.loads((_out_dir(tmp_path) / "results.json").read_text())
    # Every test set, plus the val metrics (evaluation_val/...), which never test.
    assert {key.split("/")[0] for key in metrics} == {*test_sets, "evaluation_val"}
    assert all(math.isfinite(v) for v in metrics.values())


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


def test_eval_classification__old_checkpoint_key(
    tmp_path: Path, run_config: dict[str, Any]
) -> None:
    run_config["checkpoint"] = run_config.pop("weights")
    with pytest.raises(ValueError, match="now called 'weights'"):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )


def _redirect_outputs(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    config["eval"]["logging"] = {
        "tensorboard_dir": str(out_dir / "tb"),
        "per_label_dir": str(out_dir / "eval_per_label"),
    }
    return config


def test_eval_classification__exported_model_equals_checkpoint(
    tmp_path: Path,
    run_config: dict[str, Any],
    exported_path: Path,
    fake_dataset: None,
) -> None:
    """The export holds the checkpoint's teacher: both give the same metrics."""
    run_config = _only(run_config, ["knn", "linear", "linear_slice_cls"])
    eval_classification.eval_classification(
        eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
    )
    exported_config = {**run_config, "weights": str(exported_path)}
    exported_config.update(_exported_model_keys())
    exported_dir = tmp_path / "exported_eval"
    exported_dir.mkdir()
    exported_config = _redirect_outputs(exported_config, exported_dir)
    eval_classification.eval_classification(
        eval_config=_write_config(exported_dir, exported_config), accelerator="cpu"
    )
    from_checkpoint = json.loads((_out_dir(tmp_path) / "results.json").read_text())
    from_export = json.loads((exported_dir / "results.json").read_text())
    assert from_export == from_checkpoint
    assert yaml.safe_load((exported_dir / "params.yaml").read_text()) == (
        exported_config
    )


def test_eval_classification__exported_model_settings(
    tmp_path: Path,
    run_config: dict[str, Any],
    exported_path: Path,
    fake_dataset: None,
    mocker: MockerFixture,
) -> None:
    """The exported model is built from model / model_args, without a download, and
    the adapter gets the config's normalize."""
    run_config = _only(run_config, ["knn"])
    run_config.update(_exported_model_keys())
    run_config["weights"] = str(exported_path)
    # Leaves the architecture as it is, so the weights still fit.
    run_config["model_args"] = {"drop_path_rate": 0.0}
    run_config["normalize"] = {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}
    get_wrapped_model = mocker.spy(package_helpers, "get_wrapped_model")
    adapter = mocker.patch(
        "lightly_train._commands.eval_classification.DINOv2Adapter",
        wraps=DINOv2Adapter,
    )
    evaluate = mocker.patch.object(ClassificationEvaluator, "evaluate", return_value={})
    eval_classification.eval_classification(
        eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
    )
    assert get_wrapped_model.call_args.kwargs == {
        "model": "dinov2/_vittest14",
        "num_input_channels": 3,
        "model_args": {"drop_path_rate": 0.0},
        "load_weights": False,
    }
    assert adapter.call_args.kwargs["num_channels"] == 3
    assert adapter.call_args.kwargs["normalize"] == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    # The evaluated model carries the exported weights, as the teacher (no drop path).
    model = evaluate.call_args.args[0]
    exported = torch.load(exported_path, weights_only=True)
    assert all(
        torch.equal(value, exported[key])
        for key, value in model.get_model().state_dict().items()
    )
    assert all(block.sample_drop_ratio == 0.0 for block in model.get_model().blocks)


def test_eval_classification__exported_model_num_channels_from_weights(
    tmp_path: Path,
    run_config: dict[str, Any],
    fake_dataset: None,
    mocker: MockerFixture,
) -> None:
    # A 1-channel model, as the 3D branch trains: the channel count comes from the
    # weights, not from normalize (which may have one value for all channels).
    path = tmp_path / "exported_1ch.pt"
    wrapped_model = package_helpers.get_wrapped_model(
        model="dinov2/_vittest14", num_input_channels=1
    )
    torch.save(wrapped_model.get_model().state_dict(), path)
    run_config = _only(run_config, ["knn"])
    run_config.update(_exported_model_keys())
    run_config["weights"] = str(path)
    run_config["normalize"] = {"mean": [0.5], "std": [0.5]}
    adapter = mocker.patch(
        "lightly_train._commands.eval_classification.DINOv2Adapter",
        wraps=DINOv2Adapter,
    )
    eval_classification.eval_classification(
        eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
    )
    assert adapter.call_args.kwargs["num_channels"] == 1
    assert adapter.call_args.kwargs["normalize"] == ((0.5,), (0.5,))
    metrics = json.loads((_out_dir(tmp_path) / "results.json").read_text())
    assert metrics and all(math.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize("missing", ["model", "normalize"])
def test_eval_classification__exported_model_needs_keys(
    tmp_path: Path, run_config: dict[str, Any], exported_path: Path, missing: str
) -> None:
    run_config.update(_exported_model_keys())
    run_config["weights"] = str(exported_path)
    del run_config[missing]
    with pytest.raises(ValueError, match=rf"needs \['{missing}'\]"):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )
    # Refused before params.yaml is written, so the corrected config can run.
    assert not _out_dir(tmp_path).exists()


@pytest.mark.parametrize(
    "key, value",
    [
        ("model", "dinov2/_vittest14"),
        ("model", None),
        ("model_args", {}),
        ("model_args", None),
        ("normalize", {"mean": [0.5], "std": [0.5]}),
        ("normalize", None),
    ],
)
def test_eval_classification__checkpoint_rejects_exported_model_keys(
    tmp_path: Path, run_config: dict[str, Any], key: str, value: Any
) -> None:
    # Even a null value: the key would suggest the config decides it.
    run_config[key] = value
    with pytest.raises(ValueError, match=rf"sets \['{key}'\], but 'weights' is a"):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )
    assert not _out_dir(tmp_path).exists()


def test_eval_classification__exported_model_online_encoder(
    tmp_path: Path, run_config: dict[str, Any], exported_path: Path
) -> None:
    run_config.update(_exported_model_keys())
    run_config["weights"] = str(exported_path)
    run_config["eval"]["encoder"] = "online"
    with pytest.raises(ValueError, match="'online' requires a LightlyTrain checkpoint"):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )


@pytest.mark.parametrize(
    "change",
    [
        {"model": "dinov2/vits14-notpretrained"},
        # register tokens are parameters, so this changes the state dict's keys
        {"model_args": {"num_register_tokens": 2}},
    ],
)
def test_eval_classification__exported_model_wrong_model(
    tmp_path: Path,
    run_config: dict[str, Any],
    exported_path: Path,
    change: dict[str, Any],
) -> None:
    run_config.update(_exported_model_keys())
    run_config.update(change)
    run_config["weights"] = str(exported_path)
    with pytest.raises(ValueError, match="does not fit model="):
        eval_classification.eval_classification(
            eval_config=_write_config(tmp_path, run_config), accelerator="cpu"
        )
    assert not _out_dir(tmp_path).exists()


@pytest.mark.parametrize(
    "export_args, match",
    [
        # A pickled module, not a state dict.
        ({"format": "torch_model"}, "neither a LightlyTrain checkpoint nor"),
        # Keys prefixed with the wrapper's attribute names.
        (
            {"part": "embedding_model", "format": "torch_state_dict"},
            "no 'patch_embed.proj.weight'",
        ),
    ],
)
def test_eval_classification__unsupported_export(
    tmp_path: Path,
    run_config: dict[str, Any],
    checkpoint_path: Path,
    export_args: dict[str, Any],
    match: str,
) -> None:
    path = tmp_path / "exported.pt"
    lightly_train.export(out=path, checkpoint=checkpoint_path, **export_args)
    run_config.update(_exported_model_keys())
    run_config["weights"] = str(path)
    with pytest.raises(ValueError, match=match):
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
