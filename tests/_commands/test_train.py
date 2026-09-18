#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from pytest import LogCaptureFixture
from pytest_mock import MockerFixture
from pytorch_lightning.accelerators.cpu import CPUAccelerator

from lightly_train._checkpoint import Checkpoint
from lightly_train._commands import train, train_helpers
from lightly_train._commands.train import (
    CLITrainConfig,
    FunctionTrainConfig,
    TrainConfig,
)
from lightly_train._loggers.jsonl import JSONLLogger
from lightly_train._methods.dinov2.dinov2 import DINOv2AdamWViTArgs, DINOv2Args
from lightly_train._methods.dinov2.utils import is_tokenization_param
from lightly_train._scaling import ScalingInfo

from .. import helpers
from ..helpers import DummyCustomModel

# Make the teacher change visibly within a few steps. With the default warmup the
# learning rate is so small that EMA updates vanish in float32, and with the default
# momentum schedule the momentum reaches 1.0 (no teacher update) at the last step.
FAST_UPDATE_METHOD_ARGS: dict[str, Any] = {
    "warmup_steps": 1,
    "momentum_start": 0.5,
    "momentum_end": 0.5,
}


def _pretrain_kwargs(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    """Arguments for DINOv2 pretraining on a small synthetic KneeNo dataset on CPU.

    The dataset has 8 series, so batch_size=4 results in 2 steps per epoch.
    """
    data_root = tmp_path / "data"
    data_meta = tmp_path / "meta.json"
    if not data_meta.exists():
        helpers.create_mi_dataset(tmp_path)
    pretrain_kwargs: dict[str, Any] = dict(
        out=tmp_path / "out",
        data_root=data_root,
        data_meta=data_meta,
        series_depth=8,
        resample_mode="nearest",
        model="dinov2/_vittest14",
        method="dinov2",
        batch_size=4,
        num_workers=0,
        epochs=1,
        accelerator="cpu",
        devices=1,
        transform_args=copy.deepcopy(helpers.MI_DINOV2_TRANSFORM_ARGS),
    )
    pretrain_kwargs.update(kwargs)
    return pretrain_kwargs


def test_track_training_started_event(mocker: MockerFixture) -> None:
    """Ensure training_started analytics payload stays consistent."""
    from lightly_train._events import tracker

    mock_track_event = mocker.patch("lightly_train._events.tracker.track_event")
    model = DummyCustomModel()

    tracker.track_training_started(
        task_type="ssl_pretraining",
        model=model,
        method="simclr",
        batch_size=128,
        devices="auto",
        epochs=10,
    )

    mock_track_event.assert_called_once_with(
        "training_started",
        {
            "task_type": "ssl_pretraining",
            "model_name": model.__class__.__name__,
            "method": "simclr",
            "batch_size": 128,
            "devices": 1,
            "epochs": 10,
        },
    )


def test_pretrain__cpu(tmp_path: Path) -> None:
    out = tmp_path / "out"
    # num_workers=2 checks that the dataset and worker_init_fn are picklable.
    train.pretrain(**_pretrain_kwargs(tmp_path, num_workers=2))

    # Check that the correct files were created.
    filepaths = {fp.relative_to(out) for fp in out.rglob("*")}
    expected_filepaths = {
        Path("checkpoints"),
        Path("checkpoints") / "epoch=0-step=2.ckpt",
        Path("checkpoints") / "last.ckpt",
        Path("exported_models"),
        Path("exported_models") / "exported_last.pt",
        Path("metrics.jsonl"),
        Path("train.log"),
        # Tensorboard filename is not deterministic, so we need to find it.
        next(fp for fp in filepaths if fp.name.startswith("events.out.tfevents")),
    }
    assert filepaths == expected_filepaths

    # The exported model is the 3D DINOv2 backbone.
    state_dict = torch.load(
        out / "exported_models" / "exported_last.pt", weights_only=True
    )
    assert state_dict["patch_embed.proj.weight"].shape == (8, 1, 4, 14, 14)


@pytest.mark.parametrize("gradient_accumulation_steps", [1, 4])
def test_pretrain__batch_sizes_for_gradient_accumulation(
    tmp_path: Path,
    mocker: MockerFixture,
    gradient_accumulation_steps: int,
) -> None:
    global_batch_size = 4
    total_num_devices = 1
    per_device_batch_size = global_batch_size // total_num_devices
    effective_global_batch_size = global_batch_size * gradient_accumulation_steps
    get_dataloader_spy = mocker.spy(train_helpers, "get_dataloader")
    get_method_spy = mocker.spy(train_helpers, "get_method")
    # 16 series result in 4 batches, i.e. at least one optimizer step with 4
    # accumulation steps. DINOv2 divides by the number of optimizer steps.
    helpers.create_mi_dataset(tmp_path, n_cases=8)

    train.pretrain(
        **_pretrain_kwargs(
            tmp_path,
            batch_size=global_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            devices=total_num_devices,
            epochs=0,
        )
    )

    assert get_dataloader_spy.call_args.kwargs["batch_size"] == per_device_batch_size
    assert get_dataloader_spy.call_args.kwargs["series_depth"] == 8
    assert (
        get_method_spy.call_args.kwargs["global_batch_size"]
        == effective_global_batch_size
    )


def test_pretrain__resume_interrupted(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    out = tmp_path / "out"
    kwargs = _pretrain_kwargs(tmp_path, method_args=FAST_UPDATE_METHOD_ARGS)
    train.pretrain(**kwargs)

    # Check that we can resume training
    last_ckpt_path = out / "checkpoints" / "last.ckpt"
    first_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)

    with caplog.at_level(logging.INFO):
        train.pretrain(**{**kwargs, "epochs": 2, "resume_interrupted": True})
    assert (
        f"Restoring states from the checkpoint path at {last_ckpt_path}" in caplog.text
    )
    # Epochs in checkpoint are 0-indexed. Epoch 1 is therefore the second epoch.
    # weights_only=True does not work here.
    assert torch.load(last_ckpt_path, weights_only=False)["epoch"] == 1

    # Check that exported checkpoint weights changed between first and second run.
    second_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)
    first_state_dict = first_ckpt.lightly_train.models.model.state_dict()
    second_state_dict = second_ckpt.lightly_train.models.model.state_dict()
    assert first_state_dict.keys() == second_state_dict.keys()
    assert any(
        not torch.equal(first_state_dict[key], second_state_dict[key])
        for key in first_state_dict
    )

    # Check that last.ckpt and exported_model.pt contain same information. If this fails
    # it means that checkpoint loading is not working correctly.
    exported_state_dict = torch.load(
        out / "exported_models" / "exported_last.pt", weights_only=True
    )
    assert second_state_dict.keys() == exported_state_dict.keys()
    for key in second_state_dict.keys():
        assert torch.equal(second_state_dict[key], exported_state_dict[key])


def test_pretrain__overwrite_true(tmp_path: Path) -> None:
    """Test that overwrite=True allows training with an existing output directory that
    contains files."""
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "file.txt").touch()

    train.pretrain(**_pretrain_kwargs(tmp_path, overwrite=True))


def test_pretrain__overwrite_false(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "file.txt").touch()

    with pytest.raises(ValueError):
        train.pretrain(**_pretrain_kwargs(tmp_path))


def test_pretrain__embed_dim(tmp_path: Path) -> None:
    train.pretrain(**_pretrain_kwargs(tmp_path, embed_dim=64))


@pytest.mark.parametrize("series_depth", [0, -1])
def test_pretrain__series_depth_not_positive(tmp_path: Path, series_depth: int) -> None:
    """series_depth<=0 means native per-series depth (kneeno treats every
    non-positive value the same as 0). The synthetic dataset has series of two
    different depths (6 and 9); RandomResizedCrop3D always resizes its crop to a
    fixed output size, so default collation across them works without a
    depth-bucket sampler."""
    out = tmp_path / "out"
    train.pretrain(**_pretrain_kwargs(tmp_path, series_depth=series_depth))
    assert (out / "checkpoints" / "last.ckpt").exists()


def test_pretrain__resample_mode_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resample_mode"):
        train.pretrain(**_pretrain_kwargs(tmp_path, resample_mode="cubic"))


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Requires Python 3.10 or higher for typing."
)
def test_pretrain__parameters() -> None:
    """Tests that pretrain function and TrainConfig have the same parameters and default
    values.

    This test is here to make sure we don't forget to update pretrain/TrainConfig when
    we change parameters in one of the two.
    """
    helpers.assert_same_params(a=FunctionTrainConfig, b=train.pretrain)
    helpers.assert_same_params(a=TrainConfig, b=FunctionTrainConfig, assert_type=False)
    helpers.assert_same_params(a=TrainConfig, b=CLITrainConfig, assert_type=False)


def test_pretrain__zero_epochs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    train.pretrain(**_pretrain_kwargs(tmp_path, epochs=0))
    assert (out / "checkpoints" / "last.ckpt").exists()


def test_train_from_dictconfig(tmp_path: Path) -> None:
    kwargs = _pretrain_kwargs(tmp_path)
    config = OmegaConf.create(
        dict(
            out=str(kwargs["out"]),
            data_root=str(kwargs["data_root"]),
            data_meta=str(kwargs["data_meta"]),
            series_depth=8,
            resample_mode="nearest",
            model="dinov2/_vittest14",
            method="dinov2",
            batch_size=4,
            num_workers=0,
            epochs=1,
            accelerator="cpu",
            devices=1,
            # OmegaConf does not support tuples, the CLI passes lists.
            transform_args={
                "image_size": [56, 56, 16],
                "local_view": {"view_size": [28, 28, 8], "num_views": 2},
            },
            optim_args={"lr": 0.1},
            loader_args={"shuffle": True},
            trainer_args={"min_epochs": 1},
            callbacks={"model_checkpoint": {"every_n_epochs": 5}},
            loggers={"jsonl": {"flush_logs_every_n_steps": 5}},
        )
    )
    train.pretrain_from_dictconfig(config=config)
    assert (kwargs["out"] / "checkpoints" / "last.ckpt").exists()


def test_pretrain__TrainConfig__model_dump(tmp_path: Path) -> None:
    """
    Test that TrainConfig is dumped correctly even if some of its attributes are
    subclasses of the types specified in the TrainConfig class.
    """
    method_args = DINOv2Args()
    optim_args = DINOv2AdamWViTArgs()
    method_args.resolve_auto(
        scaling_info=ScalingInfo(dataset_size=20_000, epochs=100),
        optimizer_args=optim_args,
        wrapped_model=helpers.dummy_dinov2_vit_model(),
    )
    config = TrainConfig(
        out=tmp_path / "out",
        data_root=tmp_path / "data",
        data_meta=tmp_path / "meta.json",
        series_depth=8,
        resample_mode="nearest",
        model="dinov2/_vittest14",
        method="dinov2",
        optim_args=optim_args,
        method_args=method_args,
    )
    dumped_config_direct = config.model_dump()

    # Assert that the indirect dump is the same as the direct dump.
    dumped_cofig_indirect = {
        key: value.model_dump() if hasattr(value, "model_dump") else value
        for key, value in config.__dict__.items()
    }
    assert dumped_config_direct == dumped_cofig_indirect

    # Check for some specific attributes.
    assert dumped_config_direct["series_depth"] == 8
    assert dumped_config_direct["optim_args"]["betas"] == (0.9, 0.999)
    assert dumped_config_direct["method_args"]["teacher_temp_warmup_steps"] == 37500
    assert (
        dumped_config_direct["method_args"]["student_freeze_last_layer_steps"] == 1250
    )


def test_pretrain__log_resolved_config(
    caplog: LogCaptureFixture, tmp_path: Path
) -> None:
    config = TrainConfig(
        out=tmp_path / "out",
        data_root=tmp_path / "data",
        data_meta=tmp_path / "meta.json",
        series_depth=8,
        resample_mode="nearest",
        accelerator=CPUAccelerator(),
        batch_size=4,
        model="dinov2/_vittest14",
    )

    class MemoryLogger(JSONLLogger):
        def __init__(self) -> None:
            self.logs: list[dict[str, Any]] = []

        # Type ignore because JSONLLogger.log_hyperparams has a more complicated
        # signature but we only require part of it for the thest.
        def log_hyperparams(self, params: dict[str, Any]) -> None:  # type: ignore[override]
            self.logs.append(params)

    logger = MemoryLogger()

    assert len(logger.logs) == 0
    with caplog.at_level(logging.INFO):
        train.log_resolved_config(config=config, loggers=[logger])
        expected = (
            "Resolved configuration:\n"
            "{\n"
            '    "accelerator": "CPUAccelerator",\n'
            '    "activation_checkpoint_args": null,\n'
            '    "batch_size": 4,\n'
        )
        assert expected in caplog.text

    assert len(logger.logs) == 1
    assert logger.logs[0]["accelerator"] == "CPUAccelerator"
    assert logger.logs[0]["batch_size"] == 4
    assert logger.logs[0]["series_depth"] == 8


def test_pretrain__checkpoint(mocker: MockerFixture, tmp_path: Path) -> None:
    """
    Assert that train_helpers.load_state_dict is called when a checkpoint is provided.
    """
    out = tmp_path / "out"

    # Part 1: Generate a checkpoint.
    train.pretrain(**_pretrain_kwargs(tmp_path, epochs=0))
    last_ckpt_path = out / "checkpoints" / "last.ckpt"
    first_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)

    # Part 2: Load the checkpoint
    spy_load_state_dict = mocker.spy(train_helpers, "load_state_dict")
    train.pretrain(
        **_pretrain_kwargs(
            tmp_path,
            epochs=1,
            overwrite=True,
            checkpoint=last_ckpt_path,
            method_args=FAST_UPDATE_METHOD_ARGS,
            optim_args={"lr": 1.0},  # Make sure that parameters change meaningfully.
        )
    )
    spy_load_state_dict.assert_called_once()
    call_args = spy_load_state_dict.call_args_list[0]
    args, kwargs = call_args
    assert kwargs["checkpoint"] == last_ckpt_path

    # Check that exported checkpoint weights changed between first and second run.
    second_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)
    first_state_dict = first_ckpt.lightly_train.models.model.state_dict()
    second_state_dict = second_ckpt.lightly_train.models.model.state_dict()
    assert first_state_dict.keys() == second_state_dict.keys()
    assert any(
        not torch.equal(first_state_dict[key], second_state_dict[key])
        for key in first_state_dict
    )

    # Check that last.ckpt and exported_model.pt contain same information. If this fails
    # it means that checkpoint loading is not working correctly.
    exported_state_dict = torch.load(
        out / "exported_models" / "exported_last.pt", weights_only=True
    )
    assert second_state_dict.keys() == exported_state_dict.keys()
    for key in second_state_dict.keys():
        assert torch.equal(second_state_dict[key], exported_state_dict[key]), (
            f"Parameter {key} differs between checkpoint and exported model: {second_state_dict[key]} vs. {exported_state_dict[key]}"
        )


def test_pretrain__checkpoint_n_tokenization_only_steps(tmp_path: Path) -> None:
    """End-to-end: with n_tokenization_only_steps covering every step of the second
    run, the transformer blocks and final norm must not move at all, while at least
    one tokenization parameter does."""
    out = tmp_path / "out"

    # Part 1: generate a checkpoint.
    train.pretrain(**_pretrain_kwargs(tmp_path, epochs=0))
    last_ckpt_path = out / "checkpoints" / "last.ckpt"
    first_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)
    first_state_dict = first_ckpt.lightly_train.models.model.state_dict()

    # 2 steps per epoch (see _pretrain_kwargs docstring); 100 safely covers all of
    # them for this 1-epoch run, whatever the exact step count.
    method_args = {**FAST_UPDATE_METHOD_ARGS, "n_tokenization_only_steps": 100}
    train.pretrain(
        **_pretrain_kwargs(
            tmp_path,
            epochs=1,
            overwrite=True,
            checkpoint=last_ckpt_path,
            method_args=method_args,
            optim_args={"lr": 1.0},  # Make sure that parameters change meaningfully.
        )
    )
    second_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)
    second_state_dict = second_ckpt.lightly_train.models.model.state_dict()
    assert first_state_dict.keys() == second_state_dict.keys()

    frozen_keys = [k for k in first_state_dict if not is_tokenization_param(k)]
    tokenization_keys = [k for k in first_state_dict if is_tokenization_param(k)]
    assert frozen_keys, "Expected at least one block/norm parameter"
    assert tokenization_keys, "Expected at least one tokenization parameter"

    for key in frozen_keys:
        assert torch.equal(first_state_dict[key], second_state_dict[key]), (
            f"Expected frozen parameter '{key}' to be unchanged"
        )
    assert any(
        not torch.equal(first_state_dict[key], second_state_dict[key])
        for key in tokenization_keys
    ), "Expected at least one tokenization parameter to have changed"
