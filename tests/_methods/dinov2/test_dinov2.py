#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import math
from typing import Literal

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Size

from lightly_train._methods.dinov2 import dinov2 as dinov2_module
from lightly_train._methods.dinov2.dinov2 import (
    DINOv2,
    DINOv2AdamWViTArgs,
    DINOv2Args,
)
from lightly_train._methods.dinov2.utils import is_tokenization_param
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import IMAGENET_SIZE, ScalingInfo
from lightly_train.types import Batch

from ...helpers import dummy_dinov2_vit_model


def setup_dinov2_helper(
    dinov2_args: DINOv2Args,
    mocker: MockerFixture,
    emb_model: EmbeddingModel,
    batch_size: int,
) -> DINOv2:
    optimizer_args = DINOv2AdamWViTArgs()
    scaling_info = ScalingInfo(dataset_size=1000, epochs=100)
    dinov2_args.resolve_auto(
        scaling_info=scaling_info,
        optimizer_args=optimizer_args,
        wrapped_model=emb_model.wrapped_model,
    )

    dinov2 = DINOv2(
        method_args=dinov2_args,
        optimizer_args=optimizer_args,
        embedding_model=emb_model,
        global_batch_size=batch_size,
        num_input_channels=1,
    )

    trainer_mock = mocker.Mock()
    trainer_mock.global_step = 0
    trainer_mock.max_epochs = 1
    trainer_mock.estimated_stepping_batches = 1

    dinov2.trainer = trainer_mock

    return dinov2


class TestDINOv2:
    @pytest.mark.parametrize(
        "optim_type, expected",
        [
            ("auto", DINOv2AdamWViTArgs),
            (OptimizerType.ADAMW, DINOv2AdamWViTArgs),
        ],
    )
    def test_optimizer_args_cls(
        self, optim_type: OptimizerType | Literal["auto"], expected: type[OptimizerArgs]
    ) -> None:
        assert DINOv2.optimizer_args_cls(optim_type=optim_type) == expected

    @pytest.mark.parametrize(
        "n_local_crops, ibot_separate_head, center_method",
        [
            (8, False, "softmax"),
            (0, False, "softmax"),
            (8, True, "softmax"),
            (8, True, "sinkhorn_knopp"),
        ],
    )
    def test_train_step_impl(
        self,
        mocker: MockerFixture,
        n_local_crops: int,
        ibot_separate_head: bool,
        center_method: Literal["softmax", "sinkhorn_knopp"],
    ) -> None:
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        b = 16

        # Views are (B, C, D, H, W). The dummy model has patch size 2, so global views
        # have a 4x4x4 patch grid.
        views = [torch.rand(b, 1, 8, 8, 8) for _ in range(2)] + [
            torch.rand(b, 1, 4, 4, 4) for _ in range(n_local_crops)
        ]
        batch: Batch = {
            "views": views,
            "filename": [f"img_{i}" for i in range(b)],
        }

        # run DINOv2
        dinov2_args = DINOv2Args(
            ibot_separate_head=ibot_separate_head, center_method=center_method
        )
        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, b)

        out = dinov2.training_step_impl(batch, 0)

        # check that the ibot and dino heads are the same
        if ibot_separate_head:
            assert dinov2.student_head.dino_head != dinov2.student_head.ibot_head
        else:
            assert dinov2.student_head.dino_head == dinov2.student_head.ibot_head
            assert len(list(dinov2.student_head.dino_head.parameters())) == len(
                list(dinov2.student_head.ibot_head.parameters())
            )
            for (name_dino, param_dino), (name_ibot, param_ibot) in zip(
                dinov2.student_head.dino_head.named_parameters(),
                dinov2.student_head.ibot_head.named_parameters(),
            ):
                assert name_dino == name_ibot
                assert param_dino.dtype == param_ibot.dtype
                assert param_dino.requires_grad == param_ibot.requires_grad
                assert torch.allclose(param_dino, param_ibot, rtol=1e-3, atol=1e-4)
        assert out.log_dict is not None
        if n_local_crops == 0:
            assert out.log_dict["train_loss/dino_local_loss"] == torch.tensor(0.0)
        assert out.loss.shape == Size([])
        assert out.log_dict["train_loss/dino_global_loss"].shape == Size([])
        assert out.log_dict["train_loss/dino_local_loss"].shape == Size([])
        assert out.log_dict["train_loss/ibot_loss"].shape == Size([])
        assert out.log_dict["train_loss/koleo_loss"].shape == Size([])
        assert torch.isfinite(out.loss)

    def test_train_step_impl__anisotropic_views(self, mocker: MockerFixture) -> None:
        # Patch size (H, W, D) = (4, 2, 2) with views (D, H, W) = (4, 16, 8) gives a
        # (2, 4, 4) patch grid. The mask grid must match the patch embedding grid.
        emb_model = EmbeddingModel(
            wrapped_model=dummy_dinov2_vit_model(
                patch_size=(4, 2, 2), img_size=(16, 8, 4)
            )
        )
        b = 4
        views = [torch.rand(b, 1, 4, 16, 8) for _ in range(2)] + [
            torch.rand(b, 1, 2, 8, 4) for _ in range(2)
        ]
        batch: Batch = {"views": views, "filename": [f"img_{i}" for i in range(b)]}
        dinov2 = setup_dinov2_helper(
            DINOv2Args(mask_probability=1.0), mocker, emb_model, b
        )
        spy = mocker.spy(dinov2_module, "MaskingGenerator")
        out = dinov2.training_step_impl(batch, 0)
        assert spy.call_args.kwargs["input_size"] == (2, 4, 4)
        assert torch.isfinite(out.loss)

    def test_layerwise_decay_optimizer(self, mocker: MockerFixture) -> None:
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        b = 16

        dinov2_args = DINOv2Args(warmup_steps=2)
        dinov2_args.layerwise_decay = 0.9

        trainer_mock = mocker.Mock()
        trainer_mock.global_step = 0
        trainer_mock.max_epochs = 2
        trainer_mock.estimated_stepping_batches = 4

        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, b)
        dinov2.trainer = trainer_mock

        target_lr_before_scaling = dinov2.optimizer_args.lr  # type: ignore[attr-defined]
        optim_scheduler = dinov2.configure_optimizers()
        optim = optim_scheduler[0][0]  # type: ignore[index, literal-required]

        scheduler = optim_scheduler[1][0]["scheduler"]  # type: ignore[index, literal-required]

        num_layers = emb_model.wrapped_model.get_model().n_blocks
        lr_decay_rate = dinov2_args.layerwise_decay

        # Verify that the target lr is correctly scaled
        lr_neutral = dinov2.optimizer_args.lr  # type: ignore[attr-defined]
        assert target_lr_before_scaling * math.sqrt(b / 1024) == lr_neutral

        def check_param_groups() -> None:
            for param_group in optim.param_groups:
                name = param_group["name"]
                if "ibot_head" not in name and "dino_head" not in name:
                    # This is a ViT block --> decay through the layers
                    layer_id = num_layers + 1
                    if (
                        "pos_embed" in name
                        or "patch_embed" in name
                        or "mask_token" in name
                        or "cls_token" in name
                        or "register_tokens" in name
                    ):
                        layer_id = 0
                    elif "blocks." in name and "residual." not in name:
                        layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
                    temp_target_lr = target_lr * (
                        lr_decay_rate ** (num_layers + 1 - layer_id)
                    )
                    if "patch_embed" in name:
                        temp_target_lr *= dinov2_args.patch_embed_lr_multiplier
                    # assert that the lr is close to the target lr
                    assert math.isclose(
                        param_group["lr"], temp_target_lr, rel_tol=1e-10, abs_tol=1e-10
                    )

                else:
                    # This is a head block --> no decay
                    assert math.isclose(
                        param_group["lr"], target_lr, rel_tol=1e-10, abs_tol=1e-10
                    )
                if name.endswith(".bias") or "norm" in name or "gamma" in name:
                    assert param_group["weight_decay"] == 0.0
                else:
                    assert (
                        param_group["weight_decay"]
                        == dinov2.optimizer_args.weight_decay  # type: ignore[attr-defined]
                    )

        # First batch
        target_lr = lr_neutral / (
            trainer_mock.estimated_stepping_batches / trainer_mock.max_epochs
        )
        check_param_groups()

        optim.step()
        scheduler.step()

        # Second batch
        target_lr = lr_neutral
        check_param_groups()

        optim.step()
        scheduler.step()
        optim.step()
        scheduler.step()

        # Last Batch
        target_lr = dinov2_args.min_lr
        check_param_groups()

    @pytest.mark.parametrize("layerwise_decay", [0.9, 1.0])
    def test_on_before_optimizer_step__tokenization_only(
        self, mocker: MockerFixture, layerwise_decay: float
    ) -> None:
        """While global_step < n_tokenization_only_steps, only the tokenization and
        the projection heads train; the transformer blocks and the final norm are
        held at lr=0."""
        # student_freeze_last_layer_steps defaults to 1250 and would independently
        # freeze the last_layer group at these global_steps; disable it so this test
        # isolates the n_tokenization_only_steps rule.
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov2_args = DINOv2Args(
            n_tokenization_only_steps=10,
            layerwise_decay=layerwise_decay,
            student_freeze_last_layer_steps=0,
        )
        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, batch_size=16)
        trainer_mock = mocker.Mock()
        trainer_mock.global_step = 0
        trainer_mock.estimated_stepping_batches = 100
        dinov2.trainer = trainer_mock

        optim = dinov2.configure_optimizers()[0][0]  # type: ignore[index, literal-required]
        dinov2.on_before_optimizer_step(optim)

        for group in optim.param_groups:
            name = group["name"]
            if "head" not in name and not is_tokenization_param(name):
                assert group["lr"] == 0.0, f"Expected '{name}' to be frozen"
            else:
                assert group["lr"] > 0.0, f"Expected '{name}' to keep training"

    def test_on_before_optimizer_step__tokenization_only_ends(
        self, mocker: MockerFixture
    ) -> None:
        """Once global_step reaches n_tokenization_only_steps, every group trains
        again -- the freeze is derived from global_step, not a one-way transition
        that would need an explicit "unfreeze"."""
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov2_args = DINOv2Args(
            n_tokenization_only_steps=10, student_freeze_last_layer_steps=0
        )
        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, batch_size=16)
        trainer_mock = mocker.Mock()
        trainer_mock.global_step = 10
        trainer_mock.estimated_stepping_batches = 100
        dinov2.trainer = trainer_mock

        optim = dinov2.configure_optimizers()[0][0]  # type: ignore[index, literal-required]
        dinov2.on_before_optimizer_step(optim)

        for group in optim.param_groups:
            assert group["lr"] > 0.0, f"Expected '{group['name']}' to be unfrozen"

    def test_on_before_optimizer_step__tokenization_only_default_off(
        self, mocker: MockerFixture
    ) -> None:
        """n_tokenization_only_steps=0 (the default) must leave training exactly as
        it was before this option existed: nothing is frozen by this rule."""
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov2_args = DINOv2Args(student_freeze_last_layer_steps=0)
        assert dinov2_args.n_tokenization_only_steps == 0
        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, batch_size=16)
        trainer_mock = mocker.Mock()
        trainer_mock.global_step = 0
        trainer_mock.estimated_stepping_batches = 100
        dinov2.trainer = trainer_mock

        optim = dinov2.configure_optimizers()[0][0]  # type: ignore[index, literal-required]
        dinov2.on_before_optimizer_step(optim)

        for group in optim.param_groups:
            assert group["lr"] > 0.0, f"Expected '{group['name']}' to be unfrozen"

    def test_on_train_start__logs_tokenization_only_active(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The status log at training start must compare against the current
        (possibly resumed) global_step, not against 0. This has to be on_train_start,
        not on_fit_start: Lightning restores the fit_loop's global_step/current_epoch
        (restore_training_state) only after on_fit_start but before on_train_start --
        see the comment on DINOv2.on_train_start."""
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov2_args = DINOv2Args(n_tokenization_only_steps=10)
        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, batch_size=16)
        trainer_mock = mocker.Mock()
        trainer_mock.global_step = 3  # e.g. a resumed run
        dinov2.trainer = trainer_mock

        with caplog.at_level(logging.INFO):
            dinov2.on_train_start()

        assert dinov2._tokenization_only_active is True
        assert "Training only the tokenization" in caplog.text
        assert "7 more step(s)" in caplog.text
        assert "currently at step 3" in caplog.text

    @pytest.mark.parametrize(
        "n_tokenization_only_steps, global_step",
        [
            (0, 0),  # disabled entirely
            (10, 10),  # resumed exactly at the boundary
            (10, 15),  # resumed past the boundary
        ],
    )
    def test_on_train_start__logs_all_layers(
        self,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
        n_tokenization_only_steps: int,
        global_step: int,
    ) -> None:
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov2_args = DINOv2Args(n_tokenization_only_steps=n_tokenization_only_steps)
        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, batch_size=16)
        trainer_mock = mocker.Mock()
        trainer_mock.global_step = global_step
        dinov2.trainer = trainer_mock

        with caplog.at_level(logging.INFO):
            dinov2.on_train_start()

        assert dinov2._tokenization_only_active is False
        assert "Training all layers" in caplog.text
        assert "Training only the tokenization" not in caplog.text

    def test_on_before_optimizer_step__logs_unfreeze_once(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The "unfreezing" message must fire exactly once, at the step the freeze
        actually ends, not on every subsequent step."""
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov2_args = DINOv2Args(n_tokenization_only_steps=10)
        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, batch_size=16)
        trainer_mock = mocker.Mock()
        trainer_mock.global_step = 0
        trainer_mock.estimated_stepping_batches = 100
        dinov2.trainer = trainer_mock
        dinov2.on_train_start()  # starts active, as in a fresh run

        optim_mock = mocker.Mock(param_groups=[])
        with caplog.at_level(logging.INFO):
            for step in (0, 5, 9, 10, 11):
                trainer_mock.global_step = step
                caplog.clear()
                dinov2.on_before_optimizer_step(optim_mock)
                if step == 10:
                    assert "Unfreezing the transformer blocks" in caplog.text
                else:
                    assert "Unfreezing the transformer blocks" not in caplog.text

    def test_on_before_optimizer_step__no_unfreeze_log_if_already_unfrozen(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A run that starts already past the tokenization-only phase (e.g. resumed
        past it) must not log an "unfreezing" event -- nothing happened during this
        run to report."""
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov2_args = DINOv2Args(n_tokenization_only_steps=10)
        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, batch_size=16)
        trainer_mock = mocker.Mock()
        trainer_mock.global_step = 15
        trainer_mock.estimated_stepping_batches = 100
        dinov2.trainer = trainer_mock
        dinov2.on_train_start()  # starts already unfrozen

        optim_mock = mocker.Mock(param_groups=[])
        with caplog.at_level(logging.INFO):
            dinov2.on_before_optimizer_step(optim_mock)

        assert "Unfreezing the transformer blocks" not in caplog.text


class TestDINOv2Args:
    def test_resolve_auto(self) -> None:
        args = DINOv2Args()
        args.resolve_auto(
            scaling_info=ScalingInfo(dataset_size=IMAGENET_SIZE, epochs=100),
            optimizer_args=DINOv2AdamWViTArgs(),
            wrapped_model=dummy_dinov2_vit_model(),
        )
        assert not args.has_auto()
