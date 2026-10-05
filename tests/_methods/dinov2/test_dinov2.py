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
import math
import random
from functools import partial
from typing import Any, Literal

import pytest
import torch
import torch._dynamo
from pydantic import ValidationError
from pytest_mock import MockerFixture
from torch import Size
from torch._dynamo.utils import counters

from lightly_train._methods.dinov2 import dinov2 as dinov2_module
from lightly_train._methods.dinov2.dinov2 import (
    DINOv2,
    DINOv2AdamWViTArgs,
    DINOv2Args,
)
from lightly_train._methods.dinov2.utils import is_tokenization_param
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.attention import (
    SDPAttention,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.block import (
    NestedTensorBlock,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import IMAGENET_SIZE, ScalingInfo
from lightly_train._torch_helpers import update_momentum
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
        "n_local_crops, ibot_separate_head, center_method, ibotpp",
        [
            (8, False, "softmax", None),
            (0, False, "softmax", None),
            (8, True, "softmax", None),
            (8, True, "sinkhorn_knopp", None),
            (8, False, "softmax", "all"),
            (0, False, "softmax", "masked"),
            (8, True, "sinkhorn_knopp", "all"),
            (8, False, "sinkhorn_knopp", "masked"),
        ],
    )
    def test_train_step_impl(
        self,
        mocker: MockerFixture,
        n_local_crops: int,
        ibot_separate_head: bool,
        center_method: Literal["softmax", "sinkhorn_knopp"],
        ibotpp: Literal["all", "masked"] | None,
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
            ibot_separate_head=ibot_separate_head,
            center_method=center_method,
            ibotpp=ibotpp,
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

    @pytest.mark.parametrize("ibotpp", [None, "all", "masked"])
    def test_train_step_impl__ibotpp_tokens(
        self, mocker: MockerFixture, ibotpp: Literal["all", "masked"] | None
    ) -> None:
        # Non-cubic (2, 4, 4) patch grid, see test_train_step_impl__anisotropic_views.
        emb_model = EmbeddingModel(
            wrapped_model=dummy_dinov2_vit_model(
                patch_size=(4, 2, 2), img_size=(16, 8, 4)
            )
        )
        b = 4
        n_crops = 2 * b
        n_patches = 2 * 4 * 4
        views = [torch.rand(b, 1, 4, 16, 8) for _ in range(2)] + [
            torch.rand(b, 1, 2, 8, 4) for _ in range(2)
        ]
        batch: Batch = {"views": views, "filename": [f"img_{i}" for i in range(b)]}
        dinov2 = setup_dinov2_helper(
            DINOv2Args(mask_probability=0.5, ibotpp=ibotpp), mocker, emb_model, b
        )
        masks_spy = mocker.spy(dinov2_module, "create_collated_masks")
        student_spy = mocker.spy(
            dinov2.student_embedding_model.wrapped_model, "forward_features"
        )
        loss_spy = mocker.spy(dinov2.ibot_loss, "forward_masked")

        out = dinov2.training_step_impl(batch, 0)

        assert torch.isfinite(out.loss)
        collated_masks = masks_spy.spy_return["collated_masks"]
        assert collated_masks.shape == (n_crops, n_patches)
        n_masked_crops = int(collated_masks.any(-1).sum())
        assert 0 < n_masked_crops < n_crops
        # The student input is masked the same way in every mode.
        global_call = student_spy.call_args_list[0]
        assert torch.equal(global_call.kwargs["masks"], collated_masks)

        loss_kwargs = loss_spy.call_args.kwargs
        ibot_masks = loss_kwargs["student_masks_flat"]
        if ibotpp is None:
            assert torch.equal(ibot_masks, collated_masks)
        elif ibotpp == "all":
            assert ibot_masks.shape == (n_crops, n_patches)
            assert ibot_masks.all()
        else:
            expected = collated_masks.any(-1, keepdim=True).expand_as(collated_masks)
            assert torch.equal(ibot_masks, expected)
        n_tokens = int(ibot_masks.sum())
        if ibotpp == "all":
            assert n_tokens == n_crops * n_patches
        elif ibotpp == "masked":
            assert n_tokens == n_masked_crops * n_patches
        assert loss_kwargs["n_masked_patches"] == n_tokens
        assert loss_kwargs["student_patch_tokens_masked"].shape[0] == n_tokens
        assert loss_kwargs["teacher_patch_tokens_masked"].shape[0] == n_tokens
        assert loss_kwargs["masks_weight"].shape == (n_tokens,)

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


def _compile_test_model() -> EmbeddingModel:
    """Small, but with the block layout of the shipped dinov2/vitb14-notpretrained: unchunked blocks, the same
    drop path rate in every block (drop_path_uniform), registers, LayerScale."""
    torch.manual_seed(0)
    model = DinoVisionTransformer(
        img_size=(8, 8, 8),
        patch_size=(2, 2, 2),
        in_chans=1,
        embed_dim=16,
        depth=4,
        num_heads=2,
        mlp_ratio=1,
        block_fn=partial(NestedTensorBlock, attn_class=SDPAttention),
        block_chunks=0,
        num_register_tokens=4,
        drop_path_rate=0.2,
        drop_path_uniform=True,
        init_values=1e-5,
    )
    return EmbeddingModel(wrapped_model=DINOv2ViTModelWrapper(model=model))


def _backbone_blocks(dinov2: DINOv2) -> list[torch.nn.Module]:
    return [
        block
        for embedding_model in (
            dinov2.teacher_embedding_model,
            dinov2.student_embedding_model,
        )
        for block in embedding_model.wrapped_model.get_model().blocks  # type: ignore[operator, union-attr]
    ]


class TestDINOv2CompileBlocks:
    """compile_blocks on the CPU, with Dynamo's "eager" backend: what is traced and how often.

    Inductor's numerics (compiled vs. eager vs. a higher-precision reference) are checked on a GPU by
    test_dinov2_compile_cuda.py.
    """

    @pytest.fixture(autouse=True)
    def _dynamo(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        # DINOv2.__init__ sets optimize_ddp globally; restore it afterwards.
        monkeypatch.setattr(
            torch._dynamo.config, "optimize_ddp", torch._dynamo.config.optimize_ddp
        )
        # Blocks falling back to eager after too many recompiles must fail, not pass.
        monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
        # Trace with Dynamo but run the captured graphs as they are, without Inductor.
        compile_ = torch.nn.Module.compile
        monkeypatch.setattr(
            torch.nn.Module,
            "compile",
            lambda self, *args, **kwargs: compile_(self, backend="eager"),
        )
        torch._dynamo.reset()
        counters.clear()
        yield
        torch._dynamo.reset()

    def _setup(self, mocker: MockerFixture, compile_blocks: bool) -> DINOv2:
        return setup_dinov2_helper(
            DINOv2Args(compile_blocks=compile_blocks),
            mocker,
            _compile_test_model(),
            batch_size=4,
        )

    def test_compiles_every_block(self, mocker: MockerFixture) -> None:
        eager = self._setup(mocker, compile_blocks=False)
        assert torch._dynamo.config.optimize_ddp
        compiled = self._setup(mocker, compile_blocks=True)
        assert not torch._dynamo.config.optimize_ddp
        assert all(b._compiled_call_impl is None for b in _backbone_blocks(eager))
        assert all(
            b._compiled_call_impl is not None for b in _backbone_blocks(compiled)
        )
        # Checkpoints and the exported model are unaffected.
        assert compiled.state_dict().keys() == eager.state_dict().keys()

    def test_training_step(self, mocker: MockerFixture) -> None:
        eager = self._setup(mocker, compile_blocks=False)
        compiled = self._setup(mocker, compile_blocks=True)
        compiled.load_state_dict(eager.state_dict())

        def step(dinov2: DINOv2, batch_size: int) -> dict[str, torch.Tensor]:
            torch.manual_seed(1)
            views = [torch.rand(batch_size, 1, 8, 8, 8) for _ in range(2)] + [
                torch.rand(batch_size, 1, 4, 4, 4) for _ in range(4)
            ]
            torch.manual_seed(2)  # drop path
            random.seed(2)  # iBOT masks
            dinov2.zero_grad()
            out = dinov2.training_step_impl(
                {"views": views, "filename": [""] * batch_size}, 0
            )
            out.loss.backward()
            return {"loss": out.loss.detach()} | {
                name: param.grad
                for name, param in dinov2.named_parameters()
                if param.grad is not None
            }

        # The second, smaller batch is the dynamic-shape recompile (last batch of an epoch).
        for batch_size in (4, 3):
            expected = step(eager, batch_size)
            actual = step(compiled, batch_size)
            assert actual.keys() == expected.keys()
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], msg=name)

        # The blocks of the student and the teacher share their graphs: one per mode -- teacher (no grad),
        # student global views, student local views (dynamic shapes from the second view size on) -- plus one
        # when the teacher sees its second batch size. Separate graphs per block (e.g. from per-block drop path
        # rates) would pass Dynamo's recompile limit after 8 and silently run the rest eagerly.
        assert counters["stats"]["unique_graphs"] == 4
        assert not counters["graph_break"]


def _heads(dinov2: DINOv2) -> list[torch.nn.Module]:
    """The distinct projection heads (the iBOT heads are the DINO heads unless
    ibot_separate_head)."""
    heads = {
        id(head): head
        for heads in (dinov2.teacher_head, dinov2.student_head)
        for head in (heads.dino_head, heads.ibot_head)
    }
    return list(heads.values())


class TestDINOv2CompileHeads:
    """compile_heads on the CPU, with Dynamo's "eager" backend: what is traced and how often.

    Inductor's numerics are checked on a GPU by test_dinov2_compile_cuda.py.
    """

    @pytest.fixture(autouse=True)
    def _dynamo(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        # DINOv2.__init__ sets these globally; restore them afterwards.
        monkeypatch.setattr(
            torch._dynamo.config, "optimize_ddp", torch._dynamo.config.optimize_ddp
        )
        monkeypatch.setattr(
            torch._functorch.config,
            "activation_memory_budget",
            torch._functorch.config.activation_memory_budget,
        )
        # Heads or the loss falling back to eager after too many recompiles must fail, not pass.
        monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
        # Trace with Dynamo but run the captured graphs as they are, without Inductor.
        module_compile = torch.nn.Module.compile
        monkeypatch.setattr(
            torch.nn.Module,
            "compile",
            lambda self, *args, **kwargs: module_compile(self, backend="eager"),
        )
        monkeypatch.setattr(torch, "compile", partial(torch.compile, backend="eager"))
        torch._dynamo.reset()
        counters.clear()
        yield
        torch._dynamo.reset()

    @pytest.mark.parametrize("ibot_separate_head", [False, True])
    def test_compiles_heads(
        self, mocker: MockerFixture, ibot_separate_head: bool
    ) -> None:
        eager = _chunk_test_method(mocker, ibot_separate_head=ibot_separate_head)
        assert torch._dynamo.config.optimize_ddp
        compiled = _chunk_test_method(
            mocker, ibot_separate_head=ibot_separate_head, compile_heads=True
        )
        assert not torch._dynamo.config.optimize_ddp
        assert len(_heads(compiled)) == (4 if ibot_separate_head else 2)
        assert all(head._compiled_call_impl is None for head in _heads(eager))
        assert all(head._compiled_call_impl is not None for head in _heads(compiled))
        assert compiled._ibot_chunk_fn is not dinov2_module.ibot_chunk_loss
        assert eager._ibot_chunk_fn is dinov2_module.ibot_chunk_loss
        # Checkpoints and the exported model are unaffected.
        assert compiled.state_dict().keys() == eager.state_dict().keys()

    @pytest.mark.parametrize(
        "kwargs, n_graphs",
        [
            # The head forward (shared by all heads): teacher and student (grad mode) x DINO and iBOT
            # (bottleneck_only), dynamic shapes from the second token count on, plus the teacher's DINO head at
            # its second batch size: 5. The iBOT loss: one static graph, then one with dynamic shapes and a
            # dynamic teacher temperature: 2.
            ({}, 7),
            # 48 divides neither token count of plain iBOT, but its first chunk already differs from the second
            # token count. With ibotpp="all" (256 or 192 tokens) the partial last chunk adds one: see below.
            ({"ibot_loss_chunk_size": 48}, 7),
            # The blocks' 4 graphs of TestDINOv2CompileBlocks plus the teacher's at the third step.
            ({"compile_blocks": True}, 12),
            ({"ibot_loss_chunk_size": 48, "compile_blocks": True}, 12),
        ],
    )
    @pytest.mark.parametrize("ibotpp", [None, "all"])
    def test_training_steps(
        self,
        mocker: MockerFixture,
        kwargs: dict[str, Any],
        n_graphs: int,
        ibotpp: Literal["all"] | None,
    ) -> None:
        # compile_heads always uses the bottleneck-based iBOT loss: compare with the unchunked eager one.
        eager = _chunk_test_method(mocker, ibotpp=ibotpp)
        compiled = _chunk_test_method(
            mocker, ibotpp=ibotpp, compile_heads=True, **kwargs
        )
        compiled.load_state_dict(eager.state_dict())
        expected = _chunk_test_steps(eager, torch.float32, n_local_crops=2)
        actual = _chunk_test_steps(compiled, torch.float32, n_local_crops=2)
        for step, (exp, act) in enumerate(zip(expected, actual)):
            assert act.keys() == exp.keys()
            for name in exp:
                torch.testing.assert_close(
                    act[name], exp[name], msg=f"step {step}: {name}"
                )
        if ibotpp == "all" and "ibot_loss_chunk_size" in kwargs:
            # The first chunks (48 tokens) and the partial last one have different sizes: the loss compiles a
            # static graph for the first, a dynamic one for the last and a dynamic one for the changed teacher
            # temperature.
            n_graphs += 1
        assert counters["stats"]["unique_graphs"] == n_graphs
        assert not counters["graph_break"]
        # Steady state: further steps (new teacher temperatures, the same shapes) compile nothing new.
        _chunk_test_steps(compiled, torch.float32, n_local_crops=2, first_step=3)
        assert counters["stats"]["unique_graphs"] == n_graphs

    def test_activation_memory_budget(self, mocker: MockerFixture) -> None:
        assert torch._functorch.config.activation_memory_budget == 1.0
        _chunk_test_method(mocker, compile_heads=True, activation_memory_budget=0.5)
        assert torch._functorch.config.activation_memory_budget == 0.5


def _chunk_test_method(mocker: MockerFixture, **kwargs: Any) -> DINOv2:
    """A small DINOv2 on a non-cubic (2, 4, 4) patch grid, 8 global crops of 32 tokens.

    output_dim (200) differs from every other size, so the [*, output_dim] tensors of
    the losses can be told apart.
    """
    torch.manual_seed(0)
    emb_model = EmbeddingModel(
        wrapped_model=dummy_dinov2_vit_model(patch_size=(4, 2, 2), img_size=(16, 8, 4))
    )
    args = DINOv2Args(output_dim=200, hidden_dim=32, dino_bottleneck_dim=16, **kwargs)
    return setup_dinov2_helper(args, mocker, emb_model, batch_size=4)


def _chunk_test_steps(
    method: DINOv2, dtype: torch.dtype, n_local_crops: int, first_step: int = 0
) -> list[dict[str, torch.Tensor]]:
    """Three training steps (batch sizes 4, 3, 4; global_step 0, 1, 2, so the teacher
    temperature changes), each followed by an SGD step of the student and the EMA
    update of the teacher. Records every loss term, gradient, both centers (after
    their pending update) and every parameter after the update."""
    records = []
    for step, batch_size in enumerate((4, 3, 4), start=first_step):
        method.trainer.global_step = step  # type: ignore[misc]
        g = torch.Generator().manual_seed(100 + step)
        views = [
            torch.rand(batch_size, 1, 4, 16, 8, generator=g, dtype=dtype)
            for _ in range(2)
        ] + [
            torch.rand(batch_size, 1, 2, 8, 4, generator=g, dtype=dtype)
            for _ in range(n_local_crops)
        ]
        random.seed(step)  # iBOT masks
        torch.manual_seed(step)  # drop path
        method.zero_grad(set_to_none=True)
        out = method.training_step_impl(
            {"views": views, "filename": [""] * batch_size}, step
        )
        out.loss.backward()  # type: ignore[no-untyped-call]
        assert out.log_dict is not None
        record = {"loss": out.loss.detach()}
        record |= {name: value.detach() for name, value in out.log_dict.items()}
        # The centers after this step's pending update, read from copies: the method itself must apply the
        # update at the next step.
        for name in ("dino_loss", "ibot_loss"):
            loss_module = copy.deepcopy(getattr(method, name))
            loss_module.apply_center_update()
            record[f"center/{name}"] = loss_module.center
        for name, param in method.named_parameters():
            if param.grad is not None:
                record[f"grad/{name}"] = param.grad.clone()
        with torch.no_grad():
            for param in method.parameters():
                if param.grad is not None:
                    param.sub_(0.1 * param.grad)
        update_momentum(
            method.student_embedding_model, method.teacher_embedding_model, m=0.9
        )
        update_momentum(method.student_head, method.teacher_head, m=0.9)
        for name, param in method.named_parameters():
            record[f"param/{name}"] = param.detach().clone()
        records.append(record)
    return records


class TestIBOTLossChunking:
    """ibot_loss_chunk_size: the chunked iBOT loss against the unchunked one, over
    whole training steps."""

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    @pytest.mark.parametrize("n_local_crops", [0, 2])
    @pytest.mark.parametrize("ibot_separate_head", [False, True])
    @pytest.mark.parametrize("chunk_size", [7, 64, 1_000_000])
    @pytest.mark.parametrize("ibotpp", [None, "all", "masked"])
    def test_training_steps__equal_to_unchunked(
        self,
        mocker: MockerFixture,
        ibotpp: Literal["all", "masked"] | None,
        chunk_size: int,
        ibot_separate_head: bool,
        n_local_crops: int,
        dtype: torch.dtype,
    ) -> None:
        kwargs: dict[str, Any] = dict(
            ibotpp=ibotpp, ibot_separate_head=ibot_separate_head
        )
        unchunked = _chunk_test_method(mocker, **kwargs).to(dtype)
        chunked = _chunk_test_method(
            mocker, ibot_loss_chunk_size=chunk_size, **kwargs
        ).to(dtype)
        chunked.load_state_dict(unchunked.state_dict())
        expected = _chunk_test_steps(unchunked, dtype, n_local_crops)
        actual = _chunk_test_steps(chunked, dtype, n_local_crops)

        for step, (exp, act) in enumerate(zip(expected, actual)):
            assert act.keys() == exp.keys()
            for name in exp:
                msg = f"step {step}: {name}"
                if chunk_size >= 256 and not name.startswith("center/"):
                    # A single chunk runs the unchunked operations in the same order.
                    assert torch.equal(act[name], exp[name]), msg
                elif dtype == torch.float64:
                    torch.testing.assert_close(
                        act[name], exp[name], rtol=1e-10, atol=1e-12, msg=msg
                    )
                else:
                    torch.testing.assert_close(act[name], exp[name], msg=msg)

    def test_training_step__keeps_no_full_size_tensors(
        self, mocker: MockerFixture
    ) -> None:
        """Nothing [> chunk_size, output_dim] is kept for the backward pass."""

        def saved_logits_rows(method: DINOv2) -> list[int]:
            rows: list[int] = []

            def pack(t: torch.Tensor) -> torch.Tensor:
                if t.dim() == 2 and t.shape[1] == 200:
                    rows.append(t.shape[0])
                return t

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                _chunk_test_steps(method, torch.float32, n_local_crops=2)
            return rows

        unchunked = saved_logits_rows(_chunk_test_method(mocker, ibotpp="all"))
        # All 32 tokens of the 8 (or 6) global crops.
        assert max(unchunked) == 256
        chunked = saved_logits_rows(
            _chunk_test_method(mocker, ibotpp="all", ibot_loss_chunk_size=16)
        )
        # The DINO loss keeps its [crops, output_dim] tensors (8 rows at most).
        assert max(chunked) <= 16

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"compile_heads": True},
            {"ibot_loss_chunk_size": 8},
        ],
    )
    def test_sinkhorn_knopp__raises(self, kwargs: dict[str, Any]) -> None:
        # Rejected by the args themselves, i.e. when the config is parsed.
        with pytest.raises(ValidationError, match="need center_method='softmax'"):
            DINOv2Args(center_method="sinkhorn_knopp", **kwargs)

    def test_activation_memory_budget_without_compile__raises(self) -> None:
        with pytest.raises(ValidationError, match="only applies to compiled regions"):
            DINOv2Args(activation_memory_budget=0.5)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"compile_heads": True},
            {"compile_blocks": True, "activation_memory_budget": 0.5},
            {"ibot_loss_chunk_size": 8, "compile_heads": True},
        ],
    )
    def test_args__valid(self, kwargs: dict[str, Any]) -> None:
        DINOv2Args(**kwargs)

    def test_no_ibot_tokens__center_update_still_called(
        self, mocker: MockerFixture
    ) -> None:
        # Without masked crops, plain iBOT has no tokens. update_center must still run: it all-reduces across
        # ranks under DDP, so a rank skipping it would desynchronize the collectives. It contributes the
        # current center, i.e. leaves it unchanged.
        method = _chunk_test_method(
            mocker, mask_probability=0.0, ibot_loss_chunk_size=8
        )
        method.ibot_loss.center = torch.randn(1, 1, 200)
        center = method.ibot_loss.center.clone()
        spy = mocker.spy(method.ibot_loss, "update_center")
        records = _chunk_test_steps(method, torch.float32, n_local_crops=0)
        assert spy.call_count == 3
        for record in records:
            assert record["train_loss/ibot_loss"].item() == 0.0
            assert torch.isfinite(record["loss"])
            torch.testing.assert_close(record["center/ibot_loss"], center)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ibot_loss_chunk_size": 0},
            {"activation_memory_budget": 0.0},
            {"activation_memory_budget": 1.5},
        ],
    )
    def test_args__out_of_range(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            DINOv2Args(**kwargs)


class TestDINOv2Args:
    def test_resolve_auto(self) -> None:
        args = DINOv2Args()
        args.resolve_auto(
            scaling_info=ScalingInfo(dataset_size=IMAGENET_SIZE, epochs=100),
            optimizer_args=DINOv2AdamWViTArgs(),
            wrapped_model=dummy_dinov2_vit_model(),
        )
        assert not args.has_auto()
