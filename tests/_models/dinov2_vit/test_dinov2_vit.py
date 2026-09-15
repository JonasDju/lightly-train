#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture

from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.drop_path import DropPath
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
    _vit_test,
)


def _model(**kwargs: Any) -> DinoVisionTransformer:
    # Sizes are (H, W, D). Inputs are (B, C, D, H, W). 4x4x4 patch grid at native size.
    kwargs.setdefault("patch_size", (2, 2, 2))
    kwargs.setdefault("img_size", (8, 8, 8))
    kwargs.setdefault("in_chans", 1)
    return _vit_test(**kwargs)


class TestDINOv2ViTModelWrapper:
    def test_init(self) -> None:
        feature_extractor = DINOv2ViTModelWrapper(model=_model())

        for name, param in feature_extractor.named_parameters():
            assert param.requires_grad, name

        for name, module in feature_extractor.named_modules():
            assert module.training, name

    def test_feature_dim(self) -> None:
        feature_extractor = DINOv2ViTModelWrapper(model=_model())
        assert feature_extractor.feature_dim() == 8

    def test_patch_size(self) -> None:
        feature_extractor = DINOv2ViTModelWrapper(model=_model(patch_size=(4, 2, 2)))
        assert tuple(feature_extractor.patch_size()) == (4, 2, 2)

    def test_forward_features(self) -> None:
        torch.manual_seed(0)
        feature_extractor = DINOv2ViTModelWrapper(model=_model())

        x = torch.rand(2, 1, 8, 8, 8)
        collated_masks = torch.rand(2, 4 * 4 * 4) > 0.5

        feats_cls = feature_extractor.forward_features(x)
        assert feats_cls["features"].shape == (2, 8, 4, 4, 4)
        assert feats_cls["cls_token"].shape == (2, 8)

        feats_cls_masked = feature_extractor.forward_features(x, masks=collated_masks)
        assert not torch.allclose(
            feats_cls["features"], feats_cls_masked["features"], atol=1e-6
        )
        assert not torch.allclose(
            feats_cls["cls_token"], feats_cls_masked["cls_token"], atol=1e-6
        )

    def test_forward_features__patch_order(self) -> None:
        # The features must be reshaped as (B, C, D, H, W) in the token order of the
        # patch embedding.
        feature_extractor = DINOv2ViTModelWrapper(model=_model())
        x = torch.rand(1, 1, 8, 8, 8)
        tokens = feature_extractor.get_model()(x, is_training=True)[
            "x_norm_patchtokens"
        ]
        features = feature_extractor.forward_features(x)["features"]
        torch.testing.assert_close(features.flatten(2).transpose(1, 2), tokens)

    def test_forward_features__n_blocks(self) -> None:
        feature_extractor = DINOv2ViTModelWrapper(model=_model())
        feats = feature_extractor.forward_features(
            torch.rand(2, 1, 8, 8, 8), n_blocks=2
        )
        assert feats["features"].shape == (2, 16, 4, 4, 4)
        assert feats["cls_token"].shape == (2, 16)

    def test_forward_features__non_cubic_native_size(
        self, mocker: MockerFixture
    ) -> None:
        # Regression test for swapped H and W axes: (H, W, D) = (16, 8, 4) with patch
        # size (4, 2, 2) gives a (D, H, W) = (2, 4, 4) grid. At the native size neither
        # the input nor the positional embedding must be interpolated.
        model = _model(img_size=(16, 8, 4), patch_size=(4, 2, 2))
        assert model.patch_embed.patches_resolution == (2, 4, 4)
        feature_extractor = DINOv2ViTModelWrapper(model=model)
        mocker.patch(
            "torch.nn.functional.interpolate",
            side_effect=AssertionError("unexpected interpolation"),
        )
        feats = feature_extractor.forward_features(torch.rand(2, 1, 4, 16, 8))
        assert feats["features"].shape == (2, 8, 2, 4, 4)

    @pytest.mark.parametrize("interpolate_antialias", [False, True])
    @pytest.mark.parametrize("interpolate_offset", [0.0, 0.1])
    def test_forward_features__interpolate_pos_embed(
        self, interpolate_antialias: bool, interpolate_offset: float
    ) -> None:
        # Inputs with a different size than img_size interpolate the positional
        # embedding. antialias=True must not crash for the 3D model.
        feature_extractor = DINOv2ViTModelWrapper(
            model=_model(
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
            )
        )
        feats = feature_extractor.forward_features(torch.rand(1, 1, 4, 12, 6))
        assert feats["features"].shape == (1, 8, 2, 6, 3)

    def test_forward_features__resizes_to_multiple_of_patch_size(self) -> None:
        feature_extractor = DINOv2ViTModelWrapper(model=_model())
        feats = feature_extractor.forward_features(torch.rand(1, 1, 7, 7, 5))
        assert feats["features"].shape == (1, 8, 4, 4, 3)

    def test_forward_features__dinov2_keeps_input(self) -> None:
        model = _model()
        feature_extractor = DINOv2ViTModelWrapper(model=model)
        volume = torch.rand(1, 1, 8, 8, 8)
        inputs: list[torch.Tensor] = []

        handle = model.register_forward_pre_hook(
            lambda _, args: inputs.append(args[0].detach().clone())
        )
        try:
            feature_extractor.forward_features(volume)
        finally:
            handle.remove()

        assert torch.equal(inputs[0], volume)

    def test_forward_multiscale_features(self) -> None:
        feature_extractor = DINOv2ViTModelWrapper(model=_model())
        feats = feature_extractor.forward_multiscale_features(
            torch.rand(2, 1, 8, 8, 8), layer_indices=[0, 2]
        )
        assert len(feats) == 2
        for feat in feats:
            assert feat["features"].shape == (2, 8, 4, 4, 4)
            assert feat["cls_token"].shape == (2, 8)

    def test_forward_pool(self) -> None:
        feature_extractor = DINOv2ViTModelWrapper(model=_model())

        x = torch.rand(1, 8, 4, 4, 4)
        x_cls = torch.rand(1, 8)
        pooled_features = feature_extractor.forward_pool(
            {"features": x, "cls_token": x_cls}
        )["pooled_features"]
        assert pooled_features.shape == (1, 8, 1, 1, 1)

    def test_get_model(self) -> None:
        model = _model()
        extractor = DINOv2ViTModelWrapper(model=model)
        assert extractor.get_model() is model

    @pytest.mark.parametrize(
        "model_name",
        ["_vittest14"],
    )
    def test_make_teacher(self, model_name: str) -> None:
        student = DINOv2ViTPackage.get_model(model_name, num_input_channels=1)
        feature_extractor = DINOv2ViTModelWrapper(model=copy.deepcopy(student))
        feature_extractor.make_teacher()
        teacher = feature_extractor.get_model()

        #  Ensure models are the same expect for the drop paths
        assert len(list(student.parameters())) == len(list(teacher.parameters()))
        for (name_student, param_student), (name_teacher, param_teacher) in zip(
            student.named_parameters(), teacher.named_parameters()
        ):
            assert name_student == name_teacher
            assert param_student.dtype == param_teacher.dtype
            assert param_student.requires_grad == param_teacher.requires_grad
            assert torch.allclose(param_student, param_teacher, rtol=1e-3, atol=1e-4)

        for student_block, teacher_block in zip(student.blocks, teacher.blocks):
            assert isinstance(student_block.drop_path1, DropPath)
            assert isinstance(student_block.drop_path2, DropPath)
            assert student_block.sample_drop_ratio > 0.0  # type: ignore[operator]
            assert isinstance(teacher_block.drop_path1, torch.nn.Identity)
            assert isinstance(teacher_block.drop_path2, torch.nn.Identity)
            assert teacher_block.sample_drop_ratio == 0.0  # type: ignore[operator]

    def test__device(self) -> None:
        # If this test fails it means the wrapped model doesn't move all required
        # modules to the correct device. This happens if not all required modules
        # are registered as attributes of the class.
        wrapped_model = DINOv2ViTModelWrapper(model=_model())
        wrapped_model.to("meta")
        wrapped_model.forward_features(torch.rand(1, 1, 8, 8, 8, device="meta"))


class TestDINOv2ViTPackageConfigs:
    def test_vittest14__3d(self) -> None:
        model = DINOv2ViTPackage.get_model("_vittest14", num_input_channels=1)
        assert tuple(model.patch_size) == (14, 14, 4)
        assert model.patch_embed.patch_size == (4, 14, 14)
        assert model.patch_embed.in_chans == 1
        feats = DINOv2ViTModelWrapper(model=model).forward_features(
            torch.rand(1, 1, 16, 56, 56)
        )
        assert feats["features"].shape == (1, 8, 4, 4, 4)
