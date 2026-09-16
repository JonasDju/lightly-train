#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
import torch
from monai.transforms import (
    RandAdjustContrast,
    RandGaussianNoise,
    RandGibbsNoise,
    RandHistogramShift,
)

from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2ViTTransform,
)
from lightly_train._transforms.monai_wrappers import AnisotropyAwareRandGaussianSharpen
from lightly_train._transforms.transform import (
    RandAdjustContrastArgs,
    RandGaussianNoiseArgs,
    RandGaussianSharpenArgs,
    RandGibbsNoiseArgs,
    RandHistogramShiftArgs,
)
from lightly_train.types import TransformInput


def test_dinov2_transform_args__photometric_2d_fields_have_no_dino_defaults() -> None:
    # channel_drop/color_jitter/random_gray_scale/solarize are RGB-only 2D fields
    # inherited from the shared MethodTransformArgs base. DINOTransformArgs used to
    # re-declare them with DINO-specific defaults (DINOColorJitterArgs,
    # random_gray_scale=0.2, ...) despite never actually forwarding them to
    # ViewTransform -- pure clutter. That re-declaration is now gone, so they fall
    # back to the base class's own (inert) None default.
    fields = DINOv2ViTTransform.transform_args_cls().model_fields
    for name in ("channel_drop", "color_jitter", "random_gray_scale", "solarize"):
        assert fields[name].default is None
    transform_args = DINOv2ViTTransform.transform_args_cls()()
    assert transform_args.channel_drop is None
    assert transform_args.color_jitter is None
    assert transform_args.random_gray_scale is None
    assert transform_args.solarize is None


def test_dinov2_transform_shapes() -> None:
    # Volumes enter the transform as (C, H, W, D) uint8 arrays.
    volume = np.random.uniform(0, 255, size=(1, 300, 280, 30)).astype(np.uint8)
    input: TransformInput = {"image": volume}

    transform_args = DINOv2ViTTransform.transform_args_cls()()
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    assert transform_args.num_channels == 1
    transform = DINOv2ViTTransform(transform_args)

    views = transform(input)
    assert len(views) == 2 + 8
    # Views leave the transform as (C, D, H, W) float tensors.
    for view in views[:2]:
        assert view["image"].shape == (1, 16, 224, 224)
    for view in views[2:]:
        assert view["image"].shape == (1, 8, 98, 98)
    for view in views:
        assert type(view["image"]) is torch.Tensor
        assert view["image"].dtype == torch.float32


def test_dinov2_transform__new_monai_ops_disabled_by_default() -> None:
    transform_args = DINOv2ViTTransform.transform_args_cls()()
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    transform = DINOv2ViTTransform(transform_args)

    new_op_types = (
        AnisotropyAwareRandGaussianSharpen,
        RandGibbsNoise,
        RandHistogramShift,
        RandAdjustContrast,
        RandGaussianNoise,
    )
    for view_transform in transform.transforms:
        ops = view_transform.transform.transforms
        for cls in new_op_types:
            assert not any(isinstance(op, cls) for op in ops)


def test_dinov2_transform__new_monai_ops_shared_across_all_views() -> None:
    # gaussian_blur has per-view overrides (global_view_1, local_view); the five new
    # augmentations do not -- one top-level setting reaches every view.
    transform_args = DINOv2ViTTransform.transform_args_cls()(
        gaussian_sharpen=RandGaussianSharpenArgs(prob=1.0),
        gibbs_noise=RandGibbsNoiseArgs(prob=1.0),
        histogram_shift=RandHistogramShiftArgs(prob=1.0),
        adjust_contrast=RandAdjustContrastArgs(prob=1.0),
        gaussian_noise=RandGaussianNoiseArgs(prob=1.0),
    )
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    transform = DINOv2ViTTransform(transform_args)

    assert len(transform.transforms) == 2 + 8
    new_op_types = (
        AnisotropyAwareRandGaussianSharpen,
        RandGibbsNoise,
        RandHistogramShift,
        RandAdjustContrast,
        RandGaussianNoise,
    )
    for view_transform in transform.transforms:
        ops = view_transform.transform.transforms
        for cls in new_op_types:
            assert any(isinstance(op, cls) for op in ops)

    # Sanity check that shapes are unaffected by the new augmentations.
    volume = np.random.uniform(0, 255, size=(1, 300, 280, 30)).astype(np.uint8)
    views = transform({"image": volume})
    for view in views[:2]:
        assert view["image"].shape == (1, 16, 224, 224)
    for view in views[2:]:
        assert view["image"].shape == (1, 8, 98, 98)
