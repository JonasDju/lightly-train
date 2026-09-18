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
    OneOf,
    RandAdjustContrast,
    RandGaussianNoise,
    RandGibbsNoise,
    RandHistogramShift,
    Transform,
)

from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2ViTTransform,
)
from lightly_train._transforms.monai_wrappers import (
    AlphaRandHistogramShift,
    AnisotropyAwareRandGaussianSharpen,
)
from lightly_train._transforms.transform import (
    RandAdjustContrastArgs,
    RandGaussianNoiseArgs,
    RandGaussianSharpenArgs,
    RandGibbsNoiseArgs,
    RandHistogramShiftArgs,
)
from lightly_train.types import TransformInput


def _contains(ops: list[Transform], cls: type) -> bool:
    """True if `cls` appears among `ops`, including nested inside a OneOf (used
    for the mutually-exclusive blur/sharpen selection in view_transform.py)."""
    for op in ops:
        if isinstance(op, cls):
            return True
        if isinstance(op, OneOf) and any(isinstance(t, cls) for t in op.transforms):
            return True
    return False


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


def test_dinov2_transform__default_augmentation_status() -> None:
    # histogram_shift/adjust_contrast are enabled by default (replacements for the
    # RGB-only photometric ops); gaussian_sharpen/gaussian_noise stay off by
    # default. gibbs_noise is off by default too, *except* for global view 1
    # (transforms[1]), which enables it via its own per-view override -- see
    # DINOGlobalView1TransformArgs in dino_transform.py.
    transform_args = DINOv2ViTTransform.transform_args_cls()()
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    transform = DINOv2ViTTransform(transform_args)

    always_on = (AlphaRandHistogramShift, RandAdjustContrast)
    always_off = (AnisotropyAwareRandGaussianSharpen, RandGaussianNoise)
    for view_transform in transform.transforms:
        ops = view_transform.transform.transforms
        for cls in always_on:
            assert _contains(ops, cls)
        for cls in always_off:
            assert not _contains(ops, cls)

    global_0_ops = transform.transforms[0].transform.transforms
    global_1_ops = transform.transforms[1].transform.transforms
    local_ops = transform.transforms[2].transform.transforms
    assert not _contains(global_0_ops, RandGibbsNoise)
    assert _contains(global_1_ops, RandGibbsNoise)
    assert not _contains(local_ops, RandGibbsNoise)


def test_dinov2_transform__histogram_shift_and_contrast_and_noise_shared_across_views() -> None:
    # Unlike gaussian_blur/gaussian_sharpen/gibbs_noise (which have per-view
    # overrides for global view 1), histogram_shift/adjust_contrast/gaussian_noise
    # are shared: one top-level setting reaches every view.
    transform_args = DINOv2ViTTransform.transform_args_cls()(
        histogram_shift=RandHistogramShiftArgs(prob=1.0),
        adjust_contrast=RandAdjustContrastArgs(prob=1.0),
        gaussian_noise=RandGaussianNoiseArgs(prob=1.0),
    )
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    transform = DINOv2ViTTransform(transform_args)

    assert len(transform.transforms) == 2 + 8
    shared_op_types = (AlphaRandHistogramShift, RandAdjustContrast, RandGaussianNoise)
    for view_transform in transform.transforms:
        ops = view_transform.transform.transforms
        for cls in shared_op_types:
            assert _contains(ops, cls)

    # Sanity check that shapes are unaffected by the new augmentations.
    volume = np.random.uniform(0, 255, size=(1, 300, 280, 30)).astype(np.uint8)
    views = transform({"image": volume})
    for view in views[:2]:
        assert view["image"].shape == (1, 16, 224, 224)
    for view in views[2:]:
        assert view["image"].shape == (1, 8, 98, 98)


def test_dinov2_transform__gaussian_sharpen_and_gibbs_noise_have_per_view_override() -> None:
    # A top-level gaussian_sharpen/gibbs_noise setting reaches global view 0 and
    # the local views, but NOT global view 1 -- it has its own
    # DINOGlobalView1TransformArgs.gaussian_sharpen/.gibbs_noise override instead
    # (default: gaussian_sharpen off, gibbs_noise on -- see dino_transform.py).
    # gaussian_blur is also enabled by default; since it and gaussian_sharpen end
    # up mutually exclusive per view (OneOf, see view_transform.py) rather than
    # two separate ops, _contains() below looks inside OneOf too.
    transform_args = DINOv2ViTTransform.transform_args_cls()(
        gaussian_sharpen=RandGaussianSharpenArgs(prob=1.0),
        gibbs_noise=RandGibbsNoiseArgs(prob=1.0),
    )
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    transform = DINOv2ViTTransform(transform_args)

    global_0_ops = transform.transforms[0].transform.transforms
    global_1_ops = transform.transforms[1].transform.transforms
    local_ops = transform.transforms[2].transform.transforms
    assert _contains(global_0_ops, AnisotropyAwareRandGaussianSharpen)
    assert _contains(global_0_ops, RandGibbsNoise)
    assert _contains(local_ops, AnisotropyAwareRandGaussianSharpen)
    assert _contains(local_ops, RandGibbsNoise)
    # global view 1 ignores the top-level gaussian_sharpen (its own default is off)
    # but still has its own default-enabled gibbs_noise.
    assert not _contains(global_1_ops, AnisotropyAwareRandGaussianSharpen)
    assert _contains(global_1_ops, RandGibbsNoise)
