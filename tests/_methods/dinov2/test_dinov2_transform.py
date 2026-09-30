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

from lightly_train._methods.dino.dino_transform import DINOGaussianSharpenArgs
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


def _find(ops: list[Transform], cls: type) -> Transform | None:
    """The first `cls` among `ops`, including nested inside a OneOf."""
    for op in ops:
        candidates = op.transforms if isinstance(op, OneOf) else [op]
        for t in candidates:
            if isinstance(t, cls):
                return t
    return None


def test_dinov2_transform__gaussian_sharpen_is_configured_per_view() -> None:
    # Like gaussian_blur, gaussian_sharpen is set per view: the top-level setting only
    # reaches global view 0; global view 1 and the local views read their own
    # global_view_1/local_view overrides. Distinct probs show which setting each view
    # actually got. The blur is on by default in every view, so the sharpen may sit
    # inside a OneOf (see view_transform.py) -- _find looks there too.
    transform_args = DINOv2ViTTransform.transform_args_cls()(
        gaussian_sharpen=DINOGaussianSharpenArgs(prob=0.9),
        global_view_1={"gaussian_sharpen": {"prob": 0.3}},
        local_view={"gaussian_sharpen": {"prob": 0.6}},
    )
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    transform = DINOv2ViTTransform(transform_args)

    probs = []
    for view_transform in transform.transforms:
        sharpen = _find(
            view_transform.transform.transforms, AnisotropyAwareRandGaussianSharpen
        )
        assert sharpen is not None
        probs.append(sharpen.prob)
    assert probs == [0.9, 0.3] + [0.6] * 8


def test_dinov2_transform__gaussian_sharpen_does_not_leak_into_other_views() -> None:
    # Without per-view overrides, a top-level gaussian_sharpen stays in global view 0.
    transform_args = DINOv2ViTTransform.transform_args_cls()(
        gaussian_sharpen=DINOGaussianSharpenArgs(prob=1.0),
    )
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    transform = DINOv2ViTTransform(transform_args)

    ops = [vt.transform.transforms for vt in transform.transforms]
    assert _contains(ops[0], AnisotropyAwareRandGaussianSharpen)
    for view_ops in ops[1:]:
        assert not _contains(view_ops, AnisotropyAwareRandGaussianSharpen)


def test_dinov2_transform__gibbs_noise_has_global_view_1_override() -> None:
    # A top-level gibbs_noise reaches global view 0 and the local views; global view 1
    # has its own DINOGlobalView1TransformArgs.gibbs_noise instead (on by default).
    transform_args = DINOv2ViTTransform.transform_args_cls()(
        gibbs_noise=RandGibbsNoiseArgs(prob=0.9),
    )
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    transform = DINOv2ViTTransform(transform_args)

    probs = []
    for view_transform in transform.transforms:
        gibbs = _find(view_transform.transform.transforms, RandGibbsNoise)
        assert gibbs is not None
        probs.append(gibbs.prob)
    assert probs[0] == 0.9
    assert transform_args.global_view_1.gibbs_noise is not None
    assert probs[1] == transform_args.global_view_1.gibbs_noise.prob != 0.9
    assert probs[2:] == [0.9] * 8
