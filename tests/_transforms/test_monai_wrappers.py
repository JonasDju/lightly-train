#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math

import numpy as np
import pytest
from monai.data import MetaTensor
from monai.transforms import RandRotate

from lightly_train._transforms.monai_wrappers import (
    ORIG_SHAPE_META_KEY,
    AnisotropyAwareRandGaussianSharpen,
    AnisotropyAwareRandGaussianSmooth,
    AnisotropyTrackingRandomResizedCrop3D,
)


def _volume(h: int, w: int, d: int, dtype: type = np.float32) -> np.ndarray:
    # (C, H, W, D)
    return np.random.uniform(0, 255, size=(1, h, w, d)).astype(dtype)


def _crop(size: tuple[int, int, int] = (16, 16, 16)) -> AnisotropyTrackingRandomResizedCrop3D:
    # scale=(1.0, 1.0) so the crop always covers the whole volume -- keeps the test
    # focused on the shape tag, not on RandomResizedCrop3D's own sampling.
    return AnisotropyTrackingRandomResizedCrop3D(size=size, scale=(1.0, 1.0))


def _rotate(degrees: float = 10.0, prob: float = 1.0) -> RandRotate:
    # In-plane only, matching view_transform.py: range_x/range_y left at 0.
    # range_z is the one that carries the actual in-plane (H-W) rotation for a
    # (C, H, W, D) volume -- see the module docstring in monai_wrappers.py.
    return RandRotate(range_z=math.radians(degrees), prob=prob)


def _blur(sigma: tuple[float, float] = (1.0, 1.0), prob: float = 1.0) -> AnisotropyAwareRandGaussianSmooth:
    return AnisotropyAwareRandGaussianSmooth(sigma_range=sigma, prob=prob)


def _sharpen(
    sigma1: tuple[float, float] = (1.0, 1.0),
    sigma2: float | tuple[float, float] = 0.5,
    prob: float = 1.0,
) -> AnisotropyAwareRandGaussianSharpen:
    return AnisotropyAwareRandGaussianSharpen(
        sigma1=sigma1,
        sigma2=sigma2,
        prob=prob,
    )


class TestMetaTagSurvival:
    def test_crop_tags_pre_crop_shape(self) -> None:
        crop = _crop(size=(8, 8, 8))
        out = crop(_volume(20, 24, 10))
        assert isinstance(out, MetaTensor)
        assert out.meta[ORIG_SHAPE_META_KEY] == (20, 24, 10)
        assert out.shape[-3:] == (8, 8, 8)

    def test_tag_survives_rotate_then_blur(self) -> None:
        crop = _crop(size=(8, 8, 8))
        rotate = _rotate()
        blur = _blur()

        cropped = crop(_volume(20, 24, 10))
        rotated = rotate(cropped)
        assert isinstance(rotated, MetaTensor)
        assert tuple(int(s) for s in rotated.meta[ORIG_SHAPE_META_KEY]) == (20, 24, 10)

        blurred = blur(rotated)
        assert isinstance(blurred, MetaTensor)
        assert tuple(int(s) for s in blurred.meta[ORIG_SHAPE_META_KEY]) == (20, 24, 10)

    def test_tag_survives_sharpen(self) -> None:
        crop = _crop(size=(8, 8, 8))
        sharpen = _sharpen()

        cropped = crop(_volume(20, 24, 10))
        sharpened = sharpen(cropped)
        assert isinstance(sharpened, MetaTensor)
        assert tuple(int(s) for s in sharpened.meta[ORIG_SHAPE_META_KEY]) == (20, 24, 10)


class TestAnisotropyAwareRandGaussianSmooth:
    def test_scales_sigma_z_down_for_anisotropic_volume(self) -> None:
        cropped = _crop(size=(16, 16, 16))(_volume(64, 64, 8))
        blur = _blur(sigma=(1.0, 1.0))
        blur(cropped)
        assert blur.sigma_z[0] < blur._base_sigma_z[0]
        assert blur.sigma_z[0] < blur.sigma_x[0]

    def test_cubic_volume_scale_is_one(self) -> None:
        cropped = _crop(size=(16, 16, 16))(_volume(16, 16, 16))
        blur = _blur(sigma=(1.0, 1.0))
        blur(cropped)
        assert blur.sigma_z[0] == pytest.approx(blur._base_sigma_z[0])

    def test_does_not_compound_across_calls(self) -> None:
        blur = _blur(sigma=(1.0, 1.0))
        cropped = _crop(size=(16, 16, 16))(_volume(64, 64, 8))
        blur(cropped)
        first_sigma_z = blur.sigma_z
        blur(cropped)
        assert blur.sigma_z == pytest.approx(first_sigma_z)

    def test_two_original_depths_same_crop_target_scale_differently(self) -> None:
        crop = _crop(size=(16, 16, 16))
        blur_shallow = _blur(sigma=(1.0, 1.0))
        blur_deep = _blur(sigma=(1.0, 1.0))

        blur_shallow(crop(_volume(64, 64, 6)))
        blur_deep(crop(_volume(64, 64, 40)))

        assert blur_shallow.sigma_z[0] < blur_deep.sigma_z[0]

    def test_fallback_without_meta_tag(self) -> None:
        # No preceding crop step -- must not raise, falls back to img.shape.
        blur = _blur(sigma=(1.0, 1.0))
        blur(_volume(64, 64, 8))
        assert blur.sigma_z[0] < blur._base_sigma_z[0]


class TestAnisotropyAwareRandGaussianSharpen:
    def test_scales_both_sigmas_down_for_anisotropic_volume(self) -> None:
        cropped = _crop(size=(16, 16, 16))(_volume(64, 64, 8))
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        sharpen(cropped)
        assert sharpen.sigma1_z[0] < sharpen._base_sigma1_z[0]
        assert sharpen.sigma1_z[0] < sharpen.sigma1_x[0]
        assert sharpen.sigma2_z < sharpen._base_sigma2_z
        assert sharpen.sigma2_z < sharpen.sigma2_x

    def test_scales_tuple_sigma2_down_for_anisotropic_volume(self) -> None:
        cropped = _crop(size=(16, 16, 16))(_volume(64, 64, 8))
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=(0.3, 0.9))
        sharpen(cropped)
        assert sharpen.sigma2_z[0] < sharpen._base_sigma2_z[0]
        assert sharpen.sigma2_z[1] < sharpen._base_sigma2_z[1]

    def test_cubic_volume_scale_is_one(self) -> None:
        cropped = _crop(size=(16, 16, 16))(_volume(16, 16, 16))
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        sharpen(cropped)
        assert sharpen.sigma1_z[0] == pytest.approx(sharpen._base_sigma1_z[0])
        assert sharpen.sigma2_z == pytest.approx(sharpen._base_sigma2_z)

    def test_does_not_compound_across_calls(self) -> None:
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        cropped = _crop(size=(16, 16, 16))(_volume(64, 64, 8))
        sharpen(cropped)
        first_sigma1_z = sharpen.sigma1_z
        first_sigma2_z = sharpen.sigma2_z
        sharpen(cropped)
        assert sharpen.sigma1_z == pytest.approx(first_sigma1_z)
        assert sharpen.sigma2_z == pytest.approx(first_sigma2_z)

    def test_two_original_depths_same_crop_target_scale_differently(self) -> None:
        crop = _crop(size=(16, 16, 16))
        sharpen_shallow = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        sharpen_deep = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)

        sharpen_shallow(crop(_volume(64, 64, 6)))
        sharpen_deep(crop(_volume(64, 64, 40)))

        assert sharpen_shallow.sigma1_z[0] < sharpen_deep.sigma1_z[0]
        assert sharpen_shallow.sigma2_z < sharpen_deep.sigma2_z

    def test_fallback_without_meta_tag(self) -> None:
        # No preceding crop step -- must not raise, falls back to img.shape.
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        sharpen(_volume(64, 64, 8))
        assert sharpen.sigma1_z[0] < sharpen._base_sigma1_z[0]
