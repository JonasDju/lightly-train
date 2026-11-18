#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
import torch
from monai.data import MetaTensor
from monai.transforms import CropForeground, RandRotate

from lightly_train._transforms.monai_wrappers import (
    FOREGROUND_BOX_META_KEY,
    VOXEL_SPACING_META_KEY,
    AnisotropyAwareRandGaussianSharpen,
    AnisotropyAwareRandGaussianSmooth,
    AnisotropyTrackingRandomResizedCrop3D,
    TagForegroundBox,
)


def _volume(h: int, w: int, d: int, dtype: type = np.float32) -> np.ndarray:
    # (C, H, W, D)
    return np.random.uniform(0, 255, size=(1, h, w, d)).astype(dtype)


def _crop(size: tuple[int, int, int] = (16, 16, 16)) -> AnisotropyTrackingRandomResizedCrop3D:
    # scale=ratio=(1.0, 1.0) so the crop always covers the whole volume (the default ratio
    # jitter would shrink it) -- keeps the test focused on the spacing tag, not on
    # RandomResizedCrop3D's own sampling.
    return AnisotropyTrackingRandomResizedCrop3D(size=size, scale=(1.0, 1.0), ratio=(1.0, 1.0))


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


def _spaced(volume: np.ndarray, spacing: tuple[float, float, float]) -> MetaTensor:
    """``volume`` as ``DINOTransform.__call__`` passes it to the views: tagged with its (H, W, D) spacing."""
    return MetaTensor(torch.from_numpy(volume), meta={VOXEL_SPACING_META_KEY: spacing})


def _anisotropic_view() -> MetaTensor:
    # 64x64x8 at 0.5 x 0.5 x 3 mm, whole, resized to 16x16x8: view spacing 2 x 2 x 3 mm, scale_z 2/3.
    return _crop(size=(16, 16, 8))(_spaced(_volume(64, 64, 8), (0.5, 0.5, 3.0)))


class TestCropTagsViewSpacing:
    def test_whole_volume_crop_scales_the_spacing_by_crop_over_out(self) -> None:
        out = _crop(size=(8, 8, 8))(_spaced(_volume(20, 24, 10), (0.5, 0.6, 3.0)))
        assert isinstance(out, MetaTensor)
        assert out.shape[-3:] == (8, 8, 8)
        assert out.meta[VOXEL_SPACING_META_KEY] == pytest.approx((0.5 * 20 / 8, 0.6 * 24 / 8, 3.0 * 10 / 8))

    def test_random_crop_uses_its_own_crop_size(self) -> None:
        volume = _spaced(_volume(40, 36, 12), (0.5, 0.6, 3.0))
        crop = AnisotropyTrackingRandomResizedCrop3D(size=(8, 8, 4), scale=(0.1, 0.5), seed=0)
        twin = AnisotropyTrackingRandomResizedCrop3D(size=(8, 8, 4), scale=(0.1, 0.5), seed=0)
        for _ in range(10):
            params = twin.get_params(volume.shape)
            expected = (0.5 * params.height / 8, 0.6 * params.width / 8, 3.0 * params.depth / 4)
            assert crop(volume).meta[VOXEL_SPACING_META_KEY] == pytest.approx(expected)

    def test_without_spacing_it_is_one_over_the_shape(self) -> None:
        # Every series spans the same unit extent: resized whole to a cube, its voxels are cubic.
        for volume in (_volume(20, 24, 10), MetaTensor(torch.from_numpy(_volume(20, 24, 10)))):
            out = _crop(size=(8, 8, 8))(volume)
            assert out.meta[VOXEL_SPACING_META_KEY] == pytest.approx((1 / 8, 1 / 8, 1 / 8))
        out = _crop(size=(10, 12, 4))(_volume(20, 24, 10))
        assert out.meta[VOXEL_SPACING_META_KEY] == pytest.approx((1 / 10, 1 / 12, 1 / 4))

    def test_tag_survives_rotate_then_blur(self) -> None:
        cropped = _anisotropic_view()
        rotated = _rotate()(cropped)
        assert isinstance(rotated, MetaTensor)
        assert rotated.meta[VOXEL_SPACING_META_KEY] == pytest.approx((2.0, 2.0, 3.0))
        blurred = _blur()(rotated)
        assert isinstance(blurred, MetaTensor)
        assert blurred.meta[VOXEL_SPACING_META_KEY] == pytest.approx((2.0, 2.0, 3.0))

    def test_tag_survives_sharpen(self) -> None:
        sharpened = _sharpen()(_anisotropic_view())
        assert isinstance(sharpened, MetaTensor)
        assert sharpened.meta[VOXEL_SPACING_META_KEY] == pytest.approx((2.0, 2.0, 3.0))


class TestAnisotropyAwareRandGaussianSmooth:
    def test_scale_is_sqrt_hw_over_d_of_the_view_spacing(self) -> None:
        blur = _blur(sigma=(1.0, 1.0))
        blur(_anisotropic_view())
        assert blur.sigma_z[0] == pytest.approx(blur._base_sigma_z[0] * 2 / 3)
        assert blur.sigma_x[0] == blur._base_sigma_z[0]

    def test_in_plane_anisotropy_uses_the_geometric_mean(self) -> None:
        blur = _blur(sigma=(1.0, 1.0))
        blur(_spaced(_volume(8, 8, 8), (1.0, 4.0, 4.0)))
        assert blur.sigma_z[0] == pytest.approx(0.5)

    def test_isotropic_view_scale_is_one(self) -> None:
        blur = _blur(sigma=(1.0, 1.0))
        blur(_crop(size=(16, 16, 16))(_spaced(_volume(32, 32, 8), (0.5, 0.5, 2.0))))
        assert blur.sigma_z[0] == pytest.approx(blur._base_sigma_z[0])

    def test_does_not_compound_across_calls(self) -> None:
        blur = _blur(sigma=(1.0, 1.0))
        cropped = _anisotropic_view()
        blur(cropped)
        first_sigma_z = blur.sigma_z
        blur(cropped)
        assert blur.sigma_z == pytest.approx(first_sigma_z)

    def test_thicker_slices_blur_less_along_depth(self) -> None:
        crop = _crop(size=(16, 16, 8))
        blur_thin = _blur(sigma=(1.0, 1.0))
        blur_thick = _blur(sigma=(1.0, 1.0))
        blur_thin(crop(_spaced(_volume(64, 64, 8), (0.5, 0.5, 1.0))))
        blur_thick(crop(_spaced(_volume(64, 64, 8), (0.5, 0.5, 4.0))))
        assert blur_thick.sigma_z[0] < blur_thin.sigma_z[0]

    def test_fallback_without_meta_tag(self) -> None:
        # No preceding crop step -- must not raise, falls back to 1 / img.shape: scale 8 / 64.
        blur = _blur(sigma=(1.0, 1.0))
        blur(_volume(64, 64, 8))
        assert blur.sigma_z[0] == pytest.approx(blur._base_sigma_z[0] / 8)


class TestAnisotropyAwareRandGaussianSharpen:
    def test_scales_both_sigmas_by_the_view_spacing(self) -> None:
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        sharpen(_anisotropic_view())
        assert sharpen.sigma1_z[0] == pytest.approx(sharpen._base_sigma1_z[0] * 2 / 3)
        assert sharpen.sigma1_z[0] < sharpen.sigma1_x[0]
        assert sharpen.sigma2_z == pytest.approx(sharpen._base_sigma2_z * 2 / 3)
        assert sharpen.sigma2_z < sharpen.sigma2_x

    def test_scales_tuple_sigma2(self) -> None:
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=(0.3, 0.9))
        sharpen(_anisotropic_view())
        assert sharpen.sigma2_z == pytest.approx((0.3 * 2 / 3, 0.9 * 2 / 3))

    def test_isotropic_view_scale_is_one(self) -> None:
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        sharpen(_crop(size=(16, 16, 16))(_spaced(_volume(32, 32, 8), (0.5, 0.5, 2.0))))
        assert sharpen.sigma1_z[0] == pytest.approx(sharpen._base_sigma1_z[0])
        assert sharpen.sigma2_z == pytest.approx(sharpen._base_sigma2_z)

    def test_does_not_compound_across_calls(self) -> None:
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        cropped = _anisotropic_view()
        sharpen(cropped)
        first_sigma1_z = sharpen.sigma1_z
        first_sigma2_z = sharpen.sigma2_z
        sharpen(cropped)
        assert sharpen.sigma1_z == pytest.approx(first_sigma1_z)
        assert sharpen.sigma2_z == pytest.approx(first_sigma2_z)

    def test_thicker_slices_sharpen_less_along_depth(self) -> None:
        crop = _crop(size=(16, 16, 8))
        sharpen_thin = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        sharpen_thick = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        sharpen_thin(crop(_spaced(_volume(64, 64, 8), (0.5, 0.5, 1.0))))
        sharpen_thick(crop(_spaced(_volume(64, 64, 8), (0.5, 0.5, 4.0))))
        assert sharpen_thick.sigma1_z[0] < sharpen_thin.sigma1_z[0]
        assert sharpen_thick.sigma2_z < sharpen_thin.sigma2_z

    def test_fallback_without_meta_tag(self) -> None:
        # No preceding crop step -- must not raise, falls back to 1 / img.shape.
        sharpen = _sharpen(sigma1=(1.0, 1.0), sigma2=0.5)
        sharpen(_volume(64, 64, 8))
        assert sharpen.sigma1_z[0] == pytest.approx(sharpen._base_sigma1_z[0] / 8)


def _tissue_volume() -> np.ndarray:
    """(1, H, W, D) uint8: air at 0..5, a "tissue" block at [5:25, 14:34, 2:10]."""
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 6, (1, 40, 36, 12), dtype=np.uint8)
    volume[:, 5:25, 14:34, 2:10] = 200
    return volume


class TestTagForegroundBox:
    def test_tags_monai_bounding_box(self) -> None:
        volume = _tissue_volume()
        out = TagForegroundBox(threshold=10)(volume)
        assert isinstance(out, MetaTensor)
        assert out.meta[FOREGROUND_BOX_META_KEY] == ((5, 14, 2), (25, 34, 10))
        start, end = CropForeground(select_fn=lambda x: x > 10).compute_bounding_box(
            torch.from_numpy(volume)
        )
        assert out.meta[FOREGROUND_BOX_META_KEY] == (tuple(start), tuple(end))
        # Only tags: the volume is not cropped or changed.
        assert np.array_equal(out.numpy(), volume)

    def test_float_threshold(self) -> None:
        out = TagForegroundBox(threshold=199.5)(_tissue_volume())
        assert out.meta[FOREGROUND_BOX_META_KEY] == ((5, 14, 2), (25, 34, 10))

    def test_no_foreground_no_tag(self) -> None:
        out = TagForegroundBox(threshold=250)(_tissue_volume())
        assert FOREGROUND_BOX_META_KEY not in out.meta

    def test_pickles(self) -> None:
        # DataLoader workers pickle the transform under spawn/forkserver; a lambda as
        # select_fn would not.
        restored = pickle.loads(pickle.dumps(TagForegroundBox(threshold=10)))
        out = restored(_tissue_volume())
        assert out.meta[FOREGROUND_BOX_META_KEY] == ((5, 14, 2), (25, 34, 10))


class TestCropReadsForegroundBox:
    def test_crops_land_on_the_tissue(self) -> None:
        volume = _tissue_volume()
        tagged = TagForegroundBox(threshold=10)(_spaced(volume, (0.5, 0.5, 2.0)))
        # At most 0.27 of each axis (11 x 10 x 4 voxels), so every crop fits in the
        # block and, placed inside it, holds nothing but tissue.
        crop = AnisotropyTrackingRandomResizedCrop3D(
            size=(4, 4, 2), scale=(0.01, 0.02), ratio=(1.0, 1.0), seed=0
        )
        for _ in range(50):
            out = crop(tagged)
            assert out.min() >= 199
            # The spacing tag passes TagForegroundBox and reaches the crop.
            assert out.meta[VOXEL_SPACING_META_KEY][2] > out.meta[VOXEL_SPACING_META_KEY][0]
            # The box is in pre-crop coordinates, so it is not passed on.
            assert FOREGROUND_BOX_META_KEY not in out.meta

    def test_untagged_metatensor_matches_ndarray(self) -> None:
        volume = _tissue_volume()
        crop_a = AnisotropyTrackingRandomResizedCrop3D(
            size=(8, 8, 4), scale=(0.1, 0.5), seed=0
        )
        crop_b = AnisotropyTrackingRandomResizedCrop3D(
            size=(8, 8, 4), scale=(0.1, 0.5), seed=0
        )
        for _ in range(10):
            a = crop_a(volume)
            b = crop_b(MetaTensor(torch.from_numpy(volume)))
            assert torch.equal(a.as_tensor(), b.as_tensor())
