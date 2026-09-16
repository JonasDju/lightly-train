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
import pydantic
import pytest
import torch
from monai.transforms import (
    NormalizeIntensity,
    RandAdjustContrast,
    RandGaussianNoise,
    RandGibbsNoise,
    RandHistogramShift,
    RandRotate,
)

from lightly_train._transforms.monai_wrappers import AnisotropyAwareRandGaussianSharpen
from lightly_train._transforms.transform import (
    GaussianBlurArgs,
    NormalizeArgs,
    RandAdjustContrastArgs,
    RandGaussianNoiseArgs,
    RandGaussianSharpenArgs,
    RandGibbsNoiseArgs,
    RandHistogramShiftArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    RandomResizedCropArgs,
    RandomRotationArgs,
)
from lightly_train._transforms.view_transform import ToTensor, ViewTransform, ViewTransformArgs
from lightly_train.types import TransformInput


def _get_random_resized_crop_args(
    scale: RandomResizeArgs | None = None,
) -> RandomResizedCropArgs:
    return RandomResizedCropArgs(
        size=(16, 16, 6),  # (H, W, D)
        scale=RandomResizeArgs(min_scale=0.2, max_scale=1.0)
        if scale is None
        else scale,
    )


def _get_random_rotation_args() -> RandomRotationArgs:
    return RandomRotationArgs(prob=1.0, degrees=10)


def _get_gaussian_blur_args() -> GaussianBlurArgs:
    return GaussianBlurArgs(prob=1.0, sigmas=(0.1, 2), blur_limit=0)


def _get_gaussian_sharpen_args() -> RandGaussianSharpenArgs:
    return RandGaussianSharpenArgs(prob=1.0, sigma1=(0.5, 1.0), sigma2=0.5)


def _get_gibbs_noise_args() -> RandGibbsNoiseArgs:
    return RandGibbsNoiseArgs(prob=1.0, alpha=(0.2, 0.8))


def _get_histogram_shift_args() -> RandHistogramShiftArgs:
    return RandHistogramShiftArgs(prob=1.0, num_control_points=10)


def _get_adjust_contrast_args() -> RandAdjustContrastArgs:
    return RandAdjustContrastArgs(prob=1.0, gamma=(0.5, 4.5))


def _get_gaussian_noise_args() -> RandGaussianNoiseArgs:
    return RandGaussianNoiseArgs(prob=1.0, mean=0.0, std=0.1)


def _get_normalize_args() -> NormalizeArgs:
    return NormalizeArgs(mean=(0.5,), std=(0.5,))


def _view_transform(
    random_resized_crop: RandomResizedCropArgs | None = None,
    random_rotation: RandomRotationArgs | None = None,
    gaussian_blur: GaussianBlurArgs | None = None,
    gaussian_sharpen: RandGaussianSharpenArgs | None = None,
    gibbs_noise: RandGibbsNoiseArgs | None = None,
    histogram_shift: RandHistogramShiftArgs | None = None,
    adjust_contrast: RandAdjustContrastArgs | None = None,
    gaussian_noise: RandGaussianNoiseArgs | None = None,
    **kwargs: object,
) -> ViewTransform:
    args = dict(
        random_resized_crop=random_resized_crop or _get_random_resized_crop_args(),
        random_flip=None,
        random_rotation=random_rotation,
        gaussian_blur=gaussian_blur,
        normalize=_get_normalize_args(),
        gaussian_sharpen=gaussian_sharpen,
        gibbs_noise=gibbs_noise,
        histogram_shift=histogram_shift,
        adjust_contrast=adjust_contrast,
        gaussian_noise=gaussian_noise,
    )
    args.update(kwargs)
    return ViewTransform(ViewTransformArgs(**args))  # type: ignore[arg-type]


def _volume(dtype: type = np.uint8, value: float | None = None) -> np.ndarray:
    # (C, H, W, D)
    if value is not None:
        return np.full((1, 20, 24, 10), value, dtype=dtype)
    return np.random.uniform(0, 255, size=(1, 20, 24, 10)).astype(dtype)


class TestViewTransform:
    @pytest.mark.parametrize("random_rotation", [_get_random_rotation_args(), None])
    @pytest.mark.parametrize("gaussian_blur", [_get_gaussian_blur_args(), None])
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_view_transform_args_combinations(
        self,
        random_rotation: RandomRotationArgs | None,
        gaussian_blur: GaussianBlurArgs | None,
        dtype: type,
    ) -> None:
        view_transform = _view_transform(
            random_rotation=random_rotation, gaussian_blur=gaussian_blur
        )
        tr_input: TransformInput = {"image": _volume(dtype)}
        tr_output = view_transform(tr_input)
        assert isinstance(tr_output, dict)
        assert set(tr_output) == {"image"}
        img = tr_output["image"]
        # (C, D, H, W), a plain tensor (no MONAI MetaTensor) for default collation.
        assert img.shape == (1, 6, 16, 16)
        assert img.dtype == torch.float32
        assert type(img) is torch.Tensor
        assert img.is_contiguous()

    def test_view_transform__unsupported_2d_args_are_ignored(self) -> None:
        view_transform = _view_transform(
            random_flip=RandomFlipArgs(horizontal_prob=0.5, vertical_prob=0.5),
        )
        assert view_transform({"image": _volume()})["image"].shape == (1, 6, 16, 16)

    def test_view_transform_args__photometric_2d_fields_removed(self) -> None:
        # channel_drop/color_jitter/random_gray_scale/solarize were RGB-only 2D
        # fields that ViewTransform never actually used (always hardcoded to None
        # upstream in DINOTransform); removed as clutter, not just disabled.
        with pytest.raises(pydantic.ValidationError):
            ViewTransformArgs(
                random_resized_crop=_get_random_resized_crop_args(),
                random_flip=None,
                random_rotation=None,
                gaussian_blur=None,
                normalize=_get_normalize_args(),
                color_jitter=None,  # type: ignore[call-arg]
            )

    @pytest.mark.parametrize("value, expected", [(255, 1.0), (0, -1.0), (127.5, 0.0)])
    def test_view_transform__normalize(self, value: float, expected: float) -> None:
        # Normalize args are for the [0, 1] range while volumes are in [0, 255].
        view_transform = _view_transform(
            random_resized_crop=_get_random_resized_crop_args(
                scale=RandomResizeArgs(min_scale=1.0, max_scale=1.0)
            )
        )
        img = view_transform({"image": _volume(np.float32, value=value)})["image"]
        torch.testing.assert_close(img, torch.full_like(img, expected))

    def test_view_transform__rotation_in_radians(self) -> None:
        view_transform = _view_transform(random_rotation=_get_random_rotation_args())
        rotations = [
            t for t in view_transform.transform.transforms if isinstance(t, RandRotate)
        ]
        assert len(rotations) == 1
        assert max(rotations[0].range_x) == pytest.approx(math.radians(10))

    def test_view_transform__rotation_is_in_plane_only(self) -> None:
        # range_y/range_z (the H-D/W-D planes) must stay at 0: rotating a plane
        # that mixes the depth axis with an in-plane axis is not a true physical
        # rotation on anisotropic volumes (it becomes shear + stretch in mm
        # space unless voxels are cubic).
        view_transform = _view_transform(random_rotation=_get_random_rotation_args())
        rotations = [
            t for t in view_transform.transform.transforms if isinstance(t, RandRotate)
        ]
        assert len(rotations) == 1
        assert max(rotations[0].range_y) == 0.0
        assert max(rotations[0].range_z) == 0.0

    def test_view_transform__reproducible_with_random_state(self) -> None:
        view_transform = _view_transform(
            random_rotation=_get_random_rotation_args(),
            gaussian_blur=_get_gaussian_blur_args(),
        )
        volume = _volume()
        view_transform.transform.set_random_state(seed=0)
        out0 = view_transform({"image": volume})["image"]
        view_transform.transform.set_random_state(seed=0)
        out1 = view_transform({"image": volume})["image"]
        torch.testing.assert_close(out0, out1)

    def test_view_transform__record_geometry_not_supported(self) -> None:
        with pytest.raises(NotImplementedError):
            ViewTransform(
                ViewTransformArgs(
                    random_resized_crop=_get_random_resized_crop_args(),
                    random_flip=None,
                    random_rotation=None,
                    gaussian_blur=None,
                    normalize=_get_normalize_args(),
                ),
                record_geometry=True,
            )

    @pytest.mark.parametrize(
        "gaussian_sharpen", [_get_gaussian_sharpen_args(), None]
    )
    @pytest.mark.parametrize("gibbs_noise", [_get_gibbs_noise_args(), None])
    @pytest.mark.parametrize(
        "histogram_shift", [_get_histogram_shift_args(), None]
    )
    @pytest.mark.parametrize(
        "adjust_contrast", [_get_adjust_contrast_args(), None]
    )
    @pytest.mark.parametrize("gaussian_noise", [_get_gaussian_noise_args(), None])
    def test_view_transform__monai_intensity_args_combinations(
        self,
        gaussian_sharpen: RandGaussianSharpenArgs | None,
        gibbs_noise: RandGibbsNoiseArgs | None,
        histogram_shift: RandHistogramShiftArgs | None,
        adjust_contrast: RandAdjustContrastArgs | None,
        gaussian_noise: RandGaussianNoiseArgs | None,
    ) -> None:
        view_transform = _view_transform(
            gaussian_sharpen=gaussian_sharpen,
            gibbs_noise=gibbs_noise,
            histogram_shift=histogram_shift,
            adjust_contrast=adjust_contrast,
            gaussian_noise=gaussian_noise,
        )
        img = view_transform({"image": _volume(np.float32)})["image"]
        assert img.shape == (1, 6, 16, 16)
        assert img.dtype == torch.float32
        assert type(img) is torch.Tensor
        assert img.is_contiguous()

    def test_view_transform__new_ops_absent_when_none(self) -> None:
        view_transform = _view_transform()
        ops = view_transform.transform.transforms
        for cls in (
            AnisotropyAwareRandGaussianSharpen,
            RandGibbsNoise,
            RandHistogramShift,
            RandAdjustContrast,
            RandGaussianNoise,
        ):
            assert not any(isinstance(op, cls) for op in ops)

    def test_view_transform__gaussian_sharpen_uses_anisotropy_aware_wrapper(
        self,
    ) -> None:
        view_transform = _view_transform(gaussian_sharpen=_get_gaussian_sharpen_args())
        sharpens = [
            t
            for t in view_transform.transform.transforms
            if isinstance(t, AnisotropyAwareRandGaussianSharpen)
        ]
        assert len(sharpens) == 1
        # Not just any RandGaussianSharpen -- must be the anisotropy-aware wrapper.
        assert type(sharpens[0]) is AnisotropyAwareRandGaussianSharpen

    def test_view_transform__gaussian_noise_mean_and_std_scaled_by_255(self) -> None:
        view_transform = _view_transform(
            gaussian_noise=RandGaussianNoiseArgs(prob=1.0, mean=0.1, std=0.2)
        )
        noises = [
            t
            for t in view_transform.transform.transforms
            if isinstance(t, RandGaussianNoise)
        ]
        assert len(noises) == 1
        assert noises[0].mean == pytest.approx(0.1 * 255)
        assert noises[0].std == pytest.approx(0.2 * 255)

    def test_view_transform__monai_ops_order(self) -> None:
        view_transform = _view_transform(
            random_rotation=_get_random_rotation_args(),
            gaussian_blur=_get_gaussian_blur_args(),
            gaussian_sharpen=_get_gaussian_sharpen_args(),
            gibbs_noise=_get_gibbs_noise_args(),
            histogram_shift=_get_histogram_shift_args(),
            adjust_contrast=_get_adjust_contrast_args(),
            gaussian_noise=_get_gaussian_noise_args(),
        )
        op_types = [type(t) for t in view_transform.transform.transforms]
        # Spatial filters (blur, sharpen) -> k-space artifact (Gibbs) -> intensity
        # remap (histogram shift, contrast) -> additive noise last -> normalize/tensor.
        expected_order = [
            RandRotate,
            AnisotropyAwareRandGaussianSharpen,  # gaussian_blur precedes it below
            RandGibbsNoise,
            RandHistogramShift,
            RandAdjustContrast,
            RandGaussianNoise,
            NormalizeIntensity,
            ToTensor,
        ]
        # gaussian_blur (AnisotropyAwareRandGaussianSmooth) sits right before sharpen;
        # check its position explicitly, then check the remaining relative order.
        from lightly_train._transforms.monai_wrappers import (
            AnisotropyAwareRandGaussianSmooth,
        )

        blur_idx = op_types.index(AnisotropyAwareRandGaussianSmooth)
        sharpen_idx = op_types.index(AnisotropyAwareRandGaussianSharpen)
        assert blur_idx < sharpen_idx
        # Remaining ops appear, in order, after sharpen.
        remaining = [t for t in op_types if t in expected_order[2:]]
        assert remaining == expected_order[2:]

    def test_view_transform__enabling_all_new_ops_changes_output(self) -> None:
        volume = _volume(np.float32)

        disabled = _view_transform()
        disabled.transform.set_random_state(seed=0)
        out_disabled = disabled({"image": volume})["image"]

        enabled = _view_transform(
            gaussian_sharpen=_get_gaussian_sharpen_args(),
            gibbs_noise=_get_gibbs_noise_args(),
            histogram_shift=_get_histogram_shift_args(),
            adjust_contrast=_get_adjust_contrast_args(),
            gaussian_noise=_get_gaussian_noise_args(),
        )
        enabled.transform.set_random_state(seed=0)
        out_enabled = enabled({"image": volume})["image"]

        assert not torch.allclose(out_disabled, out_enabled)

    def test_view_transform__reproducible_with_random_state__new_ops(self) -> None:
        view_transform = _view_transform(
            gaussian_sharpen=_get_gaussian_sharpen_args(),
            gibbs_noise=_get_gibbs_noise_args(),
            histogram_shift=_get_histogram_shift_args(),
            adjust_contrast=_get_adjust_contrast_args(),
            gaussian_noise=_get_gaussian_noise_args(),
        )
        volume = _volume()
        view_transform.transform.set_random_state(seed=0)
        out0 = view_transform({"image": volume})["image"]
        view_transform.transform.set_random_state(seed=0)
        out1 = view_transform({"image": volume})["image"]
        torch.testing.assert_close(out0, out1)


class TestRandAdjustContrastArgs:
    def test_defaults_match_monai(self) -> None:
        args = RandAdjustContrastArgs()
        assert args.prob == 0.1
        assert args.gamma == (0.5, 4.5)

    def test_accepts_cli_shaped_list(self) -> None:
        args = RandAdjustContrastArgs(gamma=[0.2, 0.3])  # type: ignore[arg-type]
        assert args.gamma == (0.2, 0.3)


class TestRandGaussianNoiseArgs:
    def test_defaults_match_monai(self) -> None:
        args = RandGaussianNoiseArgs()
        assert args.prob == 0.1
        assert args.mean == 0.0
        assert args.std == 0.1


class TestRandHistogramShiftArgs:
    def test_defaults_match_monai(self) -> None:
        args = RandHistogramShiftArgs()
        assert args.prob == 0.1
        assert args.num_control_points == 10

    def test_accepts_cli_shaped_list(self) -> None:
        args = RandHistogramShiftArgs(num_control_points=[5, 15])  # type: ignore[arg-type]
        assert args.num_control_points == (5, 15)


class TestRandGaussianSharpenArgs:
    def test_defaults_match_monai(self) -> None:
        args = RandGaussianSharpenArgs()
        assert args.prob == 0.1
        assert args.sigma1 == (0.5, 1.0)
        assert args.sigma2 == 0.5
        assert args.alpha == (10.0, 30.0)

    def test_accepts_cli_shaped_list_for_scalar_sigma2(self) -> None:
        args = RandGaussianSharpenArgs(sigma2=[0.3, 0.9])  # type: ignore[arg-type]
        assert args.sigma2 == (0.3, 0.9)

    def test_accepts_scalar_sigma2(self) -> None:
        args = RandGaussianSharpenArgs(sigma2=0.7)
        assert args.sigma2 == 0.7


class TestRandGibbsNoiseArgs:
    def test_defaults_match_monai(self) -> None:
        args = RandGibbsNoiseArgs()
        assert args.prob == 0.1
        assert args.alpha == (0.0, 1.0)

    def test_accepts_cli_shaped_list(self) -> None:
        args = RandGibbsNoiseArgs(alpha=[0.2, 0.8])  # type: ignore[arg-type]
        assert args.alpha == (0.2, 0.8)

    def test_accepts_scalar_alpha(self) -> None:
        args = RandGibbsNoiseArgs(alpha=0.5)
        assert args.alpha == 0.5


class TestRandomRotationArgs:
    @pytest.mark.parametrize(
        "degrees, expected",
        [
            (10, (-10.0, 10.0)),
            (7.5, (-7.5, 7.5)),
            ((1, 2), (1.0, 2.0)),
            ([1.0, 2.0], (1.0, 2.0)),
        ],
    )
    def test_degrees_tuple(
        self, degrees: float | tuple[float, float], expected: tuple
    ) -> None:
        args = RandomRotationArgs(prob=1.0, degrees=degrees)
        assert args.degrees_tuple() == expected
