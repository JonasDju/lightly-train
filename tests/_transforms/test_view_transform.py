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
import torch
from monai.transforms import RandRotate

from lightly_train._transforms.transform import (
    ChannelDropArgs,
    ColorJitterArgs,
    GaussianBlurArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    RandomResizedCropArgs,
    RandomRotationArgs,
    SolarizeArgs,
)
from lightly_train._transforms.view_transform import ViewTransform, ViewTransformArgs
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


def _get_normalize_args() -> NormalizeArgs:
    return NormalizeArgs(mean=(0.5,), std=(0.5,))


def _view_transform(
    random_resized_crop: RandomResizedCropArgs | None = None,
    random_rotation: RandomRotationArgs | None = None,
    gaussian_blur: GaussianBlurArgs | None = None,
    **kwargs: object,
) -> ViewTransform:
    args = dict(
        channel_drop=None,
        random_resized_crop=random_resized_crop or _get_random_resized_crop_args(),
        random_flip=None,
        random_rotation=random_rotation,
        color_jitter=None,
        random_gray_scale=None,
        gaussian_blur=gaussian_blur,
        solarize=None,
        normalize=_get_normalize_args(),
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
            channel_drop=ChannelDropArgs(num_channels_keep=1, weight_drop=(1.0,)),
            random_flip=RandomFlipArgs(horizontal_prob=0.5, vertical_prob=0.5),
            color_jitter=ColorJitterArgs(
                prob=0.8,
                strength=1.0,
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            random_gray_scale=0.2,
            solarize=SolarizeArgs(prob=0.5, threshold=0.5),
        )
        assert view_transform({"image": _volume()})["image"].shape == (1, 6, 16, 16)

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
                    channel_drop=None,
                    random_resized_crop=_get_random_resized_crop_args(),
                    random_flip=None,
                    random_rotation=None,
                    color_jitter=None,
                    random_gray_scale=None,
                    gaussian_blur=None,
                    solarize=None,
                    normalize=_get_normalize_args(),
                ),
                record_geometry=True,
            )


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
