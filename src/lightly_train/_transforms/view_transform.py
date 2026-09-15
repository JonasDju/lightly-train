#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Any, cast

import cv2
import numpy as np
import torch
from albumentations import (
    BasicTransform,
    ColorJitter,
    GaussianBlur,
    HorizontalFlip,
    Rotate,
    Solarize,
    ToGray,
    VerticalFlip,
)
from albumentations.pytorch.transforms import ToTensorV2
from lightning_utilities.core.imports import RequirementCache
from monai.data import MetaTensor
from monai.transforms import (
    Transform,
    RandRotate, RandGaussianSmooth, NormalizeIntensity, Compose
)

from lightly_train._configs.config import PydanticConfig
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.normalize import NormalizeDtypeAware as Normalize
from lightly_train._transforms.random_resized_crop import RandomResizedCrop3D
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    ColorJitterArgs,
    GaussianBlurArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    RandomResizedCropArgs,
    RandomRotationArgs,
    SolarizeArgs
)
from lightly_train.types import TransformInput, TransformOutputSingleView

ALBUMENTATIONS_VERSION_2XX = RequirementCache("albumentations>=2.0.0")
ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_22 = RequirementCache("albumentations>=1.4.22")


class ToTensor(Transform):

    def __call__(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        # MONAI transforms (e.g. RandGaussianSmooth, RandRotate) may already return a
        # MetaTensor. Convert to a plain tensor so that the default collate function
        # does not carry MONAI metadata around.
        tensor = torch.as_tensor(data)
        if isinstance(tensor, MetaTensor):
            tensor = tensor.as_tensor()
        # Input shape is (C, H, W, D), but our 3D implementation of DINOv2 expects (C, D, H, W)
        return tensor.permute(0, 3, 1, 2).contiguous()


class ViewTransformArgs(PydanticConfig):
    channel_drop: ChannelDropArgs | None
    random_resized_crop: RandomResizedCropArgs  # only its .scale attribute can be None
    random_flip: RandomFlipArgs | None
    random_rotation: RandomRotationArgs | None
    color_jitter: ColorJitterArgs | None
    random_gray_scale: float | None
    gaussian_blur: GaussianBlurArgs | None
    solarize: SolarizeArgs | None
    normalize: NormalizeArgs


def _get_RandomResizedCrop(args: RandomResizedCropArgs) -> Transform:
    # A lot of though went into the choice of interpolation method here.
    # See details in https://github.com/lightly-ai/lightly-train-old/pull/284
    assert args.scale is not None
    return RandomResizedCrop3D(
        size=(args.size[0], args.size[1], args.size[2]),
        scale=args.scale.as_tuple(),
        interpolation="area",
        upscale_interpolation="linear",     # Deviates from CV2 INTER_AREA slightly, but looks better in my opinion.
                                            # Select None for the closest 3D approximation of CV2s' INTER_AREA
    )


def _get_Solarize(args: SolarizeArgs) -> Solarize:
    if ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_22:
        return Solarize(
            threshold_range=(args.threshold, args.threshold),
            p=args.prob,
        )
    return Solarize(
        # Old albumentations versions require the threshold to be in the range [0, 255]
        # for uint8 images. New versions automatically scale the threshold from [0, 1.0]
        # depending on the image type.
        threshold=args.threshold * 255,
        p=args.prob,
    )


def build_photometric_ops(
    color_jitter: ColorJitterArgs | None,
    random_gray_scale: float | None,
    gaussian_blur: GaussianBlurArgs | None,
    solarize: SolarizeArgs | None,
) -> list[BasicTransform]:
    """Builds the standard DINO photometric augmentation ops in fixed order.

    Shared by ``ViewTransform`` and the dinov31 PaKA clean/local renders so the
    two stay in sync by construction.
    """
    ops: list[BasicTransform] = []
    if color_jitter:
        ops.append(
            ColorJitter(
                brightness=color_jitter.strength * color_jitter.brightness,
                contrast=color_jitter.strength * color_jitter.contrast,
                saturation=color_jitter.strength * color_jitter.saturation,
                hue=color_jitter.strength * color_jitter.hue,
                p=color_jitter.prob,
            )
        )
    if random_gray_scale:
        ops.append(ToGray(p=random_gray_scale))
    if gaussian_blur:
        ops.append(
            GaussianBlur(
                # Setting blur_limit=0 is necessary for older versions of albumentations.
                # See details in https://linear.app/lightly/issue/LIG-5871/look-into-albumentations-gaussian-blur-difference
                blur_limit=gaussian_blur.blur_limit,
                sigma_limit=gaussian_blur.sigmas,
                p=gaussian_blur.prob,
            )
        )
    if solarize:
        ops.append(_get_Solarize(solarize))
    return ops


class ViewTransform:
    def __init__(
        self,
        args: ViewTransformArgs,
        record_geometry: bool = False,
    ):
        if record_geometry:
            raise NotImplementedError(
                "record_geometry=True is not supported by the 3D view transform."
            )
        transform: list[Transform] = []

        # .scale here corresponds to MethodTransformArgs.random_resize and may be None
        # .size here corresponds to MethodTransformArgs.image_size and may not be None
        if args.random_resized_crop.scale is None:
            args.random_resized_crop.scale = RandomResizeArgs(
                min_scale=1.0, max_scale=1.0
            )
        transform += [_get_RandomResizedCrop(args.random_resized_crop)]

        # Disable flipping for now, as the MRI volumes should always have the same orientation
        # if args.random_flip:
        #     transform += [
        #         HorizontalFlip(p=args.random_flip.horizontal_prob),
        #         VerticalFlip(p=args.random_flip.vertical_prob),
        #     ]

        if args.random_rotation:
            # MONAI expects the rotation ranges in radians.
            degrees_x, degrees_y, degrees_z = args.random_rotation.degrees_tuple()
            transform += [
                RandRotate(
                    range_x=math.radians(degrees_x),
                    range_y=math.radians(degrees_y),
                    range_z=math.radians(degrees_z),
                    prob=args.random_rotation.prob,
                    mode="bilinear",
                    padding_mode="border"
                )
            ]


        # Gaussian blur
        if args.gaussian_blur:
            transform += [
                RandGaussianSmooth(
                    sigma_x=args.gaussian_blur.sigmas,
                    sigma_y=args.gaussian_blur.sigmas,
                    sigma_z=args.gaussian_blur.sigmas,
                    prob=args.gaussian_blur.prob
                )
            ]

        transform += [
            NormalizeIntensity(
                subtrahend=[m * 255 for m in args.normalize.mean],
                divisor=[s * 255 for s in args.normalize.std],
                channel_wise=True,
            )
        ]
        transform += [ToTensor()]

        self.transform = Compose(transform)

    def __call__(self, input: TransformInput) -> TransformOutputSingleView:
        transformed: TransformOutputSingleView = {
            "image": self.transform(input["image"])
        }
        return transformed
