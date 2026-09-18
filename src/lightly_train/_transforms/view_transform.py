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
import torch
from albumentations import (
    BasicTransform,
    ColorJitter,
    GaussianBlur,
    Solarize,
    ToGray,
)
from lightning_utilities.core.imports import RequirementCache
from monai.data import MetaTensor
from monai.transforms import (
    Transform,
    RandRotate, NormalizeIntensity, Compose, ToNumpy,
    RandAdjustContrast, RandGaussianNoise, RandGibbsNoise, OneOf,
)

from lightly_train._configs.config import PydanticConfig
from lightly_train._transforms.monai_wrappers import (
    AnisotropyAwareRandGaussianSharpen,
    AnisotropyAwareRandGaussianSmooth,
    AnisotropyTrackingRandomResizedCrop3D,
    AlphaRandHistogramShift
)
from lightly_train._transforms.transform import (
    ColorJitterArgs,
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
    random_resized_crop: RandomResizedCropArgs  # only its .scale attribute can be None
    random_rotation: RandomRotationArgs | None
    gaussian_blur: GaussianBlurArgs | None
    normalize: NormalizeArgs
    random_flip: RandomFlipArgs | None = None
    gaussian_sharpen: RandGaussianSharpenArgs | None = None
    gibbs_noise: RandGibbsNoiseArgs | None = None
    histogram_shift: RandHistogramShiftArgs | None = None
    adjust_contrast: RandAdjustContrastArgs | None = None
    gaussian_noise: RandGaussianNoiseArgs | None = None


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

        # Normalize first, before any resampling. RandomResizedCrop3D (below) is a
        # from-scratch NumPy transform that round-trips its output back to the input
        # dtype, clipping/rounding to the integer range if the input is an integer
        # type -- so if this ran last (as upstream lightly-train's 2D pipeline does),
        # a `uint8` volume (what KneeNo returns under resample_mode="nearest") would
        # get quantized to 256 discrete levels at the very first op, before rotate/
        # blur/sharpen/etc. ever see it. NormalizeIntensity promotes to float32
        # regardless of input dtype, so running it first means the crop always sees
        # float input and never hits that integer round-trip. ToNumpy bridges
        # NormalizeIntensity's MetaTensor output back to a plain ndarray, since
        # RandomResizedCrop3D cannot consume a MetaTensor directly.
        transform += [
            NormalizeIntensity(
                subtrahend=[m * 255 for m in args.normalize.mean],
                divisor=[s * 255 for s in args.normalize.std],
                channel_wise=True,
            ),
            ToNumpy(),
        ]

        # .scale here corresponds to MethodTransformArgs.random_resize and may be None
        # .size here corresponds to MethodTransformArgs.image_size and may not be None
        if args.random_resized_crop.scale is None:
            args.random_resized_crop.scale = RandomResizeArgs(
                min_scale=1.0, max_scale=1.0
            )
        transform += [
            AnisotropyTrackingRandomResizedCrop3D(
                size=args.random_resized_crop.size,
                scale=args.scale.as_tuple(),
                interpolation="area",
                upscale_interpolation="linear",
                # Deviates from CV2 INTER_AREA slightly, but looks better in my opinion.
                # Select None for the closest 3D approximation of CV2s' INTER_AREA
            )
        ]

        if args.random_rotation:
            # MONAI expects the rotation ranges in radians, in-plane rotation only
            transform += [
                RandRotate(
                    prob=args.random_rotation.prob,
                    range_z=tuple(
                        math.radians(deg)
                        for deg in args.random_rotation.degrees_tuple()
                    ),
                    mode=args.random_rotation.interpolation,
                    padding_mode="border"
                )
            ]

        # Intensity augmentations to replace the photometric ops
        if args.histogram_shift:
            transform += [
                AlphaRandHistogramShift(
                    alpha=args.histogram_shift.alpha,
                    prob=args.histogram_shift.prob,
                    num_control_points=args.histogram_shift.num_control_points,
                )
            ]

        if args.adjust_contrast:
            transform += [
                RandAdjustContrast(
                    prob=args.adjust_contrast.prob,
                    gamma=args.adjust_contrast.gamma,
                )
            ]


        # If both blur and sharpen is enabled, select one of them. If only one of them is
        # enabled, only add this one
        if (args.gaussian_blur and args.gaussian_blur.prob > 0
                and args.gaussian_sharpen and args.gaussian_sharpen.prob > 0):
            transform += [
                OneOf([
                    AnisotropyAwareRandGaussianSmooth(
                        prob=args.gaussian_blur.prob,
                        sigma_range=args.gaussian_blur.sigma_range,
                    ),
                    AnisotropyAwareRandGaussianSharpen(
                        prob=args.gaussian_sharpen.prob,
                        sigma1=args.gaussian_sharpen.sigma1,
                        sigma2=args.gaussian_sharpen.sigma2,
                        alpha=args.gaussian_sharpen.alpha,
                    )
                ])
            ]
        elif args.gaussian_blur and args.gaussian_blur.prob > 0:
            transform += [
                AnisotropyAwareRandGaussianSmooth(
                    prob=args.gaussian_blur.prob,
                    sigma_range=args.gaussian_blur.sigma_range,
                )
            ]
        elif args.gaussian_sharpen and args.gaussian_sharpen.prob > 0:
            transform += [
                AnisotropyAwareRandGaussianSharpen(
                    prob=args.gaussian_sharpen.prob,
                    sigma1=args.gaussian_sharpen.sigma1,
                    sigma2=args.gaussian_sharpen.sigma2,
                    alpha=args.gaussian_sharpen.alpha,
                )
            ]


        # Noise
        if args.gibbs_noise:
            transform += [
                RandGibbsNoise(
                    prob=args.gibbs_noise.prob,
                    alpha=args.gibbs_noise.alpha,
                )
            ]

        if args.gaussian_noise:
            transform += [
                RandGaussianNoise(
                    prob=args.gaussian_noise.prob,
                    mean=args.gaussian_noise.mean,
                    std=args.gaussian_noise.std,
                )
            ]


        transform += [ToTensor()]
        self.transform = Compose(transform)

    def __call__(self, input: TransformInput) -> TransformOutputSingleView:
        transformed: TransformOutputSingleView = {
            "image": self.transform(input["image"])
        }
        return transformed
