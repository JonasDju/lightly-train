#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

from pydantic import Field

from lightly_train._configs.config import PydanticConfig
from lightly_train._transforms.transform import (
    GaussianBlurArgs,
    MethodTransform,
    MethodTransformArgs,
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
from lightly_train._transforms.view_transform import (
    ViewTransform,
    ViewTransformArgs,
)
from lightly_train.types import (
    ImageSizeTuple,
    TransformInput,
    TransformOutput,
)


class DINORandomResizeArgs(RandomResizeArgs):
    min_scale: float = 0.14


class DINOLocalViewRandomResizeArgs(RandomResizeArgs):
    min_scale: float = 0.05
    max_scale: float = 0.14


class DINOGaussianBlurArgs(GaussianBlurArgs):
    prob: float = 1.0
    sigma_range: tuple[float, float] = Field(default=(0.1, 2), strict=False)


class DINOGlobalView1GaussianBlurArgs(DINOGaussianBlurArgs):
    prob: float = 0.1


class DINOLocalViewGaussianBlurArgs(DINOGaussianBlurArgs):
    prob: float = 0.5


class DINOGlobalView1TransformArgs(PydanticConfig):
    gaussian_blur: DINOGlobalView1GaussianBlurArgs | None = Field(
        default_factory=DINOGlobalView1GaussianBlurArgs
    )
    gaussian_sharpen: RandGaussianSharpenArgs | None = None
    gibbs_noise: RandGibbsNoiseArgs | None = Field(
        default_factory=RandGibbsNoiseArgs
    )


class DINOLocalViewTransformArgs(PydanticConfig):
    num_views: int = 6
    view_size: ImageSizeTuple = (96, 96, 8)
    random_resize: DINOLocalViewRandomResizeArgs | None = Field(
        default_factory=DINOLocalViewRandomResizeArgs
    )
    gaussian_blur: DINOLocalViewGaussianBlurArgs | None = Field(
        default_factory=DINOLocalViewGaussianBlurArgs
    )


class DINOTransformArgs(MethodTransformArgs):
    image_size: ImageSizeTuple = (224, 224, 16)
    num_channels: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)

    # Resizing & rotation
    random_resize: DINORandomResizeArgs | None = Field(         # overwritten by DINOv2ViTTransformArgs
        default_factory=DINORandomResizeArgs
    )
    random_rotation: RandomRotationArgs | None = None

    # Replacement for photometric ops
    histogram_shift: RandHistogramShiftArgs | None = Field(
        default_factory=RandHistogramShiftArgs
    )
    adjust_contrast: RandAdjustContrastArgs | None = Field(
        default_factory=RandAdjustContrastArgs
    )

    # Smoothing / Sharpening
    gaussian_blur: DINOGaussianBlurArgs | None = Field(
        default_factory=DINOGaussianBlurArgs
    )
    gaussian_sharpen: RandGaussianSharpenArgs | None = None

    # Noise
    gibbs_noise: RandGibbsNoiseArgs | None = None
    gaussian_noise: RandGaussianNoiseArgs | None = None

    # Special settings for the second global view and the local views
    global_view_1: DINOGlobalView1TransformArgs = Field(
        default_factory=DINOGlobalView1TransformArgs
    )
    local_view: DINOLocalViewTransformArgs | None = Field(      # overwritten by DINOv2ViTTransformArgs
        default_factory=DINOLocalViewTransformArgs
    )

    # Ignore: Only relevant for DINOv3
    record_geometry: bool = False


class DINOTransform(MethodTransform):
    """

    equivalent to the lightly.transforms.dino_transform.py:DINOTransform class
    """

    def __init__(self, transform_args: DINOTransformArgs):
        super().__init__(transform_args=transform_args)
        # Default from https://github.com/lightly-ai/lightly/blob/fac3dcb56745d8e5edcc59307866060cf7530bfa/lightly/transforms/dino_transform.py#L115

        global_transform_0 = ViewTransform(
            ViewTransformArgs(
                random_resized_crop=RandomResizedCropArgs(
                    size=transform_args.image_size,
                    scale=transform_args.random_resize,
                ),
                random_rotation=transform_args.random_rotation,
                gaussian_blur=transform_args.gaussian_blur,
                normalize=transform_args.normalize,
                gaussian_sharpen=transform_args.gaussian_sharpen,
                gibbs_noise=transform_args.gibbs_noise,
                histogram_shift=transform_args.histogram_shift,
                adjust_contrast=transform_args.adjust_contrast,
                gaussian_noise=transform_args.gaussian_noise,
            ),
            record_geometry=transform_args.record_geometry,
        )

        global_transform_1 = ViewTransform(
            ViewTransformArgs(
                random_resized_crop=RandomResizedCropArgs(
                    size=transform_args.image_size,
                    scale=transform_args.random_resize,
                ),
                random_rotation=transform_args.random_rotation,
                gaussian_blur=transform_args.global_view_1.gaussian_blur,
                normalize=transform_args.normalize,
                gaussian_sharpen=transform_args.global_view_1.gaussian_sharpen,
                gibbs_noise=transform_args.global_view_1.gibbs_noise,
                histogram_shift=transform_args.histogram_shift,
                adjust_contrast=transform_args.adjust_contrast,
                gaussian_noise=transform_args.gaussian_noise,
            ),
            record_geometry=transform_args.record_geometry,
        )

        transforms = [global_transform_0, global_transform_1]

        # Only add local transforms if local_view is provided
        if transform_args.local_view is not None:
            local_transform = ViewTransform(
                ViewTransformArgs(
                    random_resized_crop=RandomResizedCropArgs(
                        size=transform_args.local_view.view_size,
                        scale=transform_args.local_view.random_resize,
                    ),
                    random_rotation=transform_args.random_rotation,
                    gaussian_blur=transform_args.local_view.gaussian_blur,
                    normalize=transform_args.normalize,
                    gaussian_sharpen=transform_args.gaussian_sharpen,
                    gibbs_noise=transform_args.gibbs_noise,
                    histogram_shift=transform_args.histogram_shift,
                    adjust_contrast=transform_args.adjust_contrast,
                    gaussian_noise=transform_args.gaussian_noise,
                ),
                record_geometry=transform_args.record_geometry,
            )
            local_transforms = [local_transform] * transform_args.local_view.num_views
            transforms.extend(local_transforms)

        self.transforms = transforms

    def __call__(self, input: TransformInput) -> TransformOutput:
        return [transform(input) for transform in self.transforms]

    @staticmethod
    def transform_args_cls() -> type[DINOTransformArgs]:
        return DINOTransformArgs
