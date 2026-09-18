#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Literal,
    Type,
    TypeVar,
)

import cv2
import pydantic
from lightly.transforms.utils import IMAGENET_NORMALIZE
from pydantic import Field, field_validator, model_validator

from lightly_train._configs.config import PydanticConfig
from lightly_train._configs.validate import no_auto
from lightly_train.types import ImageSizeTuple, TransformInput, TransformOutput

logger = logging.getLogger(__name__)


class ChannelDropArgs(PydanticConfig):
    num_channels_keep: int
    weight_drop: tuple[float, ...] = Field(strict=False)


class ResizeArgs(PydanticConfig):
    height: int | Literal["auto"]
    width: int | Literal["auto"]

    def resolve_auto(self, height: int, width: int) -> None:
        if self.height == "auto":
            self.height = height
        if self.width == "auto":
            self.width = width


class RandomResizeArgs(PydanticConfig):
    min_scale: float = 0.08
    max_scale: float = 1.0

    def as_tuple(self) -> tuple[float, float]:
        return self.min_scale, self.max_scale


class RandomResizedCropArgs(PydanticConfig):
    # don't allow None for .size since it comes from MethodTransformArgs.image_size
    # however .scale comes from MethodTransformArgs.random_resize which may be None
    size: tuple[int, int, int]
    scale: RandomResizeArgs | None


class RandomFlipArgs(PydanticConfig):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class ActivationPolicyArgs(PydanticConfig):
    step_start: int | Literal["auto"] = 0
    step_stop: int | Literal["auto"] | None = None

    @field_validator("step_start")
    @classmethod
    def validate_step_start(cls, v: int | str) -> int | str:
        if isinstance(v, int) and v < 0:
            raise ValueError("step_start must be >= 0.")
        return v

    @field_validator("step_stop")
    @classmethod
    def validate_step_stop(cls, v: int | str | None) -> int | str | None:
        if isinstance(v, int) and v <= 0:
            raise ValueError("step_stop must be > 0.")
        return v

    @model_validator(mode="after")
    def validate_step_window(self) -> ActivationPolicyArgs:
        if self.step_start == "auto" or self.step_stop == "auto":
            return self
        if self.step_stop is not None and self.step_start >= self.step_stop:
            raise ValueError("activation policy requires step_start < step_stop.")
        return self

    def is_active(self, step: int) -> bool:
        if self.step_start == "auto":
            return False
        if step < self.step_start:
            return False
        if self.step_stop is None or self.step_stop == "auto":
            return True
        return step < self.step_stop


class RandomIoUCropArgs(ActivationPolicyArgs):
    min_scale: float
    max_scale: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    sampler_options: Sequence[float] | None
    crop_trials: int
    iou_trials: int
    prob: float = Field(ge=0.0, le=1.0)


class RandomPhotometricDistortArgs(ActivationPolicyArgs):
    brightness: tuple[float, float] = Field(strict=False)
    contrast: tuple[float, float] = Field(strict=False)
    saturation: tuple[float, float] = Field(strict=False)
    hue: tuple[float, float] = Field(strict=False)
    prob: float = Field(ge=0.0, le=1.0)


class RandomRotate90Args(PydanticConfig):
    prob: float = Field(ge=0.0, le=1.0)


class RandomRotationArgs(PydanticConfig):
    prob: float = Field(ge=0.0, le=1.0)
    degrees: float | tuple[float, float]
    interpolation: str = "bilinear"

    # Required because of: https://github.com/pydantic/pydantic/issues/10571
    @field_validator("degrees", mode="before")
    @classmethod
    def validate_degrees(cls, v: Any) -> Any:
        if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
            return tuple(v)
        return v

    def degrees_tuple(self) -> tuple[float, float]:
        if isinstance(self.degrees, (int, float)):
            return float(-self.degrees), float(self.degrees)
        assert len(self.degrees) == 2
        return tuple(float(d) for d in self.degrees)  # type: ignore[return-value]


class RandomZoomOutArgs(ActivationPolicyArgs):
    prob: float = Field(ge=0.0, le=1.0)
    fill: float
    side_range: tuple[float, float] = Field(strict=False)


class ColorJitterArgs(PydanticConfig):
    prob: float = Field(ge=0.0, le=1.0)  # Probability to apply ColorJitter
    strength: float  # Multiplier for the parameters below
    brightness: float
    contrast: float
    saturation: float
    hue: float


class GaussianBlurArgs(PydanticConfig):
    prob: float = Field(ge=0.0, le=1.0)
    sigma_range: tuple[float, float]


class SolarizeArgs(PydanticConfig):
    prob: float = Field(ge=0.0, le=1.0)
    threshold: float


# --- MONAI intensity/artifact augmentation args (3D volume pipeline) ---
#
# These mirror MONAI's own `Rand*` transform defaults (see each transform's docstring
# in `monai.transforms`) so that enabling an augmentation without specifying every
# parameter still reproduces MONAI's recommended behavior. Unlike the 2D
# albumentations-based args above, none of these are enabled by default anywhere
# (every `MethodTransformArgs`/`ViewTransformArgs` field for them defaults to `None`).


class RandAdjustContrastArgs(PydanticConfig):
    prob: float = Field(default=0.8, ge=0.0, le=1.0)
    gamma: tuple[float, float] = Field(default=(0.8, 1.2), strict=False)


class RandGaussianNoiseArgs(PydanticConfig):
    prob: float = Field(default=0.1, ge=0.0, le=1.0)
    mean: float = 0.0
    std: float = 0.075


class RandHistogramShiftArgs(PydanticConfig):
    alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    prob: float = Field(default=0.8, ge=0.0, le=1.0)
    num_control_points: int | tuple[int, int] = 10

    # Required because of: https://github.com/pydantic/pydantic/issues/10571
    @pydantic.field_validator("num_control_points", mode="before")
    @classmethod
    def cast_list_to_tuple(cls, value: int | Sequence[int]) -> int | tuple[int, int]:
        if isinstance(value, int):
            return value
        elif (
            isinstance(value, Sequence)
            and (len(value) == 2)
            and all(isinstance(v, int) for v in value)
        ):
            return tuple(value)  # type: ignore[return-value]
        else:
            raise ValueError("num_control_points must be an int or a tuple of ints")


class RandGaussianSharpenArgs(PydanticConfig):
    prob: float = Field(default=0.1, ge=0.0, le=1.0)
    sigma1: tuple[float, float] = Field(default=(0.5, 1.0), strict=False)
    sigma2: float | tuple[float, float] = 0.5
    alpha: tuple[float, float] = Field(default=(5.0, 10.0), strict=False)

    # Required because of: https://github.com/pydantic/pydantic/issues/10571
    @pydantic.field_validator("sigma2", mode="before")
    @classmethod
    def cast_list_to_tuple(cls, value: float | Sequence[float]) -> float | tuple[float, float]:
        if isinstance(value, (int, float)):
            return value
        elif (
            isinstance(value, Sequence)
            and (len(value) == 2)
            and all(isinstance(v, (int, float)) for v in value)
        ):
            return tuple(float(v) for v in value)  # type: ignore[return-value]
        else:
            raise ValueError("sigma2 must be a float or a tuple of floats")


class RandGibbsNoiseArgs(PydanticConfig):
    # Note: no anisotropy-aware wrapper. MONAI's k-space mask is a sphere of radius
    # (1 - alpha) * max(shape) * sqrt(2) / 2 in voxel-index space; on a strongly
    # anisotropic view (e.g. 224x224x16) that sphere always fully contains the short
    # depth axis, so alpha only ever truncates in-plane k-space. That matches the
    # physically expected Gibbs artifact for 2D multi-slice MRI acquisitions (ringing
    # from in-plane readout/phase-encode truncation, not across slices), so no
    # rescaling is applied here.
    prob: float = Field(default=0.2, ge=0.0, le=1.0)
    alpha: float | tuple[float, float] = (0.5, 0.75)

    # Required because of: https://github.com/pydantic/pydantic/issues/10571
    @pydantic.field_validator("alpha", mode="before")
    @classmethod
    def cast_list_to_tuple(cls, value: float | Sequence[float]) -> float | tuple[float, float]:
        if isinstance(value, (int, float)):
            return value
        elif (
            isinstance(value, Sequence)
            and (len(value) == 2)
            and all(isinstance(v, (int, float)) for v in value)
        ):
            return tuple(float(v) for v in value)  # type: ignore[return-value]
        else:
            raise ValueError("alpha must be a float or a tuple of floats")


class NormalizeArgs(PydanticConfig):
    # Strict is set to False because OmegaConf does not support parsing tuples from the
    # CLI. Setting strict to False allows Pydantic to convert lists to tuples.
    mean: tuple[float, ...] = Field(
        default=(
            0.5,
        ),
        strict=False,
    )
    std: tuple[float, ...] = Field(
        default=(
            0.5,
        ),
        strict=False,
    )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, config: dict[str, list[float]]) -> NormalizeArgs:
        return cls(mean=tuple(config["mean"]), std=tuple(config["std"]))


class ScaleJitterArgs(PydanticConfig):
    sizes: Sequence[tuple[int, int]] | None
    min_scale: float | None
    max_scale: float | None
    num_scales: int | None
    prob: float = Field(ge=0.0, le=1.0)
    divisible_by: int | None | Literal["auto"]

    # ScaleJitter does not have `step_start`, only `step_stop`.
    step_stop: int | Literal["auto"] | None = None

    @field_validator("step_stop")
    @classmethod
    def validate_step_stop(cls, v: int | str | None) -> int | str | None:
        if isinstance(v, int) and v <= 0:
            raise ValueError("step_stop must be > 0.")
        return v

    @property
    def scale_range(self) -> tuple[float, float] | None:
        if self.min_scale is not None and self.max_scale is not None:
            return self.min_scale, self.max_scale
        return None

    def is_active(self, step: int) -> bool:
        if self.step_stop is None or self.step_stop == "auto":
            return True
        return step < self.step_stop


class MixUpArgs(ActivationPolicyArgs):
    prob: float = Field(ge=0.0, le=1.0)


class CopyBlendArgs(ActivationPolicyArgs):
    prob: float = Field(ge=0.0, le=1.0)
    area_threshold: int = Field(ge=0)
    num_objects: int = Field(gt=0)
    expand_ratios: tuple[float, float] = Field(strict=False)


class MosaicArgs(ActivationPolicyArgs):
    prob: float = Field(ge=0.0, le=1.0)

    output_size: int = Field(gt=0)
    max_size: int | None = Field(gt=0)
    rotation_range: float
    translation_range: tuple[float, float] = Field(strict=False)
    scaling_range: tuple[float, float] = Field(strict=False)
    fill_value: int | float
    max_cached_images: int = Field(gt=0)
    random_pop: bool

    @model_validator(mode="after")
    def validate_ranges(self) -> MosaicArgs:
        if (
            self.scaling_range[0] <= 0.0
            or self.scaling_range[1] < self.scaling_range[0]
        ):
            raise ValueError(
                "mosaic scaling_range must be positive with scaling_range[0] <= scaling_range[1]."
            )
        if any(v < 0.0 for v in self.translation_range):
            raise ValueError("mosaic translation_range values must be non-negative.")
        return self


class SmallestMaxSizeArgs(PydanticConfig):
    # Maximum size of the smallest side of the image.
    max_size: int | list[int] | Literal["auto"]
    prob: float = Field(ge=0.0, le=1.0)

    def resolve_auto(self, height: int, width: int) -> None:
        if self.max_size == "auto":
            self.max_size = min(height, width)


class RandomCropArgs(PydanticConfig):
    height: int | Literal["auto"]
    width: int | Literal["auto"]
    pad_position: str
    pad_if_needed: bool  # Pad if crop size exceeds image size.
    fill: tuple[float, ...] | float  # Padding value for images.
    prob: float = Field(ge=0.0, le=1.0)  # Probability to apply RandomCrop.

    def resolve_auto(self, height: int, width: int) -> None:
        if self.height == "auto":
            self.height = height
        if self.width == "auto":
            self.width = width


class MethodTransformArgs(PydanticConfig):
    image_size: ImageSizeTuple
    # Defaulted to None (unlike the fields below) so that a subclass that has no use
    # for these RGB-only 2D transforms (e.g. DINOTransformArgs, for 3D single-channel
    # MRI volumes) can simply not re-declare them instead of wiring up dead defaults.
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"]
    random_resize: RandomResizeArgs | None
    random_flip: RandomFlipArgs | None = None
    random_rotation: RandomRotationArgs | None
    color_jitter: ColorJitterArgs | None = None
    random_gray_scale: float | None = None
    normalize: NormalizeArgs
    gaussian_blur: GaussianBlurArgs | None
    solarize: SolarizeArgs | None = None

    def resolve_auto(self) -> None:
        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                self.num_channels = len(self.normalize.mean)

    def resolve_incompatible(self) -> None:
        # Adjust normalization mean and std to match num_channels.
        if len(self.normalize.mean) != no_auto(self.num_channels):
            logger.debug(
                "Adjusting mean of normalize transform to match num_channels. "
                f"num_channels is {self.num_channels} but "
                f"normalize.mean has length {len(self.normalize.mean)}."
            )
            # Repeat the values until they match num_channels.
            self.normalize.mean = tuple(
                self.normalize.mean[i % len(self.normalize.mean)]
                for i in range(no_auto(self.num_channels))
            )
        if len(self.normalize.std) != no_auto(self.num_channels):
            logger.debug(
                "Adjusting std of normalize transform to match num_channels. "
                f"num_channels is {self.num_channels} but "
                f"normalize.std has length {len(self.normalize.std)}."
            )
            # Repeat the values until they match num_channels.
            self.normalize.std = tuple(
                self.normalize.std[i % len(self.normalize.std)]
                for i in range(no_auto(self.num_channels))
            )

        # Disable transforms if necessary.
        if self.color_jitter is not None and no_auto(self.num_channels) != 3:
            logger.debug(
                "Disabling color jitter transform as it only supports 3-channel "
                f"images but num_channels is {self.num_channels}."
            )
            self.color_jitter = None
        if self.random_gray_scale is not None and no_auto(self.num_channels) != 3:
            logger.debug(
                "Disabling random gray scale transform as it only supports 3-channel "
                f"images but num_channels is {self.num_channels}."
            )
            self.random_gray_scale = None
        if self.solarize is not None and no_auto(self.num_channels) != 3:
            logger.debug(
                "Disabling solarize transform as it only supports 3-channel "
                f"images but num_channels is {self.num_channels}."
            )
            self.solarize = None


_T = TypeVar("_T", covariant=True)


class MethodTransform:
    transform_args: MethodTransformArgs

    def __init__(self, transform_args: MethodTransformArgs):
        self.transform_args = transform_args

    def __call__(self, input: TransformInput) -> TransformOutput:
        raise NotImplementedError

    @staticmethod
    def transform_args_cls() -> Type[MethodTransformArgs]:
        raise NotImplementedError
