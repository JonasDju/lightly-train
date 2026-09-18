#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Anisotropy-aware wrappers around MONAI's ``RandGaussianSmooth``/``RandGaussianSharpen``.

Knee-MRI volumes are highly anisotropic: the depth axis (``D``, slice count)
typically has far fewer voxels than the in-plane axes (``H``, ``W``). MONAI's
``RandGaussianSmooth`` applies the same sigma to every axis, which is too strong
for ``D``. No real physical voxel-spacing metadata is available anywhere in this
data pipeline (KneeNo's source is plain JPEG slices, not DICOM/NIfTI), so we use
each volume's own voxel-count shape ratio as a proxy for its anisotropy -- the
same proxy ``RandomResizedCrop3D`` and ``MaskingGenerator`` already use to anchor
their own sampled ratios.

Rotation is deliberately *not* made anisotropy-aware this way: a rotation in
voxel-index space is only a true physical rotation when voxels are cubic.
Scaling a rotation range down by the voxel-count ratio still leaves a
resampling that is, in physical (mm) space, a combination of rotation, shear,
and stretching along any axis pair with unequal spacing -- not the rotation
it's meant to be. ``view_transform.py`` instead uses a plain MONAI ``RandRotate``
restricted to the H-W in-plane rotation only, via ``range_z`` (``range_x``/
``range_y`` left at their default 0). Verified empirically, not just assumed
from the axis names: for a ``(C, H, W, D)`` volume, rotating with ``range_z``
alone leaves every voxel's value constant along the ``D`` axis (a true in-plane
H-W rotation), while ``range_x``/``range_y`` both mix ``D`` into the rotation.

``RandGaussianSharpen`` is made anisotropy-aware the same way as
``RandGaussianSmooth`` (``AnisotropyAwareRandGaussianSharpen`` below), since it too
takes a per-axis sigma (in fact two: a pre-blur ``sigma1`` and a post-blur
``sigma2``). ``RandHistogramShift`` is wrapped too, but not for anisotropy --
``AlphaRandHistogramShift`` below blends its output with the original image by a
fixed ``alpha`` because the unwrapped default is too strong (an intensity-strength
knob, unrelated to axis anisotropy). The other new MONAI intensity/artifact
augmentations in this pipeline (``RandAdjustContrast``, ``RandGaussianNoise``,
``RandGibbsNoise``) are voxelwise or already shape-relative and are used
unwrapped -- see the "no anisotropy wrapper" note on ``RandGibbsNoiseArgs`` in
``transform.py`` for the one case (Gibbs ringing) where this was a deliberate
choice rather than an oversight.

Why a ``MetaTensor`` tag instead of reading ``img.shape`` directly: in the real
pipeline (``view_transform.py``), ``RandomResizedCrop3D`` runs immediately before
``RandGaussianSmooth`` and always resizes to a *fixed configured output size* --
so by the time ``RandGaussianSmooth`` would see it, every volume's shape is
identical regardless of its real anisotropy. ``AnisotropyTrackingRandomResizedCrop3D``
tags its output with the *pre-crop* shape as `MetaTensor` metadata; MONAI's own
meta-propagation machinery (``convert_to_dst_type``) carries that tag, untouched,
through ``RandGaussianSmooth``. ``ToTensor`` (the last step in
``view_transform.py``) already strips ``MetaTensor`` down to a plain tensor, so
the tag never reaches the model.

This voxel-count ratio is a heuristic approximation, not true physical spacing.
When a real per-case anisotropy signal becomes available (e.g. loaded from a
separate dataset-metadata file), only what populates ``ORIG_SHAPE_META_KEY``
needs to change -- the consuming code below stays the same.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from monai.data import MetaTensor
from monai.transforms import RandGaussianSharpen, RandGaussianSmooth, RandHistogramShift
from numpy.typing import NDArray

from lightly_train._transforms.random_resized_crop import RandomResizedCrop3D

__all__ = [
    "ORIG_SHAPE_META_KEY",
    "AnisotropyTrackingRandomResizedCrop3D",
    "AnisotropyAwareRandGaussianSmooth",
    "AnisotropyAwareRandGaussianSharpen",
    "AlphaRandHistogramShift"
]

# (H, W, D) of the volume before RandomResizedCrop3D's crop+resize.
ORIG_SHAPE_META_KEY = "lt_orig_spatial_shape"


def _get_anisotropy_shape(img: Any) -> tuple[int, int, int]:
    """Read the (H, W, D) shape to derive anisotropy ratios from.

    Prefers the pre-crop shape tagged by ``AnisotropyTrackingRandomResizedCrop3D``
    (reached through MONAI's meta propagation); falls back to the input's own
    shape so these transforms stay usable standalone (e.g. in tests) without a
    preceding tagging step.
    """
    meta = getattr(img, "meta", None)
    if meta is not None and ORIG_SHAPE_META_KEY in meta:
        h, w, d = meta[ORIG_SHAPE_META_KEY]
        return int(h), int(w), int(d)
    h, w, d = img.shape[-3:]
    return int(h), int(w), int(d)


class AlphaRandHistogramShift(RandHistogramShift):
    """ The default RandHistogramShift but with an alpha value to control its strength

    The default augmentation is way too strong.
    """

    def __init__(self, alpha: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def __call__(self, img, randomize: bool = True):
        transformed = super().__call__(img, randomize=randomize)
        return self.alpha * transformed + (1 - self.alpha) * img


class AnisotropyTrackingRandomResizedCrop3D(RandomResizedCrop3D):
    """``RandomResizedCrop3D`` that tags its output with the pre-crop shape.

    Kept separate from ``RandomResizedCrop3D`` itself (which is general-purpose,
    also used to replay a crop onto a label volume, and separately tested) so
    this anisotropy-tracking behavior stays scoped to this file's pipeline.
    """

    def __call__(self, data: NDArray) -> MetaTensor:
        orig_shape = tuple(int(s) for s in data.shape[-3:])
        out = super().__call__(data)
        return MetaTensor(out, meta={ORIG_SHAPE_META_KEY: orig_shape})


class AnisotropyAwareRandGaussianSmooth(RandGaussianSmooth):
    """``RandGaussianSmooth`` with a ``sigma_z`` scaled down by the volume's own
    ``D / sqrt(H * W)`` ratio, so the effective blur extent along the coarse
    depth axis roughly matches the in-plane blur extent.
    """

    def __init__(self, sigma_range: tuple[float, float], *args: Any, **kwargs: Any) -> None:
        super().__init__(sigma_x=sigma_range, sigma_y=sigma_range, sigma_z=sigma_range, *args, **kwargs)
        # The configured (isotropic) bounds, kept around so each call rescales
        # from the original value rather than compounding on the previous call's
        # already-scaled sigma_z.
        self._base_sigma_z = self.sigma_z

    def __call__(self, img: Any, randomize: bool = True) -> Any:
        if randomize:
            h, w, d = _get_anisotropy_shape(img)
            scale_z = d / math.sqrt(h * w)
            self.sigma_z = tuple(s * scale_z for s in self._base_sigma_z)
        return super().__call__(img, randomize=randomize)


class AnisotropyAwareRandGaussianSharpen(RandGaussianSharpen):
    """``RandGaussianSharpen`` with its two z-axis sigmas (``sigma1_z``, the
    pre-blur sigma, and ``sigma2_z``, the post-blur sigma) scaled down by the same
    ``D / sqrt(H * W)`` ratio as ``AnisotropyAwareRandGaussianSmooth``.
    """

    def __init__(self,
                 sigma1: tuple[float, float],
                 sigma2: float | tuple[float, float],
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(
            sigma1_x=sigma1,
            sigma1_y=sigma1,
            sigma1_z=sigma1,
            sigma2_x=sigma2,
            sigma2_y=sigma2,
            sigma2_z=sigma2,
            *args, **kwargs
        )
        # The configured (isotropic) bounds, kept around so each call rescales
        # from the original value rather than compounding on the previous call's
        # already-scaled sigma.
        self._base_sigma1_z = self.sigma1_z
        self._base_sigma2_z = self.sigma2_z

    def __call__(self, img: Any, randomize: bool = True) -> Any:
        if randomize:
            h, w, d = _get_anisotropy_shape(img)
            scale_z = d / math.sqrt(h * w)
            self.sigma1_z = tuple(s * scale_z for s in self._base_sigma1_z)
            # sigma2_z may be a plain float (MONAI then samples it against sigma1_z's
            # sampled value at call time) or a tuple, unlike sigma1_z which is always
            # a tuple -- scale whichever form it is.
            if isinstance(self._base_sigma2_z, Iterable):
                self.sigma2_z = tuple(s * scale_z for s in self._base_sigma2_z)
            else:
                self.sigma2_z = self._base_sigma2_z * scale_z
        return super().__call__(img, randomize=randomize)
