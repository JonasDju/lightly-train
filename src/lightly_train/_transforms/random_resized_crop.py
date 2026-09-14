"""Anisotropy-aware 3D analogue of albumentations' ``RandomResizedCrop``.

Works on volumes of shape ``(C, H, W, D)``
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple, Sequence

import numpy as np
from monai.transforms import Transform
from numpy.typing import NDArray

__all__ = ["RandomResizedCrop3D", "CropParams3D"]

InterpolationMode = Literal["area", "linear", "nearest", "cubic"]


class CropParams3D(NamedTuple):
    """Fully describes a crop, so it can be replayed on other volumes.

    Fields are ordered to match the ``(C, H, W, D)`` axis layout.
    """

    h_start: int
    w_start: int
    d_start: int
    height: int
    width: int
    depth: int


# --------------------------------------------------------------------------- #
# Separable resampling weights.
#
# For every axis we build a dense ``(out_size, in_size)`` matrix whose rows sum
# to one, then contract the volume with it. All three axes are handled the same
# way, so a single implementation covers arbitrary (an)isotropic rescaling.
#
# The conventions match OpenCV / ``torch.nn.functional.interpolate`` with
# ``align_corners=False``:
#   * "area"    -> cv2.INTER_AREA   (exact, incl. fractional pixel coverage)
#   * "linear"  -> cv2.INTER_LINEAR (half-pixel centers)
#   * "nearest" -> cv2.INTER_NEAREST
#   * "cubic"   -> cv2.INTER_CUBIC  (Catmull-Rom-like, a = -0.75)
# --------------------------------------------------------------------------- #
def _area_weights(in_size: int, out_size: int) -> NDArray[np.float64]:
    scale = in_size / out_size
    out_idx = np.arange(out_size, dtype=np.float64)
    starts = out_idx * scale
    ends = (out_idx + 1) * scale
    in_idx = np.arange(in_size, dtype=np.float64)
    # Length of the overlap between output bin i and input pixel j.
    left = np.maximum(starts[:, None], in_idx[None, :])
    right = np.minimum(ends[:, None], in_idx[None, :] + 1.0)
    weights = np.clip(right - left, 0.0, None)
    return weights / weights.sum(axis=1, keepdims=True)


def _linear_weights(in_size: int, out_size: int) -> NDArray[np.float64]:
    scale = in_size / out_size
    src = (np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5
    lo = np.floor(src)
    frac = src - lo
    lo = lo.astype(np.int64)
    weights = np.zeros((out_size, in_size), dtype=np.float64)
    rows = np.arange(out_size)
    np.add.at(weights, (rows, np.clip(lo, 0, in_size - 1)), 1.0 - frac)
    np.add.at(weights, (rows, np.clip(lo + 1, 0, in_size - 1)), frac)
    return weights


def _nearest_weights(in_size: int, out_size: int) -> NDArray[np.float64]:
    scale = in_size / out_size
    src = np.floor(np.arange(out_size, dtype=np.float64) * scale).astype(np.int64)
    weights = np.zeros((out_size, in_size), dtype=np.float64)
    weights[np.arange(out_size), np.clip(src, 0, in_size - 1)] = 1.0
    return weights


def _cubic_weights(in_size: int, out_size: int, a: float = -0.75) -> NDArray[np.float64]:
    scale = in_size / out_size
    src = (np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5
    base = np.floor(src)
    frac = (src - base)[:, None]
    offsets = np.arange(-1, 3, dtype=np.float64)[None, :]
    t = np.abs(frac - offsets)
    t2, t3 = t**2, t**3
    kernel = np.where(
        t <= 1.0,
        (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0,
        np.where(t < 2.0, a * t3 - 5.0 * a * t2 + 8.0 * a * t - 4.0 * a, 0.0),
    )
    kernel /= kernel.sum(axis=1, keepdims=True)
    cols = np.clip(base[:, None].astype(np.int64) + offsets.astype(np.int64), 0, in_size - 1)
    weights = np.zeros((out_size, in_size), dtype=np.float64)
    rows = np.repeat(np.arange(out_size), 4)
    np.add.at(weights, (rows, cols.ravel()), kernel.ravel())
    return weights


_WEIGHT_FNS = {
    "area": _area_weights,
    "linear": _linear_weights,
    "nearest": _nearest_weights,
    "cubic": _cubic_weights,
}


def _resample(
    volume: NDArray[np.floating],
    out_shape: tuple[int, int, int],
    mode: InterpolationMode,
    upscale_mode: InterpolationMode | None = None,
) -> NDArray:
    """Resample the last three axes of ``volume`` to ``out_shape``.

    ``volume`` is ``(C, H, W, D)`` and ``out_shape`` is the target ``(H, W, D)``;
    axis 0 is carried along untouched.

    ``upscale_mode`` overrides ``mode`` on axes that are being enlarged.

    Note on ``"area"`` and cv2 parity: the area weights reproduce
    ``cv2.INTER_AREA`` exactly on every axis, in both directions -- they collapse
    to nearest-neighbour at integer enlarging factors and blend across the
    boundary otherwise, which is what cv2 does. The one deliberate divergence is
    that cv2 only runs true area resampling when *every* axis is reduced; if any
    axis is enlarged it drops the area path for the whole image and emulates it
    with a bilinear variant, so the *reducing* axes lose their anti-aliasing. We
    always apply exact area per axis. This matters here because the depth axis is
    often enlarged while H and W are heavily reduced, and cv2's rule would throw
    away the anti-aliasing exactly where it is needed.
    """
    out = volume
    # Shrink the most-reduced axis first: every later contraction then runs on
    # less data. Resampling is separable, so the order does not affect the result.
    # ``i`` indexes out_shape (H, W, D); the matching array axis is ``i + 1``.
    order = sorted(range(3), key=lambda i: out_shape[i] / volume.shape[i + 1])
    for i in order:
        axis = i + 1
        out_size = out_shape[i]
        in_size = out.shape[axis]
        if in_size == out_size:
            continue
        axis_mode = upscale_mode if (upscale_mode is not None and out_size > in_size) else mode
        weights = _WEIGHT_FNS[axis_mode](in_size, out_size)
        # tensordot dispatches to a single BLAS gemm; it is several times
        # faster here than transposing into a matmul.
        out = np.moveaxis(np.tensordot(weights.astype(out.dtype), out, axes=([1], [axis])), 0, axis)
    return out


class RandomResizedCrop3D(Transform):
    """Crop a random sub-volume and resize it to a fixed shape.

    Mirrors ``albumentations.RandomResizedCrop`` but for ``(C, H, W, D)``
    volumes, and — crucially — keeps the crop's aspect ratio *anchored to the
    aspect ratio of the input volume* instead of aiming for a cube. The sampled
    ratios are multiplied by the volume's current ``H/D`` and ``W/D`` ratios, so
    on a ``(C, 512, 512, 24)`` scan the crops stay flat rather than collapsing
    the depth axis.

    With ``r_hd``/``r_wd`` the sampled ratios and ``s`` the sampled scale, the
    crop sizes work out to a shape-independent fraction of each axis::

        h = H * (s * r_hd**2 / r_wd) ** (1/3)
        w = W * (s * r_wd**2 / r_hd) ** (1/3)
        d = D * (s / (r_hd * r_wd)) ** (1/3)

    Args:
        size: Output spatial shape ``(H, W, D)``, in the same axis order as the
            volumes themselves. A single int means a cube.
        scale: Range of the crop *volume* as a fraction of the input volume,
            sampled uniformly. Same semantics as albumentations' area scale.
        ratio: Range of the aspect-ratio jitter applied on top of the volume's
            own aspect ratio. Sampled log-uniformly, as in albumentations.
        interpolation: One of ``"area"``, ``"linear"``, ``"nearest"``,
            ``"cubic"``. ``"area"`` is the exact 3D equivalent of
            ``cv2.INTER_AREA`` and is the default, matching lightly-train's 2D
            pipeline. It is exact for enlarging axes too, where it reproduces
            cv2's behaviour of collapsing towards nearest-neighbour. See
            ``_resample`` for the one deliberate divergence from cv2.
        upscale_interpolation: Optional override used only on axes that are being
            *enlarged*. ``None`` (the default) applies ``interpolation``
            everywhere. Set it to
            ``"linear"`` if you would rather blend than duplicate when the crop
            is shallower than the output depth -- a common case for thin volumes,
            but a deliberate departure from the 2D pipeline.
        max_attempts: Rejection-sampling attempts before falling back to a crop
            that is shrunk isotropically until it fits.
        seed: Optional seed or ``np.random.Generator``.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = "area",
        upscale_interpolation: InterpolationMode | None = None,
        max_attempts: int = 10,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if isinstance(size, int):
            size = (size, size, size)
        size = tuple(int(s) for s in size)
        if len(size) != 3 or any(s <= 0 for s in size):
            raise ValueError(f"size must be three positive ints (H, W, D), got {size}.")
        if not 0.0 < scale[0] <= scale[1]:
            raise ValueError(f"scale must satisfy 0 < scale[0] <= scale[1], got {scale}.")
        if not 0.0 < ratio[0] <= ratio[1]:
            raise ValueError(f"ratio must satisfy 0 < ratio[0] <= ratio[1], got {ratio}.")
        if interpolation not in _WEIGHT_FNS:
            raise ValueError(f"interpolation must be one of {sorted(_WEIGHT_FNS)}, got {interpolation!r}.")
        if upscale_interpolation is not None and upscale_interpolation not in _WEIGHT_FNS:
            raise ValueError(
                f"upscale_interpolation must be None or one of {sorted(_WEIGHT_FNS)}, "
                f"got {upscale_interpolation!r}."
            )

        self.size: tuple[int, int, int] = size  # type: ignore[assignment]
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        self.log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        self.interpolation: InterpolationMode = interpolation
        self.upscale_interpolation: InterpolationMode | None = upscale_interpolation
        self.max_attempts = int(max_attempts)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Parameter sampling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sizes_from_ratios(target_volume: float, aspect_ratio_hd: float, aspect_ratio_wd: float) -> tuple[int, int, int]:
        # NOTE: h and w are derived from the *rounded* d, so that the returned
        # sizes are exactly consistent with each other.
        d = int(round((target_volume / (aspect_ratio_hd * aspect_ratio_wd)) ** (1.0 / 3.0)))
        return int(round(d * aspect_ratio_hd)), int(round(d * aspect_ratio_wd)), d

    def get_params(self, volume_shape: Sequence[int]) -> CropParams3D:
        """Sample crop parameters for a volume of shape ``(..., H, W, D)``."""
        height, width, depth = (int(s) for s in volume_shape[-3:])
        volume = float(depth) * height * width
        # Aspect ratio of the *input*, used to anchor the sampled ratios.
        current_ratio_hd = height / depth
        current_ratio_wd = width / depth

        h = w = d = 0
        for _ in range(self.max_attempts):
            target_volume = self.rng.uniform(*self.scale) * volume
            aspect_ratio_hd = math.exp(self.rng.uniform(*self.log_ratio)) * current_ratio_hd
            aspect_ratio_wd = math.exp(self.rng.uniform(*self.log_ratio)) * current_ratio_wd
            h, w, d = self._sizes_from_ratios(target_volume, aspect_ratio_hd, aspect_ratio_wd)
            if 0 < h <= height and 0 < w <= width and 0 < d <= depth:
                return self._random_position(h, w, d, height, width, depth)

        # Fallback: shrink the last candidate isotropically until it fits, which
        # preserves its aspect ratio (albumentations instead center-crops).
        shrink = min(height / max(h, 1), width / max(w, 1), depth / max(d, 1), 1.0)
        h = min(max(int(round(h * shrink)), 1), height)
        w = min(max(int(round(w * shrink)), 1), width)
        d = min(max(int(round(d * shrink)), 1), depth)
        return self._random_position(h, w, d, height, width, depth)

    def _random_position(self, h: int, w: int, d: int, height: int, width: int, depth: int) -> CropParams3D:
        return CropParams3D(
            h_start=int(self.rng.integers(0, height - h + 1)),
            w_start=int(self.rng.integers(0, width - w + 1)),
            d_start=int(self.rng.integers(0, depth - d + 1)),
            height=h,
            width=w,
            depth=d,
        )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    def apply(self, volume: NDArray, params: CropParams3D) -> NDArray:
        """Apply given crop parameters — useful to replay a crop on a label volume."""
        squeeze_channel = volume.ndim == 3
        if squeeze_channel:
            volume = volume[None]
        if volume.ndim != 4:
            raise ValueError(f"Expected a (C, H, W, D) volume, got shape {volume.shape}.")

        crop = volume[
            :,
            params.h_start : params.h_start + params.height,
            params.w_start : params.w_start + params.width,
            params.d_start : params.d_start + params.depth,
        ]

        in_dtype = crop.dtype
        work_dtype = np.float64 if in_dtype == np.float64 else np.float32
        out = _resample(
            crop.astype(work_dtype, copy=False), self.size, self.interpolation, self.upscale_interpolation
        )

        if np.issubdtype(in_dtype, np.integer):
            info = np.iinfo(in_dtype)
            out = np.clip(np.rint(out), info.min, info.max)
        elif in_dtype == np.bool_:
            out = out > 0.5
        out = out.astype(in_dtype, copy=False)
        return out[0] if squeeze_channel else out

    def __call__(self, data: NDArray) -> NDArray:
        """Crop ``volume`` of shape ``(C, H, W, D)`` and resize it to ``self.size``."""
        return self.apply(data, self.get_params(data.shape))