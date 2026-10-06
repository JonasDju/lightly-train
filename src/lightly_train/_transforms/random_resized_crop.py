"""Anisotropy-aware 3D analogue of albumentations' ``RandomResizedCrop``.

Works on volumes of shape ``(C, H, W, D)``
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, NamedTuple, Sequence

import numpy as np
from monai.transforms import Randomizable, Transform
from numpy.typing import DTypeLike, NDArray
from pydantic import AfterValidator
from scipy import sparse

__all__ = ["RandomResizedCrop3D", "CropParams3D"]

InterpolationMode = Literal["area", "linear", "nearest", "cubic"]

# Contract an axis with a sparse matrix product instead of a dense ``tensordot`` when at most this fraction of the
# weight matrix is non-zero. The weights are banded (an output voxel only reads its few nearest inputs), so a strong
# reduction, e.g. a 512-wide crop to 224, is mostly zeros: there the sparse product does the same multiply-adds
# minus the zeros and is 2-3x faster per global view. Near-identity resizes stay dense, where BLAS wins.
# 0 disables the sparse path.
_SPARSE_MAX_DENSITY = 0.25


#: ``((h0, w0, d0), (h1, w1, d1))``: a bounding box with exclusive end, as
#: ``monai.transforms.CropForeground.compute_bounding_box`` returns it.
ForegroundBox = tuple[Sequence[int], Sequence[int]]


def _start_range(crop: int, size: int, box: tuple[int, int] | None) -> tuple[int, int]:
    """Inclusive range of start indices of a ``crop``-long window on a ``size``-long
    axis.

    Without a box, anywhere in the axis. With a foreground box ``[b0, b1)``: inside the
    box if the crop fits there, otherwise anywhere it contains the box. Both stay inside
    the axis, and the second is never empty since ``b1 - b0 < crop <= size``. An empty
    box (``b1 <= b0`` after clipping) counts as none.
    """
    low, high = 0, size - crop
    if box is None:
        return low, high
    b0, b1 = max(box[0], 0), min(box[1], size)
    if b1 <= b0:
        return low, high
    if crop <= b1 - b0:
        return b0, b1 - crop
    return max(low, b1 - crop), min(high, b0)


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


def _nearest_slice_weights(in_size: int, out_size: int) -> NDArray[np.float64]:
    """Nearest-neighbour along the depth axis, mapped like KneeNo's ``KneeNoDataset._resample_depth``.

    ``np.linspace(0, in_size - 1, out_size).round()`` keeps both end slices and spreads the dropped (or doubled)
    slices evenly, whereas ``_nearest_weights`` (cv2's floor mapping) always drops the last ones on a reduction.
    """
    src = np.linspace(0, in_size - 1, out_size).round().astype(np.int64)
    weights = np.zeros((out_size, in_size), dtype=np.float64)
    weights[np.arange(out_size), src] = 1.0
    return weights


_WEIGHT_FNS = {
    "area": _area_weights,
    "linear": _linear_weights,
    "nearest": _nearest_weights,
    "cubic": _cubic_weights,
}
# The depth axis (out-of-plane) only differs in its nearest-neighbour mapping.
_OUT_OF_PLANE_WEIGHT_FNS = {**_WEIGHT_FNS, "nearest": _nearest_slice_weights}


def parse_interpolation(spec: str) -> tuple[InterpolationMode, InterpolationMode]:
    """Split an interpolation spec into its ``(in_plane, out_of_plane)`` modes.

    ``"<in-plane>+<out-of-plane>"`` sets the H/W axes and the depth axis separately, e.g. ``"linear+nearest"``
    resamples every slice bilinearly but only picks (drops or doubles) whole slices along depth. A single mode
    ``"X"`` is shorthand for ``"X+X"``.
    """
    modes = spec.split("+") if isinstance(spec, str) else []
    if len(modes) == 1:
        modes = modes * 2
    if len(modes) != 2 or any(m not in _WEIGHT_FNS for m in modes):
        raise ValueError(
            f"Invalid interpolation {spec!r}: expected one of {sorted(_WEIGHT_FNS)}, or "
            "'<in-plane>+<out-of-plane>' with two of them, e.g. 'linear+nearest'."
        )
    return modes[0], modes[1]  # type: ignore[return-value]


def _validate_interpolation(spec: str) -> str:
    parse_interpolation(spec)
    return spec


# Pydantic field type for an interpolation spec: validated, but kept as written so run configs round-trip.
ResizeInterpolation = Annotated[str, AfterValidator(_validate_interpolation)]


def _resample(
    volume: NDArray[np.floating],
    out_shape: tuple[int, int, int],
    mode: str,
    upscale_mode: str | None = None,
) -> NDArray:
    """Resample the last three axes of ``volume`` to ``out_shape``.

    ``volume`` is ``(C, H, W, D)`` and ``out_shape`` is the target ``(H, W, D)``;
    axis 0 is carried along untouched.

    ``mode`` and ``upscale_mode`` are specs as accepted by ``parse_interpolation``:
    their in-plane mode applies to H and W, their out-of-plane mode to D.
    ``upscale_mode`` overrides ``mode`` on axes that are being enlarged. As the
    resampling is separable, ``"<in-plane>+nearest"`` is exactly a slice-by-slice
    in-plane resample of the slices that the depth mapping selects. Along depth,
    ``"nearest"`` uses KneeNo's mapping (see ``_nearest_slice_weights``).

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
    modes = parse_interpolation(mode)
    upscale_modes = None if upscale_mode is None else parse_interpolation(upscale_mode)
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
        axis_modes = upscale_modes if (upscale_modes is not None and out_size > in_size) else modes
        # H and W (i = 0, 1) are in-plane, D (i = 2) is out-of-plane.
        if i == 2:
            weight_fn = _OUT_OF_PLANE_WEIGHT_FNS[axis_modes[1]]
        else:
            weight_fn = _WEIGHT_FNS[axis_modes[0]]
        weights = weight_fn(in_size, out_size).astype(out.dtype)
        if np.count_nonzero(weights) <= _SPARSE_MAX_DENSITY * weights.size:
            moved = np.moveaxis(out, axis, 0)
            contracted = sparse.csr_array(weights) @ moved.reshape(in_size, -1)
            out = np.moveaxis(contracted.reshape((out_size, *moved.shape[1:])), 0, axis)
        else:
            # tensordot dispatches to a single BLAS gemm; it is several times
            # faster here than transposing into a matmul.
            out = np.moveaxis(np.tensordot(weights, out, axes=([1], [axis])), 0, axis)
    return out


class RandomResizedCrop3D(Randomizable, Transform):
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

    The crop position is uniform, unless ``get_params`` is given a foreground box (see
    there): then the crop is placed on the tissue, with its size left untouched.

    Args:
        size: Output spatial shape ``(H, W, D)``, in the same axis order as the
            volumes themselves. A single int means a cube.
        scale: Range of the crop *volume* as a fraction of the input volume,
            sampled uniformly. Same semantics as albumentations' area scale.
        ratio: Range of the aspect-ratio jitter applied on top of the volume's
            own aspect ratio. Sampled log-uniformly, as in albumentations.
        interpolation: One of ``"area"``, ``"linear"``, ``"nearest"``,
            ``"cubic"``, or ``"<in-plane>+<out-of-plane>"`` to set the H/W axes
            and the depth axis separately (see ``parse_interpolation``), e.g.
            ``"area+nearest"``. ``"area"`` is the exact 3D equivalent of
            ``cv2.INTER_AREA`` and is the default, matching lightly-train's 2D
            pipeline. It is exact for enlarging axes too, where it reproduces
            cv2's behaviour of collapsing towards nearest-neighbour. See
            ``_resample`` for the one deliberate divergence from cv2.
        upscale_interpolation: Optional override used only on axes that are being
            *enlarged*, in the same format as ``interpolation``. ``None`` (the
            default) applies ``interpolation`` everywhere. Set it to
            ``"linear"`` if you would rather blend than duplicate when the crop
            is shallower than the output depth -- a common case for thin volumes,
            but a deliberate departure from the 2D pipeline.
        max_attempts: Rejection-sampling attempts before falling back to a crop
            that is shrunk isotropically until it fits.
        seed: Optional seed or ``np.random.RandomState``. Randomness is drawn from
            MONAI's ``self.R`` so that ``set_random_state`` (e.g. called per
            DataLoader worker) reseeds this transform.
        output_dtype: dtype of the output. ``None`` (the default) round-trips to the
            input dtype, rounding and clipping to the integer range for integer
            inputs -- what replaying a crop onto a label volume needs. Anything else
            casts the resampled result directly, e.g. ``np.float32`` keeps the full
            interpolation precision of a ``uint8`` input.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = "area",
        upscale_interpolation: str | None = None,
        max_attempts: int = 10,
        seed: int | np.random.RandomState | None = None,
        output_dtype: DTypeLike | None = None,
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
        _validate_interpolation(interpolation)
        if upscale_interpolation is not None:
            _validate_interpolation(upscale_interpolation)

        self.size: tuple[int, int, int] = size  # type: ignore[assignment]
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        self.log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        self.interpolation = interpolation
        self.upscale_interpolation = upscale_interpolation
        self.max_attempts = int(max_attempts)
        self.output_dtype = None if output_dtype is None else np.dtype(output_dtype)
        if isinstance(seed, np.random.RandomState):
            self.set_random_state(state=seed)
        else:
            self.set_random_state(seed=seed)

    # ------------------------------------------------------------------ #
    # Parameter sampling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sizes_from_ratios(target_volume: float, aspect_ratio_hd: float, aspect_ratio_wd: float) -> tuple[int, int, int]:
        # NOTE: h and w are derived from the *rounded* d, so that the returned
        # sizes are exactly consistent with each other.
        d = int(round((target_volume / (aspect_ratio_hd * aspect_ratio_wd)) ** (1.0 / 3.0)))
        return int(round(d * aspect_ratio_hd)), int(round(d * aspect_ratio_wd)), d

    def get_params(
        self,
        volume_shape: Sequence[int],
        foreground_box: ForegroundBox | None = None,
    ) -> CropParams3D:
        """Sample crop parameters for a volume of shape ``(..., H, W, D)``.

        ``foreground_box`` is the tissue's ``((h0, w0, d0), (h1, w1, d1))`` bounding box
        (exclusive end, as ``monai.transforms.CropForeground.compute_bounding_box``
        returns it). It only moves the crop, never resizes it: the sizes are sampled
        exactly as without it, so the crop is not stretched any differently. See
        ``_start_range`` for the placement rule.
        """
        height, width, depth = (int(s) for s in volume_shape[-3:])
        volume = float(depth) * height * width
        # Aspect ratio of the *input*, used to anchor the sampled ratios.
        current_ratio_hd = height / depth
        current_ratio_wd = width / depth

        h = w = d = 0
        for _ in range(self.max_attempts):
            target_volume = self.R.uniform(*self.scale) * volume
            aspect_ratio_hd = math.exp(self.R.uniform(*self.log_ratio)) * current_ratio_hd
            aspect_ratio_wd = math.exp(self.R.uniform(*self.log_ratio)) * current_ratio_wd
            h, w, d = self._sizes_from_ratios(target_volume, aspect_ratio_hd, aspect_ratio_wd)
            if 0 < h <= height and 0 < w <= width and 0 < d <= depth:
                return self._random_position(
                    h, w, d, height, width, depth, foreground_box
                )

        # Fallback: shrink the last candidate isotropically until it fits, which
        # preserves its aspect ratio (albumentations instead center-crops).
        shrink = min(height / max(h, 1), width / max(w, 1), depth / max(d, 1), 1.0)
        h = min(max(int(round(h * shrink)), 1), height)
        w = min(max(int(round(w * shrink)), 1), width)
        d = min(max(int(round(d * shrink)), 1), depth)
        return self._random_position(h, w, d, height, width, depth, foreground_box)

    def _random_position(
        self,
        h: int,
        w: int,
        d: int,
        height: int,
        width: int,
        depth: int,
        foreground_box: ForegroundBox | None = None,
    ) -> CropParams3D:
        starts = []
        for axis, (crop, size) in enumerate(((h, height), (w, width), (d, depth))):
            box = None
            if foreground_box is not None:
                box = (int(foreground_box[0][axis]), int(foreground_box[1][axis]))
            low, high = _start_range(crop, size, box)
            # One draw per axis either way, so without a box (or with one spanning the
            # whole volume) the crops are exactly those of plain uniform placement.
            starts.append(int(self.R.randint(low, high + 1)))
        return CropParams3D(
            h_start=starts[0],
            w_start=starts[1],
            d_start=starts[2],
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

        if self.output_dtype is not None:
            out = out.astype(self.output_dtype, copy=False)
        else:
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