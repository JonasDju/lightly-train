#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""The transform speed-ups must not change what the transforms compute.

Each fast path is compared with the code it replaced, run on the same input with the same random state:

* ``_resample``'s sparse contraction vs. the dense ``tensordot`` (``_SPARSE_MAX_DENSITY = 0``, which is exactly
  the previous code),
* ``AlphaRandHistogramShift.interp`` (``np.interp``) vs. MONAI's ``RandHistogramShift.interp`` (``searchsorted``),
* ``AnisotropyAwareRandGaussianSmooth`` / ``AnisotropyAwareRandGaussianSharpen`` (scipy ``correlate1d``) vs. their
  previous ``__call__``, which handed the filtering to MONAI's ``RandGaussianSmooth`` / ``GaussianSmooth`` and
  ``RandGaussianSharpen`` / ``GaussianSharpen`` (``torch.conv3d``).

Bit-identical where the arithmetic is the same: the random draws (and so every random parameter), the paths that
do not filter (transform skipped, unity kernels, the dense resampling path) and nearest-neighbour resampling
(one weight of exactly 1 per output voxel). Otherwise the fast paths sum in a different order or in float64
before rounding, so they are compared up to float rounding: ``|new - old| <= ULPS * eps * max|old|``, a few
units in the last place of the data's magnitude. Measured over 200 random shapes / sigmas: at most 3.5 for cubic
resampling and the Gaussian, 0.5 for the histogram shift. The sharpen, ``blurred + a * (blurred - filtered)``,
amplifies the blurs' rounding by up to ``1 + 2a`` (``a`` in [5, 10] by default), so its scale is
``(1 + 2a) * max|input|`` instead (measured: at most 1.1).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch
from monai.data import MetaTensor
from monai.transforms import RandGaussianSharpen, RandGaussianSmooth, RandHistogramShift
from pytest_mock import MockerFixture

from lightly_train._commands import train_helpers
from lightly_train._data.mi_dataset import reseed_randomizables
from lightly_train._transforms import monai_wrappers
from lightly_train._transforms import random_resized_crop as rrc
from lightly_train._transforms.monai_wrappers import (
    ORIG_SHAPE_META_KEY,
    AlphaRandHistogramShift,
    AnisotropyAwareRandGaussianSharpen,
    AnisotropyAwareRandGaussianSmooth,
)
from lightly_train._transforms.random_resized_crop import RandomResizedCrop3D

ULPS = 8


def _numpy(x: Any) -> np.ndarray:
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def assert_rounding_equal(
    new: Any, old: Any, ulps: float = ULPS, scale: float | None = None
) -> None:
    """``|new - old| <= ulps * eps * scale``; ``scale`` defaults to ``max|old|``."""
    new, old = _numpy(new), _numpy(old)
    assert new.dtype == old.dtype
    assert new.shape == old.shape
    scale = float(np.abs(old).max()) if scale is None else scale
    atol = ulps * np.finfo(old.dtype).eps * scale
    np.testing.assert_allclose(new, old, rtol=0, atol=atol)


def assert_bit_equal(new: Any, old: Any) -> None:
    new, old = _numpy(new), _numpy(old)
    assert new.dtype == old.dtype
    np.testing.assert_array_equal(new, old)


def _volume(
    shape: tuple[int, ...], seed: int = 0, dtype: Any = np.float32
) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 255, size=shape).astype(dtype)


# --------------------------------------------------------------------------- _resample / RandomResizedCrop3D


@pytest.fixture
def dense(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Runs a callable with the sparse path disabled, i.e. the previous, all-dense ``_resample``."""

    def run(fn: Any, *args: Any, **kwargs: Any) -> Any:
        with monkeypatch.context() as m:
            m.setattr(rrc, "_SPARSE_MAX_DENSITY", 0)
            return fn(*args, **kwargs)

    return run


# (C, H, W, D) in, (H, W, D) out: strong reductions (the global / local crops), enlargements, mixed, unchanged axes.
RESAMPLE_CASES = [
    ((1, 140, 120, 24), (56, 48, 16)),
    ((1, 60, 70, 10), (24, 28, 16)),
    ((2, 33, 47, 9), (64, 20, 9)),
    ((1, 512, 384, 24), (224, 224, 16)),
]


@pytest.mark.parametrize(
    "mode", ["area", "linear", "cubic", "linear+nearest", "area+nearest"]
)
@pytest.mark.parametrize("in_shape, out_shape", RESAMPLE_CASES)
def test_resample__sparse_equals_dense(
    dense: Any, mode: Any, in_shape: tuple[int, ...], out_shape: tuple[int, int, int]
) -> None:
    volume = _volume(in_shape)
    assert_rounding_equal(
        rrc._resample(volume, out_shape, mode),
        dense(rrc._resample, volume, out_shape, mode),
    )


@pytest.mark.parametrize("in_shape, out_shape", RESAMPLE_CASES)
def test_resample__nearest_bit_equal(
    dense: Any, in_shape: tuple[int, ...], out_shape: tuple[int, int, int]
) -> None:
    volume = _volume(in_shape)
    assert_bit_equal(
        rrc._resample(volume, out_shape, "nearest"),
        dense(rrc._resample, volume, out_shape, "nearest"),
    )


def test_resample__float64(dense: Any) -> None:
    volume = _volume((1, 90, 80, 12), dtype=np.float64)
    new = rrc._resample(volume, (40, 30, 16), "area", "linear")
    assert new.dtype == np.float64
    assert_rounding_equal(
        new, dense(rrc._resample, volume, (40, 30, 16), "area", "linear")
    )


def test_resample__paths(dense: Any, mocker: MockerFixture) -> None:
    """Strong reductions take the sparse path; near-identity resizes stay on the dense one, bit-identically."""
    spy = mocker.spy(rrc.sparse, "csr_array")
    # H 100 -> 20 (area: 5-6 of 100 weights per row non-zero) is sparse, W unchanged, D 4 -> 6 (linear: 2 of 4
    # non-zero) dense.
    rrc._resample(_volume((1, 100, 8, 4)), (20, 8, 6), "area", "linear")
    assert spy.call_count == 1

    spy.reset_mock()
    small = _volume((1, 4, 4, 4))  # every axis 4 -> 5..7: half of the weights non-zero
    near_identity = rrc._resample(small, (5, 6, 7), "area", "linear")
    assert spy.call_count == 0
    assert_bit_equal(
        near_identity, dense(rrc._resample, small, (5, 6, 7), "area", "linear")
    )


@pytest.mark.parametrize(
    "dtype, output_dtype",
    [(np.uint8, np.float32), (np.float32, None), (np.uint8, None)],
)
def test_random_resized_crop__equals_dense(
    dense: Any, dtype: Any, output_dtype: Any
) -> None:
    volume = _volume((1, 200, 180, 24), dtype=dtype)

    def crop() -> tuple[np.ndarray, Any]:
        transform = RandomResizedCrop3D(
            size=(56, 48, 16),
            scale=(0.05, 1.0),
            interpolation="area",
            upscale_interpolation="linear",
            output_dtype=output_dtype,
        )
        transform.set_random_state(3)
        out = transform(volume)
        return out, transform.R.get_state()[1]

    new, new_state = crop()
    old, old_state = dense(crop)
    assert_bit_equal(new_state, old_state)  # same random draws
    if np.issubdtype(new.dtype, np.integer):
        # Rounded to integers: a value within rounding of .5 could round either way; allow that, nothing more.
        assert np.abs(new.astype(np.int64) - old.astype(np.int64)).max() <= 1
        assert np.mean(new != old) < 1e-3
    else:
        assert_rounding_equal(new, old)


# --------------------------------------------------------------------------- AlphaRandHistogramShift


def _hist_shift(
    x: Any, seed: int, monai_interp: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, AlphaRandHistogramShift]:
    transform = AlphaRandHistogramShift(
        alpha=0.25, prob=1.0, num_control_points=(4, 12)
    )
    transform.set_random_state(seed)
    with monkeypatch.context() as m:
        if monai_interp:
            m.setattr(AlphaRandHistogramShift, "interp", RandHistogramShift.interp)
        return transform(x), transform


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hist_shift__equals_monai(
    seed: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = torch.from_numpy(_volume((1, 40, 30, 12), seed=seed)).to(dtype)
    new, t_new = _hist_shift(x, seed, monai_interp=False, monkeypatch=monkeypatch)
    old, t_old = _hist_shift(x, seed, monai_interp=True, monkeypatch=monkeypatch)
    assert_bit_equal(t_new.floating_control_points, t_old.floating_control_points)
    assert_rounding_equal(new, old)


def test_hist_shift__interp_equals_monai() -> None:
    """The map itself, without the alpha blend that shrinks its contribution: incl. the clamping outside xp and
    the control points themselves."""
    transform = AlphaRandHistogramShift(alpha=0.25, prob=1.0)
    xp = torch.tensor([10.0, 50.0, 60.0, 200.0])
    fp = torch.tensor([10.0, 20.0, 55.0, 200.0])
    x = torch.cat([torch.linspace(0.0, 255.0, 10_001), xp])
    assert_rounding_equal(
        transform.interp(x, xp, fp), RandHistogramShift.interp(transform, x, xp, fp)
    )
    assert_bit_equal(transform.interp(xp, xp, fp), fp)


def test_hist_shift__fallback_bit_equal() -> None:
    # Not a CPU float32/float64 tensor: MONAI's own implementation, untouched.
    transform = AlphaRandHistogramShift(alpha=0.25, prob=1.0)
    xp, fp = torch.tensor([0.0, 100.0, 255.0]), torch.tensor([0.0, 60.0, 255.0])
    x = torch.linspace(0, 255, 1000)
    for x_, xp_, fp_ in [
        (x.half(), xp.half(), fp.half()),
        (x.numpy(), xp.numpy(), fp.numpy()),
    ]:
        assert_bit_equal(
            transform.interp(x_, xp_, fp_),
            RandHistogramShift.interp(transform, x_, xp_, fp_),
        )


# --------------------------------------------------------------------------- AnisotropyAwareRandGaussianSmooth


class _OldSmooth(AnisotropyAwareRandGaussianSmooth):
    """The previous ``AnisotropyAwareRandGaussianSmooth.__call__``: MONAI does the filtering."""

    def __call__(self, img: Any, randomize: bool = True) -> Any:
        if randomize:
            h, w, d = monai_wrappers._get_anisotropy_shape(img)
            scale_z = d / math.sqrt(h * w)
            self.sigma_z = tuple(s * scale_z for s in self._base_sigma_z)
        return RandGaussianSmooth.__call__(self, img, randomize=randomize)


def _smooths(
    seed: int, prob: float = 1.0, sigma_range: tuple[float, float] = (0.4, 2.0)
) -> tuple[AnisotropyAwareRandGaussianSmooth, _OldSmooth]:
    new = AnisotropyAwareRandGaussianSmooth(prob=prob, sigma_range=sigma_range)
    old = _OldSmooth(prob=prob, sigma_range=sigma_range)
    new.set_random_state(seed)
    old.set_random_state(seed)
    return new, old


def _tagged(
    shape: tuple[int, ...], orig_shape: tuple[int, int, int], seed: int = 0
) -> MetaTensor:
    return MetaTensor(
        torch.from_numpy(_volume(shape, seed=seed)),
        meta={ORIG_SHAPE_META_KEY: orig_shape},
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "shape, orig_shape",
    [
        ((1, 56, 48, 16), (512, 512, 24)),
        ((1, 24, 24, 8), (384, 384, 30)),
        ((2, 20, 30, 12), (20, 30, 12)),
    ],
)
def test_smooth__equals_monai(
    seed: int, shape: tuple[int, ...], orig_shape: tuple[int, int, int]
) -> None:
    img = _tagged(shape, orig_shape, seed=seed)
    new_t, old_t = _smooths(seed)
    new, old = new_t(img), old_t(img)
    assert (new_t.x, new_t.y, new_t.z) == (
        old_t.x,
        old_t.y,
        old_t.z,
    )  # same draws, bit for bit
    assert_bit_equal(new_t.R.get_state()[1], old_t.R.get_state()[1])
    assert type(new) is type(old) is MetaTensor
    assert new.meta[ORIG_SHAPE_META_KEY] == old.meta[ORIG_SHAPE_META_KEY] == orig_shape
    assert_rounding_equal(new, old)


def test_smooth__numpy_input() -> None:
    img = _volume((1, 30, 20, 10))
    new_t, old_t = _smooths(0)
    new, old = new_t(img), old_t(img)
    assert type(new) is type(old)
    assert_rounding_equal(new, old)


def test_smooth__not_applied_bit_equal() -> None:
    img = _tagged((1, 30, 20, 10), (512, 512, 24))
    new_t, old_t = _smooths(0, prob=0.0)
    assert_bit_equal(new_t(img), old_t(img))
    assert_bit_equal(new_t(img), img)


def test_smooth__unity_kernels_bit_equal() -> None:
    # sigmas this small give the kernel [1.0] on every axis, which both skip: the volume comes back unfiltered.
    img = _tagged((1, 30, 20, 10), (30, 20, 10))
    new_t, old_t = _smooths(0, sigma_range=(1e-6, 2e-6))
    assert_bit_equal(new_t(img), old_t(img))
    assert_bit_equal(new_t(img), img)


def test_smooth__randomize_false() -> None:
    """randomize=False filters with the previous draw (as MONAI does) -- and returns a result."""
    new_t, old_t = _smooths(1)
    first = _tagged((1, 40, 40, 12), (512, 512, 24), seed=1)
    new_t(first)
    old_t(first)
    second = _tagged((1, 40, 40, 12), (128, 128, 48), seed=2)
    new, old = new_t(second, randomize=False), old_t(second, randomize=False)
    assert new is not None
    assert (new_t.x, new_t.y, new_t.z) == (old_t.x, old_t.y, old_t.z)
    assert_rounding_equal(new, old)


class _OldSharpen(AnisotropyAwareRandGaussianSharpen):
    """The previous ``AnisotropyAwareRandGaussianSharpen.__call__``: MONAI does the filtering."""

    def __call__(self, img: Any, randomize: bool = True) -> Any:
        if randomize:
            h, w, d = monai_wrappers._get_anisotropy_shape(img)
            scale_z = d / math.sqrt(h * w)
            self.sigma1_z = tuple(s * scale_z for s in self._base_sigma1_z)
            if isinstance(self._base_sigma2_z, tuple):
                self.sigma2_z = tuple(s * scale_z for s in self._base_sigma2_z)
            else:
                self.sigma2_z = self._base_sigma2_z * scale_z
        return RandGaussianSharpen.__call__(self, img, randomize=randomize)


def _sharpens(
    seed: int,
    prob: float = 1.0,
    sigma1: tuple[float, float] = (0.5, 2.0),
    sigma2: float | tuple[float, float] = (0.3, 1.0),
) -> tuple[AnisotropyAwareRandGaussianSharpen, _OldSharpen]:
    kwargs: dict[str, Any] = dict(
        prob=prob, sigma1=sigma1, sigma2=sigma2, alpha=(5.0, 10.0)
    )
    new, old = AnisotropyAwareRandGaussianSharpen(**kwargs), _OldSharpen(**kwargs)
    new.set_random_state(seed)
    old.set_random_state(seed)
    return new, old


def _sharpen_draws(t: RandGaussianSharpen) -> tuple[Any, ...]:
    return (t.x1, t.y1, t.z1, t.x2, t.y2, t.z2, t.a)


def assert_sharpen_rounding_equal(new: Any, old: Any, img: Any, a: float) -> None:
    assert_rounding_equal(
        new, old, scale=(1 + 2 * a) * float(np.abs(_numpy(img)).max())
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "sigma2", [(0.3, 1.0), 0.3]
)  # a float is sampled between it and the drawn sigma1
@pytest.mark.parametrize(
    "shape, orig_shape",
    [
        ((1, 56, 48, 16), (512, 512, 24)),
        ((1, 24, 24, 8), (384, 384, 30)),
        ((2, 20, 30, 12), (20, 30, 12)),
    ],
)
def test_sharpen__equals_monai(
    seed: int, sigma2: Any, shape: tuple[int, ...], orig_shape: tuple[int, int, int]
) -> None:
    img = _tagged(shape, orig_shape, seed=seed)
    new_t, old_t = _sharpens(seed, sigma2=sigma2)
    new, old = new_t(img), old_t(img)
    assert _sharpen_draws(new_t) == _sharpen_draws(old_t)  # same draws, bit for bit
    assert_bit_equal(new_t.R.get_state()[1], old_t.R.get_state()[1])
    assert type(new) is type(old) is MetaTensor
    assert new.meta[ORIG_SHAPE_META_KEY] == old.meta[ORIG_SHAPE_META_KEY] == orig_shape
    assert_sharpen_rounding_equal(new, old, img, new_t.a)


def test_sharpen__numpy_input() -> None:
    img = _volume((1, 30, 20, 10))
    new_t, old_t = _sharpens(0)
    new, old = new_t(img), old_t(img)
    assert type(new) is type(old)
    assert_sharpen_rounding_equal(new, old, img, new_t.a)


def test_sharpen__not_applied_bit_equal() -> None:
    img = _tagged((1, 30, 20, 10), (512, 512, 24))
    new_t, old_t = _sharpens(0, prob=0.0)
    assert_bit_equal(new_t(img), old_t(img))
    assert_bit_equal(new_t(img), img)


def test_sharpen__unity_kernels_bit_equal() -> None:
    # Unity kernels everywhere: blurred = filtered = img, so both return img + a * 0 = img exactly.
    img = _tagged((1, 30, 20, 10), (30, 20, 10))
    new_t, old_t = _sharpens(0, sigma1=(1e-6, 2e-6), sigma2=(1e-6, 2e-6))
    assert_bit_equal(new_t(img), old_t(img))
    assert_bit_equal(new_t(img), img)


def test_sharpen__randomize_false() -> None:
    new_t, old_t = _sharpens(1)
    first = _tagged((1, 40, 40, 12), (512, 512, 24), seed=1)
    new_t(first)
    old_t(first)
    second = _tagged((1, 40, 40, 12), (128, 128, 48), seed=2)
    new, old = new_t(second, randomize=False), old_t(second, randomize=False)
    assert new is not None
    assert _sharpen_draws(new_t) == _sharpen_draws(old_t)
    assert_sharpen_rounding_equal(new, old, second, new_t.a)


def test_sharpen__elementwise_step_bit_equal() -> None:
    """Given the same blurs, the sharpening step itself is GaussianSharpen's expression bit for bit (float32,
    alpha rounded to float32 like torch's tensor * python-float)."""
    rng = np.random.default_rng(0)
    blurred, filtered = (
        rng.uniform(0, 255, (1, 20, 20, 8)).astype(np.float32) for _ in range(2)
    )
    a = 7.123456789
    expected = torch.from_numpy(blurred) + a * (
        torch.from_numpy(blurred) - torch.from_numpy(filtered)
    )
    assert_bit_equal(blurred + np.float32(a) * (blurred - filtered), expected)


# --------------------------------------------------------------------------- the whole DINOv2 transform


def test_dinov2_transform__equals_previous(monkeypatch: pytest.MonkeyPatch) -> None:
    """All fast paths at once, through the DINOv2 views as training builds them (the shipped config's
    augmentations plus the sharpen, smaller sizes): every view equal up to float rounding.

    More ulps than one transform: the views chain several ops (histogram shift, gamma contrast, Gaussian blur or
    sharpen, Gibbs noise, normalization), each rescales by its own min / max, and the sharpen amplifies its
    rounding by up to 1 + 2a = 21.
    """
    args = train_helpers.get_transform_args(
        "dinov2",
        {
            "image_size": (48, 40, 8),
            "random_resize": {"min_scale": 0.32, "max_scale": 1.0},
            "gaussian_blur": {"prob": 1.0, "sigma_range": (0.4, 0.6)},
            "global_view_1": {
                "gaussian_blur": {"prob": 0.1, "sigma_range": (0.4, 0.6)},
                # Blur and sharpen both enabled -> OneOf.
                "gaussian_sharpen": {"prob": 0.5},
                "gibbs_noise": {"prob": 0.2, "alpha": (0.5, 0.7)},
            },
            "local_view": {
                "num_views": 4,
                "view_size": (24, 20, 4),
                "random_resize": {"min_scale": 0.05, "max_scale": 0.32},
                "gaussian_blur": {"prob": 0.0, "sigma_range": (0.4, 0.6)},
                "gaussian_sharpen": {"prob": 0.5},
            },
        },
    )
    transform = train_helpers.get_transform("dinov2", args)
    volume = _volume((1, 160, 140, 24), dtype=np.uint8)

    def views() -> list[np.ndarray]:
        reseed_randomizables(transform, np.random.RandomState(0))
        return [
            _numpy(view["image"])
            for _ in range(3)
            for view in transform({"image": volume})
        ]

    new = views()
    with monkeypatch.context() as m:
        m.setattr(rrc, "_SPARSE_MAX_DENSITY", 0)
        m.setattr(AlphaRandHistogramShift, "interp", RandHistogramShift.interp)
        m.setattr(AnisotropyAwareRandGaussianSmooth, "__call__", _OldSmooth.__call__)
        m.setattr(AnisotropyAwareRandGaussianSharpen, "__call__", _OldSharpen.__call__)
        old = views()
    assert len(new) == len(old) == 3 * 6
    for n, o in zip(new, old):
        # Which views were sharpened is up to OneOf, so every view gets the worst-case sharpen factor (1 + 2 * 10).
        assert_rounding_equal(n, o, ulps=4 * ULPS * 21)
