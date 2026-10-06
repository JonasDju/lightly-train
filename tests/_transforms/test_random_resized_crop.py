#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lightly_train._transforms.random_resized_crop import (
    _OUT_OF_PLANE_WEIGHT_FNS,
    _WEIGHT_FNS,
    CropParams3D,
    RandomResizedCrop3D,
    _nearest_slice_weights,
    _resample,
    _start_range,
    parse_interpolation,
)


class TestRandomResizedCrop3D:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"size": (8, 8)},
            {"size": (8, 0, 8)},
            {"size": 8, "scale": (0.0, 1.0)},
            {"size": 8, "scale": (0.9, 0.1)},
            {"size": 8, "ratio": (0.0, 1.0)},
            {"size": 8, "interpolation": "bicubic"},
            {"size": 8, "upscale_interpolation": "bicubic"},
            {"size": 8, "interpolation": "linear+bicubic"},
            {"size": 8, "upscale_interpolation": "area+nearest+linear"},
        ],
    )
    def test_init__invalid_args(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            RandomResizedCrop3D(**kwargs)

    def test_init__int_size(self) -> None:
        assert RandomResizedCrop3D(size=8).size == (8, 8, 8)

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64, np.bool_])
    @pytest.mark.parametrize(
        "interpolation", ["area", "linear", "nearest", "cubic", "linear+nearest"]
    )
    def test_call__shape_and_dtype(self, dtype: type, interpolation: Any) -> None:
        rng = np.random.default_rng(0)
        volume = rng.uniform(0, 255, size=(1, 20, 24, 10)).astype(dtype)
        transform = RandomResizedCrop3D(
            size=(8, 12, 16), interpolation=interpolation, seed=0
        )
        out = transform(volume)
        assert out.shape == (1, 8, 12, 16)
        assert out.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_call__output_dtype(self, dtype: type) -> None:
        # No rounding back to the input dtype: a uint8 input gives exactly what the same
        # values as float32 give.
        volume = (
            np.random.default_rng(0).uniform(0, 255, size=(1, 20, 24, 10)).astype(dtype)
        )
        transform = RandomResizedCrop3D(
            size=(8, 12, 16), seed=0, output_dtype=np.float32
        )
        out = transform(volume)
        assert out.dtype == np.float32
        transform.set_random_state(seed=0)
        expected = transform(volume.astype(np.float32))
        np.testing.assert_array_equal(out, expected)
        if dtype == np.uint8:
            assert not np.array_equal(out, np.rint(out))

    def test_call__multi_channel(self) -> None:
        volume = np.random.rand(3, 20, 24, 10).astype(np.float32)
        out = RandomResizedCrop3D(size=(8, 8, 4), seed=0)(volume)
        assert out.shape == (3, 8, 8, 4)

    def test_call__without_channel_axis(self) -> None:
        volume = np.random.rand(20, 24, 10).astype(np.float32)
        out = RandomResizedCrop3D(size=(8, 8, 4), seed=0)(volume)
        assert out.shape == (8, 8, 4)

    def test_apply__invalid_ndim(self) -> None:
        transform = RandomResizedCrop3D(size=4)
        with pytest.raises(ValueError, match="Expected a"):
            transform.apply(np.zeros((1, 1, 2, 2, 2)), CropParams3D(0, 0, 0, 1, 1, 1))

    def test_get_params__within_bounds(self) -> None:
        transform = RandomResizedCrop3D(size=8, scale=(0.05, 1.0), seed=0)
        for _ in range(200):
            p = transform.get_params((1, 64, 48, 8))
            assert p.height >= 1 and p.width >= 1 and p.depth >= 1
            assert 0 <= p.h_start and p.h_start + p.height <= 64
            assert 0 <= p.w_start and p.w_start + p.width <= 48
            assert 0 <= p.d_start and p.d_start + p.depth <= 8

    def test_get_params__anchored_to_volume_aspect_ratio(self) -> None:
        # Without ratio jitter and scale=1 the crop must cover the whole, strongly
        # anisotropic volume instead of collapsing towards a cube.
        transform = RandomResizedCrop3D(size=8, scale=(1.0, 1.0), ratio=(1.0, 1.0))
        assert transform.get_params((1, 64, 48, 8)) == CropParams3D(0, 0, 0, 64, 48, 8)

    def test_get_params__scale(self) -> None:
        transform = RandomResizedCrop3D(size=8, scale=(0.125, 0.125), ratio=(1.0, 1.0))
        p = transform.get_params((1, 40, 40, 40))
        assert (p.height, p.width, p.depth) == (20, 20, 20)

    def test_get_params__fallback(self) -> None:
        transform = RandomResizedCrop3D(size=8, max_attempts=0, seed=0)
        p = transform.get_params((1, 16, 16, 4))
        assert (p.height, p.width, p.depth) == (1, 1, 1)
        assert p.h_start + p.height <= 16
        assert p.d_start + p.depth <= 4

    def test_seed__reproducible(self) -> None:
        shape = (1, 64, 64, 16)
        params_a = [RandomResizedCrop3D(size=8, seed=1).get_params(shape)]
        params_b = [RandomResizedCrop3D(size=8, seed=1).get_params(shape)]
        assert params_a == params_b

        transform = RandomResizedCrop3D(size=8, seed=2)
        transform.set_random_state(seed=1)
        assert [transform.get_params(shape)] == params_a

    def test_apply__replays_params(self) -> None:
        volume = np.random.rand(1, 20, 24, 10).astype(np.float32)
        transform = RandomResizedCrop3D(size=(6, 6, 6), seed=0)
        params = transform.get_params(volume.shape)
        np.testing.assert_array_equal(
            transform.apply(volume, params), transform.apply(volume.copy(), params)
        )

        # Output size equal to the crop size means no resampling, only slicing.
        identity = RandomResizedCrop3D(size=(params.height, params.width, params.depth))
        expected = volume[
            :,
            params.h_start : params.h_start + params.height,
            params.w_start : params.w_start + params.width,
            params.d_start : params.d_start + params.depth,
        ]
        np.testing.assert_array_equal(identity.apply(volume, params), expected)

    def test_upscale_interpolation(self) -> None:
        # Depth 4 -> 8 is enlarged, height and width stay the same.
        volume = np.broadcast_to(
            np.arange(4, dtype=np.float32) * 10, (1, 8, 8, 4)
        ).copy()
        kwargs: dict[str, Any] = dict(
            size=(8, 8, 8), scale=(1.0, 1.0), ratio=(1.0, 1.0)
        )
        nearest = RandomResizedCrop3D(interpolation="nearest", **kwargs)(volume)
        assert set(np.unique(nearest)) <= {0.0, 10.0, 20.0, 30.0}
        linear = RandomResizedCrop3D(
            interpolation="nearest", upscale_interpolation="linear", **kwargs
        )(volume)
        assert not set(np.unique(linear)) <= {0.0, 10.0, 20.0, 30.0}


@pytest.mark.parametrize("fns", [_WEIGHT_FNS, _OUT_OF_PLANE_WEIGHT_FNS])
@pytest.mark.parametrize("mode", sorted(_WEIGHT_FNS))
@pytest.mark.parametrize("in_size, out_size", [(10, 4), (4, 10), (7, 7), (9, 3)])
def test_weights__rows_sum_to_one(
    fns: dict[str, Any], mode: str, in_size: int, out_size: int
) -> None:
    weights = fns[mode](in_size, out_size)
    assert weights.shape == (out_size, in_size)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("area", ("area", "area")),
        ("cubic", ("cubic", "cubic")),
        ("linear+nearest", ("linear", "nearest")),
        ("nearest+area", ("nearest", "area")),
    ],
)
def test_parse_interpolation(spec: str, expected: tuple[str, str]) -> None:
    assert parse_interpolation(spec) == expected


@pytest.mark.parametrize(
    "spec",
    ["", "+", "linear+", "+nearest", "area+nearest+linear", "trilinear", "linear+bicubic"],
)
def test_parse_interpolation__invalid(spec: str) -> None:
    with pytest.raises(ValueError, match="Invalid interpolation"):
        parse_interpolation(spec)


@pytest.mark.parametrize("in_size, out_size", [(24, 16), (16, 24), (30, 24), (7, 20), (5, 2)])
def test_nearest_slice_weights__matches_kneeno(in_size: int, out_size: int) -> None:
    """Depth "nearest" picks the same slices as KneeNo's resample_mode="nearest"."""
    from kneeno.dataset import KneeNoDataset

    vol = np.random.default_rng(0).standard_normal((in_size, 3, 4))  # KneeNo: (D, H, W)
    dataset = SimpleNamespace(resample_mode="nearest", series_depth=out_size)
    expected = KneeNoDataset._resample_depth(dataset, vol)  # type: ignore[arg-type]
    ours = np.tensordot(_nearest_slice_weights(in_size, out_size), vol, axes=([1], [0]))
    np.testing.assert_array_equal(ours, expected)


@pytest.mark.parametrize("depth_out", [16, 24, 40])
def test_resample__in_plane_plus_nearest_is_slice_by_slice(depth_out: int) -> None:
    """Every output slice is the in-plane resample of one whole input slice."""
    volume = np.random.default_rng(0).standard_normal((2, 40, 30, 24))
    out = _resample(volume, (20, 50, depth_out), "linear+nearest")
    idx = np.linspace(0, 23, depth_out).round().astype(int)
    for k, j in enumerate(idx):
        expected = _resample(volume[..., j : j + 1], (20, 50, 1), "linear")
        np.testing.assert_allclose(out[..., k : k + 1], expected, atol=1e-12)


@pytest.mark.parametrize("mode", ["area", "linear", "cubic"])
def test_resample__single_mode_equals_pair(mode: str) -> None:
    volume = np.random.default_rng(0).standard_normal((1, 30, 20, 12))
    np.testing.assert_array_equal(
        _resample(volume, (14, 26, 16), mode, "linear"),
        _resample(volume, (14, 26, 16), f"{mode}+{mode}", "linear+linear"),
    )


def test_resample__upscale_spec_per_plane() -> None:
    """H is reduced, W enlarged, D enlarged: each axis takes its plane's mode of the matching spec."""
    volume = np.random.default_rng(0).standard_normal((1, 30, 10, 12))
    out = _resample(volume, (12, 25, 20), "area+cubic", "linear+nearest")
    expected = np.tensordot(_WEIGHT_FNS["area"](30, 12), volume, axes=([1], [1]))
    expected = np.moveaxis(expected, 0, 1)
    expected = np.moveaxis(
        np.tensordot(_WEIGHT_FNS["linear"](10, 25), expected, axes=([1], [2])), 0, 2
    )
    expected = np.moveaxis(
        np.tensordot(_nearest_slice_weights(12, 20), expected, axes=([1], [3])), 0, 3
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize(
    "mode, torch_mode", [("linear", "trilinear"), ("nearest", "nearest")]
)
@pytest.mark.parametrize("out_shape", [(5, 7, 3), (12, 20, 9), (6, 14, 6)])
def test_resample__matches_torch_interpolate(
    mode: Any, torch_mode: str, out_shape: tuple[int, int, int]
) -> None:
    volume = np.random.default_rng(0).standard_normal((2, 6, 10, 4))
    if mode == "nearest":
        # Only in-plane: depth "nearest" follows KneeNo's mapping, not torch's floor.
        out_shape = (*out_shape[:2], volume.shape[3])
    kwargs = {"align_corners": False} if torch_mode == "trilinear" else {}
    expected = F.interpolate(
        torch.from_numpy(volume)[None], size=out_shape, mode=torch_mode, **kwargs
    )[0].numpy()
    np.testing.assert_allclose(_resample(volume, out_shape, mode), expected, atol=1e-10)


def test_resample__area_integer_factors() -> None:
    volume = np.random.default_rng(0).standard_normal((1, 8, 12, 4))
    # Integer downscaling is block averaging.
    expected = F.avg_pool3d(torch.from_numpy(volume)[None], kernel_size=2)[0].numpy()
    np.testing.assert_allclose(_resample(volume, (4, 6, 2), "area"), expected)
    # Integer upscaling duplicates voxels.
    expected = volume.repeat(2, axis=1).repeat(2, axis=2).repeat(2, axis=3)
    np.testing.assert_allclose(_resample(volume, (16, 24, 8), "area"), expected)


@pytest.mark.parametrize(
    "crop, size, box, expected",
    [
        (4, 10, None, (0, 6)),  # no box: anywhere
        (3, 10, (2, 8), (2, 5)),  # fits in the box: inside it
        (6, 10, (2, 8), (2, 2)),  # exactly the box
        (8, 10, (2, 8), (0, 2)),  # larger than the box: contains it
        (8, 10, (0, 3), (0, 0)),  # box at the start, crop must not leave the axis
        (8, 10, (7, 10), (2, 2)),  # box at the end
        (10, 10, (4, 5), (0, 0)),  # crop spans the axis
        (4, 10, (0, 10), (0, 6)),  # box spans the axis: same as no box
        (4, 10, (5, 5), (0, 6)),  # empty box: same as no box
        (4, 10, (-3, 20), (0, 6)),  # clipped to the axis
    ],
)
def test_start_range(
    crop: int, size: int, box: tuple[int, int] | None, expected: tuple[int, int]
) -> None:
    assert _start_range(crop, size, box) == expected


# Volume (H, W, D) and boxes with exclusive end, (start, end).
_SHAPE = (40, 36, 12)
_BOXES = [
    ((10, 4, 2), (30, 30, 10)),  # centred
    ((0, 0, 0), (12, 9, 3)),  # in a corner, small
    ((5, 20, 0), (40, 36, 12)),  # touching the far edges
    ((17, 17, 5), (18, 18, 6)),  # a single voxel
]


@pytest.mark.parametrize("scale", [(0.05, 0.32), (0.32, 1.0), (1.0, 1.0)])
@pytest.mark.parametrize("box", _BOXES)
def test_get_params__foreground_box_places_without_resizing(
    scale: tuple[float, float], box: tuple[tuple[int, ...], tuple[int, ...]]
) -> None:
    for seed in range(300):
        # Freshly seeded per crop: the sizes are drawn before the position, but numpy's
        # randint consumes a range-dependent amount of randomness, so the streams of the
        # two crops part after the first position. ratio up to 3 drives some draws into
        # the shrink fallback, which must honour the box too.
        kwargs: dict[str, Any] = dict(size=8, scale=scale, ratio=(1 / 3, 3), seed=seed)
        params = RandomResizedCrop3D(**kwargs).get_params(_SHAPE, foreground_box=box)
        plain = RandomResizedCrop3D(**kwargs).get_params(_SHAPE)
        sizes = (params.height, params.width, params.depth)
        # Same sizes as uniform placement: the box never stretches a crop.
        assert sizes == (plain.height, plain.width, plain.depth)
        starts = (params.h_start, params.w_start, params.d_start)
        for start, crop, n, b0, b1 in zip(starts, sizes, _SHAPE, *box):
            assert 0 <= start and start + crop <= n
            if crop <= b1 - b0:
                assert b0 <= start and start + crop <= b1  # inside the box
            else:
                assert start <= b0 and b1 <= start + crop  # around the box


@pytest.mark.parametrize("scale", [(0.05, 0.32), (0.32, 1.0)])
def test_get_params__whole_volume_box_matches_no_box(
    scale: tuple[float, float],
) -> None:
    """A box spanning the volume draws exactly today's uniformly placed crops."""
    with_box = RandomResizedCrop3D(size=8, scale=scale, seed=0)
    without_box = RandomResizedCrop3D(size=8, scale=scale, seed=0)
    whole = ((0, 0, 0), _SHAPE)
    for _ in range(100):
        expected = without_box.get_params(_SHAPE)
        assert with_box.get_params(_SHAPE, foreground_box=whole) == expected
