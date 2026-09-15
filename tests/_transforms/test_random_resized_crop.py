#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lightly_train._transforms.random_resized_crop import (
    _WEIGHT_FNS,
    CropParams3D,
    RandomResizedCrop3D,
    _resample,
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
        ],
    )
    def test_init__invalid_args(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            RandomResizedCrop3D(**kwargs)

    def test_init__int_size(self) -> None:
        assert RandomResizedCrop3D(size=8).size == (8, 8, 8)

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64, np.bool_])
    @pytest.mark.parametrize("interpolation", ["area", "linear", "nearest", "cubic"])
    def test_call__shape_and_dtype(self, dtype: type, interpolation: Any) -> None:
        rng = np.random.default_rng(0)
        volume = rng.uniform(0, 255, size=(1, 20, 24, 10)).astype(dtype)
        transform = RandomResizedCrop3D(
            size=(8, 12, 16), interpolation=interpolation, seed=0
        )
        out = transform(volume)
        assert out.shape == (1, 8, 12, 16)
        assert out.dtype == dtype

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


@pytest.mark.parametrize("mode", sorted(_WEIGHT_FNS))
@pytest.mark.parametrize("in_size, out_size", [(10, 4), (4, 10), (7, 7), (9, 3)])
def test_weights__rows_sum_to_one(mode: str, in_size: int, out_size: int) -> None:
    weights = _WEIGHT_FNS[mode](in_size, out_size)
    assert weights.shape == (out_size, in_size)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "mode, torch_mode", [("linear", "trilinear"), ("nearest", "nearest")]
)
@pytest.mark.parametrize("out_shape", [(5, 7, 3), (12, 20, 9), (6, 14, 6)])
def test_resample__matches_torch_interpolate(
    mode: Any, torch_mode: str, out_shape: tuple[int, int, int]
) -> None:
    volume = np.random.default_rng(0).standard_normal((2, 6, 10, 4))
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
