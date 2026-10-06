#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Pins the DINOv2 augmentations to a stored reference.

``dinov2_views_reference.npz`` holds the views ``_views()`` produced at commit d1aab3e,
before ``transform.crop_foreground`` existed. With the option off (its default), the
pipeline must still produce exactly those views. Regenerate the file only for an
intended change of the augmentations:

    .venv/bin/python tests/_methods/dinov2/test_dinov2_transform_reference.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lightly_train._data.mi_dataset import reseed_randomizables
from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2ViTTransform,
    DINOv2ViTTransformArgs,
)

REFERENCE = Path(__file__).with_name("dinov2_views_reference.npz")
NUM_SAMPLES = 3


def _transform_args(**overrides: object) -> DINOv2ViTTransformArgs:
    # Every augmentation on, with probabilities high enough that each one fires in some
    # view; non-cubic sizes so that an axis mix-up shows.
    args = DINOv2ViTTransformArgs.model_validate(
        {
            "image_size": (20, 16, 8),
            "num_channels": 1,
            "resize_interpolation": "area+nearest",
            "resize_upscale_interpolation": "linear",
            "random_rotation": {"prob": 0.5, "degrees": 15},
            "histogram_shift": {"prob": 0.8},
            "adjust_contrast": {"prob": 0.8},
            "gaussian_blur": {"prob": 0.5, "sigma_range": (0.4, 0.6)},
            "gaussian_sharpen": {"prob": 0.5},
            "gibbs_noise": {"prob": 0.5},
            "gaussian_noise": {"prob": 0.5},
            "local_view": {"num_views": 3, "view_size": (10, 8, 4)},
            **overrides,
        }
    )
    args.resolve_auto()
    args.resolve_incompatible()
    return args


def _volume(index: int) -> np.ndarray:
    """A ``(1, H, W, D)`` uint8 volume: noisy "tissue" block in air, off-centre."""
    rng = np.random.default_rng(index)
    volume = (rng.random((1, 48, 40, 12)) * 8).astype(np.uint8)
    volume[:, 6:30, 12:38, 1:11] = rng.integers(
        40, 256, (1, 24, 26, 10), dtype=np.uint8
    )
    return volume


def _views(transform_args: DINOv2ViTTransformArgs) -> np.ndarray:
    """All views of ``NUM_SAMPLES`` volumes, flattened and concatenated, from one
    seeded transform (so later samples also check the random streams' state)."""
    transform = DINOv2ViTTransform(transform_args)
    reseed_randomizables(transform, rng=np.random.RandomState(0))
    torch.manual_seed(0)
    views = [
        view["image"].numpy().ravel()
        for index in range(NUM_SAMPLES)
        for view in transform({"image": _volume(index)})
    ]
    return np.concatenate(views)


def test_dinov2_transform__matches_reference_without_crop_foreground() -> None:
    expected = np.load(REFERENCE)["views"]
    actual = _views(_transform_args())
    assert actual.shape == expected.shape
    # Not bit-exact: BLAS summation order may differ between machines.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_dinov2_transform__reference_detects_crop_placement() -> None:
    """The reference is sensitive to where the crops land: placing them on the tissue
    changes the views."""
    expected = np.load(REFERENCE)["views"]
    actual = _views(_transform_args(crop_foreground={"threshold": 10}))
    assert not np.allclose(actual, expected, rtol=0, atol=1e-5)


if __name__ == "__main__":
    np.savez_compressed(REFERENCE, views=_views(_transform_args()))
    print(f"Wrote {REFERENCE}")
