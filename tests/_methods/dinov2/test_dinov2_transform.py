#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
import torch

from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2ViTTransform,
)
from lightly_train.types import TransformInput


def test_dinov2_transform_shapes() -> None:
    # Volumes enter the transform as (C, H, W, D) uint8 arrays.
    volume = np.random.uniform(0, 255, size=(1, 300, 280, 30)).astype(np.uint8)
    input: TransformInput = {"image": volume}

    transform_args = DINOv2ViTTransform.transform_args_cls()()
    transform_args.resolve_auto()
    transform_args.resolve_incompatible()
    assert transform_args.num_channels == 1
    transform = DINOv2ViTTransform(transform_args)

    views = transform(input)
    assert len(views) == 2 + 8
    # Views leave the transform as (C, D, H, W) float tensors.
    for view in views[:2]:
        assert view["image"].shape == (1, 16, 224, 224)
    for view in views[2:]:
        assert view["image"].shape == (1, 8, 98, 98)
    for view in views:
        assert type(view["image"]) is torch.Tensor
        assert view["image"].dtype == torch.float32
