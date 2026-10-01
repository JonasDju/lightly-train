#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Bridges the 3D DINOv2 encoder to ``kneeno.evaluation``'s unified feature format.

``DINOv2Adapter`` is the ``kneeno.evaluation.adapter.EncoderAdapter`` implementation this
repo hands to ``kneeno.evaluation.ClassificationEvaluator``. It is the DINOv2 counterpart
of ``vjepa2/src/datasets/kneeno_adapter.py::VJepa21Adapter``.

``prepare_input`` is the deterministic (non-augmenting) counterpart of
``_transforms/view_transform.py::ViewTransform``: evaluation must apply the same
normalization and the same fixed output size, but none of the random augmentation.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any, Sequence

import numpy as np
import torch
from kneeno.evaluation.adapter import EncoderAdapter
from torch import Tensor

from lightly_train._transforms.random_resized_crop import _resample
from lightly_train._transforms.view_transform import (
    RESIZE_INTERPOLATION,
    RESIZE_UPSCALE_INTERPOLATION,
)
from lightly_train.types import ImageSizeTuple

logger = getLogger(__name__)


class DINOv2Adapter(EncoderAdapter):  # type: ignore[misc]  # untyped base class
    """``EncoderAdapter`` for this repo's 3D ``DINOv2ViTModelWrapper``.

    Unlike V-JEPA 2.1, DINOv2 produces a cls token (``x_norm_clstoken``), so
    ``has_cls_token`` is True and KneeNo's ``linear`` task is available in addition to
    ``linear_pool``.
    """

    has_cls_token = True

    def __init__(
        self,
        embed_dim: int,
        image_size: ImageSizeTuple | Sequence[int],
        normalize: tuple[Sequence[float], Sequence[float]] = ((0.5,), (0.5,)),
    ) -> None:
        """
        Args:
            embed_dim:
                Feature dimension of the encoder, i.e. ``wrapped_model.feature_dim()``.
            image_size:
                The training global-crop size as ``(H, W, D)`` -- the repo-wide config
                axis order, matching ``DINOTransformArgs.image_size``. Volumes are
                resized to exactly this size, since ``RandomResizedCrop3D`` always
                resizes its crop to a fixed configured output size during training.
            normalize:
                ``(mean, std)`` per channel, as in ``NormalizeArgs``. Scaled by 255
                internally to match ``ViewTransform``'s
                ``NormalizeIntensity(subtrahend=[m * 255], divisor=[s * 255])``.
        """
        self._embed_dim = embed_dim
        # Config order is (H, W, D); tensors are (D, H, W). This is the one conversion.
        height, width, depth = (
            int(image_size[0]),
            int(image_size[1]),
            int(image_size[2]),
        )
        self.size = (depth, height, width)
        mean, std = normalize
        # ViewTransform normalizes raw [0, 255] intensities, hence the * 255.
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1) * 255.0
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1) * 255.0

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def prepare_input(self, volume: Tensor) -> Tensor:
        """``(1, D, H, W)`` raw volume -> ``(1, D', H', W')`` normalized float32.

        Resize all three axes to the configured training crop size, then normalize -- the
        same order and the same resampler as training, i.e. ``RandomResizedCrop3D`` at
        ``scale=ratio=1`` (whose crop is the whole volume). Resizing depth as well (V-JEPA
        2.1's adapter leaves it native) is what makes this the faithful counterpart of
        *this* repo's training pipeline: the random resized crop there always emits a fixed
        output size, so the encoder never saw a variable depth. It also keeps every
        evaluation batch uniformly shaped, so the positional embedding is not
        re-interpolated on every batch.

        Not ``F.interpolate(mode="trilinear")``: it has no anti-aliasing in 3D, so
        downsampling (e.g. 672 -> 224 in-plane) keeps noise and aliasing that the
        area-resampled training views never contain. An axis that already has its target
        size is left untouched, so slices are never mixed when the depth already matches.

        Runs in DataLoader worker processes: stays on the CPU and holds nothing
        unpicklable.
        """
        if not torch.is_tensor(volume):
            volume = torch.as_tensor(volume)
        buffer = volume.float()  # (C, D, H, W), C == 1

        if tuple(buffer.shape[1:]) != self.size:
            # The resampler works on (C, H, W, D) numpy arrays, like the training transforms.
            resized = _resample(
                buffer.permute(0, 2, 3, 1).numpy(),
                (self.size[1], self.size[2], self.size[0]),  # H W D
                RESIZE_INTERPOLATION,
                RESIZE_UPSCALE_INTERPOLATION,
            )
            buffer = torch.from_numpy(np.ascontiguousarray(resized.transpose(0, 3, 1, 2))) # back to (C, D, H, W)

        # Normalize last, as ViewTransform does.
        return (buffer - self.mean) / self.std

    def forward_features(self, model: Any, batch: Tensor) -> dict[str, Tensor | None]:
        """``batch``: ``(B, 1, D, H, W)`` -- the inherited ``collate`` stacks
        ``prepare_input`` outputs along a new batch axis.

        Unwraps DDP (if wrapped) before calling the wrapper, so a frozen forward pass
        under ``torch.no_grad()`` does not trip DDP's backward-pass bookkeeping.
        """
        wrapper = model.module if hasattr(model, "module") else model
        out = wrapper.forward_features(batch)
        # (B, embed_dim, patD, patH, patW) -> (B, P, embed_dim), P = patD * patH * patW.
        # Same reshape DINOv2._forward_teacher already applies to these features.
        patches = out["features"].flatten(2).permute(0, 2, 1)
        return {"cls": out["cls_token"], "patches": patches}
