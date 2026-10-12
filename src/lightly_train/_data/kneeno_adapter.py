#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Bridges the 2D DINOv2 encoder to ``kneeno.evaluation``'s unified feature format.

``DINOv2Adapter`` is the ``kneeno.evaluation.adapter.EncoderAdapter`` implementation this
repo hands to ``kneeno.evaluation.ClassificationEvaluator``. It is the DINOv2 counterpart
of ``vjepa2/src/datasets/kneeno_adapter.py::VJepa21Adapter``.

KneeNo hands the adapter whole ``(1, D, H, W)`` volumes, while this branch's encoder is
the stock 2D one trained on single slices. The adapter therefore treats a volume as ``D``
independent slices: ``prepare_input`` runs each slice through lightly-train's
``EmbeddingTransform`` (resize to ``image_size`` + normalize, the deterministic eval
counterpart of the training transform), and ``forward_features`` encodes all slices in one
batch. It returns the per-slice cls and patch tokens (``has_slice_tokens``); KneeNo derives the
volume-level views from them (slice cls tokens pooled per ``knn.pooling`` / ``linear.pooling``,
all slices' patch tokens flattened in slice order) and additionally runs its slice tasks on them.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from kneeno import LabeledExternalKneeMRIDataset
from kneeno.evaluation.adapter import EncoderAdapter
from kneeno.evaluation.dataset import SOURCES
from torch import Tensor

from lightly_train._embedding.embedding_transform import EmbeddingTransform
from lightly_train.types import ImageSizeTuple, TransformInput


class DINOv2Adapter(EncoderAdapter):  # type: ignore[misc]  # untyped base class
    """``EncoderAdapter`` for the stock 2D ``DINOv2ViTModelWrapper``.

    Unlike V-JEPA 2.1, DINOv2 produces a cls token (``x_norm_clstoken``), so
    ``has_cls_token`` is True and KneeNo's ``linear`` task is available in addition to
    ``linear_pool``. Every slice is encoded on its own, so ``has_slice_tokens`` is True as
    well, which makes KneeNo's slice tasks (``linear_slice_cls``, ``attentive_slice_cls``,
    ``attentive_slice_pool``) available.
    """

    has_cls_token = True
    has_slice_tokens = True

    def __init__(
        self,
        embed_dim: int,
        num_channels: int,
        image_size: ImageSizeTuple | Sequence[int],
        normalize: tuple[Sequence[float], Sequence[float]] = (
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ) -> None:
        """
        Args:
            embed_dim:
                Feature dimension of the encoder, i.e. ``wrapped_model.feature_dim()``.
            num_channels:
                Input channels of the encoder, i.e. the resolved
                ``DINOv2ViTTransformArgs.num_channels`` the model was built with. KneeNo's
                volumes are grayscale (1 channel); with ``num_channels > 1`` (e.g. 3 for
                the official RGB checkpoints) they have to be repeated along the channel
                axis to match.
            image_size:
                The training global-crop size as ``(H, W)``, matching
                ``DINOv2ViTTransformArgs.image_size``.
            normalize:
                ``(mean, std)`` per channel, as in ``NormalizeArgs``. Defaults match
                ``NormalizeArgs``' (ImageNet).
        """
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        mean, std = tuple(normalize[0]), tuple(normalize[1])
        # One (mean, std) per input channel, or a single pair shared by all of them.
        if len(mean) != len(std) or len(mean) not in (1, num_channels):
            raise ValueError(
                f"normalize has {len(mean)} mean / {len(std)} std values, expected 1 "
                f"or num_channels={num_channels}"
            )
        self._embed_dim = embed_dim
        self.num_channels = num_channels
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.mean, self.std = mean, std

        self.embed_transform = EmbeddingTransform(
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def prepare_input(
        self, volume: Tensor, orientation: str | None = None, source: str | None = None
    ) -> Tensor:
        """``(1, D, H, W)`` raw volume (in [0, 255] for internal / unscaled for external)
        -> ``(C, D, image_size[0], image_size[1])`` float32.

        ``source`` is the dataset the volume came from (``kneeno.evaluation.dataset.SOURCES``):
        an ``"external"`` volume (raw NIfTI intensities) is first quantized like the internal
        JPEGs. Anything else raises, since it would silently take the internal path, i.e.
        treat raw NIfTI intensities as 0..255.
        """
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        if source == "external":
            volume = LabeledExternalKneeMRIDataset.to_uint8(volume)
            volume = torch.from_numpy(volume)

        if self.num_channels > 1:
            if volume.shape[0] != 1:
                raise ValueError(
                    f"Can only repeat a 1-channel volume to {self.num_channels} "
                    f"channels, got {volume.shape[0]}"
                )
            # (1, D, H, W) -> (num_channels, D, H, W)
            volume = volume.repeat(self.num_channels, 1, 1, 1)

        # Albumentations expects images in (H, W, C)
        # (C, D, H, W) -> (D, H, W, C)
        depth_slices = volume.permute(1, 2, 3, 0).numpy()
        inputs = [TransformInput(image=depth_slice) for depth_slice in depth_slices]
        outputs = [self.embed_transform(transform_input) for transform_input in inputs]
        # [(C, H, W)] of length D
        slices = [transform_output[0]["image"] for transform_output in outputs]
        volume = torch.stack(slices)  # (D, C, H, W)

        return volume.permute(1, 0, 2, 3)  # (C, D, H, W)

    def forward_features(self, model: Any, batch: Tensor) -> dict[str, Tensor | None]:
        """``batch``: ``(B, C, D, image_size[0], image_size[1])`` -> per-slice tokens
        ``{"slice_cls": (B, D, embed_dim), "slice_patches": (B, D, P, embed_dim)}``.

        Unwraps DDP (if wrapped) before calling the encoder directly, so a frozen
        forward pass under ``torch.no_grad()`` does not trip DDP's backward-pass
        bookkeeping.
        """
        b, c, d, h, w = batch.shape

        wrapper = model.module if hasattr(model, "module") else model
        # (B, C, D, H, W) -> (B*D, C, H, W)
        batch = batch.permute(0, 2, 1, 3, 4).flatten(0, 1)
        out = wrapper.forward_features(batch)

        # CLS tokens
        # (B*D, embed_dim) -> (B, D, embed_dim)
        slice_cls = out["cls_token"].reshape(b, d, self._embed_dim)

        # Patches
        # (B*D, embed_dim, patH, patW) -> (B, D, P, embed_dim), P = patH * patW
        slice_patches = (
            out["features"]
            .flatten(2)
            .permute(0, 2, 1)
            .reshape(b, d, -1, self._embed_dim)
        )
        return {"slice_cls": slice_cls, "slice_patches": slice_patches}
