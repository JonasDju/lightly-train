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
batch. A volume's ``cls`` is the mean of its slices' cls tokens, its ``patches`` are all
slices' patch tokens concatenated in slice order (``D * P`` tokens).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from kneeno import LabeledExternalKneeMRIDataset
from kneeno.evaluation.adapter import EncoderAdapter
from kneeno.evaluation.features import pool_patches
from torch import Tensor

from lightly_train._embedding.embedding_transform import EmbeddingTransform
from lightly_train.types import ImageSizeTuple, TransformInput

#: ``eval.data.dataset_type`` values.
DATASET_TYPES = ("internal", "external")


class DINOv2Adapter(EncoderAdapter):  # type: ignore[misc]  # untyped base class
    """``EncoderAdapter`` for the stock 2D ``DINOv2ViTModelWrapper``.

    Unlike V-JEPA 2.1, DINOv2 produces a cls token (``x_norm_clstoken``), so
    ``has_cls_token`` is True and KneeNo's ``linear`` task is available in addition to
    ``linear_pool``.
    """

    has_cls_token = True

    def __init__(
        self,
        dataset_type: str,
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
            dataset_type:
                Describes the type of eval dataset, either "internal" or "external"
                (anything else raises a ``ValueError``). Controls the preprocessing of
                the raw incoming volume.
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
        if dataset_type not in DATASET_TYPES:
            raise ValueError(
                f"dataset_type must be one of {DATASET_TYPES}, got {dataset_type!r}"
            )
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        mean, std = tuple(normalize[0]), tuple(normalize[1])
        # One (mean, std) per input channel, or a single pair shared by all of them.
        if len(mean) != len(std) or len(mean) not in (1, num_channels):
            raise ValueError(
                f"normalize has {len(mean)} mean / {len(std)} std values, expected 1 "
                f"or num_channels={num_channels}"
            )
        self.dataset_type = dataset_type
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

    def prepare_input(self, volume: Tensor, orientation: str | None = None) -> Tensor:
        """``(1, D, H, W)`` raw volume (in [0, 255] for internal / unscaled for external)
        -> ``(C, D, image_size[0], image_size[1])`` float32.
        """
        if self.dataset_type == "external":
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
        """``batch``: ``(B, C, D, image_size[0], image_size[1])``

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
        cls_tokens = out["cls_token"].reshape(b, d, self._embed_dim)

        # Patches
        # (B*D, embed_dim, patH, patW) -> (B, D, P, embed_dim), P = patH * patW
        patches = (
            out["features"]
            .flatten(2)
            .permute(0, 2, 1)
            .reshape(b, d, -1, self._embed_dim)
        )

        # This basic return value is the minimum required to make in-training
        # evaluation work with KneeNo
        # TODO: Replace this with something more powerful
        return {
            "cls": pool_patches(cls_tokens, "avg"),
            "patches": patches.flatten(1, 2),
        }
