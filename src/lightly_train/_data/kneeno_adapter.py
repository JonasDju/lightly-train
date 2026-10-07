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

STUB: KneeNo hands the adapter whole ``(1, D, H, W)`` volumes, while this branch's encoder
is the stock 2D one trained on single slices. How a volume becomes 2D encoder input (and
how the per-slice features are pooled back into one volume's features) is not designed
yet, so ``prepare_input`` and ``forward_features`` raise ``NotImplementedError``. A run
whose config enables evaluation therefore fails at the first epoch an eval task is due.
"""

from __future__ import annotations

from typing import Any, Sequence

from kneeno.evaluation.adapter import EncoderAdapter
from torch import Tensor

from lightly_train.types import ImageSizeTuple

#: ``eval.data.dataset_type`` values.
DATASET_TYPES = ("internal", "external")

_NOT_IMPLEMENTED = (
    "DINOv2Adapter is a stub on this branch: how KneeNo's 3D volumes are fed to the 2D "
    "DINOv2 encoder is not implemented yet. Remove the eval block from the run config "
    "(or disable every eval.freq task) to train without evaluation."
)


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

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def prepare_input(self, volume: Tensor, orientation: str | None = None) -> Tensor:
        """``(1, D, H, W)`` raw volume -> encoder input. Not implemented yet."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def forward_features(self, model: Any, batch: Tensor) -> dict[str, Tensor | None]:
        """Collated ``prepare_input`` outputs -> ``{"cls": (B, D), "patches": (B, P, D)}``.
        Not implemented yet."""
        raise NotImplementedError(_NOT_IMPLEMENTED)
