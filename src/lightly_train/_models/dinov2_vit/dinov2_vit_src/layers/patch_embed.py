#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

# Modifications Copyright 2025 Lightly AG:
# - Modified load_state_dict to handle different number of input channels

from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor

from lightly_train import _torch_helpers
from lightly_train._models import _model_helpers

logger = logging.getLogger(__name__)

class PatchEmbed(nn.Module):
    """
    3D image to patch embedding: (B,C,D,H,W) -> (B,N,embed_dim)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (224, 224, 24),
        patch_size: Tuple[int, int, int] = (14, 14, 4),
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        assert isinstance(img_size, (tuple, list, ListConfig)) and len(img_size) == 3, "img_size must specify a length in all three spatial dimensions"
        assert isinstance(patch_size, (tuple, list, ListConfig)) and len(patch_size) == 3, "patch_size must specify a length in all three spatial dimensions"

        # Reorder W H D -> D H W
        img_size = (img_size[2], img_size[1], img_size[0])
        patch_size = (patch_size[2], patch_size[1], patch_size[0])

        patch_grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )

        self.img_size = img_size        # D H W
        self.patch_size = patch_size    # D H W
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        _torch_helpers.register_load_state_dict_pre_hook(
            self, _model_helpers.patch_embed_adjust_input_channels_hook
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int, int]:
        _, _, D, H, W = x.shape
        patch_D, patch_H, patch_W = self.patch_size

        # Find the next multiple of patch size for height, width & depth.
        new_D = math.ceil(D / patch_D) * patch_D
        new_H = math.ceil(H / patch_H) * patch_H
        new_W = math.ceil(W / patch_W) * patch_W

        if new_H != H or new_W != W or new_D != D:
            # Resize image to nearest valid resolution
            logger.info("Resizing image to nearest valid resolution")
            x = F.interpolate(
                x, size=(new_D, new_H, new_W), mode="trilinear", align_corners=False
            )
            D, H, W = new_D, new_H, new_W

        assert D % patch_D == 0, (
            f"Input image depth {D} is not a multiple of patch depth: {patch_D}"
        )
        assert H % patch_H == 0, (
            f"Input image height {H} is not a multiple of patch height {patch_H}"
        )
        assert W % patch_W == 0, (
            f"Input image width {W} is not a multiple of patch width: {patch_W}"
        )

        x = self.proj(x)  # B C D H W
        D, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)  # B DHW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, D, H, W, self.embed_dim)  # B D H W C
        return x, D, H, W

    def flops(self) -> float:
        Do, Ho, Wo = self.patches_resolution
        flops = (
            Do
            * Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        )
        if self.norm is not None:
            flops += Do * Ho * Wo * self.embed_dim
        return flops
