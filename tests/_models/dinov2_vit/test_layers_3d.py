#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import pytest
import torch

from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.attention import (
    Attention,
    SDPAttention,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.patch_embed import (
    PatchEmbed,
)


def _patch_embed(**kwargs: Any) -> PatchEmbed:
    # Sizes are given as (H, W, D). H != W on purpose to catch swapped axes.
    return PatchEmbed(
        img_size=(16, 12, 8), patch_size=(4, 2, 2), in_chans=1, embed_dim=6, **kwargs
    )


class TestPatchEmbed:
    def test_init__axis_order(self) -> None:
        patch_embed = _patch_embed()
        assert patch_embed.img_size == (8, 16, 12)  # (D, H, W)
        assert patch_embed.patch_size == (2, 4, 2)  # (D, H, W)
        assert patch_embed.patches_resolution == (4, 4, 6)
        assert patch_embed.num_patches == 96
        assert patch_embed.proj.kernel_size == (2, 4, 2)

    @pytest.mark.parametrize("size", [16, (16, 16)])
    def test_init__requires_3d_sizes(self, size: Any) -> None:
        with pytest.raises(AssertionError):
            PatchEmbed(img_size=size, patch_size=(4, 4, 4))
        with pytest.raises(AssertionError):
            PatchEmbed(img_size=(16, 16, 16), patch_size=size)

    def test_forward(self) -> None:
        tokens, d, h, w = _patch_embed()(torch.rand(2, 1, 8, 16, 12))
        assert tokens.shape == (2, 96, 6)
        assert (d, h, w) == (4, 4, 6)

    def test_forward__resizes_to_multiple_of_patch_size(self) -> None:
        tokens, d, h, w = _patch_embed()(torch.rand(1, 1, 7, 15, 11))
        assert (d, h, w) == (4, 4, 6)
        assert tokens.shape == (1, 96, 6)

    def test_forward__no_flatten(self) -> None:
        tokens, _, _, _ = _patch_embed(flatten_embedding=False)(
            torch.rand(2, 1, 8, 16, 12)
        )
        assert tokens.shape == (2, 4, 4, 6, 6)

    @pytest.mark.parametrize(
        "shape", [(2, 1, 8, 16, 12), (1, 1, 7, 15, 11), (3, 1, 2, 4, 2)]
    )
    def test_compute_out_dims(self, shape: tuple[int, ...]) -> None:
        patch_embed = _patch_embed()
        x = torch.rand(shape)
        _, d, h, w = patch_embed(x)
        assert patch_embed.compute_out_dims(x) == (shape[0], 6, d, h, w)


class TestSDPAttention:
    def test_forward__matches_attention(self) -> None:
        torch.manual_seed(0)
        attention = Attention(dim=16, num_heads=4, qkv_bias=True).eval()
        sdpa = SDPAttention(dim=16, num_heads=4, qkv_bias=True).eval()
        sdpa.load_state_dict(attention.state_dict())
        x = torch.rand(2, 10, 16)
        torch.testing.assert_close(sdpa(x), attention(x), atol=1e-6, rtol=1e-5)

    def test_forward__attn_bias(self) -> None:
        torch.manual_seed(0)
        sdpa = SDPAttention(dim=16, num_heads=4).eval()
        x = torch.rand(2, 10, 16)
        torch.testing.assert_close(
            sdpa(x, attn_bias=torch.zeros(1, 4, 10, 10)), sdpa(x)
        )

    def test_forward__training_with_dropout(self) -> None:
        sdpa = SDPAttention(dim=16, num_heads=4, attn_drop=0.5).train()
        assert sdpa(torch.rand(2, 10, 16)).shape == (2, 10, 16)
