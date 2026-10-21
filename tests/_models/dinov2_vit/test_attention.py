#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Callable

import pytest
import torch

from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.attention import (
    Attention,
    SDPAttention,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.models import (
    vision_transformer as vits,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)


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


# Every builder uses SDPAttention, in training and evaluation alike: without xFormers,
# MemEffAttention falls back to Attention, which materializes the full attention matrix.
@pytest.mark.parametrize("builder", [vits.vit_small, vits._vit_test])
def test_builders__use_sdp_attention(
    builder: Callable[..., DinoVisionTransformer],
) -> None:
    model = builder(img_size=28, patch_size=14)
    attentions = [m for m in model.modules() if isinstance(m, Attention)]
    assert len(attentions) == model.n_blocks
    assert all(type(attn) is SDPAttention for attn in attentions)
