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
from torch import Tensor
from torch.nn import Conv2d, Module
from torch.testing import assert_close

from lightly_train._models import _model_helpers
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.patch_embed import (
    PatchEmbed,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
    _vit_test,
)

from ..helpers import dinov2_2d_checkpoint as _2d_checkpoint

# Sizes are given as (H, W, D), patch grids are (D, H, W). H != W != D everywhere so
# that a swapped axis cannot pass unnoticed.
PATCH_SIZE = (4, 2, 2)
SOURCE_IMG_SIZE = (16, 8, 4)  # patch grid (2, 4, 4), 32 patches
TARGET_IMG_SIZE = (24, 16, 8)  # patch grid (4, 6, 8), 192 patches
# 2 * 4 * 8 = 64 = 8**2. A perfect square, which the old 2D hook would have happily
# reshaped into an 8x8 grid.
SQUARE_IMG_SIZE = (16, 16, 4)  # patch grid (2, 4, 8), 64 patches


def _model(
    img_size: tuple[int, int, int] = SOURCE_IMG_SIZE, **kwargs: Any
) -> DinoVisionTransformer:
    kwargs.setdefault("in_chans", 1)
    return _vit_test(PATCH_SIZE, img_size=img_size, **kwargs)


def _block_keys(state_dict: dict[str, Any]) -> list[str]:
    return [
        key
        for key in state_dict
        if key.startswith("blocks.") or key.startswith("norm.")
    ]


def _index_encoded_pos_embed(grid: tuple[int, int, int], dim: int) -> Tensor:
    """A pos_embed whose channels 0, 1, 2 hold the patch's own d, h, w index.

    Interpolating it must leave channel 0 varying only along D, channel 1 only along H
    and channel 2 only along W. Comparing against a reference ``F.interpolate`` call
    would not catch a permuted reshape, this does.
    """
    d, h, w = grid
    patches = torch.zeros(d, h, w, dim)
    patches[..., 0] = torch.arange(d, dtype=torch.float32)[:, None, None]
    patches[..., 1] = torch.arange(h, dtype=torch.float32)[None, :, None]
    patches[..., 2] = torch.arange(w, dtype=torch.float32)[None, None, :]
    return torch.cat(
        [torch.zeros(1, 1, dim), patches.reshape(1, d * h * w, dim)], dim=1
    )


def _assert_index_encoding_preserved(
    pos_embed: Tensor, grid: tuple[int, int, int], source_grid: tuple[int, int, int]
) -> None:
    patches = pos_embed[0, 1:].reshape(*grid, pos_embed.shape[-1])
    for axis in range(3):
        channel = patches[..., axis]
        # Constant along the two other axes ...
        other_axes = tuple(a for a in range(3) if a != axis)
        assert_close(
            channel, channel.amax(dim=other_axes, keepdim=True).expand_as(channel)
        )
        # ... and spanning the source index range along its own.
        assert channel.min().item() == pytest.approx(0.0)
        assert channel.max().item() == pytest.approx(source_grid[axis] - 1)


class TestInterpolatePosEmbedHook:
    def test__trilinear_3d(self) -> None:
        source, target = _model(SOURCE_IMG_SIZE), _model(TARGET_IMG_SIZE)
        assert source.patch_embed.patches_resolution == (2, 4, 4)
        assert target.patch_embed.patches_resolution == (4, 6, 8)
        checkpoint = source.state_dict()

        target.load_state_dict(checkpoint, strict=True)

        assert target.pos_embed.shape == (1, 1 + 4 * 6 * 8, source.embed_dim)
        # The cls position is carried over untouched.
        assert_close(target.pos_embed[:, :1], checkpoint["pos_embed"][:, :1])
        # The buffer describes the new grid, not the checkpoint's.
        assert target.pos_embed_grid.tolist() == [4, 6, 8]
        # Everything else is a plain load.
        loaded = target.state_dict()
        for key in _block_keys(checkpoint):
            assert_close(loaded[key], checkpoint[key])

    def test__axis_order(self) -> None:
        source, target = _model(SOURCE_IMG_SIZE), _model(TARGET_IMG_SIZE)
        source_grid = source.patch_embed.patches_resolution
        checkpoint = source.state_dict()
        checkpoint["pos_embed"] = _index_encoded_pos_embed(
            source_grid, source.embed_dim
        )

        target.load_state_dict(checkpoint, strict=True)

        _assert_index_encoding_preserved(
            target.pos_embed, target.patch_embed.patches_resolution, source_grid
        )

    def test__square_product_grid_not_treated_as_2d(self) -> None:
        # Regression test: the old hook recovered the grid with sqrt(n_patches) and
        # would have silently accepted this one, since 2 * 4 * 8 == 8**2.
        source, target = _model(SQUARE_IMG_SIZE), _model(TARGET_IMG_SIZE)
        source_grid = source.patch_embed.patches_resolution
        assert source_grid == (2, 4, 8)
        assert source.pos_embed.shape[1] - 1 == 64 == 8**2
        checkpoint = source.state_dict()
        checkpoint["pos_embed"] = _index_encoded_pos_embed(
            source_grid, source.embed_dim
        )

        target.load_state_dict(checkpoint, strict=True)

        _assert_index_encoding_preserved(
            target.pos_embed, target.patch_embed.patches_resolution, source_grid
        )

    def test__noop_matching_grid(self) -> None:
        source, target = _model(), _model()
        checkpoint = source.state_dict()
        pos_embed = checkpoint["pos_embed"]

        _model_helpers.interpolate_pos_embed_hook(target, checkpoint, "")

        # Untouched when checkpoint and model grids already match.
        assert checkpoint["pos_embed"] is pos_embed

    @pytest.mark.parametrize("img_size", [SOURCE_IMG_SIZE, TARGET_IMG_SIZE])
    def test__missing_grid_buffer_raises(self, img_size: tuple[int, int, int]) -> None:
        # Raises for a matching grid too: a 3D checkpoint without the buffer predates
        # this repo's interpolation support and should fail loudly either way.
        source, target = _model(SOURCE_IMG_SIZE), _model(img_size)
        checkpoint = source.state_dict()
        del checkpoint["pos_embed_grid"]

        with pytest.raises(RuntimeError, match="does not record the patch grid"):
            target.load_state_dict(checkpoint, strict=True)

    def test__inconsistent_grid_raises(self) -> None:
        source, target = _model(SOURCE_IMG_SIZE), _model(TARGET_IMG_SIZE)
        checkpoint = source.state_dict()
        checkpoint["pos_embed_grid"] = torch.tensor([2, 4, 5])

        with pytest.raises(RuntimeError, match="Inconsistent checkpoint"):
            target.load_state_dict(checkpoint, strict=True)

    def test__prefixed(self) -> None:
        # Loaded through the wrapper, so every key is prefixed with "_model.".
        source = DINOv2ViTModelWrapper(model=_model(SOURCE_IMG_SIZE))
        target = DINOv2ViTModelWrapper(model=_model(TARGET_IMG_SIZE))
        checkpoint = source.state_dict()

        target.load_state_dict(checkpoint, strict=True)

        model = target.get_model()
        assert model.pos_embed.shape == (1, 1 + 4 * 6 * 8, model.embed_dim)
        assert_close(model.pos_embed[:, :1], checkpoint["_model.pos_embed"][:, :1])
        assert model.pos_embed_grid.tolist() == [4, 6, 8]

    def test__ignores_non_pos_embed_keys(self) -> None:
        weight = torch.randn(3, 8)
        state_dict = {"blocks.0.attn.qkv.weight": weight}

        _model_helpers.interpolate_pos_embed_hook(_model(), state_dict, "")

        assert state_dict == {"blocks.0.attn.qkv.weight": weight}


class TestInterpolatePosEmbedHook2D:
    """Loading an original 2D DINOv2 checkpoint into the 3D model."""

    TOKENIZATION_KEYS = [
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "pos_embed",
        "pos_embed_grid",
        "cls_token",
        "mask_token",
        "register_tokens",
    ]

    def test__keeps_tokenization_random(self) -> None:
        model = _model(num_register_tokens=4)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        checkpoint = _2d_checkpoint(model)

        model.load_state_dict(checkpoint, strict=True)

        after = model.state_dict()
        for key in self.TOKENIZATION_KEYS:
            assert_close(after[key], before[key])
        assert after["pos_embed_grid"].tolist() == list(
            model.patch_embed.patches_resolution
        )

    def test__loads_blocks_and_norm(self) -> None:
        model = _model(num_register_tokens=4)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        checkpoint = _2d_checkpoint(model)

        model.load_state_dict(checkpoint, strict=True)

        after = model.state_dict()
        keys = _block_keys(checkpoint)
        assert keys
        for key in keys:
            assert_close(after[key], checkpoint[key])
            assert not torch.equal(after[key], before[key])

    def test__noreg_checkpoint_into_reg_model(self) -> None:
        # A "-noreg" 2D checkpoint has no register_tokens; the model's stay random.
        model = _model(num_register_tokens=4)
        assert model.register_tokens is not None
        before = model.register_tokens.clone()
        checkpoint = _2d_checkpoint(model, num_register_tokens=0)
        assert "register_tokens" not in checkpoint

        model.load_state_dict(checkpoint, strict=True)

        assert_close(model.register_tokens, before)

    def test__reg_checkpoint_into_noreg_model(self) -> None:
        # The reverse: a reg4 checkpoint into a model without registers must not trip
        # strict=True over an unexpected key.
        model = _model()
        assert model.register_tokens is None
        checkpoint = _2d_checkpoint(model, num_register_tokens=4)

        model.load_state_dict(checkpoint, strict=True)

    def test__forward_after_load(self) -> None:
        model = _model(num_register_tokens=4)
        model.load_state_dict(_2d_checkpoint(model), strict=True)

        # Inputs are (B, C, D, H, W) for an (H, W, D) img_size of (16, 8, 4).
        out = model(torch.rand(2, 1, 4, 16, 8), is_training=True)

        assert out["x_norm_patchtokens"].shape == (2, 32, model.embed_dim)
        assert out["x_norm_regtokens"].shape == (2, 4, model.embed_dim)


class _Conv2dPatchEmbed(Module):
    """Stand-in for DINOv3's 2D PatchEmbed, which shares the channel hook."""

    def __init__(self, in_chans: int, embed_dim: int = 4) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=2, stride=2)


def _patch_embed_3d(in_chans: int, embed_dim: int = 4) -> PatchEmbed:
    # Sizes are (H, W, D).
    return PatchEmbed(
        img_size=(16, 8, 4),
        patch_size=(4, 2, 2),
        in_chans=in_chans,
        embed_dim=embed_dim,
    )


class TestPatchEmbedAdjustInputChannelsHook:
    @pytest.mark.parametrize("dim", [2, 3])
    def test__truncate(self, dim: int) -> None:
        module = _patch_embed_3d(in_chans=1) if dim == 3 else _Conv2dPatchEmbed(1)
        weight = torch.randn(4, 3, 2, 2, 2) if dim == 3 else torch.randn(4, 3, 2, 2)
        state_dict = {"proj.weight": weight}

        _model_helpers.patch_embed_adjust_input_channels_hook(module, state_dict, "")

        assert_close(state_dict["proj.weight"], weight[:, :1])

    @pytest.mark.parametrize("dim", [2, 3])
    def test__repeat(self, dim: int) -> None:
        module = _patch_embed_3d(in_chans=3) if dim == 3 else _Conv2dPatchEmbed(3)
        weight = torch.randn(4, 1, 2, 2, 2) if dim == 3 else torch.randn(4, 1, 2, 2)
        state_dict = {"proj.weight": weight}

        _model_helpers.patch_embed_adjust_input_channels_hook(module, state_dict, "")

        assert state_dict["proj.weight"].shape[1] == 3
        for channel in range(3):
            assert_close(state_dict["proj.weight"][:, channel : channel + 1], weight)

    def test__repeat_with_remainder(self) -> None:
        module = _patch_embed_3d(in_chans=5)
        weight = torch.randn(4, 2, 2, 2, 2)
        state_dict = {"proj.weight": weight}

        _model_helpers.patch_embed_adjust_input_channels_hook(module, state_dict, "")

        assert_close(
            state_dict["proj.weight"], torch.cat([weight, weight, weight[:, :1]], dim=1)
        )

    def test__rank_mismatch_keeps_random_init(self) -> None:
        # A 2D checkpoint kernel cannot become a 3D patch embedding under any reshaping.
        module = _patch_embed_3d(in_chans=1)
        state_dict = {
            "proj.weight": torch.randn(4, 3, 14, 14),
            "proj.bias": torch.randn(4),
        }

        _model_helpers.patch_embed_adjust_input_channels_hook(module, state_dict, "")

        assert_close(state_dict["proj.weight"], module.proj.weight)
        assert_close(state_dict["proj.bias"], module.proj.bias)

    def test__ignores_missing_key(self) -> None:
        state_dict: dict[str, Any] = {"norm.weight": torch.randn(4)}

        _model_helpers.patch_embed_adjust_input_channels_hook(
            _patch_embed_3d(in_chans=1), state_dict, ""
        )

        assert list(state_dict) == ["norm.weight"]
