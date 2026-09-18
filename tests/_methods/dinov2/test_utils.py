#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
import random
import re

import numpy as np
import pytest
import torch

from lightly_train._methods.dinov2 import utils
from lightly_train._methods.dinov2.dinov2 import DINOv2AdamWViTArgs
from lightly_train._methods.dinov2.utils import (
    MaskingGenerator,
    create_collated_masks,
    is_tokenization_param,
)
from lightly_train._optim.trainable_modules import TrainableModules

from ... import helpers
from ...helpers import dummy_dinov2_vit_model


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(0)


class TestMaskingGenerator:
    def setup_method(self) -> None:
        # Cubic grid, the 3D analogue of the square grid used by the 2D tests.
        self.grid_size = 8
        self.num_patches = self.grid_size**3

    @pytest.mark.parametrize("grid_size", [14, 16])
    def test_get_shape_and_repr(self, grid_size: int) -> None:
        masking_generator = MaskingGenerator(
            input_size=(4, grid_size, grid_size),
            max_num_patches=int(0.5 * 4 * grid_size**2),
        )

        assert masking_generator.get_shape() == (4, grid_size, grid_size)
        assert masking_generator.num_patches == 4 * grid_size**2

        repr_str = repr(masking_generator)
        assert re.match(
            rf"Generator\(4,\s*{grid_size},\s*{grid_size}\s*->\s*\[\d+\s*~\s*\d+\],\s*max\s*=\s*[-\d\.]+\s*~\s*[-\d\.]+\)",
            repr_str,
        )

    def test_init__int_input_size(self) -> None:
        assert MaskingGenerator(input_size=5, max_num_patches=10).get_shape() == (
            5,
            5,
            5,
        )

    @pytest.mark.parametrize(
        [
            "n_masked_patch_tokens_min",
            "n_masked_patch_tokens_max",
            "masking_ratio",
        ],
        [
            (0, 0, 0.0),
            (0, 64, 0.0),
            (8, 8, 1.0),
            (8, 64, 0.125),
            (8, 64, 0.25),
            (8, 64, 0.5),
            (8, 64, 1.0),
        ],
    )
    def test_masking_generator_call(
        self,
        n_masked_patch_tokens_min: int,
        n_masked_patch_tokens_max: int,
        masking_ratio: float,
    ) -> None:
        n_masked_patch_tokens = int(masking_ratio * self.num_patches)

        masking_generator = MaskingGenerator(
            input_size=(self.grid_size,) * 3,
            min_num_patches=n_masked_patch_tokens_min,
            max_num_patches=n_masked_patch_tokens_max,
        )

        mask = masking_generator(n_masked_patch_tokens)

        assert mask.dtype == np.bool_
        assert mask.shape == (self.grid_size,) * 3
        assert n_masked_patch_tokens_min <= mask.sum() <= n_masked_patch_tokens

    @pytest.mark.parametrize(
        "aspect_ratio, n_masked_patch_tokens, is_masked",
        [
            # d = round((V / A^2)^(1/3)), h = w = round(d * A) on an 8x8x8 grid. The
            # cuboid is only accepted if 0 < d, h, w < 8 and d * h * w <= V.
            (1.0, 1, True),  # 1x1x1
            (1.0, 27, True),  # 3x3x3
            (1.0, 512, False),  # 8x8x8 does not fit strictly inside the grid
            (4.0, 16, True),  # 1x4x4
            (4.0, 2, False),  # d rounds to 0, or a 1x4x4 block exceeds the budget
            (0.5, 2, True),  # 2x1x1
            (0.5, 64, True),  # 6x3x3 = 54
            (2.0, 16, False),  # 2x4x4 = 32 exceeds the budget
        ],
    )
    def test_masking_generator__aspect_ratio_validity(
        self,
        aspect_ratio: float,
        n_masked_patch_tokens: int,
        is_masked: bool,
    ) -> None:
        masking_generator = MaskingGenerator(
            input_size=(self.grid_size,) * 3,
            min_num_patches=n_masked_patch_tokens,
            max_num_patches=n_masked_patch_tokens,
            min_aspect=aspect_ratio,
            max_aspect=aspect_ratio,
        )

        mask = masking_generator(n_masked_patch_tokens)
        assert mask.any() == is_masked

    @pytest.mark.parametrize("cube_size", [2, 3, 4])
    def test_masking_generator__aspect_ratio_cube(self, cube_size: int) -> None:
        """With aspect ratio 1.0 and num_mask=min_num_masks_per_block we expect a single, cubic masked block."""
        masking_generator = MaskingGenerator(
            input_size=(self.grid_size,) * 3,
            max_num_patches=cube_size**3,
            min_num_patches=cube_size**3,
            min_aspect=1.0,
            max_aspect=1.0,
        )

        mask = masking_generator(cube_size**3)
        assert mask.sum() == cube_size**3
        coords = np.argwhere(mask)
        extent = coords.max(axis=0) - coords.min(axis=0) + 1
        assert tuple(extent) == (cube_size,) * 3

    def test_masking_generator__cuboid_anchored_to_grid(self) -> None:
        """Without aspect ratio jitter the block has the aspect ratio of the grid."""
        masking_generator = MaskingGenerator(
            input_size=(2, 8, 8),
            max_num_patches=16,
            min_num_patches=16,
            min_aspect=1.0,
            max_aspect=1.0,
        )
        mask = masking_generator(16)
        coords = np.argwhere(mask)
        extent = coords.max(axis=0) - coords.min(axis=0) + 1
        assert tuple(extent) == (1, 4, 4)

    @pytest.mark.parametrize("grid", [(4, 16, 16), (6, 16, 16), (8, 8, 8)])
    @pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5])
    def test_masking_generator__achieves_target_ratio(
        self, grid: tuple[int, int, int], ratio: float
    ) -> None:
        """Regression test: on anisotropic grids such as the default 4x16x16 grid of
        the global views, unanchored aspect ratios produced masks far below the
        requested ratio and many empty masks."""
        num_patches = math.prod(grid)
        masking_generator = MaskingGenerator(
            input_size=grid, max_num_patches=int(0.5 * num_patches)
        )
        ratios = np.array(
            [
                masking_generator(int(ratio * num_patches)).sum() / num_patches
                for _ in range(100)
            ]
        )
        assert ratios.mean() >= 0.9 * ratio
        assert (ratios == 0).mean() <= 0.05


class TestCreateCollatedMasks:
    def setup_method(self) -> None:
        self.grid = (4, 16, 16)
        self.num_patches = math.prod(self.grid)
        self.masking_generator = MaskingGenerator(
            input_size=self.grid,
            max_num_patches=int(0.5 * self.num_patches),
        )

    @pytest.mark.parametrize("expected_n_crops", [1, 2, 4, 8])
    def test_create_collated_masks__dtype_output_size(
        self, expected_n_crops: int
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=0.1,
            mask_ratio_max=0.5,
            n_masked_crops=min(2, expected_n_crops),
            n_crops=expected_n_crops,
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]
        assert collated_masks.dtype == torch.bool
        assert collated_masks.shape == (expected_n_crops, self.num_patches)
        assert masks["mask_indices_list"].shape == masks["masks_weight"].shape

    @pytest.mark.parametrize("expected_n_masked_crops", [0, 1, 2, 3, 4])
    def test_create_collated_masks__n_masked_crops(
        self, expected_n_masked_crops: int
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=0.1,
            mask_ratio_max=0.5,
            n_masked_crops=expected_n_masked_crops,
            n_crops=max(4, expected_n_masked_crops),
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]

        n_masked_crops = sum(m.sum() > 0 for m in collated_masks)
        assert n_masked_crops == expected_n_masked_crops

    @pytest.mark.parametrize(
        "mask_ratio_min, mask_ratio_max",
        [(0.1, 0.5), (0.5, 0.8), (1.0, 1.0)],
    )
    def test_create_collated_masks__mask_ratio_min_max(
        self, mask_ratio_min: float, mask_ratio_max: float
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            n_masked_crops=2,
            n_crops=4,
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]
        for mask in collated_masks:
            n_patch_tokens = mask.numel()
            n_masked_patch_tokens = mask.sum().item()
            if n_masked_patch_tokens == 0:
                continue

            # Divide lower bound by 4 because the bound is not strict as fewer patches than
            # min_image_mask_ratio * num_patches can be masked. This is because there is a
            # limited number of attempts to find a valid mask that satisfies all constraints.
            assert (
                mask_ratio_min * n_patch_tokens / 4
                <= n_masked_patch_tokens
                <= mask_ratio_max * n_patch_tokens
            )

    def test_create_collated_masks__mask_ratio_zero(
        self,
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=0.0,
            mask_ratio_max=0.0,
            n_masked_crops=4,
            n_crops=4,
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]
        for mask in collated_masks:
            assert not mask.any()


def test_get_optimizer_with_decay() -> None:
    dinov2 = helpers.get_method_dinov2()
    trainable_modules = dinov2.trainable_modules()
    optim = utils.get_optimizer_with_decay(
        optim_args=DINOv2AdamWViTArgs(),
        trainable_modules=trainable_modules,
        layerwise_decay=dinov2.method_args.layerwise_decay,
        patch_embed_lr_multiplier=dinov2.method_args.patch_embed_lr_multiplier,
    )

    # Map fused params back to their original names.
    param_to_name = {}
    for module in list(trainable_modules.modules) + list(
        trainable_modules.modules_no_weight_decay
    ):
        param_to_name.update({p: n for n, p in module.named_parameters()})
    groups = []
    for group in optim.param_groups:
        groups.append({param_to_name[p] for p in group["params"]})

    # Hardcoded to make 100% sure that the groups are correct. If something fails here
    # then there is probably an issue in the grouping logic or the way we set lr, wd, or
    # other parameters for the different parameters.
    expected_groups = [
        {"cls_token", "pos_embed", "mask_token"},
        {"patch_embed.proj.weight"},
        {"patch_embed.proj.bias"},
        {
            "blocks.0.norm1.weight",
            "blocks.0.norm1.bias",
            "blocks.0.attn.qkv.bias",
            "blocks.0.attn.proj.bias",
            "blocks.0.ls1.gamma",
            "blocks.0.norm2.weight",
            "blocks.0.norm2.bias",
            "blocks.0.mlp.fc1.bias",
            "blocks.0.mlp.fc2.bias",
            "blocks.0.ls2.gamma",
        },
        {
            "blocks.0.attn.qkv.weight",
            "blocks.0.attn.proj.weight",
            "blocks.0.mlp.fc1.weight",
            "blocks.0.mlp.fc2.weight",
        },
        {
            "blocks.1.norm1.weight",
            "blocks.1.norm1.bias",
            "blocks.1.attn.qkv.bias",
            "blocks.1.attn.proj.bias",
            "blocks.1.ls1.gamma",
            "blocks.1.norm2.weight",
            "blocks.1.norm2.bias",
            "blocks.1.mlp.fc1.bias",
            "blocks.1.mlp.fc2.bias",
            "blocks.1.ls2.gamma",
        },
        {
            "blocks.1.attn.qkv.weight",
            "blocks.1.attn.proj.weight",
            "blocks.1.mlp.fc1.weight",
            "blocks.1.mlp.fc2.weight",
        },
        {
            "blocks.2.norm1.weight",
            "blocks.2.norm1.bias",
            "blocks.2.attn.qkv.bias",
            "blocks.2.attn.proj.bias",
            "blocks.2.ls1.gamma",
            "blocks.2.norm2.weight",
            "blocks.2.norm2.bias",
            "blocks.2.mlp.fc1.bias",
            "blocks.2.mlp.fc2.bias",
            "blocks.2.ls2.gamma",
        },
        {
            "blocks.2.attn.qkv.weight",
            "blocks.2.attn.proj.weight",
            "blocks.2.mlp.fc1.weight",
            "blocks.2.mlp.fc2.weight",
        },
        {
            "norm.weight",
            "norm.bias",
        },
        {
            "dino_head.mlp.0.weight",
            "dino_head.mlp.2.weight",
            "dino_head.mlp.4.weight",
        },
        {
            "dino_head.mlp.0.bias",
            "dino_head.mlp.2.bias",
            "dino_head.mlp.4.bias",
        },
        {
            "dino_head.last_layer.parametrizations.weight.original0",
            "dino_head.last_layer.parametrizations.weight.original1",
        },
    ]
    assert groups == expected_groups


@pytest.mark.parametrize(
    "name, expected",
    [
        ("pos_embed", True),
        ("pos_embed_grid", True),
        ("cls_token", True),
        ("mask_token", True),
        ("register_tokens", True),
        ("patch_embed.proj.weight", True),
        ("patch_embed.proj.bias", True),
        ("blocks.0.2.attn.qkv.weight", False),
        ("blocks.5.mlp.fc1.weight", False),
        ("norm.weight", False),
        ("dino_head.mlp.0.weight", False),
    ],
)
def test_is_tokenization_param(name: str, expected: bool) -> None:
    assert is_tokenization_param(name) is expected


@pytest.mark.parametrize("layerwise_decay", [0.9, 1.0])
def test_get_fused_param_groups__never_mixes_tokenization(
    layerwise_decay: float,
) -> None:
    """Regression test: with layerwise_decay=1.0 every layer gets the same lr, so
    without the "tokenization" fusion key, pos_embed/cls_token/mask_token fuse into
    the same group as decayed block weights (verified: 4 groups instead of 10, with
    12 block weight tensors sharing a group with the tokenization). A name-based
    tokenization-only freeze in DINOv2.on_before_optimizer_step is only correct if no
    fused group mixes tokenization and non-tokenization parameters."""
    vit_wrapper = dummy_dinov2_vit_model()
    vit = vit_wrapper.get_model()
    id_to_name = {id(p): n for n, p in vit.named_parameters()}

    optim = utils.get_optimizer_with_decay(
        optim_args=DINOv2AdamWViTArgs(),
        trainable_modules=TrainableModules(modules=[vit]),
        layerwise_decay=layerwise_decay,
        patch_embed_lr_multiplier=0.2,
    )

    for group in optim.param_groups:
        names = [id_to_name[id(p)] for p in group["params"]]
        kinds = {is_tokenization_param(n) for n in names}
        assert len(kinds) == 1, (
            f"Group '{group['name']}' mixes tokenization and non-tokenization "
            f"params: {names}"
        )
