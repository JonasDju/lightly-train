#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from math import prod
from typing import Any, Sequence, cast

import torch
from torch import Tensor
from torch.nn import Module

logger = logging.getLogger(__name__)

# Parameters of the DINOv2 ViT that are specific to how a volume is turned into a token
# sequence. None of them can be carried over from a 2D checkpoint: the patch embedding
# is a Conv2d there, and the positional embedding describes a 2D grid. They are left at
# their random initialization when a 2D checkpoint is loaded into the 3D model. The
# patch embedding itself is handled one level down, by
# ``patch_embed_adjust_input_channels_hook``.
_TOKENIZATION_KEYS = (
    "pos_embed",
    "pos_embed_grid",
    "cls_token",
    "mask_token",
    "register_tokens",
)


def _checkpoint_is_2d(state_dict: dict[str, Any], prefix: str) -> bool:
    """Whether ``state_dict`` holds an original 2D DINOv2 checkpoint.

    Decided by the rank of the patch embedding kernel (4D for ``Conv2d``, 5D for
    ``Conv3d``) rather than by token counts, which are ambiguous.
    """
    proj_weight = state_dict.get(f"{prefix}patch_embed.proj.weight")
    return proj_weight is not None and proj_weight.dim() == 4


def patch_embed_adjust_input_channels_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Hook to reconcile a checkpoint's patch embedding with the module's own.

    Two cases:

    - The checkpoint's kernel has a different rank than the module's, i.e. a 2D
      (``Conv2d``) checkpoint is being loaded into the 3D (``Conv3d``) patch embedding.
      A 2D kernel is not a 3D patch embedding under any reshaping, so the module keeps
      its random initialization and the checkpoint's patch embedding is discarded.
    - Same rank, different number of input channels: the channels are cut or repeated,
      as before. Works for both ``Conv2d`` and ``Conv3d`` weights.

    Mutates ``state_dict`` in place.
    """
    proj_weight_key = f"{prefix}proj.weight"
    proj_weight = state_dict.get(proj_weight_key)
    if proj_weight is None:
        return

    proj = cast(Module, module.proj)
    target_weight = cast(Tensor, proj.weight)
    if proj_weight.dim() != target_weight.dim():
        logger.info(
            f"Loading pretrained weights with a {proj_weight.dim() - 2}D patch "
            f"embedding kernel into a {target_weight.dim() - 2}D one. The patch "
            "embedding cannot be converted and is left randomly initialized."
        )
        state_dict[proj_weight_key] = target_weight.detach().clone()
        target_bias = cast("Tensor | None", proj.bias)
        proj_bias_key = f"{prefix}proj.bias"
        if target_bias is None:
            state_dict.pop(proj_bias_key, None)
        else:
            state_dict[proj_bias_key] = target_bias.detach().clone()
        return

    in_chans = cast(int, module.in_chans)
    weights_in_chans = proj_weight.shape[1]
    if weights_in_chans > in_chans:
        # Drop last channels
        logger.info(
            f"Loading pretrained weights with {weights_in_chans} input channels, "
            f"but model has {in_chans} input channels. Keeping only the "
            f"first {in_chans} channels of the pretrained weights."
        )
        proj_weight = proj_weight[:, :in_chans]
    elif weights_in_chans < in_chans:
        # Repeat channels to initialize extra channels
        logger.info(
            f"Loading pretrained weights with {weights_in_chans} input channels, "
            f"but model has {in_chans} input channels. Repeating the "
            "channels of the pretrained weights to initialize the extra "
            "channels."
        )
        repeat_times = in_chans // weights_in_chans
        remainder = in_chans % weights_in_chans
        repeats = [1] * proj_weight.dim()
        repeats[1] = repeat_times
        proj_weight = proj_weight.repeat(*repeats)
        if remainder > 0:
            proj_weight = torch.cat([proj_weight, proj_weight[:, :remainder]], dim=1)
    state_dict[proj_weight_key] = proj_weight


def interpolate_pos_embed_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Make a mismatched DINOv2 ViT checkpoint loadable into this 3D model.

    Handles two cases, distinguished by the rank of the checkpoint's patch embedding
    kernel:

    - **3D -> 3D, different image size** (the high-resolution adaptation phase, e.g. a
      ``(224, 224, 16)`` checkpoint into a ``(280, 280, 20)`` model): ``pos_embed`` is
      trilinearly resized from the checkpoint's patch grid to this model's. The source
      grid is read from the checkpoint's own ``pos_embed_grid`` buffer and is never
      inferred: a flat token count does not determine a ``(D, H, W)`` factorization, and
      guessing one silently corrupts the embedding. A checkpoint that does not carry the
      buffer is rejected with a ``RuntimeError``.
    - **2D -> 3D** (an original DINOv2 checkpoint, to initialize the transformer blocks
      from public weights): everything specific to tokenizing a volume --
      ``pos_embed``, ``cls_token``, ``mask_token``, ``register_tokens`` and, via
      ``patch_embed_adjust_input_channels_hook``, the patch embedding -- keeps this
      model's random initialization. Only ``blocks.*`` and ``norm.*``, whose
      architecture is unchanged between 2D and 3D DINOv2, are taken from the checkpoint.

    In both cases ``state_dict`` is rewritten so that a ``strict=True`` load sees
    neither a missing nor an unexpected key. Mutates ``state_dict`` in place.
    """
    key = f"{prefix}pos_embed"
    value = state_dict.get(key)
    if value is None:
        return

    if _checkpoint_is_2d(state_dict, prefix):
        left_random = []
        for name in _TOKENIZATION_KEYS:
            tensor = getattr(module, name, None)
            if tensor is None:
                # The model has no such parameter (e.g. a checkpoint with register
                # tokens loaded into a model without them). Drop the checkpoint's.
                state_dict.pop(f"{prefix}{name}", None)
            else:
                state_dict[f"{prefix}{name}"] = tensor.detach().clone()
                left_random.append(name)
        logger.info(
            "Loading a 2D DINOv2 checkpoint into a 3D model: only the transformer "
            "blocks and the final norm are restored. Left randomly initialized: "
            f"{', '.join(left_random)}, patch_embed."
        )
        return

    grid_key = f"{prefix}pos_embed_grid"
    source_grid_value = state_dict.get(grid_key)
    if source_grid_value is None:
        raise RuntimeError(
            f"Cannot load '{key}' with shape {tuple(value.shape)}: the checkpoint does "
            f"not record the patch grid its positional embedding was trained on "
            f"('{grid_key}' is missing)."
        )
    source_grid = tuple(int(size) for size in source_grid_value)
    patch_embed = cast(Module, module.patch_embed)
    target_grid = tuple(cast("Sequence[int]", patch_embed.patches_resolution))
    # The buffer must always end up describing this model's grid, not the checkpoint's.
    state_dict[grid_key] = cast(Tensor, module.pos_embed_grid).detach().clone()

    n_source = value.shape[1] - 1
    if n_source != prod(source_grid):
        raise RuntimeError(
            f"Inconsistent checkpoint: '{key}' has {n_source} patch positions but "
            f"'{grid_key}' describes a {source_grid} grid with {prod(source_grid)} "
            "patches."
        )
    if source_grid == target_grid:
        return

    target = cast(Tensor, module.pos_embed)
    if value.shape[-1] != target.shape[-1]:
        # Different embedding dimension; nothing sensible to do. Let torch report it.
        return

    dim = value.shape[-1]
    cls_pos_embed = value[:, :1]
    # (1, N, C) -> (1, C, D, H, W) -> resize -> (1, N', C)
    patch_pos_embed = (
        value[:, 1:].reshape(1, *source_grid, dim).permute(0, 4, 1, 2, 3).float()
    )
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        size=target_grid,
        mode="trilinear",
        align_corners=False,
        # Note: antialias must never be passed here. PyTorch only supports it for
        # bilinear/bicubic/lanczos, and several DINOv2 configs set
        # interpolate_antialias: true as a 2D-era default.
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).reshape(
        1, prod(target_grid), dim
    )
    state_dict[key] = torch.cat([cls_pos_embed, patch_pos_embed], dim=1).to(value.dtype)
    logger.info(
        f"Interpolated '{key}' from a {source_grid} to a {target_grid} (D, H, W) patch "
        f"grid: {tuple(value.shape)} -> {tuple(state_dict[key].shape)}."
    )
