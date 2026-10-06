#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#

# References:
#   - https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/dino_clstoken_loss.py
#   - https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py
#
# Modifications Copyright (c) Lightly AG and affiliates:
#   - Import xFormers' cross entropy only if XFORMERS_ENABLED is True
#   - Use dist.is_initialized() to control the all_reduce operation of B in distributed setting
#     in the IBOTPatchLoss' sinkhorn_knopp_teacher
#   - Rename iBOTPatchLoss to IBOTPatchLoss
#   - Add type hints to the functions
#   - Remove dead code
#   - Add TODO for investigating the casting of self.center in IBOTPatchLoss
#   - Add ibot_chunk_loss and chunked_ibot_loss (iBOT loss from bottleneck features, in token chunks)

from __future__ import annotations

import os
from typing import Callable, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import cross_entropy  # type: ignore[import]

        def lossfunc(t: Tensor, s: Tensor, temp: float):  # type: ignore[no-untyped-def]
            s = s.float()
            t = t.float()
            if s.ndim == 2:
                return -cross_entropy(
                    s.unsqueeze(0), t.unsqueeze(0), temp, bw_inplace=True
                ).squeeze(0)
            elif s.ndim == 3:
                return -cross_entropy(s, t, temp, bw_inplace=True)
            else:
                raise ValueError(
                    f"Invalid tensor shape: {s.shape}. Expected 2D or 3D tensor."
                )

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:

    def lossfunc(t: Tensor, s: Tensor, temp: float):  # type: ignore[no-untyped-def]
        return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)

    XFORMERS_AVAILABLE = False


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center: torch.Tensor  # Type hint for self.center
        self.updated = True
        self.reduce_handle = None

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_output: Tensor, teacher_temp: float
    ) -> Tensor:
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, teacher_output: Tensor, teacher_temp: float, n_iterations: int = 3
    ) -> Tensor:
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(
        self,
        student_output_list: Tuple[Tensor, ...] | List[Tensor],
        teacher_out_softmaxed_centered_list: Tensor | List[Tensor],
    ) -> Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        # torch.zeros, not torch.tensor(0.0, device=...): that copies from pageable host memory, which waits for all
        # queued GPU work and so stops the CPU from running ahead of the GPU. float32 as before, not the (possibly
        # bf16) dtype of the student output.
        total_loss: Tensor = torch.zeros((), device=student_output_list[0].device)
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output: Tensor) -> None:
        self.updated = False
        self.len_teacher_output = teacher_output.shape[0]
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self) -> None:
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (
                1 - self.center_momentum
            )

            self.updated = True


class IBOTPatchLoss(nn.Module):
    def __init__(
        self,
        patch_out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.center: torch.Tensor  # Type hint for self.center
        self.updated = True
        self.reduce_handle = None

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_patch_tokens: Tensor, teacher_temp: float
    ) -> Tensor:
        self.apply_center_update()

        # TODO: self.center uses float32 which might cause unnecessary upcasting in fp16 settings which could slow down training
        # we need to investigate how we should handle the casting in this case
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self,
        teacher_output: Tensor,
        teacher_temp: float,
        n_masked_patches_tensor: Tensor,
        n_iterations: int = 3,
    ) -> Tensor:
        teacher_output = teacher_output.float()
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = n_masked_patches_tensor
        if dist.is_initialized():
            dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(
        self,
        student_patch_tokens: Tensor,
        teacher_patch_tokens: Tensor,
        student_masks_flat: Tensor,
    ) -> Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(
            loss * student_masks_flat.float(), dim=-1
        ) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked: Tensor,
        teacher_patch_tokens_masked: Tensor,
        student_masks_flat: Tensor,
        n_masked_patches: int | None = None,
        masks_weight: Tensor | None = None,
    ) -> Tensor:
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss: Tensor = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight

        B: int = student_masks_flat.shape[0]
        return -loss.sum() / B

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens: Tensor) -> None:
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens: Tensor) -> None:
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(
            teacher_patch_tokens.mean(1), dim=0, keepdim=True
        )
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self) -> None:
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (
                1 - self.center_momentum
            )

            self.updated = True


def ibot_chunk_loss(
    student_bottleneck: Tensor,
    teacher_bottleneck: Tensor,
    student_weight: Tensor,
    teacher_weight: Tensor,
    center: Tensor,
    teacher_temp: float,
    student_temp: float,
    token_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """The iBOT loss of a chunk of tokens, from the heads' bottleneck features.

    Computes what the unchunked path computes with the heads' last layers,
    IBOTPatchLoss.softmax_center_teacher and IBOTPatchLoss.forward_masked (with the
    non-xFormers lossfunc), with the same operations in the same order.

    Args:
        student_bottleneck: [C, d] student bottleneck features of the chunk's tokens.
        teacher_bottleneck: [C, d] teacher bottleneck features of the same tokens.
        student_weight: [K, d] weight of the student's iBOT last layer.
        teacher_weight: [K, d] weight of the teacher's iBOT last layer.
        center: [1, 1, K] iBOT center (softmax centering).
        token_weights: [C] loss weight of every token.

    Returns:
        (weighted sum of the per-token cross-entropies, not yet negated or divided by
        the number of crops, [K] sum of the teacher logits over the chunk's tokens
        for the center update).
    """
    with torch.no_grad():
        teacher_logits = F.linear(teacher_bottleneck, teacher_weight)  # [C, K]
        teacher_probs = F.softmax(
            (teacher_logits - center.reshape(1, -1)) / teacher_temp, dim=-1
        )  # [C, K]
        teacher_logits_sum = teacher_logits.sum(
            dim=0, dtype=torch.promote_types(teacher_logits.dtype, torch.float32)
        )  # [K]
    student_logits = F.linear(student_bottleneck, student_weight)  # [C, K]
    loss = torch.sum(
        teacher_probs * F.log_softmax(student_logits / student_temp, dim=-1), dim=-1
    )  # [C]
    return torch.sum(loss * token_weights), teacher_logits_sum


def chunked_ibot_loss(
    student_bottleneck: Tensor,
    teacher_bottleneck: Tensor,
    student_weight: Tensor,
    teacher_weight: Tensor,
    center: Tensor,
    teacher_temp: float,
    student_temp: float,
    token_weights: Tensor,
    n_crops: int,
    chunk_size: int | None,
    checkpoint: bool,
    chunk_fn: Callable[..., tuple[Tensor, Tensor]] = ibot_chunk_loss,
) -> tuple[Tensor, Tensor | None]:
    """The iBOT loss over T tokens, computed in chunks of chunk_size tokens.

    Equals IBOTPatchLoss.forward_masked on the full [T, K] teacher probabilities and
    student logits, but only one chunk's [chunk_size, K] tensors exist at a time when
    checkpoint is True: every chunk then runs under activation checkpointing, so only
    its bottleneck inputs are kept for the backward pass and its logits, teacher
    probabilities and log-softmax are recomputed there.

    Args:
        student_bottleneck: [T, d], see ibot_chunk_loss.
        teacher_bottleneck: [T, d].
        student_weight: [K, d].
        teacher_weight: [K, d].
        center: [1, 1, K].
        token_weights: [T].
        n_crops: Number of crops the loss is averaged over.
        chunk_size: Tokens per chunk. None computes all tokens in one chunk.
        checkpoint: Whether to run every chunk under activation checkpointing.
        chunk_fn: ibot_chunk_loss, or a compiled version of it.

    Returns:
        (loss, [1, 1, K] mean teacher logits for IBOTPatchLoss.update_center, or None
        if there are no tokens).
    """
    n_tokens = student_bottleneck.shape[0]
    if chunk_size is None:
        chunk_size = max(n_tokens, 1)
    device_type = teacher_weight.device.type
    if torch.is_autocast_enabled(device_type):
        # F.linear would cast the [K, d] teacher weight to the autocast dtype in every chunk, and again when a
        # chunk is recomputed (it is no leaf, so autocast does not cache the cast). It has no gradient: cast it
        # once, the same rounding. The student weight is still cast per chunk, so that the per-chunk gradients
        # are summed in its own dtype (one bf16 weight would sum them in bf16).
        teacher_weight = teacher_weight.to(torch.get_autocast_dtype(device_type))
    loss_sum: Tensor | None = None
    teacher_logits_sum: Tensor | None = None
    # One (empty) chunk without tokens, so that the loss is still part of the graph.
    for start in range(0, max(n_tokens, 1), chunk_size):
        end = start + chunk_size
        args = (
            student_bottleneck[start:end],
            teacher_bottleneck[start:end],
            student_weight,
            teacher_weight,
            center,
            teacher_temp,
            student_temp,
            token_weights[start:end],
        )
        if checkpoint:
            # The chunk draws no random numbers: no need to stash and restore the RNG states.
            chunk_loss, chunk_logits_sum = torch.utils.checkpoint.checkpoint(
                chunk_fn, *args, use_reentrant=False, preserve_rng_state=False
            )
        else:
            chunk_loss, chunk_logits_sum = chunk_fn(*args)
        if loss_sum is None or teacher_logits_sum is None:
            loss_sum, teacher_logits_sum = chunk_loss, chunk_logits_sum
        else:
            loss_sum = loss_sum + chunk_loss
            teacher_logits_sum = teacher_logits_sum + chunk_logits_sum
    assert loss_sum is not None and teacher_logits_sum is not None
    loss = -loss_sum / n_crops
    if n_tokens == 0:
        return loss, None
    return loss, (teacher_logits_sum / n_tokens).view(1, 1, -1)
