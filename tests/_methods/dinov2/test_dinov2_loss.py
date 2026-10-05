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
import torch.distributed as dist
from _pytest.monkeypatch import MonkeyPatch

from lightly_train._methods.dinov2.dinov2_head import DINOv2ProjectionHead
from lightly_train._methods.dinov2.dinov2_loss import (
    DINOLoss,
    IBOTPatchLoss,
    chunked_ibot_loss,
    ibot_chunk_loss,
)
from lightly_train._methods.dinov2.utils import (
    ibotpp_loss_mask,
    mask_indices_and_weights,
)


@pytest.fixture
def no_dist(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    return


@pytest.mark.usefixtures("no_dist")
class TestDINOLoss:
    def test_softmax_center_teacher(self) -> None:
        """Test that the softmax_center_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """
        batch_size = 4
        out_dim = 2

        dino_loss = DINOLoss(out_dim=out_dim, student_temp=0.1, center_momentum=0.9)

        teacher_output = torch.randn(batch_size, out_dim)
        softmax = dino_loss.softmax_center_teacher(teacher_output, teacher_temp=0.04)

        sums = softmax.sum(dim=-1)

        assert torch.allclose(sums, torch.ones(batch_size))

    def test_sinkhorn_knopp_teacher(self) -> None:
        """Test that the sinkhorn_knopp_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """
        batch_size = 4
        out_dim = 2

        dino_loss = DINOLoss(out_dim=out_dim, student_temp=0.1, center_momentum=0.9)

        teacher_output = torch.randn(batch_size, out_dim)
        Q = dino_loss.sinkhorn_knopp_teacher(
            teacher_output, teacher_temp=0.04, n_iterations=4
        )

        # Q shape = [B, K]
        assert Q.shape == (batch_size, out_dim)

        # row sums ≈ 1
        row_sums = Q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(batch_size))

    def test_update_center_momentum(self) -> None:
        """Test that the update_center method updates the center
        correctly with the given momentum.
        """
        batch_size = 4
        out_dim = 2
        center_momentum = 0.9
        mean = 2

        dino_loss = DINOLoss(
            out_dim=out_dim, student_temp=0.1, center_momentum=center_momentum
        )

        # call update & apply on a known tensor
        teacher_output = torch.ones(batch_size, out_dim) * mean
        dino_loss.update_center(teacher_output)
        dino_loss.apply_center_update()

        expected_center = mean * (1 - center_momentum) * torch.ones(out_dim)
        assert torch.allclose(dino_loss.center, expected_center)

    def test_forward(self) -> None:
        """Test that the forward method returns correct values"""
        out_dim = 2
        teacher_temp = 0.04

        dino_loss = DINOLoss(out_dim=out_dim, student_temp=0.1, center_momentum=0.9)

        teacher_output = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        student_output = [
            torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]) for _ in range(2)
        ]

        teacher_softmaxed = dino_loss.softmax_center_teacher(
            teacher_output, teacher_temp=teacher_temp
        )
        dino_loss.update_center(teacher_output)

        loss = dino_loss.forward(student_output, [teacher_softmaxed, teacher_softmaxed])

        assert loss == pytest.approx(1.5565, rel=0.0001)


@pytest.mark.usefixtures("no_dist")
class TestIBotPatchLoss:
    def test_softmax_center_teacher(self) -> None:
        """Test that the softmax_center_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """
        batch_size = 4
        patch_out_dim = 2

        ibot_loss = IBOTPatchLoss(
            patch_out_dim=patch_out_dim,
            student_temp=0.1,
            center_momentum=0.9,
        )

        teacher_output = torch.randn(batch_size, patch_out_dim)
        softmax = ibot_loss.softmax_center_teacher(teacher_output, teacher_temp=0.04)

        sums = softmax.sum(dim=-1)

        assert torch.allclose(sums, torch.ones(batch_size))

    def test_sinkhorn_knopp_teacher(self) -> None:
        """Test that the sinkhorn_knopp_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """
        batch_size = 4
        patch_out_dim = 2

        ibot_loss = IBOTPatchLoss(
            patch_out_dim=patch_out_dim,
            student_temp=0.1,
            center_momentum=0.9,
        )

        teacher_output = torch.randn(batch_size, patch_out_dim)
        n_masked_patches_tensor = torch.randint(low=1, high=batch_size, size=(1,))
        Q = ibot_loss.sinkhorn_knopp_teacher(
            teacher_output,
            teacher_temp=0.04,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )

        # Q shape = [B, K]
        assert Q.shape == (batch_size, patch_out_dim)

        # row sums ≈ 1
        row_sums = Q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(batch_size))

    def test_update_center_momentum(self) -> None:
        """Test that the update_center method updates the center
        correctly with the given momentum.
        """
        batch_size = 4
        patch_out_dim = 2
        center_momentum = 0.9
        mean = 2

        ibot_loss = IBOTPatchLoss(
            patch_out_dim=patch_out_dim,
            student_temp=0.1,
            center_momentum=center_momentum,
        )

        # call update & apply on a known tensor
        teacher_output = torch.ones(batch_size, patch_out_dim) * mean
        ibot_loss.update_center(teacher_output)
        ibot_loss.apply_center_update()

        expected_center = mean * (1 - center_momentum) * torch.ones(patch_out_dim)
        assert torch.allclose(ibot_loss.center, expected_center)

    def test_forward_masked(self) -> None:
        """Test that the forward method returns correct values"""
        out_dim = 2
        teacher_temp = 0.1

        ibot_loss = IBOTPatchLoss(
            patch_out_dim=out_dim,
            student_temp=0.2,
            center_momentum=0.9,
        )

        masked_teacher_output = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        masked_student_output = torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])

        mask = torch.tensor(
            [
                [True, False, True, False],
                [False, False, False, True],
                [False, False, False, False],
            ]
        )

        masked_teacher_softmaxed = ibot_loss.softmax_center_teacher(
            masked_teacher_output.unsqueeze(0), teacher_temp=teacher_temp
        )
        ibot_loss.update_center(masked_teacher_output.unsqueeze(0))

        loss = ibot_loss.forward_masked(
            teacher_patch_tokens_masked=masked_teacher_softmaxed,
            student_patch_tokens_masked=masked_student_output,
            student_masks_flat=mask,
        )
        assert loss == pytest.approx(0.4057, rel=0.0001)


# The chunked iBOT loss (compile_heads / ibot_loss_chunk_size) against the unchunked
# pipeline it replaces: last layers, IBOTPatchLoss.softmax_center_teacher,
# forward_masked and the center update. Small sizes, but K (the output dim) differs
# from every other size so that [*, K] tensors can be told apart.
_T_CROPS, _N_PATCHES, _D, _K = 5, 8, 6, 32
_N_TOKENS = _T_CROPS * _N_PATCHES


def _token_mask(mode: str) -> torch.Tensor:
    """The [crops, patches] token mask of the loss, as the method builds it."""
    g = torch.Generator().manual_seed(0)
    collated = torch.rand(_T_CROPS, _N_PATCHES, generator=g) < 0.4
    collated[1] = False  # an unmasked crop
    collated[3, :2] = True  # every masked crop has at least two masked tokens
    if mode == "ibot":
        return collated
    return ibotpp_loss_mask(collated, mode=mode)  # type: ignore[arg-type]


class _Setup:
    """Heads' last layers, bottleneck features, a nonzero center and token weights."""

    def __init__(self, mode: str, dtype: torch.dtype) -> None:
        torch.manual_seed(0)
        self.student_head = DINOv2ProjectionHead(
            in_dim=4, out_dim=_K, hidden_dim=8, bottleneck_dim=_D
        ).to(dtype)
        self.teacher_head = DINOv2ProjectionHead(
            in_dim=4, out_dim=_K, hidden_dim=8, bottleneck_dim=_D
        ).to(dtype)
        for head in (self.student_head, self.teacher_head):
            # Not the init values (g = 1), so that weight norm does something.
            g = head.last_layer.parametrizations.weight.original0
            g.data.uniform_(0.5, 1.5)
        self.teacher_head.requires_grad_(False)
        self.mask = _token_mask(mode)
        self.n_tokens = int(self.mask.sum())
        self.token_weights = mask_indices_and_weights(self.mask)[1]
        self.student_bottleneck = torch.randn(self.n_tokens, _D, dtype=dtype)
        self.teacher_bottleneck = torch.randn(self.n_tokens, _D, dtype=dtype)
        self.center = torch.randn(1, 1, _K, dtype=dtype)
        self.teacher_temp = 0.05
        self.dtype = dtype

    def loss_module(self) -> IBOTPatchLoss:
        loss = IBOTPatchLoss(patch_out_dim=_K, student_temp=0.1, center_momentum=0.9)
        loss.center = self.center.clone()
        # A pending update from the previous step, which both pipelines must apply first.
        loss.update_center(self.center.flip(-1))
        return loss

    def student_leaves(self) -> dict[str, torch.Tensor]:
        bottleneck = self.student_bottleneck.clone().requires_grad_(True)
        weight = self.student_head.last_layer.parametrizations.weight
        for p in (weight.original0, weight.original1):
            p.grad = None
        return {
            "bottleneck": bottleneck,
            "g": weight.original0,
            "v": weight.original1,
        }

    def legacy(self) -> dict[str, torch.Tensor]:
        """What DINOv2 computes without compile_heads and ibot_loss_chunk_size."""
        loss_module = self.loss_module()
        leaves = self.student_leaves()
        teacher_logits = self.teacher_head.last_layer(self.teacher_bottleneck)
        teacher_logits = teacher_logits.unsqueeze(0)
        teacher_probs = loss_module.softmax_center_teacher(
            teacher_logits, teacher_temp=self.teacher_temp
        ).squeeze(0)
        loss_module.update_center(teacher_logits)
        student_logits = self.student_head.last_layer(leaves["bottleneck"])
        loss = loss_module.forward_masked(
            student_patch_tokens_masked=student_logits,
            teacher_patch_tokens_masked=teacher_probs,
            student_masks_flat=self.mask,
            n_masked_patches=self.n_tokens,
            masks_weight=self.token_weights,
        )
        return self._record(loss, loss_module, leaves)

    def chunked(
        self,
        chunk_size: int | None,
        checkpoint: bool,
        chunk_fn: Any = ibot_chunk_loss,
    ) -> dict[str, torch.Tensor]:
        """What DINOv2 computes with compile_heads or ibot_loss_chunk_size."""
        loss_module = self.loss_module()
        leaves = self.student_leaves()
        loss_module.apply_center_update()
        loss, teacher_logits_mean = chunked_ibot_loss(
            student_bottleneck=leaves["bottleneck"],
            teacher_bottleneck=self.teacher_bottleneck,
            student_weight=self.student_head.last_layer.weight,
            teacher_weight=self.teacher_head.last_layer.weight,
            center=loss_module.center,
            teacher_temp=self.teacher_temp,
            student_temp=loss_module.student_temp,
            token_weights=self.token_weights,
            n_crops=_T_CROPS,
            chunk_size=chunk_size,
            checkpoint=checkpoint,
            chunk_fn=chunk_fn,
        )
        if teacher_logits_mean is not None:
            loss_module.update_center(teacher_logits_mean)
        return self._record(loss, loss_module, leaves)

    @staticmethod
    def _record(
        loss: torch.Tensor, loss_module: IBOTPatchLoss, leaves: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        loss.backward()  # type: ignore[no-untyped-call]
        loss_module.apply_center_update()
        record = {"loss": loss.detach(), "center": loss_module.center.clone()}
        for name, leaf in leaves.items():
            assert leaf.grad is not None, name
            record[f"grad/{name}"] = leaf.grad.clone()
        return record


def _assert_records(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> None:
    assert actual.keys() == expected.keys()
    for name in expected:
        if dtype == torch.float64:
            torch.testing.assert_close(
                actual[name], expected[name], rtol=1e-12, atol=1e-14, msg=name
            )
        else:
            torch.testing.assert_close(actual[name], expected[name], msg=name)


@pytest.mark.usefixtures("no_dist")
class TestChunkedIBOTLoss:
    @pytest.mark.parametrize("mode", ["ibot", "all", "masked"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("checkpoint", [False, True])
    @pytest.mark.parametrize("chunk_size", [None, "n_tokens", "n_tokens+5"])
    def test_single_chunk__bitwise_equal_to_unchunked(
        self, mode: str, dtype: torch.dtype, checkpoint: bool, chunk_size: Any
    ) -> None:
        setup = _Setup(mode, dtype)
        size = {
            None: None,
            "n_tokens": setup.n_tokens,
            "n_tokens+5": setup.n_tokens + 5,
        }[chunk_size]
        expected = setup.legacy()
        actual = setup.chunked(chunk_size=size, checkpoint=checkpoint)
        # One chunk runs exactly the operations of the unchunked pipeline, in the same
        # order. Only the center update averages the teacher logits differently.
        for name in expected:
            if name == "center":
                _assert_records(
                    {name: actual[name]}, {name: expected[name]}, dtype=dtype
                )
            else:
                assert torch.equal(actual[name], expected[name]), name

    @pytest.mark.parametrize("mode", ["ibot", "all", "masked"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("checkpoint", [False, True])
    @pytest.mark.parametrize("chunk_size", [1, 3, "n_tokens-1"])
    def test_chunks__equal_to_unchunked(
        self, mode: str, dtype: torch.dtype, checkpoint: bool, chunk_size: Any
    ) -> None:
        setup = _Setup(mode, dtype)
        size = setup.n_tokens - 1 if chunk_size == "n_tokens-1" else chunk_size
        _assert_records(
            setup.chunked(chunk_size=size, checkpoint=checkpoint),
            setup.legacy(),
            dtype=dtype,
        )

    @pytest.mark.parametrize("chunk_size", [1, 3, 16])
    def test_chunks__bf16_autocast(self, chunk_size: int) -> None:
        # Under bfloat16 both are only accurate to ~1 % (the weight gradient sums
        # bf16-rounded per-chunk parts when chunked), so both are compared against
        # the float32 result: chunking must not make anything less accurate.
        setup = _Setup("all", torch.float32)
        reference = setup.legacy()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            unchunked = setup.legacy()
            chunked = setup.chunked(chunk_size=chunk_size, checkpoint=True)

        def rel_err(x: torch.Tensor, name: str) -> float:
            ref = reference[name].double()
            return float((x.double() - ref).norm() / ref.norm())

        for name in reference:
            err_unchunked = rel_err(unchunked[name], name)
            err_chunked = rel_err(chunked[name], name)
            assert err_chunked <= 1.5 * err_unchunked + 1e-4, (
                name,
                err_chunked,
                err_unchunked,
            )

    @pytest.mark.parametrize("chunk_size", [None, "n_tokens"])
    def test_single_chunk__bf16_autocast_bitwise_equal(self, chunk_size: Any) -> None:
        # The teacher weight is cast to bfloat16 once instead of by F.linear in every
        # chunk: the same rounding, so one chunk still equals the unchunked pipeline.
        setup = _Setup("all", torch.float32)
        size = None if chunk_size is None else setup.n_tokens
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            expected = setup.legacy()
            actual = setup.chunked(chunk_size=size, checkpoint=size is not None)
        for name in expected:
            if name != "center":
                assert torch.equal(actual[name], expected[name]), name

    def test_chunks__checkpoint_without_rng_state(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        setup = _Setup("all", torch.float64)
        checkpoint = torch.utils.checkpoint.checkpoint
        kwargs_seen: list[dict[str, Any]] = []

        def spy(*args: Any, **kwargs: Any) -> Any:
            kwargs_seen.append(kwargs)
            return checkpoint(*args, **kwargs)

        monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", spy)
        setup.chunked(chunk_size=16, checkpoint=True)
        assert len(kwargs_seen) == 3
        assert all(
            kw == {"use_reentrant": False, "preserve_rng_state": False}
            for kw in kwargs_seen
        )

    @pytest.mark.parametrize(
        "chunk_size, n_chunks", [(None, 1), (1, _N_TOKENS), (16, 3), (40, 1)]
    )
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_chunks__recomputed_in_backward(
        self, chunk_size: int | None, n_chunks: int, checkpoint: bool
    ) -> None:
        setup = _Setup("all", torch.float64)
        assert setup.n_tokens == _N_TOKENS
        calls = []

        def chunk_fn(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(args[0].shape[0])
            return ibot_chunk_loss(*args)

        setup.chunked(chunk_size=chunk_size, checkpoint=checkpoint, chunk_fn=chunk_fn)
        # In the forward pass, and with checkpointing once more in the backward pass.
        assert len(calls) == n_chunks * (2 if checkpoint else 1)
        assert sum(calls) == _N_TOKENS * (2 if checkpoint else 1)

    @pytest.mark.parametrize("chunk_size", [1, 7, 16])
    def test_chunks__keep_no_full_size_tensors(self, chunk_size: int) -> None:
        """With checkpointing, nothing of shape [> chunk_size, K] is kept for backward."""
        setup = _Setup("all", torch.float64)

        def saved_logits_rows(run: Any) -> list[int]:
            rows: list[int] = []

            def pack(t: torch.Tensor) -> torch.Tensor:
                if t.dim() == 2 and t.shape[1] == _K:
                    rows.append(t.shape[0])
                return t

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                run()
            return rows

        unchunked = saved_logits_rows(setup.legacy)
        assert max(unchunked) == _N_TOKENS  # the hook sees the [T, K] tensors
        chunked = saved_logits_rows(
            lambda: setup.chunked(chunk_size=chunk_size, checkpoint=True)
        )
        assert all(rows <= chunk_size for rows in chunked)

    def test_no_tokens(self) -> None:
        setup = _Setup("all", torch.float64)
        bottleneck = torch.zeros(0, _D, dtype=torch.float64, requires_grad=True)
        loss_module = setup.loss_module()
        loss_module.apply_center_update()
        center = loss_module.center.clone()
        loss, teacher_logits_mean = chunked_ibot_loss(
            student_bottleneck=bottleneck,
            teacher_bottleneck=torch.zeros(0, _D, dtype=torch.float64),
            student_weight=setup.student_head.last_layer.weight,
            teacher_weight=setup.teacher_head.last_layer.weight,
            center=loss_module.center,
            teacher_temp=0.05,
            student_temp=0.1,
            token_weights=torch.zeros(0),
            n_crops=_T_CROPS,
            chunk_size=4,
            checkpoint=True,
        )
        assert teacher_logits_mean is None
        assert loss.item() == 0.0
        loss.backward()  # type: ignore[no-untyped-call]  # still part of the graph
        assert bottleneck.grad is not None
        assert torch.equal(loss_module.center, center)
