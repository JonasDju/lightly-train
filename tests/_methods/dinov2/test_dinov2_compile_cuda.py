#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""CUDA-only: ``method.compile_blocks: true`` must not change what a DINOv2 training step computes.

Run on a GPU node (skipped without CUDA), from the repo root:

    .venv/bin/python -m pytest -v -s tests/_methods/dinov2/test_dinov2_compile_cuda.py

``-s`` shows the error table of every comparison, which is worth keeping even when the tests pass.

Environment (all optional):
    LT_COMPILE_TEST_CONFIG        run config (default cluster/configs/pretrain-MI-vitb14-24f.yaml); ``model``,
                                  ``method`` and the view sizes / number of local views of ``transform`` come
                                  from it (nothing is loaded from its data paths)
    LT_COMPILE_TEST_CHECKPOINT    a lightly-train checkpoint (``<out>/checkpoints/last.ckpt``) of a run with
                                  this config's model, for realistic activations (recommended: at random init
                                  the LayerScale gammas are 1e-5 and their bf16 gradients are ~50 % off for
                                  eager and compiled alike); otherwise random init
    LT_COMPILE_TEST_STRICT_BATCH  batch size of the float32 check (default 3; the second step runs one less, and
                                  Dynamo specialises a batch of 1 instead of compiling it dynamically)
    LT_COMPILE_TEST_BATCH         batch size of the training-setting check (default 8; the shipped config's
                                  per-GPU batch is 128 / 4 = 32, which needs ~3x the memory of training for
                                  the uncompiled float32 reference)
    LT_COMPILE_TEST_DEVICE        default cuda. ``cpu`` runs the checks on the CPU instead of skipping -- only
                                  to debug the test itself with a small config, as Inductor's CPU code is not
                                  what training runs

What is compared. Three ``DINOv2`` methods built like ``lightly_train.pretrain`` builds them, with the same
weights: the uncompiled ("eager") one, the one with ``compile_blocks=True`` and an uncompiled reference in
higher precision. Each runs the same training steps (same views, same iBOT masks, same stochastic-depth
subsets) through ``training_step_impl`` followed by ``loss.backward()``, and per step the comparison covers

    * the backbone outputs (final-norm tokens) of the teacher, the student's global and local views,
    * the total loss and every loss term (dino global / local, iBOT, KoLeo),
    * the gradient of every student parameter, of all of them as one vector, and the median of the
      per-tensor errors (the vector alone is dominated by the LayerScale gammas, see Comparison.compare),

and finally a no-grad teacher forward like the KneeNo evaluation's. The second step has a smaller batch, so
it exercises the dynamic-shape recompile (the last batch of an epoch, eval batches) and not only the first
compile. No optimizer step is taken: the steps only differ in their input.

Randomness. The iBOT masks come from Python's ``random`` (seeded per step). Stochastic depth draws
``torch.randperm`` (and, for drop rates <= 0.1, ``bernoulli_``) on the device, which Inductor may draw
differently from eager. Both are therefore replaced by deterministic stand-ins for all three runs: the
residual branches still drop the same samples with the same rescaling, so the train-mode drop-path code path
is compiled and compared, just not its random number generator.

Why not bit-identical: Inductor fuses ops and keeps intermediates in float32 registers, so rounding differs
from eager by design, and under bfloat16 autocast both are only accurate to ~0.4 %. So every comparison is
against the higher-precision reference and requires (relative L2 errors)

    err(compiled, reference) <= FACTOR * err(eager, reference) + FLOOR

i.e. compiling may not make any output, loss or gradient measurably less accurate than the eager model already
is. Wrong indexing, a dropped or duplicated term or a broken recomputation shows up as an error of order 1.

1. ``test_fp32_strict``: float32 with TF32 off, small batch, vs. a float64 reference -- the pure "same math"
   check with a tight floor.
2. ``test_training_setting``: bfloat16 autocast as Lightning's ``bf16-mixed`` (``precision: auto`` on a GPU),
   vs. the uncompiled model in float32 (TF32 off).

Compilation uses the method's own ``Module.compile()`` (Inductor). ``fail_on_recompile_limit_hit`` is set,
so blocks silently falling back to eager after too many recompiles fails the test instead of passing it.
"""

from __future__ import annotations

import contextlib
import math
import os
import random
import statistics
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch._dynamo.utils import counters

from lightly_train._methods.dinov2.dinov2 import (
    DINOv2,
    DINOv2AdamWViTArgs,
    DINOv2Args,
)
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers import block as block_module
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers import (
    drop_path as drop_path_module,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._scaling import ScalingInfo

DEFAULT_CONFIG = (
    Path(__file__).parents[3] / "cluster" / "configs" / "pretrain-MI-vitb14-24f.yaml"
)
DEVICE = os.environ.get("LT_COMPILE_TEST_DEVICE", "cuda")

# err(compiled) <= FACTOR[kind] * err(eager) + FLOOR[check]. Outputs, losses and the global gradient are large
# aggregates whose rounding error is stable. A single parameter tensor's gradient can be a small sum with heavy
# cancellation (e.g. LayerScale gammas), where both errors are essentially random under bfloat16 -- hence the
# looser factor (the same values as vjepa2's tests/vjepa_2_1/test_compile_equivalence_cuda.py).
FACTOR = {"output": 2.0, "grad_global": 2.0, "grad": 5.0}
FLOOR = {"strict": 1e-5, "training": 1e-3}

pytestmark = pytest.mark.skipif(
    DEVICE == "cuda" and not torch.cuda.is_available(), reason="CUDA not available"
)


# --------------------------------------------------------------------------- setup (mirrors pretrain)


def _load_config() -> dict[str, Any]:
    path = Path(os.environ.get("LT_COMPILE_TEST_CONFIG", DEFAULT_CONFIG))
    with path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def _view_shapes(
    transform: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """(global view (C, D, H, W), local view (C, D, H, W), number of local views); sizes are (H, W, D)."""
    h, w, d = transform.get("image_size", (224, 224, 16))
    local = transform.get("local_view") or {}
    lh, lw, ld = local.get("view_size", (98, 98, 8))
    return (1, d, h, w), (1, ld, lh, lw), local.get("num_views", 8)


def _build_method(config: dict[str, Any], compile_blocks: bool) -> DINOv2:
    method_args = DINOv2Args.model_validate(
        {**(config.get("method") or {}), "compile_blocks": compile_blocks}
    )
    optimizer_args = DINOv2AdamWViTArgs()
    wrapped_model = package_helpers.get_wrapped_model(
        model=config["model"], num_input_channels=1
    )
    method_args.resolve_auto(
        scaling_info=ScalingInfo(dataset_size=10_000, epochs=100),
        optimizer_args=optimizer_args,
        wrapped_model=wrapped_model,
    )
    method = DINOv2(
        method_args=method_args,
        optimizer_args=optimizer_args,
        embedding_model=EmbeddingModel(wrapped_model=wrapped_model),
        global_batch_size=8,
        num_input_channels=1,
    )
    # training_step_impl only reads the step counters. Step 0 is in the first phase of every schedule.
    method.trainer = SimpleNamespace(  # type: ignore[assignment]
        global_step=0, max_epochs=100, estimated_stepping_batches=100_000
    )
    return method


@dataclass
class Models:
    eager: DINOv2
    compiled: DINOv2
    reference: DINOv2
    shapes: tuple[tuple[int, ...], tuple[int, ...], int]
    weights: str = field(default="random init")


def _build_models(reference_dtype: torch.dtype) -> Models:
    config = _load_config()
    torch.manual_seed(0)
    eager = _build_method(config, compile_blocks=False)
    checkpoint = os.environ.get("LT_COMPILE_TEST_CHECKPOINT")
    weights = "random init"
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        eager.load_state_dict(state["state_dict"], strict=True)
        weights = f"{checkpoint} (epoch {state.get('epoch')})"
    state_dict = eager.state_dict()
    compiled = _build_method(config, compile_blocks=True)
    reference = _build_method(config, compile_blocks=False)
    # compile_blocks keeps the state_dict keys, so strict loading also checks that.
    compiled.load_state_dict(state_dict, strict=True)
    reference.load_state_dict(state_dict, strict=True)

    return Models(
        eager=eager.to(DEVICE),
        compiled=compiled.to(DEVICE),
        reference=reference.to(device=DEVICE, dtype=reference_dtype),
        shapes=_view_shapes(config.get("transform") or {}),
        weights=weights,
    )


def _make_batch(
    batch_size: int,
    shapes: tuple[tuple[int, ...], tuple[int, ...], int],
    seed: int,
) -> list[Tensor]:
    """Views with some structure: per-sample smooth volumes, so samples (and their features) differ clearly.

    Pure noise gives nearly identical cls tokens at random init, which makes KoLeo's nearest-neighbour
    distances (and with them its loss and gradients) ill-conditioned for eager and compiled alike.
    """
    global_shape, local_shape, n_local = shapes
    g = torch.Generator().manual_seed(seed)
    _, d, h, w = global_shape
    coarse = torch.randn(
        batch_size, 1, max(d // 4, 1), max(h // 16, 1), max(w // 16, 1), generator=g
    )
    base = F.interpolate(coarse, size=(d, h, w), mode="trilinear", align_corners=False)
    base = base * (0.5 + torch.rand(batch_size, 1, 1, 1, 1, generator=g))
    base = base + 0.5 * torch.randn(batch_size, 1, 1, 1, 1, generator=g)

    def noisy(x: Tensor) -> Tensor:
        return x + 0.1 * torch.randn(x.shape, generator=g)

    views = [noisy(base), noisy(base.flip(-1))]
    _, ld, lh, lw = local_shape
    for _ in range(n_local):
        # A random sub-volume of half the size in every axis, resized to the local view size.
        z0, y0, x0 = (
            int(torch.randint(0, max(s - s // 2, 1), (1,), generator=g))
            for s in (d, h, w)
        )
        crop = base[
            :,
            :,
            z0 : z0 + max(d // 2, 1),
            y0 : y0 + max(h // 2, 1),
            x0 : x0 + max(w // 2, 1),
        ]
        views.append(
            noisy(
                F.interpolate(
                    crop, size=(ld, lh, lw), mode="trilinear", align_corners=False
                )
            )
        )
    return views


# --------------------------------------------------------------------------- deterministic stand-ins


def _randperm(n: int, *args: Any, device: Any = None, **kwargs: Any) -> Tensor:
    # Reversed order: a fixed permutation that is not the identity.
    return torch.arange(n - 1, -1, -1, device=device)


def _drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # Every other sample dropped, kept ones rescaled like the original.
    keep = (torch.arange(x.shape[0], device=x.device) % 2 == 0).to(x.dtype)
    return x * (keep / keep_prob).view((x.shape[0],) + (1,) * (x.ndim - 1))


@contextlib.contextmanager
def _deterministic_drop_path() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(block_module.torch, "randperm", _randperm)
        mp.setattr(drop_path_module, "drop_path", _drop_path)
        yield


@contextlib.contextmanager
def _no_tf32() -> Iterator[None]:
    """float32 matmuls and convolutions without TF32.

    Training leaves ``float32_matmul_precision: auto`` at torch's default, "highest" (no TF32 matmuls); under
    bf16 autocast the convolution runs in bfloat16 anyway, so cuDNN's TF32 default does not matter there.
    """
    old = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old[0]
        torch.backends.cudnn.allow_tf32 = old[1]
        torch.set_float32_matmul_precision(old[2])


# --------------------------------------------------------------------------- running one model


Record = dict[str, Tensor]


def _backbone(method: DINOv2, which: str) -> torch.nn.Module:
    embedding_model = getattr(method, f"{which}_embedding_model")
    return embedding_model.wrapped_model.get_model()  # type: ignore[no-any-return]


@contextlib.contextmanager
def _capture_backbone_outputs(method: DINOv2, out: list[Tensor]) -> Iterator[None]:
    """Every backbone forward ends in its final ``norm``: collect those outputs in call order."""
    handles = [
        _backbone(method, which).norm.register_forward_hook(
            lambda module, args, output: out.append(output.detach())
        )
        for which in ("teacher", "student")
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _train_step(
    method: DINOv2,
    views: list[Tensor],
    step: int,
    dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
) -> Record:
    random.seed(step)  # iBOT masks
    method.zero_grad(set_to_none=True)
    outputs: list[Tensor] = []
    autocast = (
        torch.autocast(device_type=torch.device(DEVICE).type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    batch = {
        "views": [v.to(device=DEVICE, dtype=dtype) for v in views],
        "filename": [str(i) for i in range(len(views[0]))],
    }
    with _capture_backbone_outputs(method, outputs), autocast:
        result = method.training_step_impl(batch, step)  # type: ignore[arg-type]
    result.loss.backward()

    # Call order in training_step_impl: teacher (global views), student global, student local.
    assert len(outputs) == 3, f"expected 3 backbone forwards, got {len(outputs)}"
    record: Record = {
        "out/teacher": outputs[0],
        "out/student_global": outputs[1],
        "out/student_local": outputs[2],
        "loss/total": result.loss.detach(),
    }
    for name, value in (result.log_dict or {}).items():
        if name.startswith("train_loss/"):
            record[f"loss/{name.removeprefix('train_loss/')}"] = torch.as_tensor(
                value
            ).detach()
    for name, param in method.named_parameters():
        if param.grad is not None:
            record[f"grad/{name}"] = param.grad.detach().clone()
    return record


def _eval_forward(
    method: DINOv2,
    views: list[Tensor],
    dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
) -> Record:
    """A no-grad teacher forward like the KneeNo evaluation's (``encoder: target``)."""
    autocast = (
        torch.autocast(device_type=torch.device(DEVICE).type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        tokens = _backbone(method, "teacher").forward_features(
            views[0].to(device=DEVICE, dtype=dtype)
        )
    return {
        "eval/cls": tokens["x_norm_clstoken"].detach(),
        "eval/patches": tokens["x_norm_patchtokens"].detach(),
    }


# --------------------------------------------------------------------------- comparison


def _rel_err(x: Tensor, ref: Tensor) -> float:
    x, ref = x.double(), ref.double()
    return float((x - ref).norm() / ref.norm().clamp_min(1e-30))


def _kind(name: str) -> str:
    return "grad" if name.startswith("grad/") else "output"


@dataclass
class Row:
    name: str
    kind: str
    err_eager: float
    err_compiled: float
    ok: bool


@dataclass
class Comparison:
    floor: float
    rows: list[Row] = field(default_factory=list)

    def add(self, name: str, kind: str, err_eager: float, err_compiled: float) -> None:
        ok = (
            math.isfinite(err_compiled)
            and err_compiled <= FACTOR[kind] * err_eager + self.floor
        )
        self.rows.append(Row(name, kind, err_eager, err_compiled, ok))

    def compare(
        self, prefix: str, eager: Record, compiled: Record, reference: Record
    ) -> None:
        assert eager.keys() == compiled.keys() == reference.keys(), (
            sorted(set(eager) ^ set(compiled)),
            sorted(set(eager) ^ set(reference)),
        )
        for name in reference:
            self.add(
                f"{prefix}{name}",
                _kind(name),
                _rel_err(eager[name], reference[name]),
                _rel_err(compiled[name], reference[name]),
            )
        grads = [name for name in reference if name.startswith("grad/")]
        if grads:

            def flat(record: Record) -> Tensor:
                return torch.cat([record[name].double().flatten() for name in grads])

            ref = flat(reference)
            self.add(
                f"{prefix}grad (all {len(grads)} tensors)",
                "grad_global",
                _rel_err(flat(eager), ref),
                _rel_err(flat(compiled), ref),
            )

            # The plain vector is dominated by the LayerScale gammas (initialised to 1e-5, so their gradients are
            # by far the largest), whose bf16 gradients are ~50 % off at random init for eager and compiled alike.
            # The median of the per-tensor errors weights every tensor equally and ignores the few that are pure
            # rounding noise at random init (the norm2 parameters behind those gammas: gradients ~1e-6 after heavy
            # cancellation, ~700x off under bf16 for both). So it shows a broad degradation of the other ~180
            # tensors even when each of them stays within the per-tensor FACTOR["grad"].
            def median(record: Record) -> float:
                return statistics.median(
                    _rel_err(record[name], reference[name]) for name in grads
                )

            self.add(
                f"{prefix}grad (median of per-tensor errors)",
                "grad_global",
                median(eager),
                median(compiled),
            )

    def report(self, title: str) -> None:
        print(f"\n{title}\n{'':<90} {'eager':>10} {'compiled':>10}")
        # Outputs, losses and the global gradients first, then the per-tensor gradients worst first (largest
        # margin used of the allowed FACTOR * err(eager) + floor).
        summary = [r for r in self.rows if r.kind != "grad"]
        grads = sorted(
            (r for r in self.rows if r.kind == "grad"),
            key=lambda r: r.err_compiled / (FACTOR["grad"] * r.err_eager + self.floor),
            reverse=True,
        )
        for r in summary + grads[:10]:
            print(
                f"{r.name:<90} {r.err_eager:10.2e} {r.err_compiled:10.2e} {'' if r.ok else 'FAIL'}"
            )
        if len(grads) > 10:
            print(f"... and {len(grads) - 10} more gradient tensors")

    def assert_ok(self) -> None:
        failed = [r for r in self.rows if not r.ok]
        assert not failed, (
            f"{len(failed)} of {len(self.rows)} comparisons failed "
            f"(err(compiled) > FACTOR * err(eager) + {self.floor}):\n"
            + "\n".join(
                f"  {r.name}: eager {r.err_eager:.2e}, compiled {r.err_compiled:.2e}"
                for r in failed[:20]
            )
        )


def _run_check(
    *,
    batch_size: int,
    dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
    reference_dtype: torch.dtype,
    floor: float,
    title: str,
) -> None:
    torch._dynamo.reset()
    counters.clear()
    with pytest.MonkeyPatch.context() as mp:
        # Restored afterwards: DINOv2.__init__ sets optimize_ddp globally when compiling.
        mp.setattr(
            torch._dynamo.config, "optimize_ddp", torch._dynamo.config.optimize_ddp
        )
        mp.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
        models = _build_models(reference_dtype)
        print(
            f"\n{title}: weights {models.weights}, batch sizes {batch_size}, {batch_size - 1}"
        )
        comparison = Comparison(floor=floor)
        with _deterministic_drop_path(), _no_tf32():
            for step, b in enumerate((batch_size, batch_size - 1)):
                views = _make_batch(b, models.shapes, seed=step)
                reference = _train_step(
                    models.reference, views, step, reference_dtype, None
                )
                eager = _train_step(models.eager, views, step, dtype, autocast_dtype)
                compiled = _train_step(
                    models.compiled, views, step, dtype, autocast_dtype
                )
                comparison.compare(f"step {step} ", eager, compiled, reference)
                del reference, eager, compiled

            views = _make_batch(batch_size + 1, models.shapes, seed=99)
            reference = _eval_forward(models.reference, views, reference_dtype, None)
            eager = _eval_forward(models.eager, views, dtype, autocast_dtype)
            compiled = _eval_forward(models.compiled, views, dtype, autocast_dtype)
            comparison.compare("", eager, compiled, reference)

        n_graphs = counters["stats"]["unique_graphs"]
        print(f"compiled graphs: {n_graphs}")
        comparison.report(title)
        # Something was compiled at all (the eager and reference models never are).
        assert n_graphs > 0
        comparison.assert_ok()


def test_fp32_strict() -> None:
    _run_check(
        batch_size=int(os.environ.get("LT_COMPILE_TEST_STRICT_BATCH", 3)),
        dtype=torch.float32,
        autocast_dtype=None,
        reference_dtype=torch.float64,
        floor=FLOOR["strict"],
        title="float32 (TF32 off) vs float64 reference",
    )


def test_training_setting() -> None:
    _run_check(
        batch_size=int(os.environ.get("LT_COMPILE_TEST_BATCH", 8)),
        dtype=torch.float32,
        autocast_dtype=torch.bfloat16,
        reference_dtype=torch.float32,
        floor=FLOOR["training"],
        title="bfloat16 autocast (bf16-mixed) vs float32 reference",
    )
