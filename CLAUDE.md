# CLAUDE.md

## What this repo is

A fork of **Lightly's `lightly-train`** (DINOv2 trainer), adapted to make **DINOv2
3D-native** so it can be pretrained on knee-MRI volumes from the sibling `KneeNo`
package, mirroring what `vjepa2` already does for V-JEPA 2.1. Unlike vjepa2 (already
video/3D-native), the original codebase was 2D-only, so this fork touches far more of
the internals: patch embedding, positional encoding, attention, masking, the transform
pipeline, and the training command's config surface.

**Scope rule: DINOv2 only.** All other methods/models in this large multi-method repo
(SimCLR, DINO, DINOv3(.1), distillation*, DenseCL, DetCon, all task models —
LT-DETR/EoMT/depth/classification/segmentation — and their 2D datasets/transforms) are
**expected to be broken** by these changes and are explicitly **out of scope to fix**.
Do not "fix" them back to 2D; do not spend effort making them coexist with the 3D
changes. See "Tests" below for how the broken ones are quarantined.

**Current status:** the 3D DINOv2 rewrite (patch embed, pos encoding, SDPA, masking),
the KneeNo dataset/transform wiring, and the KneeNo *evaluation* wiring (adapter,
epoch-end callback, standalone `eval_classification` command) are all done and working —
`lightly-train pretrain` with `model=dinov2/<name>` runs end to end on CPU against a real
KneeNo-layout dataset; evaluation runs when an `eval:` config is passed (off otherwise).
See "KneeNo classification evaluation" below.

## Branches

The repo has three long-lived branches (restructured 2026-10-07):

- **`main`**: stock lightly-train plus the KneeNo dependency (`KNEENO.md`, `exclude-newer` pin) and the
  `pyproject.toml`/`uv.lock` shared by all branches. It's the common base, with no training changes.
- **`2d_dinov2`**: the vanilla 2D DINOv2 baseline for the thesis comparison. Training stays as close to stock as
  possible (lightly-train's own image-folder dataset over the JPEG slices). It adds only the KneeNo in-training
  eval wiring and a `cluster/` folder. It has its own `CLAUDE.md`.
- **`3d_dinov2`** (this branch): the 3D-native DINOv2 described in this file. It merges what used to be
  `DINOv2-3D-adaptation` and `add_evaluation`.

**This `CLAUDE.md` is tracked per branch**, so it describes this branch only and moves with the code it describes.
A reverted feature takes its docs with it. Facts that don't depend on the branch go in the global `MA/CLAUDE.md`.

**Dependencies are shared through `main`.** `pyproject.toml` and `uv.lock` are identical on all three branches,
so one `.venv` serves every branch without re-syncing. Never `uv add`/`uv lock` on a single branch: change the
dependencies on `main` and merge it into the others.

## Environment

- Python **3.13**, `torch==2.13.0+cu130`, dependencies managed with **`uv`**
  (`pyproject.toml` + `uv.lock`, `[tool.uv] exclude-newer` pinned — see `KNEENO.md`).
  Run Python as **`.venv/bin/python`**.
- **`kneeno`** is deliberately *not* a normal dependency (would explode `uv.lock`, see
  `KNEENO.md`): after `uv sync`, always `uv pip install -e ../KneeNo` again (or use
  `uv sync --inexact`). Check `import kneeno` works before trusting any test run —
  a plain `uv sync` silently uninstalls it.
- `monai`, `torchmetrics[detection]` etc. are normal dependencies now (`monai>=1.3.2`
  added for the 3D transform pipeline).
- **No usable GPU on this dev box** (`torch.cuda` reports an old driver / is otherwise
  unavailable in this sandbox) — verify everything on CPU with the smallest model
  (`dinov2/_vittest14`, embed_dim=8, depth=3) and small synthetic volumes. See Testing.
- **Fragile venv, watch for this:** at least once this session, `import torch` failed
  with `ImportError: libcudnn.so.9: ... No such file` (and, after fixing that,
  `libnccl.so.2`, `libcusparseLt`) because files were missing from already-"installed"
  `nvidia-*-cu13` wheel directories in `.venv/lib/python3.13/site-packages/` — the
  dist-info was there but the actual `.so` files weren't. Fix is
  `uv pip install --python .venv/bin/python --reinstall --no-deps <pkg>==<version>`
  for the affected `nvidia-*` packages (versions are in `uv.lock`); this does not touch
  `uv.lock`. If `import torch` fails, check this before assuming a real regression.

## The 3D adaptation — key files

Axis convention throughout: **configs/transform args use `(H, W, D)`**;
**tensors and internal shapes use `(D, H, W)`** (i.e. `(B, C, D, H, W)` volumes). Getting
this swap backwards is the single easiest way to reintroduce a silent H/W-swap bug (see
Gotchas).

- `src/lightly_train/_data/mi_dataset.py` — `MIDataset`, thin wrapper over
  `kneeno.UnlabeledKneeMRIDataset` producing `(1, D, H, W)` volumes, permuted to
  `(1, H, W, D)` for MONAI transforms. Also owns `reseed_randomizables()` and
  `worker_init_fn` (see Gotchas — RNG reseeding).
- `src/lightly_train/_transforms/random_resized_crop.py` — `RandomResizedCrop3D`, a
  from-scratch anisotropy-aware 3D `RandomResizedCrop` (own separable resampling
  weights for area/linear/nearest/cubic, no external volume-augmentation dependency).
  Subclasses MONAI's `Randomizable`, uses `self.R` for randomness so per-worker
  reseeding works. Returns the input's dtype by default (rounded/clipped for integer
  inputs, what replaying a crop onto a label volume needs); `output_dtype=np.float32`
  keeps the resampled values unquantized, since resampling runs in float anyway.
  `_resample` contracts each axis with a scipy sparse product when the weight matrix is
  mostly zeros (see "CPU fast paths" below).
  Interpolation specs (`interpolation`/`upscale_interpolation`, i.e. the transform args'
  `resize_interpolation`/`resize_upscale_interpolation`) take one mode or
  `<in-plane>+<out-of-plane>` (since 2026-10-05, `parse_interpolation`; `"X"` = `"X+X"`):
  H/W use the first, depth D the second, e.g. `area+nearest` / `linear+nearest` resamples
  slices in-plane but only picks whole slices along depth (separable, so this is exactly
  slice-by-slice). Depth `nearest` uses KneeNo's `_resample_depth` mapping
  (`linspace(0, D-1, n).round()`, both end slices kept; `_nearest_slice_weights`), in-plane
  `nearest` cv2's floor mapping -- so plain `"nearest"` changed along depth on 2026-10-05.
  Pydantic fields use the `ResizeInterpolation` type (validated, stored as written).
  **Foreground-guided placement** (since 2026-10-06, `transform.crop_foreground:
  {threshold: t}`, off by default): `get_params(shape, foreground_box=...)` places the
  crop per axis inside the tissue box if it fits there, else so that it contains the box,
  always inside the volume (`_start_range`). Sizes are drawn exactly as without it, so
  there is no extra stretching and no extra fit failure. The volume is deliberately *not*
  cropped to the box (MONAI `CropForeground`): resizing a box with the tissue's
  proportions to the fixed view size would stretch the anatomy. Off, or with a box that
  spans the volume, the crops are identical to before for the same seed
  (`tests/_methods/dinov2/test_dinov2_transform_reference.py` pins all DINOv2 views to
  `dinov2_views_reference.npz`, generated at d1aab3e; regenerate it only for an intended
  augmentation change). **Evaluation needs nothing**: its counterpart is the scale-1 crop,
  the whole volume, which the box cannot move -- so `DINOv2Adapter`, the callback and
  `eval_classification` ignore the option on purpose. Measured on one real exam
  (`MA/88irr8`, threshold 10, real pipeline): air patches in local views 15–28% → 10–17%,
  global views −1…4 points; the box spans all slices and, except transversal, the full
  height, and 35–43% of it is still air; thresholds 5–50 give nearly the same box; the
  whole-volume eval view stays within the training distribution of global crops.
- `src/lightly_train/_transforms/view_transform.py` — `ViewTransform` is now a MONAI
  `Compose` pipeline, not albumentations: `RandomResizedCrop3D` (float32 output) →
  `RandRotate` (optional) → `AlphaRandHistogramShift` → `RandAdjustContrast` →
  [`RandGaussianSmooth` xor `RandGaussianSharpen`, via `OneOf` when both are enabled] →
  `RandGibbsNoise` → `RandGaussianNoise` → `NormalizeIntensity` → `ToTensor`.
  Bracketed/optional steps only run when their args are set; `histogram_shift`/
  `adjust_contrast` are the only two enabled *by default* (see `DINOTransformArgs`
  in `dino_transform.py`). **`NormalizeIntensity` runs last, as upstream does, and the
  crop emits `float32`.** Every augmentation therefore sees raw `[0, 255]` intensities,
  in full float precision: with the crop's default dtype round-trip, a `uint8` volume
  (what KneeNo returns under `resample_mode="nearest"`) would be quantized to ≤256 levels
  at the very first op (`test_view_transform__uint8_input_is_not_quantized`). Normalize
  used to run first, on the full volume, once per view (10× per DINOv2 sample, ~40% of the
  loading time); on the view it is ~free, and a sample loads ~1.5–2× faster. Where it
  sits barely matters otherwise: the resampling weights sum to one, rotation pads by edge
  replication, histogram shift and gamma contrast are min/max-relative and Gibbs is
  linear, so all of them commute with Normalize's affine map (verified with equal seeds:
  identical to float rounding with blur off). **The exception is MONAI's Gaussian
  blur/sharpen, which zero-pads**: zero is black here (it would be mid-grey after
  normalizing), so near the view border — ~4σ in-plane (up to 8 voxels) and the
  first/last slice along depth — blurred views are pulled towards black. An
  edge-replicating wrapper would make the position fully irrelevant; deliberately not
  done (yet). The opt-in `RandGaussianNoise`'s `mean`/`std` are in **raw `[0, 255]`
  intensity units** and passed through unscaled (unlike `NormalizeArgs`, which is for
  `[0, 1]`); in the normalized input they shrink by `255 * normalize.std` (127.5 by
  default). It is the only augmentation with intensity-unit parameters — every other one is
  min/max-relative, a blend weight, or in voxels/degrees — so it was the only default that
  needed re-tuning when Normalize moved to the end.
  MONAI's intensity ops return float32 anyway, so uint8 would only ever survive through
  the crop — no RAM/VRAM argument for staying in uint8 (measured: same float32 output,
  ~same worker peak memory).
  **Blur and sharpen are mutually exclusive per sample** when both are enabled
  (`prob > 0`): MONAI's `OneOf` picks one instead of applying both, since they pull
  intensity in opposite directions — check for either by also looking inside any
  `OneOf` node, not just top-level `Compose` entries. `GaussianBlurArgs.sigma_range`
  and `RandGaussianSharpenArgs.sigma1`/`.sigma2` are passed directly to the
  anisotropy-aware wrappers below (which fan a single range out to x/y/z
  internally) — `GaussianBlurArgs.sigmas`/`.blur_limit` (the old CV2/albumentations
  fields, still real fields with real defaults on `SimCLRGaussianBlurArgs` etc. for
  the other, out-of-scope methods) default to `None` on the shared base so
  `DINOGaussianBlurArgs` doesn't need to redeclare them — same pattern as the
  RGB-only `color_jitter`/`random_gray_scale`/`solarize`/`channel_drop` fields below.
  `record_geometry=True` (dinov31's PaKA feature) is unsupported here — raises
  `NotImplementedError`. Flipping is disabled (the `random_flip` field is now fully
  dead — no construction code references it, unlike the RGB-only fields which were
  actually removed): MRI volumes have a fixed orientation. Color jitter / gray-scale
  / solarize / channel-drop fields were removed entirely (RGB-only 2D clutter, always
  dead on this 1-channel 3D path). `RandRotate` is restricted to **in-plane (H-W)
  rotation only**, via `range_z` (`range_x`/`range_y` left at MONAI's default 0 —
  see the `monai_wrappers.py` bullet below for the empirical check behind that
  mapping, and don't assume the axis name matches intuition), and uses
  `padding_mode="border"` (edge-replicate) — this makes rotation's fill value
  order-independent w.r.t. Normalize (it copies real boundary voxel values, not a fixed
  number); with MONAI's `"zeros"` default, which side of `NormalizeIntensity` the rotate
  sits on would matter — as it already does for the blur/sharpen (see above).
  `RandGaussianSmooth`/`RandGaussianSharpen` are actually `AnisotropyAwareRandGaussianSmooth`/
  `AnisotropyAwareRandGaussianSharpen` from `monai_wrappers.py`, and `RandHistogramShift`
  is actually `AlphaRandHistogramShift` (also `monai_wrappers.py` — not for anisotropy,
  see below).
- `src/lightly_train/_transforms/monai_wrappers.py` — `AnisotropyAwareRandGaussianSmooth`
  scales `RandGaussianSmooth`'s `sigma_z` down by the volume's own voxel-count shape
  ratio (`D/sqrt(H*W)`; no physical spacing metadata exists anywhere in this pipeline,
  so this is a proxy — same one `RandomResizedCrop3D` and `MaskingGenerator` already
  use). `AnisotropyTrackingRandomResizedCrop3D` tags its output with the *pre-crop*
  `(H, W, D)` shape as `MetaTensor` metadata (`ORIG_SHAPE_META_KEY`), since
  `RandomResizedCrop3D` always resizes to a fixed configured size — reading the
  post-crop shape instead would make the "per-volume" signal a constant. MONAI's own
  meta-propagation (`convert_to_dst_type`) carries that tag untouched through
  `RandRotate` and `RandGaussianSmooth`; `ToTensor` strips it at the end as usual.
  `TagForegroundBox` (run once per sample in `DINOTransform.__call__` when
  `crop_foreground` is set, not per view: ~7 ms per 672×672×24 volume) wraps the raw
  volume in a `MetaTensor` tagged with the tissue box (`FOREGROUND_BOX_META_KEY`, from
  `CropForeground.compute_bounding_box`; no tag if nothing exceeds the threshold);
  `AnisotropyTrackingRandomResizedCrop3D` reads it for placement and does not pass it on.
  Its `select_fn` is the picklable `_AboveThreshold` class -- a lambda fails to pickle into
  spawn/forkserver DataLoader workers (`test_crop_foreground__spawn_workers`).
  **Rotation is deliberately not scaled this way** — a rotation in voxel-index space
  is only a true physical rotation when voxels are cubic; scaling a rotation range by
  a voxel-count ratio would still resample a mix of rotation, shear, and stretch on
  anisotropic volumes. That's why `view_transform.py` restricts `RandRotate` to
  in-plane rotation only instead of using an anisotropy-aware rotation wrapper — via
  `range_z`, verified empirically (not just assumed from the axis name): for a
  `(C, H, W, D)` volume, only `range_z` leaves every voxel constant along `D` after a
  large rotation; `range_x`/`range_y` both mix `D` in. `AlphaRandHistogramShift`
  wraps `RandHistogramShift` for a different reason — not anisotropy, but because
  MONAI's default is too strong unblended; it linearly blends the shifted output with
  the original by a fixed `alpha` (an intensity-strength knob, see
  `RandHistogramShiftArgs.alpha` in `transform.py`).
  Future work: once real per-case anisotropy is available (a separate
  dataset-metadata file), swap what populates `ORIG_SHAPE_META_KEY` — the consuming
  blur code doesn't need to change.
- **CPU fast paths in the transforms (2026-10-01).** Four hot spots of the per-sample
  pipeline were replaced by faster CPU code that computes the same thing as before:
  - `_resample` (`random_resized_crop.py`): an axis whose weight matrix is at most
    `_SPARSE_MAX_DENSITY = 0.25` non-zero (strong reductions, nearest) is contracted
    with `scipy.sparse.csr_array @ ...` instead of the dense `np.tensordot`, which
    near-identity resizes keep. `_SPARSE_MAX_DENSITY = 0` is exactly the previous,
    all-dense code.
  - `AlphaRandHistogramShift.interp`: `np.interp` for CPU float32/float64 tensors instead
    of MONAI's `searchsorted` path; anything else falls back to MONAI's.
  - `AnisotropyAwareRandGaussianSmooth` / `AnisotropyAwareRandGaussianSharpen`: their
    `__call__` re-implements MONAI's `RandGaussianSmooth` / `RandGaussianSharpen` (same
    random draws in the same order, same `gaussian_1d` kernels truncated at 4σ, zero
    padding, axes in order, unity kernels skipped) but filters with
    `_separable_gaussian` (scipy `ndimage.correlate1d`) instead of MONAI's
    single-channel `torch.conv3d`. The sharpen's final step is GaussianSharpen's float32
    expression `blurred + a * (blurred - filtered)`, `a` rounded to float32 like torch does.
    **CPU-only and 3 spatial axes only** (`_float_volume` calls `.numpy()`; MONAI's
    `ensure_tuple_size` is not replicated) — fine for DataLoader workers on 3D volumes.
    If MONAI is upgraded, re-check its `RandGaussianSmooth/Sharpen.__call__` and
    `randomize` against these copies: a changed draw order would silently desync them,
    and only the equivalence tests would notice.

  Measured single-threaded (as in a worker) on synthetic 512×512×24 uint8 volumes with the
  shipped config: a full DINOv2 sample (2 global + 8 local views) 300 → 158 ms (**1.89x**).
  Per global 224×224×16 view: Gaussian smooth 4.7x, sharpen 4.4x (not in the shipped
  config), crop 1.7x, histogram shift 1.3x; local views gain less (1.3–2.0x).
  **Not bit-identical** except where the arithmetic is the same: random draws and generator
  state, nearest resampling, the dense-path resizes, and the skipped / unity-kernel paths
  are bit-equal; everything else sums in a different order or in float64 before rounding,
  i.e. equal up to a few ulps of the data's magnitude (measured ≤ 3.5 · eps · max|old|; the
  sharpen amplifies its blurs' rounding by up to `1 + 2a`, so ≤ 1.1 · eps · (1 + 2a) ·
  max|input|). Consequence: with an **integer** `output_dtype`/input dtype and a non-nearest
  mode, a value at .5 can round the other way (±1 on a few voxels) — training is unaffected
  (crop output is float32), label replays should use nearest anyway.
  `tests/_transforms/test_transform_equivalence.py` pins all of this against the replaced
  code as the oracle (dense path via `_SPARSE_MAX_DENSITY = 0`, MONAI's `interp`, the
  previous smooth/sharpen `__call__` as `_OldSmooth`/`_OldSharpen`), per transform and end to
  end through the DINOv2 views; mutations (reflect padding, 3σ truncation, swapped sharpen
  sigmas or sign) each fail 18–33 of its tests. Any further fast path should get the same
  kind of oracle test.
- `src/lightly_train/_models/dinov2_vit/dinov2_vit_src/layers/patch_embed.py` —
  `PatchEmbed` is `Conv3d`-based; reorders `(H,W,D)` config sizes to `(D,H,W)` tensor
  order. `compute_out_dims()` computes the post-conv `(D,H,W)` grid from `patch_size`.
- `src/lightly_train/_models/dinov2_vit/dinov2_vit_src/models/vision_transformer.py` —
  3D positional embedding interpolation (`interpolate_pos_encoding`, trilinear;
  `antialias` is silently ignored — PyTorch only allows it for
  bilinear/bicubic/lanczos). `get_intermediate_layers(reshape=True)` reshapes via
  `patch_embed.compute_out_dims()`.
- `src/lightly_train/_models/_model_helpers.py` — the two `load_state_dict` pre-hooks,
  now 3D-native. `interpolate_pos_embed_hook` (on `DinoVisionTransformer`) does two
  things: **3D → 3D** it trilinearly resizes `pos_embed` between patch grids (the
  high-resolution adaptation phase), and **2D → 3D** it loads only `blocks.*` /
  `norm.*` from an original DINOv2 checkpoint, leaving the whole tokenization
  (`patch_embed.proj.*`, `pos_embed`, `cls_token`, `mask_token`, `register_tokens`)
  randomly initialized. The two cases are told apart by the rank of the checkpoint's
  patch-embedding kernel (4D vs 5D), never by token counts.
  `patch_embed_adjust_input_channels_hook` (on `PatchEmbed`) is now rank-generic, so
  the channel cut/repeat works for `Conv3d` as well as DINOv3's `Conv2d`, and a
  rank *mismatch* makes it keep the module's own random init. Both rewrite the
  state_dict so a `strict=True` load sees neither a missing nor an unexpected key.
  Register tokens need no special handling: `pos_embed` is `[1, 1 + D*H*W, dim]` with
  exactly one leading (cls) position whether or not the model has registers — registers
  are a separate parameter, concatenated after `pos_embed` is added, and carry no
  positional embedding. The old docstring claiming otherwise was wrong.
- `src/lightly_train/_models/dinov2_vit/dinov2_vit_src/configs/train/vit{s,b,l,g}14{,_reg4}_2dinit.yaml`
  (8 configs) + the matching `vit{s,b,l,g}14-{reg4,noreg}-2dinit` `MODELS` entries —
  3D train configs pinned to whatever `ffn_layer`/`block_chunks` the corresponding
  public checkpoint actually uses (`mlp`/`0` for S/B/L, `swiglufused`/`0` for g — see
  the gotcha below) and paired with its URL, so e.g.
  `model=dinov2/vitb14-reg4-2dinit` downloads it and initializes the blocks from it.
- `DINOv2Args.n_tokenization_only_steps` (`_methods/dinov2/dinov2.py`, default `0`) —
  pairs with the `*-2dinit` configs above: for the first N steps, freezes the
  transformer blocks and the final `norm` (via `lr=0.0` in `on_before_optimizer_step`,
  the same mechanism as the pre-existing `student_freeze_backbone_steps`/
  `student_freeze_last_layer_steps`) while the randomly initialized tokenization and
  the (also random) projection heads train. `train_helpers.validate_tokenization_only_steps`
  (called from `train_from_config`) rejects it with a `ValueError` unless the model
  name ends in `-2dinit`, or `checkpoint=`/`resume_interrupted=True` is set — freezing
  the blocks only makes sense when they already hold pretrained weights.
  `DINOv2.on_train_start` logs the phase status once at the start of training
  (frozen vs. all layers, and, if frozen, how many steps remain), and
  `on_before_optimizer_step` logs once more when the blocks actually unfreeze. Both
  compare against `self.trainer.global_step` — see the gotcha below for why the
  status log has to live in `on_train_start`, not `on_fit_start`.
- `DINOv2Args.ibotpp` (`None` default / `"all"` / `"masked"`) — iBOT++ from TIPSv2: iBOT's patch loss
  without the mask indicator, so visible tokens are supervised too. Only the **loss token selection**
  changes (`utils.ibotpp_loss_mask` → `mask_indices_and_weights`, the same index/weight formulas
  `create_collated_masks` uses); the student input is still masked by `collated_masks`, and both forwards,
  both centering methods and `forward_masked` are unchanged. `"all"` = every patch of every global crop;
  `"masked"` = whole crops that have masked patches, unmasked crops add 0 as in iBOT. `None` is
  bit-identical to before (checked against HEAD for both centering methods). The logged `ibot_loss` is
  not comparable across modes (`"all"` is ~2x at init, since it also counts the half of the crops that are
  not masked). **Memory:** the iBOT head + loss now run on all selected tokens (~7x the masked ones
  at the default mask settings, each with `output_dim` logits): on the cluster `"all"` OOM'ed at 32 volumes
  per GPU, `"masked"` ran. `ibot_loss_chunk_size` (below) is the fix.
- `DINOv2Args.ibot_loss_chunk_size` / `compile_heads` / `activation_memory_budget` (all off by default).
  Two iBOT paths: the **legacy** one (both of the first two off — today's code, untouched) and the
  **fused** one (`dinov2_loss.chunked_ibot_loss` over `ibot_chunk_loss`): the heads return their L2-normalized
  bottleneck (`DINOv2ProjectionHead.forward(x, bottleneck_only=True)`), and the loss applies both last layers
  (weights taken once per step, so `weight_norm` gradients flow as before), the softmax centering and the
  cross-entropy per token chunk; the center is `apply_center_update()`d before and `update_center()`d with the
  chunks' mean teacher logits after. `ibot_loss_chunk_size` runs every chunk under non-reentrant
  `torch.utils.checkpoint`, so only `[T, bottleneck_dim]` is kept; `compile_heads` alone runs one chunk without
  checkpointing. **One chunk is bit-identical to legacy** (loss, every gradient; same ops in the same order,
  against the non-xFormers `lossfunc`), more chunks differ only by summation order (~1e-16 relative in float64):
  pinned by `TestChunkedIBOTLoss` (`test_dinov2_loss.py`) and `TestIBOTLossChunking` (`test_dinov2.py`, 3 steps
  with SGD + EMA, every gradient/parameter/center). The fused path needs `center_method="softmax"` (Sinkhorn
  normalizes over all tokens): a `DINOv2Args` model validator rejects it, and the budget without a compile option,
  and `train_from_config` validates the method args (`train_helpers.validate_method_args`) before any data is
  loaded. The DINO cls loss stays eager. A rank without iBOT tokens still calls `update_center` (with the current
  center, i.e. no change): it all-reduces under DDP, skipping it would desynchronize the ranks' collectives. Under
  autocast the teacher `[K, d]` weight is cast once per step (same rounding); the student weight is still cast per
  chunk so that its per-chunk gradients are summed in float32. Chunks are checkpointed with
  `preserve_rng_state=False` (no randomness).
  `compile_heads` compiles the **whole heads** (`Module.compile()`) and `torch.compile(ibot_chunk_loss)`.
  Compiling only `head.mlp` would silently do nothing: Dynamo does not trace a module whose forward is torch's
  own code (`nn.Sequential`). The heads are therefore called as `head(x)`, never `head.forward(x)` (that would
  bypass the compiled wrapper). Graphs: 5 for the shared head forward (grad mode x `bottleneck_only`, dynamic
  shapes), 2-3 for the loss (`teacher_temp` is specialized once, then dynamic) — `TestDINOv2CompileHeads`.
  `activation_memory_budget` sets the process-wide `torch._functorch.config.activation_memory_budget`
  (needs a compile option; `train_helpers.warn_if_activation_memory_budget_with_checkpointing` warns when it is set
  with `compile_blocks` and `activation_checkpoint_args.enabled` — the compiled heads are never checkpointed). The CUDA test `test_dinov2_compile_cuda.py` runs every variant
  (`LT_COMPILE_TEST_VARIANTS`, default `ibotpp=all`) plus `test_peak_memory`. Its CPU dry run with a tiny
  `_vittest14` config fails one `blocks.0.attn.proj.bias` gradient for every `compile_blocks` variant —
  identically on the code before these options, so a property of that tiny random-init model, not a regression.
- `src/lightly_train/_models/dinov2_vit/dinov2_vit_src/layers/attention.py` —
  `SDPAttention`, a xFormers-free `scaled_dot_product_attention`-based attention used
  by all `dinov2_vit_src` model builders (`vit_small/base/large/...`, `_vit_test`) —
  needed because xFormers' memory-efficient attention isn't 3D-input-friendly here /
  isn't a dependency of this fork.
- `src/lightly_train/_methods/dinov2/utils.py` — `MaskingGenerator` samples 3D cuboids;
  aspect ratios are anchored to the **grid's own** `height/depth`, `width/depth` ratios
  (mirroring `RandomResizedCrop3D`), not sampled in isolation — needed because the
  patch grid is strongly anisotropic (e.g. 4×16×16) and unanchored ratios make most
  candidate cuboids fail to fit, badly under-masking (see Gotchas).
- `src/lightly_train/_methods/dinov2/dinov2.py` — `training_step_impl` derives the mask
  grid `(d, h, w)` from `global_views.shape` and `self._patch_size` via `math.ceil`
  (matching `PatchEmbed`'s round-up-to-multiple behavior, not floor division).
- `src/lightly_train/_commands/train.py` / `train_helpers.py` — `pretrain()`/`train()`/
  `TrainConfig` take `data_root`, `data_meta`, `series_depth`, `resample_mode` instead
  of `data`. `series_depth <= 0` (native per-series depth) works with plain
  `DataLoader` collation — see Gotchas.
  `get_dataloader()` always sets `worker_init_fn=mi_dataset.worker_init_fn`.
- `src/lightly_train/_models/dinov2_vit/dinov2_vit_src/configs/train/*.yaml` — every
  DINOv2 train config's `patch_size` / `*_crops_size` are 3-element lists (H, W, D).
  The `eval/*.yaml` configs (2D pretrained-checkpoint configs, e.g.
  `vitb14_reg4_pretrain.yaml`) are **intentionally untouched** — they describe
  pre-existing 2D checkpoints, not this fork's 3D training.
- `src/lightly_train/_models/dinov2_vit/dinov2_vit_src/configs/ssl_default_config.yaml`
  — the defaults every model config is merged over, **stripped to what builds the model**:
  the 13 `student` keys `DINOv2ViTPackage.get_model` reads + `crops.global_crops_size`.
  Meta's original training sections (`optim`, `teacher`, `dino`, `ibot`, `train`, crop
  scales, local crops, ...) were never read — lightly-train takes them from `DINOv2Args`/
  `DINOv2AdamWViTArgs` and the transform args — and were removed (verified: the values
  `get_model` reads are unchanged for all `MODELS`). The copied `train/*.yaml`/`eval/*.yaml`
  still set some of those sections; equally dead, left as they are. `crops.global_crops_size`'s
  depth (16) is kept in sync with `DINOTransformArgs.image_size`'s default depth
  (`dino_transform.py`) — see Gotchas.

## Gotchas specific to this fork

- **H/W axis order swap.** Configs/`ImageSizeTuple`/transform args are `(H, W, D)`;
  everything downstream of `PatchEmbed` is `(D, H, W)`. `PatchEmbed.__init__` does the
  one conversion (`(s[2], s[0], s[1])`) — getting that tuple order wrong makes H and W
  silently swap whenever they differ, which is invisible on cubic test fixtures. Prefer
  non-cubic shapes (e.g. `img_size=(16, 8, 4)`, i.e. H≠W≠D) in new tests specifically to
  catch this.
- **RNG reseeding across DataLoader workers.** `RandomResizedCrop3D` (MONAI
  `Randomizable`) and the other MONAI transforms hold a `RandomState` that gets copied
  unchanged into every worker process and across epochs unless explicitly reseeded.
  MONAI's own `monai.data.utils.set_rnd` isn't sufficient here: it gives every item in a
  `list` the *same* seed, so DINO's two global-view `ViewTransform`s (stored in a list
  in `DINOTransform.transforms`) would produce identical crops. `mi_dataset.py` has its
  own `reseed_randomizables()` that seeds list/dict/attribute children independently,
  wired in as `worker_init_fn` by `train_helpers.get_dataloader`. If you add a new
  transform or dataset wrapper, make sure it's still a `Randomizable` (or contains one)
  reachable from `MIDataset` so this reseeding actually reaches it — a plain
  `np.random.default_rng` held outside MONAI's object graph won't be caught.
- **Masking under-fills on anisotropic grids.** The default global-view grid is
  something like 4×16×16 (`D` much smaller than `H`/`W`). Sampling the mask cuboid's
  aspect ratios independent of the grid's own shape makes most candidate cuboids fail
  the `0 < d < depth` etc. bounds check, so `MaskingGenerator` silently returns masks far
  below the requested ratio (verified: without the grid-anchoring fix, a 0.5 target on a
  4×16×16 grid achieved ~0.23 mean, 27% empty masks). If you touch `MaskingGenerator`,
  add/extend a test asserting the achieved ratio is close to the target ratio on a
  strongly anisotropic grid, not just a cube.
- **`pos_embed_grid` is a persistent buffer and every 3D checkpoint must have it.**
  `DinoVisionTransformer` records its own `(D, H, W)` patch grid in a persistent buffer,
  because a flat token count cannot be factored back into `(D, H, W)` — `pos_embed` can
  only be resized across image sizes if the source grid travels with the weights.
  Deliberate consequence: a 3D checkpoint written *before* this buffer existed is
  rejected with a `RuntimeError` even when the shapes happen to match, rather than
  silently guessing. Do **not** reintroduce a `sqrt`-based grid recovery as a fallback:
  the old 2D hook did that, and a `(D,H,W)` grid whose product is a perfect square
  (`4*16*16 = 1024`, `2*4*8 = 64`) passed its guard and silently produced a
  correctly-*sized*, semantically wrong `pos_embed`. Pick a grid with a square product
  for any new test here (`tests/_models/test_model_helpers.py` uses `(2, 4, 8)`).
- **Loading the public 2D DINOv2 checkpoints needs its own config, for every size.**
  Verified against the real downloaded checkpoints (`inspect_dinov2_checkpoint.py`,
  a standalone script reading state_dict shapes/keys directly, ask for it if it isn't
  around), not just inferred from the `eval/*.yaml` configs:
  the stock `train/vitb14{,_reg4}.yaml` and `train/vitl14{,_reg4}.yaml` set
  `ffn_layer: swiglufused` (and the `_reg4` ones also `block_chunks: 4`), but the
  public ViT-B/L checkpoints (reg or noreg) have plain MLP FFNs and flat `blocks.{i}`
  keys — every block parameter would be named differently (`mlp.w12` vs `mlp.fc1`,
  `blocks.0.0` vs `blocks.0`) and nothing would load. `vits14{,_reg4}.yaml` already
  defaults to `mlp`/`block_chunks: 0`, so no correction was actually needed there, but
  the `_2dinit` configs restate it explicitly anyway for consistency. ViT-g/14 is the
  one size that genuinely is SwiGLU upstream, so its `_2dinit` configs only correct
  `block_chunks` (4 -> 0), not `ffn_layer`. That is what the eight
  `*_{reg4,}_2dinit.yaml` configs exist for. The existing `vits14`/`vitb14`/`*-noreg`/
  ... entries are *not* usable for this: they point at the intentionally-2D
  `eval/*.yaml`, whose scalar `patch_size`/`global_crops_size` trip `PatchEmbed`'s
  length-3 asserts. Watch the URL when adding one of these by hand: a copy-paste
  across sizes can silently point at the wrong model's checkpoint directory while
  still downloading *something* (caught once already — `vitg14-reg4-2dinit`'s URL
  briefly pointed at `dinov2_vitl14/`); `test_2dinit_model_configs` asserts the URL
  path matches the entry's own name for exactly this reason.
- **`antialias` + `trilinear`**: PyTorch's `F.interpolate` raises if `antialias=True` is
  passed with a mode outside bilinear/bicubic/lanczos. Several DINOv2 configs
  (`vitb14_reg4`, `vits14_reg4`, `vitg14_reg4`, `vitl14_reg4`) set
  `interpolate_antialias: true` (a 2D-era default) — the 3D pos-embed interpolation code
  must never forward that flag to the trilinear `F.interpolate` call.
- **`degrees` vs radians**: `RandomRotationArgs.degrees` is degrees (matches the 2D
  albumentations convention elsewhere in this repo), but MONAI's `RandRotate`
  (`range_x/y/z`) expects **radians**. Conversion happens once, in `view_transform.py`.
- **`series_depth <= 0` (native depth) works here without batching by depth, on
  purpose.** KneeNo has no depth-bucket sampler any more (removed 2026-09-28; vjepa2
  now requires `series_depth > 0`), and this repo never needed one:
  `RandomResizedCrop3D` always resizes its crop to a *fixed configured output size*
  regardless of the input volume's native depth, so every `MIDataset` item has
  identical view shapes and plain `DataLoader` collation works across series of
  different native depths. The same holds in evaluation: `DINOv2Adapter.prepare_input`
  resizes depth too, so `eval.data.series_depth <= 0` works as well (KneeNo's evaluator
  raises a clear `ValueError` for an adapter that doesn't). Don't reintroduce a
  `series_depth <= 0` restriction without re-checking these invariants first (covered by `test_get_dataloader__mi_dataset_series_depth_not_positive` in
  `tests/_commands/test_train_helpers.py` and
  `test_pretrain__series_depth_not_positive` in `tests/_commands/test_train.py`).
- **Keep `crops.global_crops_size`'s depth and `DINOTransformArgs.image_size`'s depth
  in sync.** The model's positional embedding is sized from
  `ssl_default_config.yaml`'s `crops.global_crops_size` (H, W, D) at construction
  time; the actual global view fed to the model at train time is sized from
  `DINOTransformArgs.image_size` (`dino_transform.py`). If these depths (H/W already
  both 224) drift apart, `interpolate_pos_encoding` silently re-interpolates the
  positional embedding on every single training step — not wrong, just wasteful, and
  `PatchEmbed.forward` also logs at `info` level on every such mismatch. Covered by
  `test_vittest14__global_crops_size_matches_default_transform_image_size` in
  `tests/_models/dinov2_vit/test_dinov2_vit_package.py`. There is no local-crop
  counterpart: local views are sized by `DINOv2ViTLocalViewTransformArgs.view_size` alone
  (`crops.local_crops_size` was never read and has been removed from the default config).
- **`get_fused_param_groups` fusion trap for any new name-based param-group rule.**
  `get_optimizer_with_decay` (`_methods/dinov2/utils.py`) fuses param groups keyed on
  `(lr, weight_decay, foreach, head, last_layer, tokenization)`. Before the
  `tokenization` key was added, `layerwise_decay=1.0` (the documented "long setup"
  value — every layer then gets the *same* lr) fused `cls_token`/`pos_embed`/
  `mask_token` into the same group as 12 block weight tensors (verified: 4 groups
  instead of 10). Any future rule in `DINOv2.on_before_optimizer_step` that selects
  param groups by a substring of `group["name"]` is only correct if that same
  property is part of the fusion key — see `is_tokenization_param` and the
  `test_get_fused_param_groups__never_mixes_tokenization` regression test
  (`tests/_methods/dinov2/test_utils.py`), parametrized over both decay values for
  exactly this reason.
- **Freezing in this method is always `lr=0.0`, never `requires_grad=False`.**
  `get_optimizer_with_decay` builds param groups **once**, filtering on
  `requires_grad` (`utils.py`, `if not param.requires_grad: continue`) — a parameter
  frozen at that moment is permanently absent from the optimizer, so a later
  "unfreeze" would silently never train it unless the optimizer is rebuilt, which then
  breaks Lightning's *positional* restore of optimizer state on `resume_interrupted`.
  Separately, Lightning's `DDPStrategy.setup()` calls `configure_ddp()` (which builds
  the DDP reducer from whatever `requires_grad=True` at that instant) **before**
  `setup_optimizers()`, so unfreezing at step N under multi-GPU would give those
  params local-only gradients with silent rank divergence. `student_freeze_backbone_steps`,
  `student_freeze_last_layer_steps` and `n_tokenization_only_steps` all instead derive
  their frozen set from `self.trainer.global_step` on *every* call to
  `on_before_optimizer_step` and write `update["lr"] = 0.0` — no `__init__`-time
  freeze, no unfreeze hook, correct at any `resume_interrupted` step by construction.
  One consequence: `checkpoint=` restarts `global_step` at 0 (see `load_checkpoint`),
  so any of these three step-counted phases re-runs from the start on that path —
  already true of the two pre-existing switches, not new with
  `n_tokenization_only_steps`.
- **`on_fit_start` sees a stale `self.trainer.global_step` under
  `resume_interrupted=True`; use `on_train_start` instead for anything that needs the
  restored step.** Verified by reading `pl.Trainer._run()`: it calls
  `call._call_lightning_module_hook(self, "on_fit_start")` **before**
  `self._checkpoint_connector.restore_training_state()` — which is what actually
  restores the fit loop's `global_step`/`current_epoch` from the checkpoint (model
  weights and callback state are restored earlier, but not the loop counters).
  `on_train_start` fires later, from `fit_loop.run() -> on_run_start()`, which is
  called from `_run_stage()` — itself called *after* `restore_training_state()` — so
  by then `global_step` is correct. Confirmed empirically with a real two-run
  `resume_interrupted=True` test: a status log placed in `on_fit_start` read
  `global_step=0` on the resumed run even though the checkpoint held `global_step=2`;
  the same log in `on_train_start` correctly read `2`. This is why
  `DINOv2.on_train_start` (not `on_fit_start`) owns the `n_tokenization_only_steps`
  status log — `warn_if_steps_too_low` stays in `on_fit_start` since it only needs
  `estimated_stepping_batches`/`max_epochs`, which don't depend on this restore.
  Also note `DINOv31.on_fit_start` overrides DINOv2's without calling `super()` (to
  skip the from-scratch step recommendation), so anything placed in `on_fit_start`
  here is silently skipped for DINOv3.1 — another reason the new logging lives in
  `on_train_start`, which DINOv31 does not override.

## Testing

No GPU here, so verify with small synthetic fixtures on CPU using the smallest model
(`dinov2/_vittest14` / `_vit_test`). Run tests with:
```bash
.venv/bin/python -m pytest tests/_methods/dinov2 tests/_models/dinov2_vit \
  tests/_transforms/test_random_resized_crop.py tests/_transforms/test_view_transform.py \
  tests/_transforms/test_monai_wrappers.py tests/_transforms/test_transform_equivalence.py \
  tests/_data/test_mi_dataset.py tests/_commands/test_train.py \
  tests/_commands/test_train_helpers.py tests/_models/test_embedding_model.py \
  tests/test_activation_checkpointing.py tests/test__cli.py \
  tests/_commands/test_eval_classification.py tests/test_cluster_configs.py
```
or the whole suite with `.venv/bin/python -m pytest tests` (slow-ish; the full suite is
~2100 files/cases, mostly for methods/models this fork doesn't touch).

**Quarantine of unrelated breakage:** `tests/conftest.py` has a `collect_ignore` list
(whole files where every test broke, e.g. SimCLR/DINO/DenseCL/DetCon/distillation*
transform tests, `test_oriented_object_detection_transform.py`) and a
`_BROKEN_BY_3D_DINOV2_ADAPTATION` frozenset of individual node IDs skipped in
`pytest_collection_modifyitems` (LT-DETR/EoMT/depth-estimation/image-classification task
models, 2D dataset tests, other SSL methods' transform-shape tests, etc.) — these are
**broken by design** (2D-only code paths hitting the now-3D-only `ImageSizeTuple`,
`PatchEmbed`, `ViewTransform`, `EmbeddingModel`'s `Conv3d` head, or `pretrain()`'s new
`data_root`/`data_meta` signature) and are out of scope per the Scope rule above. If you
add a new unrelated-breakage case, add it there rather than trying to fix the 2D code
path or reverting a 3D fix to accommodate it. If you fix something that make an entry
there pass again, remove it from the list/set — don't leave a passing test skipped.

**Known-good end-to-end check** (real `lightly-train pretrain` CLI, not just unit
tests): build a small KneeNo-layout dataset with `tests/helpers.py::create_mi_dataset`,
then run `lightly-train pretrain out=... data_root=... data_meta=... series_depth=8
resample_mode=nearest model=dinov2/_vittest14 method=dinov2 epochs=1 batch_size=4
accelerator=cpu num_workers=2 transform_args.image_size=[56,56,16] ...` — confirmed
working (finishes, exports a model) as of this writing. `num_workers=2` matters: it's
what catches both dataset picklability issues and the RNG-reseeding gotcha above (with
`num_workers=0` everything runs in one process and reseeding bugs are invisible).
`series_depth=0` (native per-series depth) works the same way — see Gotchas.

**`tests/helpers.py` additions for this fork:** `dummy_dinov2_vit_model(patch_size=(2,2,2), ...)`
(3D `_vit_test`), `create_mi_dataset(root, n_cases, depths, height, width, identical=...)`
(writes a KneeNo-layout dataset — JPEG slices + metadata JSON with
`"dimensions": [H, W, depth]`, KneeNo's current schema — the `identical=True`
option makes all series bit-identical, used to prove RNG reseeding actually changes
output across workers), `get_method_dinov2()` (now 1-channel), and
`MI_DINOV2_TRANSFORM_ARGS` (small transform sizes for fast end-to-end tests), and
`dinov2_2d_checkpoint(model)` (a state_dict shaped like an original 2D DINOv2
checkpoint — `Conv2d` kernel, 37×37 `pos_embed`, no `pos_embed_grid` — verified
against the real `dinov2_vits14_reg4_pretrain.pth` key list, used to test the
2D → 3D load path without downloading 88 MB).

## KneeNo classification evaluation

Frozen-encoder, multi-label classification (k-NN, linear, linear-pool, attentive-pool)
lives in `KneeNo/kneeno/evaluation/` and is shared with vjepa2 — see `KneeNo/README.md`
for the tasks themselves. This repo supplies the bridge and the wiring, mirroring
vjepa2's. **Unlike V-JEPA 2.1, DINOv2 has a cls token**, so the `linear` task is
available here and is enabled in the config.

- `src/lightly_train/_data/kneeno_adapter.py` — `DINOv2Adapter(EncoderAdapter)`.
  `has_cls_token = True`. `prepare_input` is the deterministic counterpart of
  `_transforms/view_transform.py::ViewTransform`: `NormalizeIntensity`'s
  `(x - m*255) / (s*255)`, preceded by a resize of **all three** axes to the
  training `transform_args.image_size` with the training resampler itself
  (`random_resized_crop._resample` with the run's `transform.resize_interpolation` /
  `resize_upscale_interpolation`, `DINOTransformArgs` fields since 2026-10-05, plus
  notation supported, validated in `__init__`; default
  area, linear on enlarged axes -- the values previously hard-coded, so older runs whose
  config lacks the keys are unaffected; `null` upscale = area everywhere -- bitwise
  `RandomResizedCrop3D` at `scale=ratio=1`). The callback gets them from the transform
  args via `train.py` -> `get_callbacks`; `eval_classification` reads them from the
  top-level `resize_interpolation` / `resize_upscale_interpolation` keys of its eval
  config file, falling back to the defaults with a warning. Not `F.interpolate(trilinear)` (used until
  2026-10-01): it has no 3D anti-aliasing, and on real 672px exams its 672->224 output
  (pure every-3rd-pixel subsampling) had more in-plane high-frequency energy than nearly
  every training view, so eval numbers from before the switch are not comparable. Resizing depth too (vjepa2's adapter leaves it
  native) is what matches this repo — `RandomResizedCrop3D` always emits a fixed
  configured size during training, so the encoder never saw a variable depth.
  `forward_features` flattens the wrapper's `(B, C, pD, pH, pW)` features to KneeNo's
  `(B, P, C)`. External (raw NIfTI) volumes are first quantized with
  `LabeledExternalKneeMRIDataset.to_uint8`, decided per volume by the `source` argument KneeNo passes
  (`"internal"` / `"external"`, from the dataset's `(volumes, labels, source)` items; anything else raises).
  The adapter has no `dataset_type` argument (removed 2026-10-12), so `eval.data.dataset_type: mixed`
  (train/val on internal, test on every external exam) works without changes here.
- `src/lightly_train/_callbacks/kneeno_eval.py` — `KneeNoEval(Callback)` +
  `KneeNoEvalArgs`. **Off by default**: enabled only by `callbacks={"kneeno_eval":
  {"config": <eval dict>}}` (the `config` dict is required and deep-merged over KneeNo's
  `DEFAULT_EVAL_CONFIG`). There is deliberately **no shipped default eval config** any
  more — the cluster script passes the `eval:` block of its run config. Runs in
  `on_train_epoch_end` on whatever `kneeno.evaluation.tasks_due(epoch, config["freq"])`
  returns; each task contributes one point (its test metrics) per pretraining epoch. `on_train_end` calls `ClassificationEvaluator.cleanup()` (flushes
  and closes KneeNo's own TensorBoard writer) if an evaluator was ever built.
- `cluster/configs/pretrain-MI-vitb14-24f.yaml` — the run config's `eval:` block (the
  former `_configs/kneeno_eval.yaml`, removed), copied from vjepa2's with `linear` enabled
  and `logging.tensorboard_dir: null`. `tests/test_cluster_configs.py` checks both. It also
  spells out an `attentive_pool` block (off in training) matching the eval config's.
- `src/lightly_train/_commands/eval_classification.py` — the standalone
  `lightly-train eval_classification` command (same interface as on `2d_dinov2` since
  2026-10-11). One `evaluate(..., epoch=0)` call: each task logs its test metrics once (KneeNo's
  train/val/test protocol, heads early-stopped on val; `eval.verbose` logs the val metrics per head epoch).
  Far shorter than vjepa2's script: a LightlyTrain checkpoint stores the model objects, so
  there is no architecture to rebuild.
  - **Arguments:** only `eval_config`, `accelerator` and `overwrite`. Everything else is in
    the eval config file, validated by `EvalRunConfig` (`extra="forbid"`, so a leftover
    `transform:` or `out:` is an error; a `params-pretrain.yaml` is no longer accepted):
    `checkpoint`, `image_size` (`[H, W, D]`, required: the checkpoint doesn't record it),
    optional `resize_interpolation` / `resize_upscale_interpolation` (validated like the
    transform args; a missing key falls back to `DINOv2ViTTransformArgs`' default **with a
    warning**, detected via `model_fields_set` because `null` is a valid upscale value) and
    `eval`. Env vars are expanded in the whole file; **an unset one is an error**
    (`expand_env_vars` leaves it as it is).
  - **Tasks** are those whose `eval.freq` is set (null or `<= 0` is off, `tasks_due`'s
    semantics), computed on the merged config, so a missing key keeps KneeNo's default; no
    task is an error. Slice tasks, if enabled, are dropped by KneeNo with a warning (this
    adapter has no slice tokens). The **encoder** is `eval.encoder` (`target` | `online`).
  - **Outputs:** `results.json` (final metrics) and `params.yaml` (the config file with env
    vars expanded, written *before* evaluating) in the parent of `eval.logging.per_label_dir`;
    without a `per_label_dir`, a warning, `results.json` next to the checkpoint and no
    `params.yaml`. Existing results (`results.json`, `params.yaml`, `per_label_dir`,
    `tensorboard_dir`) make it refuse before the checkpoint is loaded; `overwrite=True` runs
    anyway (per-label CSVs are appended to, TB gets a second event file).

**`evaluate()` is called on every rank, deliberately and with no rank guard.**
`ClassificationEvaluator` does the work on rank 0 and then hits `dist.barrier()` +
`dist.broadcast_object_list()`. Guarding the call site would leave rank 0 waiting at a
barrier the other ranks never reach, deadlocking the run. Do not "optimize" this into an
`if trainer.global_rank == 0`.

**Metrics go through lightly-train's loggers, not KneeNo's.** The config sets
`logging.tensorboard_dir: null`, disabling KneeNo's own `SummaryWriter`, and the callback
re-logs the returned `{"{task}/{metric}": float}` dict via `pl_module.log_dict` under an
`eval/` prefix — so eval scalars land in the same TensorBoard/JSONL run as the training
curves, and no unpicklable `SummaryWriter` is ever held by a callback (see the
`collate_fn` picklability gotcha in `MA/CLAUDE.md`).

**Config trap:** `load_eval_config` deep-merges over `DEFAULT_EVAL_CONFIG`, so disabling a
task needs an explicit `freq: {<task>: null}` — omitting the key leaves the default on.

## Running on the cluster

`cluster/submit_pretrain_dinov2_kneeno.sh` (sbatch, 1 GPU) extracts the internal dataset to
`/dev/shm/kneeno_data/internal` with `KneeNo/data/prepare_data.py --internal-tar-dir` (pretraining and the
`dataset_type: internal` eval both read it; the external NIfTI dataset is not extracted), then runs
`cluster/pretrain_dinov2_kneeno.py --config <run config>`, which calls `lightly_train.pretrain(method="dinov2", ...)`.
**Fully config-driven:** the script's only arguments are `--config` and `--pg-timeout-minutes` (multi-GPU
process-group timeout — deliberately kept as a flag: an infrastructure knob, no effect on training).
**Run config** (vjepa2-style, default `cluster/configs/pretrain-MI-vitb14-24f.yaml`): top-level `model:`
(a string, e.g. `dinov2/vitb14-notpretrained`, -> `model`), `data:` (`out`, `data_root`, `data_meta`, `series_depth`,
`resample_mode` — named like `pretrain`'s arguments and passed as `**data`), optional `train:` (further `pretrain`
arguments — `epochs`, `batch_size`, `num_workers`, `precision`, `trainer_args`, ...; allowlisted in
`run_config.TRAIN_KEYS`, unknown keys rejected, also passed as-is), `method:` (-> `method_args`),
`transform:` (-> `transform_args`), `callbacks:` (-> `callbacks`, lightly-train's own callbacks by name, e.g.
`model_checkpoint: {every_n_epochs: 50, save_top_k: -1}` for a checkpoint every 50 epochs besides `last.ckpt`;
`save_top_k` must be `-1`, the default `1` keeps only the newest) and `eval:` (-> the eval callback's `config`; the
script merges it into `callbacks` as `kneeno_eval`, and `callbacks.kneeno_eval` is rejected), loaded and validated by
`cluster/run_config.py::load_run_config` (the script imports it as `from run_config import ...`; running the script puts
`cluster/` first on `sys.path`). **`model` and `data` with
all five keys are required.** `load_run_config` runs KneeNo's `expand_env_vars` over the whole config
(`$VAR`, `${VAR}`, leading `~`, like vjepa2's `app/main.py`), and rejects unknown top-level and `data` keys
and an `eval.data.series_depth`/`resample_mode` that differs from `data:`'s. A missing `method`/`transform`
block falls back to lightly-train's defaults; **a missing `eval:` block means no evaluation**. The shipped
config lists every `DINOv2Args`/`DINOv2ViTTransformArgs` default explicitly (verified equal to the old
implicit defaults). The script passes `params_file=<config>`, a new `pretrain` arg: global rank 0 copies the
file verbatim (unexpanded; after the out-dir check, so a fresh run's out dir isn't "non-empty") to
`<data.out>/params-pretrain.yaml`, or `params-pretrain-1.yaml`, `-2`, ... when it exists (resume), like
vjepa2's `_get_safe_path` (`common_helpers.copy_params_file`) — since nothing else sets run settings, that
copy is the complete record. Next to it, `pretrain` writes `model-config.yaml`
(`common_helpers.write_model_config`, any `dinov2/<name>` model, rank 0) **only if it doesn't exist yet**, so a
resume keeps the original: `DINOv2ViTPackage.get_model_config` — the model's own config merged over
`ssl_default_config.yaml`, restricted to `student` + `crops.global_crops_size`, which is exactly what
`get_model` builds from (it uses the same method), plus a `model_args` section if any were passed. The rest of the
merged config (optim/teacher/dino/ibot/train) is left out on purpose: lightly-train never reads it
(`DINOv2Args` drives those), so recording it would be misleading. Model size (`embed_dim`, `depth`, ...) is
hardcoded per `arch` in the builders, so `arch` is what records it. Whatever `train:` leaves out (optimizer,
epochs, batch size, ...) stays at its default. **`model:` must name a 3D model config**: `dinov2/vitb14` (and the
other plain / `-noreg` / `-pretrained` names) point at the 2D `eval/*.yaml` and fail `PatchEmbed`'s 3D asserts at
build time — the shipped config named it until 2026-10-01 and would have crashed on the first cluster run. Random
init is `dinov2/vitb14-notpretrained` (reg4, `swiglufused` FFN); `*-2dinit` loads the public 2D blocks (plain MLP FFN
— so the two are not the same architecture). `tests/test_cluster_configs.py::test_run_config__model_builds` builds
every shipped config's model. Submit
from the repo root; the submit script takes `CONFIG`, `REPO_DIR`, `KNEENO_DIR`, `PG_TIMEOUT_MINUTES` env vars
(`OUT_DIR`/`MODEL` are gone — they live in the config).
Rerunning with the same `data.out` auto-resumes from `checkpoints/last.ckpt`. The `#SBATCH` resource
lines are **guesses** (no earlier cluster script for this repo existed) — verify them on the first
submit. The eval callback only *warns* when its labeled data is missing, so grep the job log for
"Disabling KneeNo evaluation".
Verified locally end to end (tar -> `prepare_data.py` -> script with eval running) on CPU with a tiny
model; the real ViT-B/24-slice run has not been executed yet.

**`OPENBLAS_NUM_THREADS=1`** is set at the top of `pretrain_dinov2_kneeno.py`, before numpy is
imported. Without it, numpy's bundled OpenBLAS starts one thread per core in *every* DataLoader worker
(DataLoader only limits torch's threads), and `RandomResizedCrop3D`'s `np.tensordot` (its dense path; strong
reductions now go through scipy sparse, see "CPU fast paths") runs on it: the first
cluster run (31 workers on 32 cores) never delivered a batch within the 360 s DataLoader timeout (~15 s per
sample instead of ~1 s). Threads don't help here anyway: even alone, the multi-threaded version was slower.
Anything else that drives this pipeline with many workers needs the same.

**`OMP_NUM_THREADS=1`** (forced, before torch is imported) is set there too, for the *main* process. It must be
forced, not `setdefault`: the cluster's job environment already sets `OMP_NUM_THREADS=32` (not from our scripts),
so a `setdefault` version did nothing there. Without it, torch
uses one OpenMP thread per core for CPU tensor ops, on cores the workers keep busy, and each parallel
region waits for its slowest (descheduled) thread. A py-spy profile (2026-10-06) showed `mask_indices_and_weights`
(nonzero/sum/index on a tiny bool mask) at 12.6% of wall time, spinning in libgomp, with the GPU at ~75%
utilization; with the env var the GPU went to >90%. Per-core CPU stats hide this (and a CPU-bound training loop):
the backward runs on autograd's own CUDA thread while the main thread sleeps, so neither thread shows 100% — profile
the process with `py-spy record --idle --native` instead. `data_wait` can't show it either: its CUDA `batch_time` is
the GPU-stream time between two events, gaps included. Related: avoid `torch.tensor(<scalar>, device=cuda)` in the
step — the pageable host-to-device copy synchronizes with the stream (it did in `DINOLoss.forward`, now `torch.zeros`).

**Multi-GPU / multi-node** needs only SLURM options (override on the `sbatch` command line, `#SBATCH`
lines can't be computed): `--ntasks-per-node` = GPUs per node, launched with `srun` (the script does
this; it also checks the two match). `devices="auto"` and rank/world size come from Lightning's
`SLURMEnvironment` (`SLURM_NTASKS`), **but `num_nodes` does not** — lightly-train uses `pretrain(num_nodes=)`
(default 1) to split the global `batch_size` across devices, so the script passes `SLURM_NNODES`. `--cpus-per-task`
is per rank (`num_workers="auto"` = that minus 1, per rank). **Process-group timeout:** Lightning passes 30 min
explicitly (torch's NCCL default of 10 min is overridden). KneeNo's evaluator runs on rank 0 while all other ranks
sit in `dist.barrier()`, so one eval round longer than the timeout kills a multi-GPU job; the script therefore
passes `DDPStrategy(find_unused_parameters=True, timeout=...)` (what lightly-train's own `"auto"` picks, plus the
timeout) whenever `SLURM_NTASKS > 1`, default 240 min (`--pg-timeout-minutes` /
`PG_TIMEOUT_MINUTES` in the submit script) — **set it above your longest eval round**. Verified locally with a
real 2-rank CPU DDP run through the script (timeout reached `init_process_group`, eval + barrier completed);
multi-node and real NCCL have not been run.

**Standalone eval** (since 2026-10-11, same as on `2d_dinov2`): `sbatch --export=ALL,PRETRAIN_JOB_ID=<id>
cluster/submit_eval_dinov2_kneeno.sh`. **Pass env vars with `--export`**: RWTH's `sbatch` wrapper submits with
`--export=NONE`, so a `PRETRAIN_JOB_ID=<id> sbatch ...` prefix never reaches the job (the script then stops at its
check).
- The sbatch script requests 1 GPU and 1 task (single process: KneeNo evaluates on rank 0 only, there's no process
  group and no `--pg-timeout-minutes`). It extracts the internal data to the job's node-local `$TMP/kneeno_data`
  (`prepare_data.py`'s default; `/dev/shm` is for pretraining only), then runs
  `cluster/eval_dinov2_kneeno.py --config <eval config>` (default `cluster/configs/eval-MI-vitb14-24f.yaml`). Env
  vars: `PRETRAIN_JOB_ID` (required) and `TMP`, both checked up front; `CONFIG`, `REPO_DIR`, `KNEENO_DIR`.
- The launcher only sets the same thread/allocator env vars as the pretraining launcher, then calls
  `lightly_train.eval_classification(eval_config=...)` (`accelerator="auto"`, so it also runs on CPU here; verified
  end to end on CPU with a tiny model, a JPEG fixture and 2 spawn workers).
- The eval config locates the pretraining run through **`${PRETRAIN_JOB_ID}`, not `${SLURM_JOB_ID}`**, which in the
  eval job is the eval job's own ID. Its outputs go to `<run>/eval/internal/post-training/` (`tb/`,
  `eval_per_label/`, `results.json`, `params.yaml`).
- Its `eval:` block equals the pretraining config's except for `logging` (`tensorboard_dir` is set: no lightly-train
  loggers here), `freq` (the four volume tasks `1`, i.e. on, the slice tasks `null`; every key listed),
  `split` (`0.8/0.1/0.1` train/val/test), `data.subset: 1.0` (all exams), `data.data_root` (`$TMP/...`), `data.num_workers: 31`,
  and the head settings of tasks the pretraining run doesn't run (`attentive_pool`). `tests/test_cluster_configs.py`
  pins that (after merging both over KneeNo's defaults), and that `checkpoint`, `image_size` and the two resize keys
  match the pretraining config (`eval-<name>.yaml` <-> `pretrain-<name>.yaml`).

## Open risks / known follow-ups (not yet done, not blocking)

- **Eval now uses the real labeled dataset** (`KneeNo/kneeno/evaluation/dataset.py::LabeledInternalKneeMRIDataset`,
  exam-level: one sample = the four sequence volumes of one exam, and every `eval.*.batch_size` /
  `eval.data.subset` counts exams; see `MA/CLAUDE.md`). The callback tests still inject a fake
  dataset (exposing `sequences` and returning a tuple of four volumes per item), which is
  why they need neither data nor `SimpleITK`; a real run needs `SimpleITK` in this venv
  (`uv pip install -e ../KneeNo` pulls it in — it was missing here and has been installed with
  `uv pip install simpleitk`; like `kneeno`, a plain `uv sync` removes it again).
- **`eval.data.dataset_type: external` / `mixed`** need the external NIfTI data, which the cluster scripts don't
  extract (`prepare_data.py --external-tar` would have to be added); the shipped configs use `internal`.
- **`eval.data.data_root`/`label_meta` in the cluster run config are cluster paths** copied
  from vjepa2's config (`/dev/shm/kneeno_data/internal`, `.../BigKneeTar/metadata_labeled.json`), so they do not resolve on a dev box. The callback logs a warning
  and disables itself rather than failing the run; pass a run config whose `eval:` block
  has local paths to actually evaluate here.
- **`DINOv2ViTModelWrapper._input_mean`/`_input_std`** (TIPSv2 support) are still
  registered as 4-D `(1, C, 1, 1)` buffers — irrelevant to this project (no TIPSv2
  usage) but would break if the TIPSv2 `input_normalization="none"` path were ever
  exercised with a 3D input.
