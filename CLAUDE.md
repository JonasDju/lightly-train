# CLAUDE.md

## What this branch is

Branch **`2d_dinov2`** of our fork of **Lightly's `lightly-train`**: the **vanilla 2D DINOv2 baseline** of the
thesis comparison against the 3D-native DINOv2 on branch `3d_dinov2`. Both are pretrained on the same unlabeled
knee-MRI data (sibling repo `KneeNo`) and scored with the same KneeNo classification evaluation.

**Keep training stock.** The point of this branch is an unmodified DINOv2 training procedure: lightly-train's
own image-folder dataset over the JPEG slices, its own 2D transforms, its own defaults. Changes belong here only
if they don't alter training: evaluation wiring, run bookkeeping, cluster tooling. Anything that changes what
the model sees or how it learns needs a deliberate decision, and a note in this file.

**Deliberate deviation: attention uses `SDPAttention` (2026-10-21).** Every `dinov2_vit_src` builder
(`vit_small/base/large/giant2/so400m`, `_vit_test`) builds its blocks with `SDPAttention`
(`layers/attention.py`, ported verbatim from `3d_dinov2`) instead of stock `MemEffAttention`, in training and
evaluation alike. Stock needs xFormers, which isn't installed (an optional extra). Without it `MemEffAttention`
falls back to `Attention`, which materializes the full `(B, heads, N, N)` score matrix (plus a float32 softmax
copy under autocast). At 518² (1374 tokens) the official-weights eval only fit `extraction_batch_size: 4`, at
~48 GB. `SDPAttention` computes the same thing through `F.scaled_dot_product_attention`, with the same parameters
and state-dict keys. Results differ by float rounding only, older checkpoints and exports load unchanged, and a
resumed pretraining run just switches kernels. `tests/_models/dinov2_vit/test_attention.py` pins equality with
`Attention` and that every builder uses it.

**Current status (2026-10-08):** the KneeNo in-training eval is wired in, with a slice-wise adapter that hands
KneeNo the per-slice tokens, so KneeNo's 2D-only slice tasks run too (see below). `cluster/` holds SLURM scripts,
launchers and configs for pretraining and for the standalone evaluation. Pretraining and the standalone eval run
end to end on CPU through the cluster launchers. Nothing has run on the cluster yet.

## Branches

The repo has three long-lived branches (restructured 2026-10-07):

- **`main`**: stock lightly-train plus the KneeNo dependency (`KNEENO.md`, `exclude-newer` pin) and the
  `pyproject.toml`/`uv.lock` shared by all branches. It's the common base, with no training changes.
- **`2d_dinov2`** (this branch): the 2D baseline described here.
- **`3d_dinov2`**: the 3D-native DINOv2 (patch embedding, attention, masking, MONAI transforms, KneeNo volume
  dataset). It has its own, much longer `CLAUDE.md`. Several things here were copied from there (eval callback,
  `eval_classification` command, `cluster/`, `params_file`, `write_model_config`). When changing them, check whether the other branch needs the
  same change.

**This `CLAUDE.md` is tracked per branch**, so it describes this branch only and moves with the code. Facts that
don't depend on the branch go in the global `MA/CLAUDE.md`.

## Environment

- Python **3.13**, `torch==2.13.0+cu130`, dependencies managed with **`uv`**. Run Python as **`.venv/bin/python`**.
- **Dependencies are shared through `main`.** `pyproject.toml` and `uv.lock` are identical on all three branches
  (copied from `3d_dinov2` onto `main` on 2026-10-07), so one `.venv` serves every branch without re-syncing.
  `monai` is in it although this branch never imports it. Never `uv add`/`uv lock` on this branch alone: change
  the dependencies on `main` and merge it into the others. A re-lock causes a huge `uv.lock` diff, because of
  uv's resolution-marker forks.
- **`kneeno`** is deliberately *not* in `pyproject.toml`/`uv.lock` (see `KNEENO.md`): after `uv sync`, run
  `uv pip install -e ../KneeNo` again (or use `uv sync --inexact`), and check that `import kneeno` works before
  trusting a test run.
- **No usable GPU on the dev box** (old driver): verify on CPU with `dinov2/_vittest14` and small synthetic
  images.

## KneeNo classification evaluation

Frozen-encoder, multi-label classification (k-NN, linear, linear-pool, attentive-pool, plus the slice tasks
`linear_slice_cls`, `attentive_slice_cls`, `attentive_slice_pool` for slice-wise models) lives in
`KneeNo/kneeno/evaluation/` and is shared with vjepa2 and `3d_dinov2`; see `KneeNo/README.md`. KneeNo evaluates
**exams**, handing the adapter one `(1, D, H, W)` volume per sequence.

- `src/lightly_train/_data/kneeno_adapter.py`: `DINOv2Adapter(EncoderAdapter)`, treating a volume as `D`
  independent slices.
  - `has_cls_token = True` (so the `linear` task is available) and `has_slice_tokens = True` (so the slice tasks
    are). The constructor takes `embed_dim`, `num_channels`, `image_size` `(H, W)`, `normalize` (no
    `dataset_type` since 2026-10-12).
  - `num_channels` is the model's resolved input-channel count (`transform_args.num_channels`, 3 with the stock
    `auto` + ImageNet normalize), so `prepare_input` knows to repeat the grayscale volume to RGB, as vjepa2's
    `n_channels` does for the official checkpoints. `normalize` must have 1 or `num_channels` values (checked).
  - `prepare_input`: external volumes go through `LabeledExternalKneeMRIDataset.to_uint8` first, decided per
    volume by the `source` argument KneeNo passes (`"internal"` / `"external"`, from the dataset's
    `(volumes, labels, source)` items; anything else raises), so an evaluation with both test sets
    (`eval.data.test_sets: [internal, external]`: internal train/val, scored on internal and external exams) works
    without changes here. The volume is
    repeated to `num_channels`, then every slice goes through lightly-train's `EmbeddingTransform` (resize to
    `image_size`, no crop; normalize with `max_pixel_value=255`). Output `(C, D, H, W)`, depth kept, so
    `eval.data.series_depth` must be `> 0` (KneeNo raises a clear error for mixed depths otherwise).
  - `forward_features`: all `B * D` slices in one encoder call. It returns only the per-slice tokens,
    `slice_cls` `(B, D, E)` and `slice_patches` `(B, D, P, E)`. KneeNo derives the volume-level views
    (`kneeno.evaluation.features.feature`): the cls token is the slice cls tokens pooled with `knn.pooling` /
    `linear.pooling` (avg | max), and `patches` are all slices' patch tokens in slice order (`D * P` per volume).
    The slice count must equal `eval.data.series_depth`. Each encoder forward sees `series_depth` times as many
    images as volumes: with the shipped eval config, 32 exams x 4 sequences x 24 slices = 3,072 images of 224².
- `src/lightly_train/_callbacks/kneeno_eval.py`: `KneeNoEval` + `KneeNoEvalArgs`, copied from `3d_dinov2` minus
  its 3D resize-interpolation arguments.
  - **Off by default:** enabled only by `callbacks={"kneeno_eval": {"config": <eval dict>}}`, deep-merged over
    KneeNo's `DEFAULT_EVAL_CONFIG`.
  - Runs in `on_train_epoch_end` for `tasks_due(epoch, config["freq"])`, on the EMA teacher by default
    (`eval.encoder: online` = student).
  - A labeled dataset that can't be loaded only logs "Disabling KneeNo evaluation" and turns eval off.
  - Wired through `CallbackArgs.kneeno_eval` and `get_callbacks(..., num_channels, image_size)` (from
    `train.py`: the resolved `transform_args.num_channels` and `transform_args.image_size`).
- **`evaluate()` is called on every rank, deliberately with no rank guard.** `ClassificationEvaluator` works on
  rank 0, then hits `dist.barrier()` + `dist.broadcast_object_list()`. A rank guard deadlocks multi-GPU runs.
- **Metrics go through lightly-train's loggers.** `logging.tensorboard_dir: null` disables KneeNo's
  `SummaryWriter`, and the callback re-logs the metrics via `pl_module.log_dict` under `eval/`.
- **Config deep-merge:** a task left out of `freq` keeps KneeNo's default, which is null (off) for every task since
  KneeNo `77839f0`.
- `src/lightly_train/_commands/eval_classification.py`: the standalone `lightly-train eval_classification`
  command (also `lightly_train.eval_classification`), ported from `3d_dinov2`. It loads a checkpoint or an exported
  model and runs `evaluate(..., epoch=0)` once: each task logs its metrics once per test set (KneeNo's
  global-split protocol since 2026-10-14: heads train on the internal `train` split, early-stopped on val;
  `eval.verbose` logs the per-head-epoch val metrics), keys `{test_set}/{task}/{metric}`, and each trained head
  also its best epoch's `evaluation_val/{task}/val_auroc`. The pretraining config tests on `internal` only
  (`eval.data.test_sets: [internal]`), so the callback logs `eval/internal/<task>/...`: compare pretraining epochs
  by those (the run doesn't pretrain on those exams; see "Run config"). `cluster/configs/eval-MI-*.yaml` tests on
  `[internal, external]` (heads trained once, scored on both) and sets `eval.data.abort_on_io_error: true` (an
  exam that fails to load raises); `pretrain-MI-*.yaml` sets `false` (it is replaced by a random one; the
  in-training eval must not kill the run).
  - **Arguments (2026-10-08):** only `eval_config`, `accelerator` and `overwrite`. Everything else is in the
    eval config file: the top-level keys `weights`, `image_size` (`[H, W]`, required: neither file records it) and
    `eval`, plus `model`, `model_args` and `normalize` for an exported model (below). It's validated by
    `EvalRunConfig` (`extra="forbid"`, so a leftover `transform:` or `out:` is an error; the old `checkpoint:` key
    raises with a pointer to `weights`), and env vars are expanded in the whole file. **An unset env var is an
    error**:
    `expand_env_vars` leaves it as it is, which would point to a nonexistent checkpoint or create a literal
    `${...}` directory. A `params-pretrain.yaml` is no longer accepted.
  - **`weights`: checkpoint or exported model (since 2026-10-18), detected by content** (`_load_encoder`, one
    `torch.load(weights_only=False)`: a checkpoint holds pickled modules).
    - A LightlyTrain checkpoint (a dict with `state_dict` and `lightly_train`) records the model and its
      normalize args, so `model` / `model_args` / `normalize` are **rejected** with one, even as null. The shipped
      eval config therefore points to `last.ckpt` and has them commented out, with the values of this branch's
      export (`dinov2/vitb14-notpretrained`, `model_args: null`, ImageNet).
    - An exported model is the state dict of the bare `DinoVisionTransformer`: `pretrain`'s
      `exported_models/exported_last.pt`, `lightly-train export` with the default `part`/`format`, or Meta's
      official weights (`dinov2/vitb14` = `dinov2_vitb14_reg4_pretrain.pth`; MLP FFN, 518² pos_embed, vs. our
      `vitb14-notpretrained`: SwiGLU, 224²). It holds the weights only, so `model` and `normalize` (`mean` and
      `std`, no defaults) are required and `model_args` is optional. The model is built with
      `package_helpers.get_wrapped_model(..., load_weights=False)` (no download, compute nodes are offline),
      loaded `strict=True` (a mismatch raises, naming `model`/`model_args`), then `make_teacher()`. Input channels
      come from `patch_embed.proj.weight.shape[1]`. Other contents (a `torch_model` export, an `embedding_model`
      state dict) raise.
    - An export is always the EMA teacher (`DINOv2.__init__` makes the given embedding model the teacher, and
      both `pretrain` and `export` take that `wrapped_model`), so `eval.encoder: online` needs a checkpoint.
  - **Tasks** are those whose `eval.freq` is set, with `tasks_due`'s semantics: null or `<= 0` is off. They are
    computed on the merged config, so a missing key keeps KneeNo's default, and no task at all is an error. The
    **encoder** is `eval.encoder`: `target` (the EMA teacher, which the checkpoint stores) or `online` (the
    student, loaded from the method's `state_dict`).
  - **Outputs:**
    - In the parent of `eval.logging.per_label_dir`: `results.json` (final metrics) and `params.yaml` (the config
      file with env vars expanded, written after loading the weights but *before* evaluating, so a config the
      weights reject leaves no `params.yaml` that would make the corrected rerun refuse).
    - Without a `per_label_dir`: a warning, `results.json` next to the weights file, and no `params.yaml`.
    - Existing results (`results.json`, `params.yaml`, `per_label_dir`, `tensorboard_dir`) make it refuse, before
      the weights are loaded. With `overwrite=True` it runs anyway, but the per-label CSVs are appended to and
      TB gets a second event file.
  - **Differs from 3d** in what the adapter needs. `num_channels` is `len(checkpoint normalize_args.mean)` for a
    checkpoint (it stores the resolved normalize args, whose length the transform matched to the resolved
    `num_channels`, `auto` included) and the patch embedding's input channels for an exported model.
    `image_size` is `(H, W)`. 3d also reads the top-level `resize_interpolation`/`resize_upscale_interpolation`.

## Run bookkeeping (copied from `3d_dinov2`; doesn't change training)

- `pretrain(params_file=...)`: global rank 0 copies the run config verbatim to `<out>/params-pretrain.yaml`, or
  `params-pretrain-1.yaml`, `-2`, ... on resume (`common_helpers.copy_params_file`). The copy happens after the
  out-dir check, so a fresh out dir isn't "non-empty". `params_file` is also a field of `TrainConfig` /
  `CLITrainConfig`.
- `<out>/model-config.yaml` (`common_helpers.write_model_config`): for any `dinov2/<name>` model, written once
  (a resume keeps the original). It holds `DINOv2ViTPackage.get_model_config`: the model's config merged over
  `ssl_default_config.yaml`, restricted to `student` + `crops.global_crops_size`, plus `model_args` if given.
  **Unlike on `3d_dinov2`, `get_model` doesn't use `get_model_config`.** It still builds from the full merged
  config (stock), and the record is accurate because stock `get_model` reads only those keys. If `get_model`
  ever reads more, extend `get_model_config`.

## Running on the cluster

- `cluster/submit_pretrain_dinov2_kneeno.sh` (sbatch, 1 GPU; job `lightly-2d-dinov2`) extracts the internal
  dataset to `/dev/shm/kneeno_data/internal` (`KneeNo/data/prepare_data.py --internal-tar-dir`). It then runs
  `cluster/pretrain_dinov2_kneeno.py --config <run config>` (default `cluster/configs/pretrain-MI-vitb14-2d.yaml`).
  Env vars: `CONFIG`, `REPO_DIR`, `KNEENO_DIR`, `PG_TIMEOUT_MINUTES`.
- **Run config:**
  - `model:` (string);
  - `data:` with exactly `out`, `data_root` and `data_meta` (all required; names as on `3d_dinov2`). `data_root`
    (`/dev/shm/kneeno_data/internal`) holds every internal exam, the global `val` ones (the `internal` test set)
    included, so it is not passed to lightly-train as one folder (it was until 2026-10-14). The launcher's
    `pretraining_dirs` passes `pretrain(data=[...])`: the sorted list of `data_root/<case_id>` for the cases KneeNo's
    `pretraining_cases` returns for `data_meta` (`internal_unlabeled.json`: global splits `train` + `unlabeled`,
    the same rule as the 3D dataset). That is a stock input type, so lightly-train's code is untouched: it searches
    each folder recursively (every `<case>/<series>/<NNN>.jpeg` slice is one sample) and raises on a missing one.
    The launcher prints the number of exams used and of exam folders held out. The list holds strings, not
    `Path`s: lightly-train logs its arguments with `json.dumps`, which fails on a list of `Path`s.
  - optional `train:` (allowlist `run_config.TRAIN_KEYS`), `method:` (`DINOv2Args`), `transform:`
    (`DINOv2ViTTransformArgs`) and `eval:` (eval callback config; a missing block means no eval).
  - optional `callbacks:` (-> `pretrain(callbacks=)`, lightly-train's own callbacks by name), e.g.
    `model_checkpoint: {every_n_epochs: 50, save_top_k: -1}` for a checkpoint every 50 epochs besides `last.ckpt`.
    `save_top_k` must be `-1`: the default `1` keeps only the newest. The script merges `eval:` into it as
    `kneeno_eval`; `callbacks.kneeno_eval` is rejected.
  - Loaded by `cluster/run_config.py::load_run_config`, which expands env vars and rejects unknown keys.
- **The shipped config lists every train/method/transform key at its stock default.** Unlike `3d_dinov2`, that
  includes `num_channels: auto` (3 channels: the grayscale JPEGs are loaded as RGB). `tests/test_cluster_configs.py`
  pins equality with the defaults, so a deliberate deviation needs that test changed too.
- The `eval:` block is copied from `3d_dinov2`, except `per_label_dir` (follows this config's `data.out`),
  `linear.pooling` and the three slice tasks. `linear_slice_cls` / `attentive_slice_cls` run every 5 epochs
  (they train on the cached slice cls tokens); `attentive_slice_pool` is `null` like `attentive_pool`.
  `series_depth: 24` is the number of slices the encoder sees per volume, and the slice tasks' slice count.
- Same as on 3d: the env vars at the top of the launcher (`OPENBLAS_NUM_THREADS=1`, forced
  `OMP_NUM_THREADS=1`, `expandable_segments`); multi-GPU through SLURM options only (`num_nodes` from
  `SLURM_NNODES`); a longer DDP process-group timeout for the rank-0 eval; auto-resume from
  `<out>/checkpoints/last.ckpt`. The `#SBATCH` resources are 3d's values, not tuned for this run.
- **Standalone eval** (since 2026-10-08): `sbatch --export=ALL,PRETRAIN_JOB_ID=<id>
  cluster/submit_eval_dinov2_kneeno.sh`. **Pass env vars with `--export`**: RWTH's `sbatch` wrapper submits with
  `--export=NONE`, so a `PRETRAIN_JOB_ID=<id> sbatch ...` prefix never reaches the job (the script then stops at its
  check).
  - The sbatch script requests 1 GPU and 1 task (single process: KneeNo evaluates on rank 0 only, there's no
    process group and no `--pg-timeout-minutes`). It extracts the internal and the external data to the job's
    node-local `$TMP/kneeno_data` (`prepare_data.py`'s default; `/dev/shm` is for pretraining only), then runs
    `cluster/eval_dinov2_kneeno.py --config <eval config>` (default `cluster/configs/eval-MI-vitb14-2d.yaml`). The
    external data comes from a hard-coded `--external-tar /hpcwork/p0021834/workspace_roman/jonas/uka_gold_nifti.tar`; the eval config's
    `label_meta.external` is `/hpcwork/p0021834/workspace_roman/jonas/metadata/external_labeled.json`. Env vars: `PRETRAIN_JOB_ID` (required) and `TMP`,
    both checked up front; `CONFIG`, `REPO_DIR`, `KNEENO_DIR`.
  - The launcher only sets the same thread/allocator env vars as the pretraining launcher, then calls
    `lightly_train.eval_classification(eval_config=...)` (`accelerator="auto"`, so it also runs on CPU here).
  - The eval config locates the pretraining run through **`${PRETRAIN_JOB_ID}`, not `${SLURM_JOB_ID}`**, which
    in the eval job is the eval job's own ID. Its outputs go to `<run>/eval/internal/post-training/` (`tb/`,
    `eval_per_label/`, `results.json`, `params.yaml`).
  - Its `eval:` block equals the pretraining config's except for `logging` (`tensorboard_dir` is set: no
    lightly-train loggers here), `freq` (every task `1`, i.e. on; every key listed), `data.test_sets`
    (`[internal, external]`, with the external paths in `data_root` / `label_meta`), `data.subset: 1.0` (all
    exams), `data.data_root` (`$TMP/...`), `data.num_workers: 31`, `data.abort_on_io_error` (`true`),
    `compile_encoder` (`true`; `false` in training, where the evaluator gets the training loop's model), the heads'
    `max_epochs` / `warmup_epochs` (50 / 10 instead of 30 / 5: trained once, not every few epochs), and the head
    settings of the tasks the pretraining run doesn't run (`attentive_pool`, `attentive_slice_pool`).
    `tests/test_cluster_configs.py` pins that (after merging both over KneeNo's defaults; the internal
    `label_meta` must be the same), that it sets none of the exported-model keys, and that `weights` and
    `image_size` match the pretraining config
    (`eval-<name>.yaml` <-> `pretrain-<name>.yaml`). Until 2026-10-14 that test did not allow the deliberate
    `abort_on_io_error` difference and failed.
  - Both configs spell out every key of each attentive head (`q2l`, `patch_sa_depth`, `qa_depth`, `num_heads`,
    `mlp_ratio`, `qkv_bias`, `complete_block` and the training settings).

**Comparability with `3d_dinov2`:**
- Both pretrain on the same exams: the global splits `train` + `unlabeled` of the same `data_meta`, selected by
  KneeNo's `pretraining_cases` (since 2026-10-14; before, the 2D run took every JPEG in the folder, the `val`
  exams included, so the `internal/*` numbers of older 2D checkpoints are not clean, only their `external/*` ones).
  Remaining difference: the 3D run reads only the series listed in the metadata that pass KneeNo's length filter
  (`min_series_len` 2); the 2D run takes every series folder of an allowed case.
- 2D defaults vs. the 3D config: ImageNet normalisation vs `0.5/0.5`; colour jitter/solarize vs MRI-specific
  intensity augmentations; 224² crops of single slices vs `224×224×16` volumes. These are all intended stock
  behaviour, but worth stating in the thesis.

## Testing

No GPU here: CPU, `dinov2/_vittest14`. This branch has the **stock test suite with no quarantine** (unlike
`3d_dinov2`), so any failure is real. Tests that cover this branch's changes:

```bash
.venv/bin/python -m pytest tests/_callbacks tests/_data/test_kneeno_adapter.py tests/test_cluster_configs.py \
  tests/_commands/test_common_helpers.py tests/_commands/test_train.py tests/_commands/test_eval_classification.py \
  tests/test__cli.py tests/test__logging.py tests/_models/dinov2_vit tests/_methods/dinov2
```

- `tests/_models/dinov2_vit/test_attention.py` pins `SDPAttention` against `Attention` and that every builder uses
  it (see "Deliberate deviation" at the top).
- `tests/_data/test_kneeno_adapter.py` pins `prepare_input` (per-slice normalization and order, layout, the
  external rescale, equality with `EmbeddingTransform`) and `forward_features` (equality with encoding each
  slice by hand, DDP unwrap), plus the `prepare_input` → `collate` → `forward_features` path.
- `tests/_callbacks/test_kneeno_eval.py` runs the real adapter on a fake labeled dataset that stands in for the
  cluster data (one fixed depth, as with `series_depth > 0`), including all seven tasks at once.
- `tests/_commands/test_eval_classification.py` runs the command end to end on a real DINOv2 checkpoint and the same
  kind of fake dataset (task selection from `freq`, all seven tasks, `target`/`online`, `num_channels` from the
  checkpoint, output locations, `params.yaml` expansion, refusal on existing results, unset env vars, invalid
  config files). It also exports that checkpoint with the real `lightly_train.export` and pins identical metrics
  from the export, the exported-model keys (required / rejected with a checkpoint), `num_channels` from the
  weights, a wrong `model`/`model_args`, `online` with an export, and the unsupported export formats.
  `tests/test_cluster_configs.py` covers the shipped eval config and the eval launcher.
- `tests/_methods/dinov2/test_dinov2_loss.py::TestDINOLoss::test_sinkhorn_knopp_teacher` (stock) is flaky in a
  full run: unseeded `torch.randn` and an `allclose` at default tolerance, so it depends on the global RNG state the
  earlier tests leave behind. It passes on its own.
- Smoke-test DataLoader-adjacent code with `num_workers >= 2` (picklability; see `MA/CLAUDE.md`).
