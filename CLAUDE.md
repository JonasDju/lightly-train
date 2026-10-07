# CLAUDE.md

## What this branch is

Branch **`2d_dinov2`** of our fork of **Lightly's `lightly-train`**: the **vanilla 2D DINOv2 baseline** of the
thesis comparison against the 3D-native DINOv2 on branch `3d_dinov2`. Both are pretrained on the same unlabeled
knee-MRI data (sibling repo `KneeNo`) and scored with the same KneeNo classification evaluation.

**Keep training stock.** The point of this branch is an unmodified DINOv2 training procedure: lightly-train's
own image-folder dataset over the JPEG slices, its own 2D transforms, its own defaults. Changes belong here only
if they don't alter training: evaluation wiring, run bookkeeping, cluster tooling. Anything that changes what
the model sees or how it learns needs a deliberate decision, and a note in this file.

**Current status (2026-10-07):** the KneeNo in-training eval is wired in, but the adapter is a **stub** (see
below). `cluster/` holds a SLURM script, launcher and run config. Pretraining runs end to end on CPU through the
cluster script. Nothing has run on the cluster yet.

## Branches

The repo has three long-lived branches (restructured 2026-10-07):

- **`main`**: stock lightly-train plus the KneeNo dependency (`KNEENO.md`, `exclude-newer` pin) and the
  `pyproject.toml`/`uv.lock` shared by all branches. It's the common base, with no training changes.
- **`2d_dinov2`** (this branch): the 2D baseline described here.
- **`3d_dinov2`**: the 3D-native DINOv2 (patch embedding, attention, masking, MONAI transforms, KneeNo volume
  dataset). It has its own, much longer `CLAUDE.md`. Several things here were copied from there (eval callback,
  `cluster/`, `params_file`, `write_model_config`). When changing them, check whether the other branch needs the
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

Frozen-encoder, multi-label classification (k-NN, linear, linear-pool, attentive-pool) lives in
`KneeNo/kneeno/evaluation/` and is shared with vjepa2 and `3d_dinov2`; see `KneeNo/README.md`. KneeNo evaluates
**exams**, handing the adapter one `(1, D, H, W)` volume per sequence.

- `src/lightly_train/_data/kneeno_adapter.py`: `DINOv2Adapter(EncoderAdapter)`, a **stub**.
  - Real: `has_cls_token = True` (so the `linear` task is available), the constructor (`dataset_type` validated
    against `internal`/`external`, `embed_dim`, `image_size` `(H, W)`, `normalize`), `embed_dim`.
  - `prepare_input` and `forward_features` raise `NotImplementedError`. How a 3D volume becomes input for the 2D
    encoder (per slice? which slices?) and how slice features are pooled back into one volume's
    `{"cls", "patches"}` is the open design question.
  - **Consequence:** a run with an `eval:` block fails at the end of the first epoch an eval task is due. With
    the shipped config that's epoch 5, after the evaluator has loaded the labeled dataset. Leave the eval block
    out until the adapter exists.
- `src/lightly_train/_callbacks/kneeno_eval.py`: `KneeNoEval` + `KneeNoEvalArgs`, copied from `3d_dinov2` minus
  its 3D resize-interpolation arguments.
  - **Off by default:** enabled only by `callbacks={"kneeno_eval": {"config": <eval dict>}}`, deep-merged over
    KneeNo's `DEFAULT_EVAL_CONFIG`.
  - Runs in `on_train_epoch_end` for `tasks_due(epoch, config["freq"])`, on the EMA teacher by default
    (`eval.encoder: online` = student).
  - A labeled dataset that can't be loaded only logs "Disabling KneeNo evaluation" and turns eval off.
  - Wired through `CallbackArgs.kneeno_eval` and `get_callbacks(..., image_size)` (from
    `train.py`: `transform_args.image_size`).
- **`evaluate()` is called on every rank, deliberately with no rank guard.** `ClassificationEvaluator` works on
  rank 0, then hits `dist.barrier()` + `dist.broadcast_object_list()`. A rank guard deadlocks multi-GPU runs.
- **Metrics go through lightly-train's loggers.** `logging.tensorboard_dir: null` disables KneeNo's
  `SummaryWriter`, and the callback re-logs the metrics via `pl_module.log_dict` under `eval/`.
- **Config trap:** disabling a task with a nonzero default frequency needs an explicit `freq: {<task>: null}`.
- Not ported from `3d_dinov2`: the standalone `lightly-train eval_classification` command.

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
  - `data:` with exactly `out` and `data`, the image folder lightly-train searches recursively. Every
    `<case>/<series>/<NNN>.jpeg` slice is one sample.
  - optional `train:` (allowlist `run_config.TRAIN_KEYS`), `method:` (`DINOv2Args`), `transform:`
    (`DINOv2ViTTransformArgs`) and `eval:` (eval callback config; a missing block means no eval).
  - Loaded by `cluster/run_config.py::load_run_config`, which expands env vars and rejects unknown keys.
- **The shipped config lists every train/method/transform key at its stock default.** Unlike `3d_dinov2`, that
  includes `num_channels: auto` (3 channels: the grayscale JPEGs are loaded as RGB). `tests/test_cluster_configs.py`
  pins equality with the defaults, so a deliberate deviation needs that test changed too.
- The `eval:` block is copied verbatim from `3d_dinov2`, except `per_label_dir`, which follows this config's
  `data.out`. Its `series_depth: 24` / `resample_mode` describe how KneeNo loads the eval volumes; what they
  mean for a 2D encoder is part of the adapter design.
- Same as on 3d: the env vars at the top of the launcher (`OPENBLAS_NUM_THREADS=1`, forced
  `OMP_NUM_THREADS=1`, `expandable_segments`); multi-GPU through SLURM options only (`num_nodes` from
  `SLURM_NNODES`); a longer DDP process-group timeout for the rank-0 eval; auto-resume from
  `<out>/checkpoints/last.ckpt`. The `#SBATCH` resources are 3d's values, not tuned for this run.

**Comparability with `3d_dinov2`:**
- Both read `/dev/shm/kneeno_data/internal`, but the 3D run selects series through `data_meta`
  (`metadata_unlabeled.json`, plus KneeNo's length filter). The 2D run takes **every** JPEG in the folder, so the
  two training sets can differ.
- 2D defaults vs. the 3D config: ImageNet normalisation vs `0.5/0.5`; colour jitter/solarize vs MRI-specific
  intensity augmentations; 224² crops of single slices vs `224×224×16` volumes. These are all intended stock
  behaviour, but worth stating in the thesis.

## Testing

No GPU here: CPU, `dinov2/_vittest14`. This branch has the **stock test suite with no quarantine** (unlike
`3d_dinov2`), so any failure is real. Tests that cover this branch's changes:

```bash
.venv/bin/python -m pytest tests/_callbacks tests/_data/test_kneeno_adapter.py tests/test_cluster_configs.py \
  tests/_commands/test_common_helpers.py tests/_commands/test_train.py tests/test__logging.py \
  tests/_models/dinov2_vit tests/_methods/dinov2
```

- `tests/_callbacks/test_kneeno_eval.py` swaps the stub for a test-only `_FakeDINOv2Adapter` (centre slice
  through the real 2D encoder). It is not a proposal for the real adapter. One test pins that the real stub
  raises.
- The fake labeled dataset stands in for the cluster data.
- Smoke-test DataLoader-adjacent code with `num_workers >= 2` (picklability; see `MA/CLAUDE.md`).
