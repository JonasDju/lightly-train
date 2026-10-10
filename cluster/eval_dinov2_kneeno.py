"""Standalone KneeNo classification evaluation of a finished 2D DINOv2 pretraining run.

Calls ``lightly_train.eval_classification`` (``src/lightly_train/_commands/eval_classification.py``) with an eval
config (``--config``): a YAML with exactly

    checkpoint:  the LightlyTrain checkpoint to evaluate, e.g. <pretraining data.out>/checkpoints/last.ckpt
    image_size:  [H, W], the pretraining run's transform.image_size
    eval:        KneeNo's evaluation config (deep-merged over KneeNo's DEFAULT_EVAL_CONFIG)

Every task with a non-null ``eval.freq`` runs once, on ``eval.encoder``, logging the whole head fine-tuning curve.
Environment variables are expanded in every string value; ``cluster/configs/eval-MI-vitb14-2d.yaml`` locates the
pretraining run through ``${PRETRAIN_JOB_ID}``. Next to ``eval.logging.per_label_dir`` the command writes the
final metrics (``results.json``) and the expanded config (``params.yaml``); it refuses to start if earlier results
exist there.

This script exists rather than calling ``lightly-train eval_classification`` directly for the environment set below,
which has to happen before numpy and torch are imported. The evaluation runs in a single process: KneeNo evaluates
on rank 0 only, so more GPUs would not help.

Submit through ``submit_eval_dinov2_kneeno.sh``; that also prepares the node-local data first. Run by hand:

    PRETRAIN_JOB_ID=<id> .venv/bin/python cluster/eval_dinov2_kneeno.py --config cluster/configs/eval-MI-vitb14-2d.yaml
"""

from __future__ import annotations

import os

# Must be set before numpy is imported (OpenBLAS reads it once, at load time), hence above the other imports.
# numpy's bundled OpenBLAS otherwise starts one thread per core in every DataLoader worker -- torch only limits
# its own threads there -- so any BLAS work in the workers oversubscribes the cores. Workers inherit the
# environment, under fork and spawn alike.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Must be set before torch is imported. torch otherwise runs CPU ops in the main process on one OpenMP thread per
# core, on cores the eval DataLoader workers keep busy. Forced, not setdefault: the cluster's job environment
# already sets OMP_NUM_THREADS=32, which would otherwise win.
os.environ["OMP_NUM_THREADS"] = "1"

# Expandable segments let freed memory be reused for any size. setdefault, so an explicit setting in the job
# environment wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402

import lightly_train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="eval config YAML")
    args = parser.parse_args()
    lightly_train.eval_classification(eval_config=args.config)


# Required: with num_workers > 0 the DataLoader may spawn/forkserver workers, which re-import this module.
if __name__ == "__main__":
    main()
