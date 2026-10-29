"""Standalone KneeNo classification evaluation of a finished 2D DINOv2 pretraining run.

Calls ``lightly_train.eval_classification`` (``src/lightly_train/_commands/eval_classification.py``) with an eval
config (``--config``): a YAML with

    weights:     a LightlyTrain checkpoint, e.g. <pretraining data.out>/checkpoints/last.ckpt, or an exported model,
                 e.g. <pretraining data.out>/exported_models/exported_last.pt (told apart by content)
    image_size:  [H, W], the pretraining run's transform.image_size
    eval:        KneeNo's evaluation config (deep-merged over KneeNo's DEFAULT_EVAL_CONFIG)

and, for an exported model only, ``model``, ``model_args`` (optional) and ``normalize``.

Every task with a non-null ``eval.freq`` runs once, on ``eval.encoder``, logging its test metrics.
Environment variables are expanded in every string value; ``cluster/configs/eval-MI-vitb14-2d.yaml`` locates the
pretraining run through ``${PRETRAIN_JOB_ID}``. Next to ``eval.logging.per_label_dir`` the command writes the
final metrics (``results.json``) and the expanded config (``params.yaml``); it refuses to start if earlier results
exist there.

This script exists rather than calling ``lightly-train eval_classification`` directly for the environment set below,
which has to happen before numpy and torch are imported. The evaluation runs in a single process: KneeNo evaluates
on rank 0 only, so more GPUs would not help.

If ``CONFIG_FILE_NAME`` is set (the submit script exports it, and the shipped eval config names its output folders
with it), it must be the config's file name without the ``.yaml``.

Submit through ``submit_eval_dinov2_kneeno.sh``; that also prepares the node-local data first. Run by hand (the
shipped config also needs SPLIT_LEVEL, SPLIT_REP and TMP):

    PRETRAIN_JOB_ID=<id> CONFIG_FILE_NAME=<config name> .venv/bin/python cluster/eval_dinov2_kneeno.py --config cluster/configs/eval-MI-vitb14-2d.yaml
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
from pathlib import Path  # noqa: E402

import lightly_train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="eval config YAML")
    args = parser.parse_args()
    # The submit script exports the config's file name without the .yaml, which names
    # the output folders: it must name this file.
    file_name = os.environ.get("CONFIG_FILE_NAME")
    if file_name is not None and f"{file_name}.yaml" != Path(args.config).name:
        raise ValueError(
            f"CONFIG_FILE_NAME={file_name!r} does not name the eval config '{args.config}'."
        )
    lightly_train.eval_classification(eval_config=args.config)


# Required: with num_workers > 0 the DataLoader may spawn/forkserver workers, which re-import this module.
if __name__ == "__main__":
    main()
