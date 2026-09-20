"""3D DINOv2 pretraining on the unlabeled KneeNo volumes, with in-training classification eval.

Default settings throughout: only the data locations and the model name are set here, everything else
(method args, transforms, optimizer, epochs, batch size, ...) is what ``lightly_train.pretrain`` picks for
``method="dinov2"``. The KneeNo classification evaluation (k-NN, linear, linear-pool, attentive-pool on the
labeled dataset) is a default-on callback and reads ``src/lightly_train/_configs/kneeno_eval.yaml``; its
``$TMP/kneeno_data/labeled`` data must exist by the time the first eval epoch ends
(``submit_pretrain_dinov2_kneeno.sh`` extracts it). If it cannot be loaded the callback only logs a warning
and disables itself, so check the log for "Disabling KneeNo evaluation".

Submit through ``submit_pretrain_dinov2_kneeno.sh``; that also prepares the node-local data first. Run by hand:

    .venv/bin/python cluster/pretrain_dinov2_kneeno.py --out /hpcwork/va105917/lightly/dinov2_vitb14

Rerunning with the same ``--out`` resumes from ``<out>/checkpoints/last.ckpt`` if one exists.

Multi-GPU / multi-node: nothing to configure here. Devices, ranks and world size come from the SLURM job
(``--ntasks-per-node`` = GPUs per node, launched with ``srun``), and ``num_nodes`` is read from
``SLURM_NNODES``. ``--batch-size`` is left at lightly-train's default, which is the *global* batch size.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path

import lightly_train
from pytorch_lightning.strategies import DDPStrategy

# The eval config's eval.data.series_depth/resample_mode is 24/nearest -- keep pretraining identical, so the
# encoder sees the same depth distribution during evaluation as it was trained on.
SERIES_DEPTH = 24
RESAMPLE_MODE = "nearest"


def num_ranks() -> int:
    """Total number of training processes (one per GPU) in the SLURM job; 1 outside SLURM."""
    return int(os.environ.get("SLURM_NTASKS", "1"))


def build_strategy(n_ranks: int, timeout_minutes: int) -> str | DDPStrategy:
    """The strategy for ``lightly_train.pretrain``.

    Single process: ``"auto"``. Several: the same DDP variant lightly-train's own ``"auto"`` picks for more than
    one device (``train_helpers.get_strategy``), but with a longer process-group timeout. The KneeNo eval runs on
    rank 0 only while every other rank waits in ``dist.barrier()`` (``kneeno/evaluation/classification.py``), so
    one eval round has to fit inside the timeout or the NCCL watchdog kills the job. Lightning's default is 30
    minutes.
    """
    if n_ranks <= 1:
        return "auto"
    return DDPStrategy(find_unused_parameters=True, timeout=timedelta(minutes=timeout_minutes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory (logs, checkpoints, exported model)")
    parser.add_argument(
        "--data-root",
        default="$TMP/kneeno_data/unlabeled",
        help="extracted unlabeled dataset (see KneeNo/data/prepare_data.py); env vars are expanded",
    )
    parser.add_argument(
        "--data-meta",
        default="/hpcwork/p0021834/workspace_roman/jonas/BigKneeTar/metadata.json",
        help="metadata.json of the unlabeled dataset",
    )
    parser.add_argument("--model", default="dinov2/vitb14", help="any 3D dinov2/<name>, e.g. dinov2/vits14")
    parser.add_argument(
        "--pg-timeout-minutes",
        type=int,
        default=240,
        help="multi-GPU only: process-group timeout; must exceed the duration of one full eval round",
    )
    args = parser.parse_args()

    out = Path(args.out)
    # A re-queued job must continue rather than fail on the existing out dir.
    resume_interrupted = (out / "checkpoints" / "last.ckpt").is_file()

    lightly_train.pretrain(
        out=out,
        data_root=os.path.expandvars(args.data_root),
        data_meta=args.data_meta,
        series_depth=SERIES_DEPTH,
        resample_mode=RESAMPLE_MODE,
        model=args.model,
        method="dinov2",
        resume_interrupted=resume_interrupted,
        # devices stays "auto" (GPUs visible to each task); nodes must be passed, lightly-train does not
        # read them from SLURM and uses this value to split the global batch across all devices.
        num_nodes=int(os.environ.get("SLURM_NNODES", "1")),
        strategy=build_strategy(num_ranks(), args.pg_timeout_minutes),
    )


# Required: with num_workers > 0 the DataLoader may spawn/forkserver workers, which re-import this module.
if __name__ == "__main__":
    main()
