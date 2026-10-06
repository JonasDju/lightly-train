"""3D DINOv2 pretraining on the unlabeled KneeNo volumes, with in-training classification eval.

Everything about the experiment comes from a run config (``--config``), like vjepa2's ``configs/``: a YAML with

    model:      -> lightly_train.pretrain(model=...), e.g. dinov2/vitb14-notpretrained (any 3D dinov2/<name>)
    data:       -> lightly_train.pretrain(out=, data_root=, data_meta=, series_depth=, resample_mode=)
    train:      -> lightly_train.pretrain(epochs=, batch_size=, num_workers=, trainer_args=) (optional)
    method:     -> lightly_train.pretrain(method_args=...)      (DINOv2Args)
    transform:  -> lightly_train.pretrain(transform_args=...)   (DINOv2ViTTransformArgs)
    eval:       -> the KneeNo eval callback's config            (deep-merged over KneeNo's DEFAULT_EVAL_CONFIG)

``model`` and ``data`` (with all five keys) are required. A missing ``train``/``method``/``transform`` block, or
a missing key within one, falls back to lightly-train's default; a missing ``eval`` block means no evaluation at
all. Environment variables (``$VAR``, ``${VAR}``, leading ``~``) are expanded in every string value with
KneeNo's ``expand_env_vars`` (loading and validation: ``run_config.load_run_config``).
``cluster/configs/pretrain-MI-vitb14-24f.yaml`` lists every method/transform default explicitly.
lightly-train copies the file into ``data.out`` as ``params-pretrain.yaml`` (``params-pretrain-1.yaml``, ...
on each resume), so that copy is the complete record of the run's settings. The only command-line option
besides ``--config`` is the multi-GPU process-group timeout, which does not affect training. Whatever the
``train`` block (``run_config.TRAIN_KEYS``) leaves out (optimizer, epochs, batch size, ...) is
``lightly_train.pretrain``'s default for ``method="dinov2"``.

The eval block's ``/dev/shm/kneeno_data/internal`` data must exist by the time the first eval epoch ends
(``submit_pretrain_dinov2_kneeno.sh`` extracts it). If it cannot be loaded the callback only logs a warning
and disables itself, so check the log for "Disabling KneeNo evaluation".

Submit through ``submit_pretrain_dinov2_kneeno.sh``; that also prepares the node-local data first. Run by hand:

    .venv/bin/python cluster/pretrain_dinov2_kneeno.py --config cluster/configs/pretrain-MI-vitb14-24f.yaml

Rerunning with the same ``data.out`` resumes from ``<out>/checkpoints/last.ckpt`` if one exists.

Multi-GPU / multi-node: nothing to configure here. Devices, ranks and world size come from the SLURM job
(``--ntasks-per-node`` = GPUs per node, launched with ``srun``), and ``num_nodes`` is read from
``SLURM_NNODES``. ``train.batch_size`` (default 128) is the *global* batch size.
"""

from __future__ import annotations

import os

# Must be set before numpy is imported (OpenBLAS reads it once, at load time), hence above the other imports.
# numpy's bundled OpenBLAS otherwise starts one thread per core in every DataLoader worker -- torch only limits
# its own threads there -- and RandomResizedCrop3D's np.tensordot runs on it: 31 workers x 32 threads on a
# 32-core job slowed each sample ~15x, so no batch ever arrived within the DataLoader timeout. BLAS is only a
# small share of the per-sample work, so throughput comes from the workers, not from threads inside one.
# Workers inherit the environment, under fork and spawn alike.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Must be set before torch is imported. torch otherwise runs CPU ops in the main process on one OpenMP thread per
# core, on cores the DataLoader workers keep busy, so every parallel region waits for a descheduled thread. The
# training loop has no heavy CPU tensor ops, and the workers already run torch single-threaded. setdefault,
# so an explicit setting in the job environment wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Expandable segments let freed memory be reused for any size. setdefault, so an explicit setting in the job
# environment wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

from pytorch_lightning.strategies import DDPStrategy  # noqa: E402

# cluster/run_config.py: running this script puts its folder first on sys.path.
from run_config import load_run_config  # noqa: E402

import lightly_train  # noqa: E402


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
    parser.add_argument("--config", required=True, help="run config YAML (see above)")
    parser.add_argument(
        "--pg-timeout-minutes",
        type=int,
        default=240,
        help="multi-GPU only: process-group timeout; must exceed the duration of one full eval round",
    )
    args = parser.parse_args()

    config = load_run_config(Path(args.config))
    # A re-queued job must continue rather than fail on the existing out dir.
    resume_interrupted = (Path(config["data"]["out"]) / "checkpoints" / "last.ckpt").is_file()

    lightly_train.pretrain(
        **config["data"],
        **config.get("train", {}),
        model=config["model"],
        method="dinov2",
        method_args=config.get("method"),
        transform_args=config.get("transform"),
        # No eval block -> no evaluation: the callback is off unless given a config.
        callbacks={"kneeno_eval": {"config": config["eval"]}} if "eval" in config else None,
        params_file=args.config,
        resume_interrupted=resume_interrupted,
        # devices stays "auto" (GPUs visible to each task); nodes must be passed, lightly-train does not
        # read them from SLURM and uses this value to split the global batch across all devices.
        num_nodes=int(os.environ.get("SLURM_NNODES", "1")),
        strategy=build_strategy(num_ranks(), args.pg_timeout_minutes),
    )


# Required: with num_workers > 0 the DataLoader may spawn/forkserver workers, which re-import this module.
if __name__ == "__main__":
    main()
