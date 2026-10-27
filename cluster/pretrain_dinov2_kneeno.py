"""Vanilla 2D DINOv2 pretraining on the JPEG slices of the unlabeled KneeNo dataset, with in-training
classification eval.

The 2D baseline to the 3D-native DINOv2 on branch ``3d_dinov2``: training is stock lightly-train, with its own
image-folder dataset over the slices. Everything about the experiment comes from a run config (``--config``), like
vjepa2's ``configs/``: a YAML with

    model:      -> lightly_train.pretrain(model=...), e.g. dinov2/vitb14-notpretrained
    data:       -> lightly_train.pretrain(out=, data=)          (see below)
    train:      -> lightly_train.pretrain(epochs=, batch_size=, num_workers=, trainer_args=, ...) (optional)
    method:     -> lightly_train.pretrain(method_args=...)      (DINOv2Args)
    transform:  -> lightly_train.pretrain(transform_args=...)   (DINOv2ViTTransformArgs)
    callbacks:  -> lightly_train.pretrain(callbacks=...)        (CallbackArgs, all but kneeno_eval)
    eval:       -> the KneeNo eval callback's config            (deep-merged over KneeNo's DEFAULT_EVAL_CONFIG)

``model`` and ``data`` (with all its keys: ``out``, ``data_root``, ``data_meta``) are required. A missing ``train``/``method``/``transform`` block, or a
missing key within one, falls back to lightly-train's default; a missing ``eval`` block means no evaluation at
all. Environment variables (``$VAR``, ``${VAR}``, leading ``~``) are expanded in every string value with
KneeNo's ``expand_env_vars`` (loading and validation: ``run_config.load_run_config``).
``cluster/configs/pretrain-MI-vitb14-2d.yaml`` lists every train/method/transform default explicitly.

``data.data_root`` holds the JPEG slices of every internal exam (``<case>/<series>/<NNN>.jpeg``), including the
global ``val`` exams the evaluation tests on. So pretrain's ``data`` is not that folder but the list of the case
folders in it that pretraining may see: those of the global splits ``train`` + ``unlabeled`` of ``data.data_meta``,
as KneeNo's ``pretraining_cases`` (the same rule as the 3D run's dataset) returns them. lightly-train searches each
recursively, so every slice is one sample, and raises if one of them does not exist.
lightly-train copies the file into ``data.out`` as ``params-pretrain.yaml`` (``params-pretrain-1.yaml``, ...
on each resume), so that copy is the complete record of the run's settings. The only command-line option
besides ``--config`` is the multi-GPU process-group timeout, which does not affect training.

With an ``eval`` block, the KneeNo adapter (``src/lightly_train/_data/kneeno_adapter.py``) evaluates the encoder
slice by slice at the end of every epoch an eval task is due. The eval block's ``/dev/shm/kneeno_data/internal``
data must exist by then (``submit_pretrain_dinov2_kneeno.sh`` extracts it); if it cannot be loaded the callback
only logs a warning and disables itself, so check the log for "Disabling KneeNo evaluation".

``callbacks`` configures lightly-train's own callbacks by name, each a mapping over that callback's defaults, or
null to switch it off (a callback left out keeps its default). E.g. to keep a checkpoint every 50 epochs, besides
``last.ckpt``:

    callbacks:
      model_checkpoint:
        every_n_epochs: 50
        save_top_k: -1    # the default 1 keeps only the newest periodic checkpoint

Resuming needs ``<out>/checkpoints/last.ckpt``, so leave ``model_checkpoint``'s ``dirpath`` and ``save_last`` at
their defaults. It is written after every epoch, whatever ``every_n_epochs`` is. The eval callback stays in its own ``eval`` block; ``callbacks.kneeno_eval`` is an error.

Submit through ``submit_pretrain_dinov2_kneeno.sh``; that also prepares the node-local data first. Run by hand:

    .venv/bin/python cluster/pretrain_dinov2_kneeno.py --config cluster/configs/pretrain-MI-vitb14-2d.yaml

Rerunning with the same ``data.out`` resumes from ``<out>/checkpoints/last.ckpt`` if one exists.

Multi-GPU / multi-node: nothing to configure here. Devices, ranks and world size come from the SLURM job
(``--ntasks-per-node`` = GPUs per node, launched with ``srun``), and ``num_nodes`` is read from
``SLURM_NNODES``. ``train.batch_size`` (default 128) is the *global* batch size.
"""

from __future__ import annotations

import os

# Must be set before numpy is imported (OpenBLAS reads it once, at load time), hence above the other imports.
# numpy's bundled OpenBLAS otherwise starts one thread per core in every DataLoader worker -- torch only limits
# its own threads there -- so any BLAS work in the workers oversubscribes the cores (on the 3D branch this slowed
# each sample ~15x). Throughput comes from the workers, not from threads inside one. Workers inherit the
# environment, under fork and spawn alike.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Must be set before torch is imported. torch otherwise runs CPU ops in the main process on one OpenMP thread per
# core, on cores the DataLoader workers keep busy, so every parallel region waits for a descheduled thread. The
# training loop has no heavy CPU tensor ops, and the workers already run torch single-threaded. Forced, not
# setdefault: the cluster's job environment already sets OMP_NUM_THREADS=32, which would otherwise win.
os.environ["OMP_NUM_THREADS"] = "1"

# Expandable segments let freed memory be reused for any size. setdefault, so an explicit setting in the job
# environment wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

from kneeno import load_metadata, pretraining_cases  # noqa: E402
from kneeno.dataset import split_hash  # noqa: E402
from pytorch_lightning.strategies import DDPStrategy  # noqa: E402

# cluster/run_config.py: running this script puts its folder first on sys.path.
from run_config import load_run_config  # noqa: E402

import lightly_train  # noqa: E402


def num_ranks() -> int:
    """Total number of training processes (one per GPU) in the SLURM job; 1 outside SLURM."""
    return int(os.environ.get("SLURM_NTASKS", "1"))


def pretraining_dirs(data_root: Path, data_meta: Path) -> list[str]:
    """The case folders under ``data_root`` that pretraining may see, sorted by case id.

    Strings, not paths: lightly-train logs its arguments as JSON, which fails on a list of paths. Prints the
    exams' ``split_hash`` itself: KneeNo logs it too, but this runs before lightly-train gives the "kneeno" logger
    a handler, so that line is dropped.
    """
    cases = sorted(pretraining_cases(load_metadata(data_meta)))
    if not cases:
        raise ValueError(f"The metadata '{data_meta}' has no exams to pretrain on.")
    held_out = len({path.name for path in data_root.iterdir()} - set(cases))
    print(
        f"Pretraining on {len(cases):,} exams in '{data_root}'; {held_out:,} exam folders there are held out "
        f"(not in a pretraining split of the metadata); sha256 of the sorted case ids: {split_hash(cases)}.",
        flush=True,
    )
    return [str(data_root / case) for case in cases]


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
    callbacks = dict(config.get("callbacks", {}))
    # No eval block -> no evaluation: the callback is off unless given a config.
    if "eval" in config:
        callbacks["kneeno_eval"] = {"config": config["eval"]}

    data = config["data"]
    lightly_train.pretrain(
        out=data["out"],
        data=pretraining_dirs(Path(data["data_root"]), Path(data["data_meta"])),
        **config.get("train", {}),
        model=config["model"],
        method="dinov2",
        method_args=config.get("method"),
        transform_args=config.get("transform"),
        callbacks=callbacks or None,
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
