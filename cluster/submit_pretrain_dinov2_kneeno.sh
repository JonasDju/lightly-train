#!/bin/bash
# Multi-GPU / multi-node needs no code change, only matching SLURM options. #SBATCH lines cannot be computed,
# so override them on the command line; --ntasks-per-node must equal the GPUs per node:
#     sbatch --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=8 cluster/submit_pretrain_dinov2_kneeno.sh
#     sbatch --nodes=2 --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=8 cluster/submit_pretrain_dinov2_kneeno.sh
# --cpus-per-task is per GPU (= per rank): each rank runs (cpus-per-task - 1) dataloader workers, plus 16 for the
# eval loader. The run config's train.batch_size is the global batch size, split across all ranks. Set
# PG_TIMEOUT_MINUTES (default 240) above the duration of one eval round: in a multi-GPU run the other ranks wait for
# rank 0's eval.
# The data goes to $TMP, the job's folder on each node's local SSD (deleted when the job ends), once per node. Not
# $BEEOND: that BeeGFS is metadata-bound and starved the loader on millions of small JPEGs. KneeNo's prepare_data.py
# must give the extracted files fresh timestamps (FreshTimesTarFile): the cluster's nightly cleaner deletes $TMP
# files whose atime and mtime are both old, which tar-restored ones are.
#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=lightly-2d-dinov2
#SBATCH --output=slurm-%j-dinov2_2d_kneeno.out

set -euo pipefail

# --- Locations (override through --export, e.g. `sbatch --export=ALL,CONFIG=... ...`) ---------------------
# RWTH's sbatch does not export a `CONFIG=... sbatch` prefix to the job. To resume a run, pass
# PRETRAIN_JOB_ID=<id of its first job> the same way; unset, the run is named after this job (see
# cluster/pretrain_dinov2_kneeno.py).
# Model, output directory and everything else about the run are in the run config.
# SLURM runs a copy of this script, so the repo cannot be located via $0.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KNEENO_DIR="${KNEENO_DIR:-$REPO_DIR/../KneeNo}"
CONFIG="${CONFIG:-cluster/configs/pretrain-MI-vitb14-2d.yaml}"  # relative to REPO_DIR
# The MSK tars (KneeNo/data/msk/preprocess_msk.py's out_dir), extracted to $TMP/kneeno_data/msk. Set it iff the
# run config's data.data_root.msk is set (pretraining fails at startup if that folder is missing).
MSK_TAR_DIR="${MSK_TAR_DIR:-}"

# Fail here rather than after the first epoch
[[ "$CONFIG" == /* ]] || CONFIG="$REPO_DIR/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Run config not found at $CONFIG (set CONFIG)" >&2; exit 1; }
[[ -f "$KNEENO_DIR/data/prepare_data.py" ]] || { echo "KneeNo not found at $KNEENO_DIR (set KNEENO_DIR)" >&2; exit 1; }
[[ -z "$MSK_TAR_DIR" || -d "$MSK_TAR_DIR" ]] || { echo "MSK_TAR_DIR is not a directory: $MSK_TAR_DIR" >&2; exit 1; }
[[ -n "${TMP:-}" && -d "$TMP" ]] || { echo "TMP is not a directory (the job's node-local scratch): ${TMP:-unset}" >&2; exit 1; }
MSK_ARGS=()
[[ -z "$MSK_TAR_DIR" ]] || MSK_ARGS=(--msk-tar-dir "$MSK_TAR_DIR")

# Fail here rather than deep inside Lightning: one task per GPU is what DDP needs.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_NTASKS_PER_NODE:-1}" != "$SLURM_GPUS_ON_NODE" ]]; then
    echo "--ntasks-per-node (${SLURM_NTASKS_PER_NODE:-1}) must equal the GPUs per node ($SLURM_GPUS_ON_NODE)" >&2
    exit 1
fi
PG_TIMEOUT_MINUTES="${PG_TIMEOUT_MINUTES:-240}"

cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"

# --- Data -> $TMP/kneeno_data/internal (the JPEG slices for pretraining and the eval dataset alike), and
# .../msk with MSK_TAR_DIR; each skipped if already populated. $TMP is node-local, so every node extracts its own copy
# (one task per node; the job's $TMP has the same path on all its nodes). No cleanup: $TMP is deleted at job end.
srun --nodes="${SLURM_NNODES:-1}" --ntasks="${SLURM_NNODES:-1}" --ntasks-per-node=1 "$PYTHON" "$KNEENO_DIR/data/prepare_data.py" --internal-tar-dir /hpcwork/p0021834/workspace_roman/jonas/BigKneeTar \
     ${MSK_ARGS[@]+"${MSK_ARGS[@]}"} --pool-size 16 --data-root "$TMP/kneeno_data"

# --- Training ---------------------------------------------------------------------------------------------
srun "$PYTHON" cluster/pretrain_dinov2_kneeno.py --config "$CONFIG" --pg-timeout-minutes "$PG_TIMEOUT_MINUTES"
