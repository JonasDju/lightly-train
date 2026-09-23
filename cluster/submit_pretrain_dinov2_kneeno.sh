#!/bin/bash
# Multi-GPU / multi-node needs no code change, only matching SLURM options. #SBATCH lines cannot be computed,
# so override them on the command line; --ntasks-per-node must equal the GPUs per node:
#     sbatch --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=8 cluster/submit_pretrain_dinov2_kneeno.sh
#     sbatch --nodes=2 --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=8 cluster/submit_pretrain_dinov2_kneeno.sh
# --cpus-per-task is per GPU (= per rank): each rank runs (cpus-per-task - 1) dataloader workers, plus 16 for the
# eval loader. batch_size stays the global default (128), split across all ranks. Set PG_TIMEOUT_MINUTES (default
# 240) above the duration of one eval round: in a multi-GPU run the other ranks wait for rank 0's eval.
##SBATCH --account=truhnlab
##SBATCH --partition=truhnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=lightly-3d-dinov2
#SBATCH --output=slurm-%j-dinov2_kneeno.out

set -euo pipefail

# --- Locations (override by exporting before sbatch, e.g. `OUT_DIR=... sbatch ...`) --------------------
# SLURM runs a copy of this script, so the repo cannot be located via $0.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KNEENO_DIR="${KNEENO_DIR:-$REPO_DIR/../KneeNo}"
OUT_DIR="${OUT_DIR:-/hpcwork/va105917/lightly/vitb14.24f}"
MODEL="${MODEL:-dinov2/vitb14}"

# Fail here rather than after the first epoch
[[ -f "$KNEENO_DIR/data/prepare_data.py" ]] || { echo "KneeNo not found at $KNEENO_DIR (set KNEENO_DIR)" >&2; exit 1; }

# Fail here rather than deep inside Lightning: one task per GPU is what DDP needs.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_NTASKS_PER_NODE:-1}" != "$SLURM_GPUS_ON_NODE" ]]; then
    echo "--ntasks-per-node (${SLURM_NTASKS_PER_NODE:-1}) must equal the GPUs per node ($SLURM_GPUS_ON_NODE)" >&2
    exit 1
fi
PG_TIMEOUT_MINUTES="${PG_TIMEOUT_MINUTES:-240}"

cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"

# --- Data -> node-local scratch ($TMP/kneeno_data/{unlabeled,labeled}); skipped if already populated -------
srun --ntasks-per-node=1 "$PYTHON" "$KNEENO_DIR/data/prepare_data.py" --unlabeled-tar-dir /hpcwork/p0021834/workspace_roman/jonas/BigKneeTar \
                                                                      --pool-size 16

# --- Training ---------------------------------------------------------------------------------------------
srun "$PYTHON" cluster/pretrain_dinov2_kneeno.py --out "$OUT_DIR" --model "$MODEL" \
                                                --pg-timeout-minutes "$PG_TIMEOUT_MINUTES"
