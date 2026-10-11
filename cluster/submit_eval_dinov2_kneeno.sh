#!/bin/bash
# Standalone evaluation of a finished pretraining run. PRETRAIN_JOB_ID (the pretraining job's SLURM ID) is required,
# the eval config locates the run's checkpoint and output folder through it. Pass it with --export: RWTH's sbatch
# submits with --export=NONE, so a `PRETRAIN_JOB_ID=... sbatch ...` prefix never reaches the job:
#     sbatch --export=ALL,PRETRAIN_JOB_ID=4839286 cluster/submit_eval_dinov2_kneeno.sh
# The evaluation runs in a single process on one GPU (KneeNo evaluates on rank 0 only, more GPUs would not help).
# --cpus-per-task feeds the eval loader's eval.data.num_workers (31) workers.
##SBATCH --account=truhnlab
##SBATCH --partition=truhnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --job-name=lightly-3d-dinov2-eval
#SBATCH --output=slurm-%j-dinov2_kneeno_eval.out

set -euo pipefail

# --- Locations (override through --export, e.g. `sbatch --export=ALL,PRETRAIN_JOB_ID=...,CONFIG=... ...`) -------
# Checkpoint, output directories and everything else about the evaluation are in the eval config.
# SLURM runs a copy of this script, so the repo cannot be located via $0.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KNEENO_DIR="${KNEENO_DIR:-$REPO_DIR/../KneeNo}"
CONFIG="${CONFIG:-cluster/configs/eval-MI-vitb14-24f.yaml}"  # relative to REPO_DIR
: "${PRETRAIN_JOB_ID:?Set PRETRAIN_JOB_ID (sbatch --export=ALL,PRETRAIN_JOB_ID=<id>) to the ID of the pretraining job}"
# The job's node-local scratch: the data is extracted there and the eval config's data_root points there.
: "${TMP:?TMP (the node-local scratch directory of the job) is not set}"
export PRETRAIN_JOB_ID

# Fail here rather than after the data extraction
[[ "$CONFIG" == /* ]] || CONFIG="$REPO_DIR/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Eval config not found at $CONFIG (set CONFIG)" >&2; exit 1; }
[[ -f "$KNEENO_DIR/data/prepare_data.py" ]] || { echo "KneeNo not found at $KNEENO_DIR (set KNEENO_DIR)" >&2; exit 1; }
if [[ "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo "The evaluation runs in a single process; submit with one task (got ${SLURM_NTASKS})" >&2
    exit 1
fi

cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"

# --- Data -> node-local scratch ($TMP/kneeno_data/internal, prepare_data.py's default): the labeled eval dataset;
# skipped if already populated
srun "$PYTHON" "$KNEENO_DIR/data/prepare_data.py" --internal-tar-dir /hpcwork/p0021834/workspace_roman/jonas/BigKneeTar \
     --pool-size 16

# --- Evaluation -------------------------------------------------------------------------------------------
srun "$PYTHON" cluster/eval_dinov2_kneeno.py --config "$CONFIG"

# Cleanup
rm -rf "$TMP/kneeno_data"
echo "Deleted KneeNo data"
