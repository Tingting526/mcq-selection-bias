#!/bin/bash
#SBATCH --job-name=mcq-analysis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
#SBATCH --error=outputs/logs/slurm/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch; do not run on the login node}"
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/environment.sh
hostname
scontrol show job "$SLURM_JOB_ID"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
MODEL="${MODEL:-qwen2.5-7b-instruct}"
DATASET="${DATASET:-medqa}"
STEP="${STEP:-pipeline}"
args=(--model "$MODEL" --dataset "$DATASET" --batch-size "${BATCH_SIZE:-4}" --scorer real)
[[ -z "${LIMIT:-}" ]] || args+=(--limit "$LIMIT")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
ARTIFACT="${PREPROCESSED:-$(python -c 'import sys; sys.path.insert(0,"scripts"); from run_pipeline import artifact_for; print(artifact_for(sys.argv[1]))' "$DATASET")}"
args+=(--preprocessed "$ARTIFACT")
case "$STEP" in
  pipeline) script=run_pipeline.py ;;
  baseline) script=run_baseline.py ;;
  permutations) script=run_permutations.py ;;
  prior) script=estimate_prior.py; args+=(--alpha 0.05) ;;
  pride) script=run_pride.py; args+=(--prior "${PRIOR:?Set PRIOR to saved prior JSON}") ;;
  *) echo "Unknown STEP=$STEP" >&2; exit 2 ;;
esac
srun python "scripts/$script" "${args[@]}"
