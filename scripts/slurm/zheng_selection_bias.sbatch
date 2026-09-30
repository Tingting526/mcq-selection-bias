#!/bin/bash
#SBATCH --job-name=mcq-zheng
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
#SBATCH --error=outputs/logs/slurm/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch on LRZ}"
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/environment.sh
nvidia-smi
args=(--model "${MODEL:-qwen2.5-7b-instruct}" --dataset "${DATASET:?Set mmlu, arc or csqa}")
[[ -z "${LIMIT:-}" ]] || args+=(--limit "$LIMIT")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
srun python scripts/run_zheng_selection_bias.py "${args[@]}"
