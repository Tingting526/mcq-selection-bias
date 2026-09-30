#!/bin/bash
#SBATCH --job-name=mcq-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
#SBATCH --error=outputs/logs/slurm/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch; do not run on the login node}"
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/environment.sh
hostname
scontrol show job "$SLURM_JOB_ID"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
MODEL="${MODEL:-qwen2.5-7b-instruct}"
DATASET="${DATASET:-medqa}"
ARTIFACT="${PREPROCESSED:-$(python -c 'import sys; sys.path.insert(0,"scripts"); from run_pipeline import artifact_for; print(artifact_for(sys.argv[1]))' "$DATASET")}"
# The real backend performs and saves contextual A/B/C/D token diagnostics.
srun python scripts/run_baseline.py --model "$MODEL" --dataset "$DATASET" \
  --preprocessed "$ARTIFACT" --scorer real --limit 2 --batch-size 1
