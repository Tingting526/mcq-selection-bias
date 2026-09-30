#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v sbatch >/dev/null && command -v sinfo >/dev/null || { echo "Run this script on LRZ after SSH login" >&2; exit 2; }
: "${PARTITION:?Run sinfo and set PARTITION to an available GPU partition}"
case "${1:-smoke}" in
  smoke) job=scripts/slurm/gpu_smoke_test.sbatch ;;
  zheng) job=scripts/slurm/zheng_selection_bias.sbatch ;;
  analysis) job=scripts/slurm/run_analysis_step.sbatch ;;
  *) echo 'Usage: bash scripts/slurm/submit.sh [smoke|analysis|zheng]' >&2; exit 2 ;;
esac
# Slurm opens logs before the job body runs.
mkdir -p outputs/logs/slurm
sinfo -h -o '%P' | sed 's/\*$//' | grep -Fx -- "$PARTITION" >/dev/null || { echo 'Partition not found by sinfo' >&2; exit 2; }
options=(--parsable --partition="$PARTITION")
[[ -z "${QOS:-}" ]] || options+=(--qos="$QOS")
[[ -z "${ACCOUNT:-}" ]] || options+=(--account="$ACCOUNT")
sbatch "${options[@]}" "$job"
