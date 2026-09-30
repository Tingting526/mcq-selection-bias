#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p outputs/logs/slurm outputs/environment
if [[ -z "${MCQ_CONDA_ENV:-}" ]]; then
  "${PYTHON_BIN:-python3}" -m venv "${MCQ_VENV:-$PWD/.venv}"
fi
source scripts/slurm/environment.sh
python -m pip install -e '.[dev]'
python -m pip check
python -m pip freeze > "outputs/environment/requirements-$(date -u +%Y%m%dT%H%M%SZ).txt"
# No model is loaded during setup.
