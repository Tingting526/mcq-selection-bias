#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup must run on the LRZ login node, not on the local machine." >&2
  exit 2
fi

mkdir -p outputs/logs/slurm outputs/environment .micromamba/bin

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PWD/.micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$PWD/.micromamba/bin/micromamba}"
ENV_NAME="${MCQ_CONDA_ENV:-mcq-analysis}"
ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"

if [[ ! -x "$MICROMAMBA_BIN" ]]; then
  archive="outputs/environment/micromamba-linux-64-latest.tar.bz2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --output "$archive" "https://micro.mamba.pm/api/micromamba/linux-64/latest"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive" "https://micro.mamba.pm/api/micromamba/linux-64/latest"
  else
    echo "Neither curl nor wget is available to download micromamba." >&2
    exit 2
  fi
  tar -xjf "$archive" -C outputs/environment bin/micromamba
  mv outputs/environment/bin/micromamba "$MICROMAMBA_BIN"
  rmdir outputs/environment/bin
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  if [[ -d "$ENV_PREFIX" ]]; then
    "$MICROMAMBA_BIN" remove -y -p "$ENV_PREFIX" --all || rm -rf "$ENV_PREFIX"
  fi
  "$MICROMAMBA_BIN" create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 pip
fi

export MCQ_CONDA_ENV="$ENV_NAME"
source scripts/slurm/environment.sh
python - <<'PY'
import os
import sys

prefix = os.path.realpath(sys.prefix)
root = os.path.realpath(os.environ["MAMBA_ROOT_PREFIX"])
print(f"Using Python: {sys.executable}")
print(f"Python prefix: {prefix}")
if not prefix.startswith(root + os.sep):
    raise SystemExit(
        f"Expected Python from {root}, but active prefix is {prefix}. "
        "Check micromamba activation before installing packages."
    )
PY
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip check
python -m pip freeze > "outputs/environment/requirements-$(date -u +%Y%m%dT%H%M%SZ).txt"
# No model is loaded during setup.
