#!/bin/bash
# Source from the project root. Only a project-owned directory on home DSS is used.
export MCQ_CACHE_ROOT="${MCQ_CACHE_ROOT:-/dss/dsshome1/0F/ra39dik2/analysis/outputs/cache}"
[[ "$MCQ_CACHE_ROOT" == /dss/* ]] || { echo 'MCQ_CACHE_ROOT must be on DSS' >&2; return 2; }
export HF_HOME="$MCQ_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$MCQ_CACHE_ROOT/torch"
export XDG_CACHE_HOME="$MCQ_CACHE_ROOT/xdg"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"
if [[ -n "${MCQ_CONDA_ENV:-}" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    conda activate "$MCQ_CONDA_ENV"
  else
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PWD/.micromamba}"
    MICROMAMBA_BIN="${MICROMAMBA_BIN:-$PWD/.micromamba/bin/micromamba}"
    if [[ ! -x "$MICROMAMBA_BIN" ]]; then
      MICROMAMBA_BIN="$(command -v micromamba || true)"
    fi
    [[ -x "$MICROMAMBA_BIN" ]] || { echo "micromamba not found; run scripts/slurm/setup_micromamba_environment.sh first" >&2; return 2; }
    env_prefix="$("$MICROMAMBA_BIN" env list | awk -v name="$MCQ_CONDA_ENV" '$1 == name {print $NF; exit}')"
    if [[ -z "$env_prefix" && -d "$MAMBA_ROOT_PREFIX/envs/$MCQ_CONDA_ENV" ]]; then
      env_prefix="$MAMBA_ROOT_PREFIX/envs/$MCQ_CONDA_ENV"
    fi
    [[ -n "$env_prefix" && -x "$env_prefix/bin/python" ]] || { echo "micromamba env '$MCQ_CONDA_ENV' not found" >&2; return 2; }
    export PATH="$env_prefix/bin:$PATH"
    export CONDA_PREFIX="$env_prefix"
    export VIRTUAL_ENV="$env_prefix"
  fi
else
  source "${MCQ_VENV:-$PWD/.venv}/bin/activate"
fi
