#!/usr/bin/env bash
set -euo pipefail

# Ensure Conda is initialized and PMNI env is active
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "PMNI" ]]; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda.sh not found at ~/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
  fi
  conda activate PMNI
fi

echo "Conda environment active: ${CONDA_DEFAULT_ENV}"