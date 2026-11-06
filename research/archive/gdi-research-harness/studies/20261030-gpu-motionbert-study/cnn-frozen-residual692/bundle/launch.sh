#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
test -n "${SLURM_JOB_ID:-}" || { echo "Slurm GPU allocation required"; exit 2; }
mode="${1:-pilot}"
[[ "$mode" == pilot || "$mode" == train ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p logs
exec > >(tee -a "logs/${mode}.log") 2>&1
date -Is
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
timeout --kill-after=10s 300s python3 -u -B run.py check --device cuda --output checks/input_gpu
timeout --kill-after=10s 180s python3 -u -B verify.py --device cuda --output checks/synthetic_gpu
if [[ "$mode" == pilot ]]; then
    timeout --kill-after=10s 900s python3 -u -B run.py pilot --device cuda --output runs/pilot
    echo FROZEN_RESIDUAL_PILOT_FINISHED
else
    timeout --kill-after=20s 10800s python3 -u -B run.py train --device cuda --output runs/main
fi
date -Is
