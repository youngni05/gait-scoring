#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Use an allocated GPU shell or sbatch submit.slurm. Do not train on ABRM02."
    exit 2
fi
mode="${1:-pilot}"
if [[ "$mode" != "pilot" && "$mode" != "train" ]]; then
    echo "Usage: bash launch.sh pilot|train"
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p logs
exec > >(tee -a "logs/${mode}.log") 2>&1
date -Is
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
timeout --signal=TERM --kill-after=10s 120s python3 -u -B run.py check --device cpu --output checks/input
timeout --signal=TERM --kill-after=10s 180s python3 -u -B run.py smoke --device cuda --output checks/gpu_smoke
if [[ "$mode" == "pilot" ]]; then
    timeout --signal=TERM --kill-after=10s 900s python3 -u -B run.py pilot --device cuda --output runs/pilot
    echo "CNN_PILOT_FINISHED"
else
    timeout --signal=TERM --kill-after=20s 10900s python3 -u -B run.py train --device cuda --output runs/main
    python3 -u -B run.py export --output runs/main
    echo "CNN_TRAINING_AND_EXPORT_FINISHED"
fi
date -Is
