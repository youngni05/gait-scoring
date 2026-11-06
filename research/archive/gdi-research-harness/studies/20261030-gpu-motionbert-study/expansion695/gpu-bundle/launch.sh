#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run inside the allocated GPU shell, not the login shell."
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p logs
exec > >(tee -a logs/expanded_run.log) 2>&1
echo "EXPANSION_PREFLIGHT_STARTED"
date -Is
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
timeout --kill-after=10s 300s python3 -u -B run.py check --device cuda --output checks/replay
timeout --kill-after=10s 600s python3 -u -B run.py profile --device cuda --batch-size 32 --output checks/profile_b32
echo "EXPANDED_TRAINING_STARTED"
date -Is
python3 -u -B batch.py run --config protocol.ready.json --output runs/main --device cuda
python3 -u -B batch.py summarize --config protocol.ready.json --output runs/main
date -Is
echo "EXPANDED_TRAINING_AND_SUMMARY_FINISHED"
