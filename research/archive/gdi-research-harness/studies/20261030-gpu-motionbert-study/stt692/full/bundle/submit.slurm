#!/usr/bin/env bash
#SBATCH --job-name=gdi-stt-full
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec > >(tee "logs/train-${SLURM_JOB_ID}.log") 2>&1
date -Is
hostname
/usr/bin/python3.8 -B capacity.py --min-free 1 > "logs/capacity-${SLURM_JOB_ID}.json"
cat "logs/capacity-${SLURM_JOB_ID}.json"
timeout --kill-after=10s 300s /usr/bin/python3.8 -u -B run.py check --device cuda --output "checks/gpu-${SLURM_JOB_ID}"
timeout --kill-after=10s 300s /usr/bin/python3.8 -u -B verify.py --device cuda --output "checks/synthetic-${SLURM_JOB_ID}"
timeout --kill-after=10s 600s /usr/bin/python3.8 -u -B verify_nested.py --device cuda --output "checks/nested-${SLURM_JOB_ID}"
/usr/bin/python3.8 -u -B run.py train --device cuda --output runs/main
date -Is
