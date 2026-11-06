#!/usr/bin/env bash
#SBATCH --job-name=gdi-stt-pilot
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec > >(tee "logs/pilot-${SLURM_JOB_ID}.log") 2>&1
date -Is
hostname
/usr/bin/python3.8 -B capacity.py --min-free 1
timeout --kill-after=10s 300s /usr/bin/python3.8 -u -B run.py check --device cuda --output "checks/gpu-${SLURM_JOB_ID}"
timeout --kill-after=10s 300s /usr/bin/python3.8 -u -B verify.py --device cuda --output "checks/synthetic-${SLURM_JOB_ID}"
timeout --kill-after=10s 900s /usr/bin/python3.8 -u -B run.py pilot --device cuda --output runs/pilot
date -Is
