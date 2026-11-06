#!/usr/bin/env bash
#SBATCH --job-name=gdi-stt-bilateral
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
mkdir -p logs control
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec > >(tee "logs/worker-${SLURM_JOB_ID}.log") 2>&1
date -Is
hostname
/usr/bin/python3.8 -u -B worker.py
