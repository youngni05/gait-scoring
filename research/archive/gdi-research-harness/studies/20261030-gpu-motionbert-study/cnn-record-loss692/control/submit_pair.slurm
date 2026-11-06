#!/usr/bin/env bash
#SBATCH --job-name=gdi-cnn-record-pair
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=18:00:00
#SBATCH --exclude=DIS03,DIS04,DIS05,DIS06
#SBATCH --output=logs/worker-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python3 -u -B control/worker_pair.py
