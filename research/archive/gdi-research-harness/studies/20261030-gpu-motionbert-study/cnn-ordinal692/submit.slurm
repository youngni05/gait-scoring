#!/bin/bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --exclude=DIS04,DIS05,DIS06
#SBATCH --job-name=gdi-ordinal692
#SBATCH --output=logs/slurm-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
srun --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 --output=logs/worker-%j-%t.log /usr/bin/python3.8 -u -B worker.py
