#!/bin/bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --job-name=gdi-stt-person-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
timeout --kill-after=10s 3300s /usr/bin/python3.8 -u -B pilot_all.py
