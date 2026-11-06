#!/bin/bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --job-name=gdi-stt-ssl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/worker-%j.log
set -euo pipefail
cd /abr/RESEARCH_USER/gdi-stt692-ssl-20261031
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
timeout --kill-after=10s 42000s /usr/bin/python3.8 -u -B worker.py
