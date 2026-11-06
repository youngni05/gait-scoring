#!/usr/bin/env bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --exclude=DIS04,DIS05,DIS06
#SBATCH --job-name=gdi-synth-conv
#SBATCH --output=logs/worker-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /usr/bin/env /usr/bin/python3.8 -u -B worker.py
