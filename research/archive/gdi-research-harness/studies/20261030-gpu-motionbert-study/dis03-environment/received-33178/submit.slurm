#!/usr/bin/env bash
#SBATCH --job-name=gdi-dis03-envcheck
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --nodelist=DIS03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=probe-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
nvidia-smi
timeout --signal=TERM --kill-after=10s 600s /usr/bin/python3.8 -u -B probe.py
