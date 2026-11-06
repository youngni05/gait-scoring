#!/bin/bash
#SBATCH --job-name=older44-sppb-ts
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python3 -u -B gpu.py "${1:-pilot}"
