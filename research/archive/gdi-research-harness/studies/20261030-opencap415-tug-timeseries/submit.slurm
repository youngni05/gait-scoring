#!/bin/bash
#SBATCH --job-name=opencap415-tug
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mode="${1:-pilot}"
for duration in 1s 3s 6s all3s; do
  (cd "$duration" && python3 -u -B gpu.py "$mode")
done
echo "OPENCAP415_ALL_LENGTHS_${mode}_FINISHED"
