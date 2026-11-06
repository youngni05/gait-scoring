#!/bin/bash
#SBATCH --job-name=care-pd-rf-tcn
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
/usr/bin/python3.8 -u -B bootstrap.py "${1:-pilot}" --folds "${2:-0,1,2,3,4,5}"
if [ "${1:-pilot}" = train ]; then
  /usr/bin/python3.8 -u -B bootstrap.py replay --folds "${2:-0,1,2,3,4,5}"
fi
