#!/bin/bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=DIS04
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mode=${1:?pilot or train}
arm=${2:?arm}
/usr/bin/python3.8 -u -B train.py "$mode" --arm "$arm" --device cuda
echo "${mode^^}_DONE $arm"
