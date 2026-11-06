#!/bin/bash
#SBATCH --job-name=ljmu100-replay
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=DIS04,DIS05,DIS06
#SBATCH --output=logs/%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
root=$(pwd)
mkdir -p verification-data
ln -s "$root/data/data.npz" verification-data/data.npz
ln -s "$root/data/baseline" verification-data/baseline
for family in CNN STT bilateral; do
    /usr/bin/python3.8 -u experiment.py baseline --family "$family" --data-root "$root/verification-data" --device cuda
done
echo GPU_BASELINE_REPLAY_COMPLETE
