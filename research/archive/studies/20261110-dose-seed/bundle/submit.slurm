#!/bin/bash
#SBATCH --job-name=dose-seed
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=DIS04,DIS05,DIS06
#SBATCH --output=logs/%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mode=${1:?pilot or train}
root=$(pwd)
PY=/usr/bin/python3.8
echo "node $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=uuid,name --format=csv,noheader || true
if [[ "$mode" == "pilot" ]]; then
    for family in CNN STT bilateral; do
        $PY -u experiment.py check --family "$family" --arm n4_s0 --data-root "$root/data" --device cuda
        $PY -u experiment.py pilot --family "$family" --arm n4_s0 --data-root "$root/data" --device cuda
    done
    $PY -u experiment.py check --family CNN --arm n64_s2 --data-root "$root/data" --device cuda
    $PY -u experiment.py check --family CNN --arm n0_s1 --data-root "$root/data" --device cuda
    echo PILOT_COMPLETE
else
    line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" tasks.txt)
    read -r arm family fold <<< "$line"
    echo "task $SLURM_ARRAY_TASK_ID: $arm $family $fold"
    $PY -u experiment.py train --family "$family" --arm "$arm" --folds "$fold" --data-root "$root/data" --device cuda
    echo TASK_COMPLETE
fi
