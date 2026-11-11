#!/bin/bash
#SBATCH --job-name=finetune
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
echo "node $(hostname)"
if [[ "$mode" == "pilot" ]]; then
    for family in CNN STT bilateral; do
        $PY -u experiment_ft.py check --family "$family" --data-root "$root/data" --device cuda
        $PY -u experiment_ft.py pilot --family "$family" --data-root "$root/data" --device cuda
    done
    echo PILOT_COMPLETE
else
    f=$SLURM_ARRAY_TASK_ID
    if (( f < 5 )); then
        $PY -u experiment_ft.py train --family CNN --folds "$f" --data-root "$root/data" --device cuda
    else
        g=$((f - 5))
        $PY -u experiment_ft.py train --family STT --folds "$g" --data-root "$root/data" --device cuda
        $PY -u experiment_ft.py train --family bilateral --folds "$g" --data-root "$root/data" --device cuda
    fi
    echo TASK_COMPLETE
fi
