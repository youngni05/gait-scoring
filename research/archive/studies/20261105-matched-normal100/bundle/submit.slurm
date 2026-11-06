#!/bin/bash
#SBATCH --job-name=matched-normal100
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
mode=${1:?pilot or train}
arm=${2:?adult or child}
root=$(pwd)
if [[ "$mode" == "pilot" ]]; then
    for family in CNN STT bilateral; do
        /usr/bin/python3.8 -u experiment.py check --arm "$arm" --family "$family" --data-root "$root/data/$arm" --device cuda
        /usr/bin/python3.8 -u experiment.py smoke --arm "$arm" --family "$family" --data-root "$root/data/$arm" --device cuda
        /usr/bin/python3.8 -u experiment.py pilot --arm "$arm" --family "$family" --data-root "$root/data/$arm" --device cuda
    done
    echo PILOT_COMPLETE
elif [[ "$mode" == "train" ]]; then
    test -f main-approved.json
    exec 9>"data/train-${arm}.lock"
    flock -n 9
    for family in CNN STT bilateral; do
        /usr/bin/python3.8 -u experiment.py train --arm "$arm" --family "$family" --data-root "$root/data/$arm" --device cuda
    done
    echo ASSIGNED_ARM_COMPLETE
else
    exit 2
fi
