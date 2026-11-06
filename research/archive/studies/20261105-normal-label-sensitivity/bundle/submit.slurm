#!/bin/bash
#SBATCH --job-name=normal-label-sensitivity
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
family=${2:?CNN or STT}
if [[ "$mode" == "pilot" ]]; then
    /usr/bin/python3.8 -u experiment.py pilot --family "$family" --folds 0 --device cuda
    echo PILOT_COMPLETE
elif [[ "$mode" == "train" ]]; then
    test -f main-approved.json
    exec 9>"train-${family}.lock"
    flock -n 9
    /usr/bin/python3.8 -u experiment.py train --family "$family" --conditions B_y90_w1_adult,B_y110_w1_adult,B_y100_w025_adult,B_y100_w4_adult --device cuda
    echo ASSIGNED_TRAINING_COMPLETE
else
    exit 2
fi
