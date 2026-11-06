#!/bin/bash
#SBATCH --job-name=ljmu100-gdi
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
folds=${2:-0,1,2,3,4}
root=$(pwd)
if [[ "$mode" == "pilot" ]]; then
    for family in CNN STT bilateral; do
        /usr/bin/python3.8 -u experiment.py check --family "$family" --data-root "$root/data" --device cuda
        /usr/bin/python3.8 -u experiment.py smoke --family "$family" --data-root "$root/data" --device cuda
        /usr/bin/python3.8 -u experiment.py pilot --family "$family" --data-root "$root/data" --device cuda
        /usr/bin/python3.8 -u experiment.py baseline --family "$family" --data-root "$root/data" --device cuda
    done
    echo PILOT_AND_BASELINE_COMPLETE
else
    test -f main-approved.json
    for family in CNN STT bilateral; do
        /usr/bin/python3.8 -u experiment.py train --family "$family" --data-root "$root/data" --device cuda --folds "$folds"
    done
    echo ASSIGNED_FOLDS_COMPLETE
fi
