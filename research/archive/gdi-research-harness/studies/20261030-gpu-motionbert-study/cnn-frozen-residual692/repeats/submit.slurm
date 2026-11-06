#!/usr/bin/env bash
#SBATCH --job-name=gdi-residual-seeds
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --nodelist=DIS03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/seeds-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
root="$PWD"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec > >(tee -a "$root/logs/sequential.log") 2>&1
date -Is
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
# Guard again AFTER allocation, leaving at least one other schedulable GPU free.
/usr/bin/python3.8 -B capacity.py --min-free 1
for seed in 20261029 20261030; do
    cd "$root/seed$seed"
    mkdir -p logs
    echo "SEED_START=$seed"
    timeout --kill-after=10s 300s /usr/bin/python3.8 -u -B run.py check --device cuda --output checks/gpu
    timeout --kill-after=10s 180s /usr/bin/python3.8 -u -B verify.py --device cuda --output checks/synthetic_gpu
    if [[ "$seed" == 20261029 ]]; then
        timeout --kill-after=10s 900s /usr/bin/python3.8 -u -B run.py pilot --device cuda --output runs/pilot
        /usr/bin/python3.8 -B -c 'import json; p=json.load(open("runs/pilot/pilot_summary.json")); assert p["passed"] and not p["outer_evaluation_used"]'
    fi
    timeout --kill-after=20s 5400s /usr/bin/python3.8 -u -B run.py train --device cuda --output runs/main
    echo "SEED_FINISHED=$seed"
done
echo BOTH_RESIDUAL_SEEDS_FINISHED
date -Is
