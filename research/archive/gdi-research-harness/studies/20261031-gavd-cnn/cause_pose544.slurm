#!/usr/bin/env bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --exclude=DIS04,DIS05,DIS06
#SBATCH --job-name=gavd-cause544
#SBATCH --output=logs/pose-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec /usr/bin/python3.8 -u -B gpu_pose_worker544.py "$@"
