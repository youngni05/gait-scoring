#!/bin/bash
#SBATCH --job-name=care-video
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
cd /abr/RESEARCH_USER/care-pd-video-stability-20261031
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONPATH=/abr/RESEARCH_USER/.local/lib/python3.8/site-packages
/usr/bin/python3 -u -B infer.py "${1:-pilot}"
