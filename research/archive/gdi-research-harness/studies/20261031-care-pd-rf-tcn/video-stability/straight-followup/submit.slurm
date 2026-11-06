#!/bin/bash
#SBATCH --job-name=care-straight
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%j.log
set -euo pipefail
export PYTHONPATH=/abr/RESEARCH_USER/.local/lib/python3.8/site-packages
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/usr/bin/python3 -u infer.py "$1"
