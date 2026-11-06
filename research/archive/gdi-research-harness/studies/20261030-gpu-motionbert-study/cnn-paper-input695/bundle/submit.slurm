#!/usr/bin/env bash
#SBATCH --job-name=gdi-cnn-paper
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --exclude=DIS06
#SBATCH --output=logs/slurm-%j.log
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR}"
bash launch.sh "${1:-pilot}"
