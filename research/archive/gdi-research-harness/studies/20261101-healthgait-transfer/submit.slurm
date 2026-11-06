#!/bin/bash
#SBATCH --job-name=hg-transfer
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --exclude=DIS04,DIS05,DIS06
#SBATCH --output=logs/worker-%j.log
set -euo pipefail
bash launch.sh "$@"
