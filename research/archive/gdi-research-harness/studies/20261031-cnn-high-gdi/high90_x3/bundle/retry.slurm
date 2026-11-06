#!/usr/bin/env bash
#SBATCH --job-name=gdi-cnn-high90
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=18:00:00
#SBATCH --exclude=DIS03,DIS04,DIS05,DIS06
#SBATCH --output=/abr/RESEARCH_USER/gdi-cnn-high90-20261031/logs/worker-%j.log
set -euo pipefail
cd /abr/RESEARCH_USER/gdi-cnn-high90-20261031
/usr/bin/python3.8 -u -B worker_v2.py
