#!/bin/bash
#SBATCH --job-name=tug-speed-stability
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --output=logs/stability-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [ "$1" = pilot ]; then
  /usr/bin/python3.8 -u cnn/r00-s20261030/gpu.py pilot
  cp cnn/r00-s20261030/pilot-summary.json pilot-summary.json
else
  for run in r00-s20261030 r00-s20261031 r00-s20261032 r01-s20261030 r01-s20261031 r01-s20261032 r02-s20261030 r02-s20261031 r02-s20261032 r03-s20261030 r03-s20261031 r03-s20261032 r04-s20261030 r04-s20261031 r04-s20261032; do
    echo "RUN_STARTED $run"
    /usr/bin/python3.8 -u "cnn/$run/gpu.py" train
  done
fi
echo "STABILITY_${1}_FINISHED"
