#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Slurm allocation required"; exit 2; }
mode="${1:-pilot}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p logs
exec > >(tee -a "logs/${mode}.log") 2>&1
if [[ "$mode" == "pilot" ]]; then
  timeout --kill-after=10s 600s python3 -u -B verify.py --device cuda --quick --output checks/gpu
  for variant in window_control record_mean; do
    timeout --kill-after=10s 1800s python3 -u -B run.py pilot --variant "$variant" --device cuda --output "runs/pilot/$variant"
    python3 -u -B review_pilot.py --variant "$variant" --device cuda --run "runs/pilot/$variant" --output "checks/pilot-$variant-gpu.json"
    python3 -u -B review_pilot.py --variant "$variant" --device cpu --run "runs/pilot/$variant" --output "checks/pilot-$variant-cpu.json"
  done
  echo RECORD_CNN_PAIR_PILOT_FINISHED
elif [[ "$mode" == "train" ]]; then
  variant="${2:?Specify window_control or record_mean}"
  [[ "$variant" == "window_control" || "$variant" == "record_mean" ]] || exit 2
  python3 - "$variant" <<'PY'
import json,sys
for device in ('cpu','gpu'):
    report=json.load(open('checks/pilot-%s-%s.json'%(sys.argv[1],device)))
    assert report['passed'] and report['variant']==sys.argv[1]
PY
  timeout --kill-after=20s 27100s python3 -u -B run.py train --variant "$variant" --device cuda --output "runs/main/$variant"
  python3 -u -B run.py export --variant "$variant" --output "runs/main/$variant"
  echo RECORD_CNN_TRAINING_FINISHED
else
  echo "Usage: launch.sh pilot | train VARIANT"; exit 2
fi
