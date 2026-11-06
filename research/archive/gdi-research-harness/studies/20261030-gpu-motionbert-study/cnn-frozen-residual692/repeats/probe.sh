#!/usr/bin/env bash
set -euo pipefail
date -Is
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
for candidate in python3 python3.8 python3.9 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        "$candidate" -B -c 'import sys,importlib.util; print(sys.executable,sys.version); print("torch:",importlib.util.find_spec("torch"))'
    fi
done
ls -ld /abr/RESEARCH_USER/.local/lib/python3.8/site-packages/torch || true
ls -l /usr/bin/python3* || true
date -Is
