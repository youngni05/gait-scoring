#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p control logs
mode="${1:-pilot}"
round="${2:-0}"
folds="${3:-0,1,2,3,4}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
status() {
 /usr/bin/python3.8 - "$1" "$mode" "$folds" <<'PY'
import json,os,pathlib,sys,datetime
p=pathlib.Path('control')/('worker-'+os.environ.get('SLURM_JOB_ID','local')+'.json')
p.write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),phase=sys.argv[1],mode=sys.argv[2],folds=sys.argv[3],updated_at=datetime.datetime.now().isoformat())))
PY
}
trap 'status training_or_summary_failed' ERR
status training
/usr/bin/python3.8 -B train.py check
/usr/bin/python3.8 -B train.py "$mode" --round "$round" --folds "$folds"
if [[ "$mode" == pilot ]]; then status pilot_complete; else status assigned_folds_complete; fi
