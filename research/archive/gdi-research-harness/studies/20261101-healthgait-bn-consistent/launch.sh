#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p control logs
mode="${1:-pilot}"
round="${2:-0}"
folds="${3:-0,1,2,3,4}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
status() {
  /usr/bin/python3.8 - "$1" "$mode" "$round" "$folds" <<'PY'
import json,os,pathlib,sys,datetime
p=pathlib.Path('control')/('worker-'+os.environ.get('SLURM_JOB_ID','local')+'.json')
tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),phase=sys.argv[1],mode=sys.argv[2],round=int(sys.argv[3]),folds=sys.argv[4],updated_at=datetime.datetime.now().isoformat())))
tmp.replace(p)
PY
}
trap 'status training_or_summary_failed' ERR
status training
/usr/bin/python3.8 -B train.py check
if [[ "$mode" == pilot ]]; then
  /usr/bin/python3.8 -B train.py pilot
  status pilot_complete
elif [[ "$mode" == source ]]; then
  /usr/bin/python3.8 -B train.py source
  status assigned_folds_complete
elif [[ "$mode" == train ]]; then
  /usr/bin/python3.8 -B train.py train --round "$round" --folds "$folds"
  status assigned_folds_complete
else
  echo "Unknown mode: $mode" >&2
  exit 2
fi
