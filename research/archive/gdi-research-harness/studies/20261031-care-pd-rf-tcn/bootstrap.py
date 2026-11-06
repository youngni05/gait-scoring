"""Unpack fixed Python 3.8 dependencies on node-local temporary storage."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

here=Path(__file__).resolve().parent
wheels=sorted((here/'wheels').glob('*.whl'))
assert len(wheels)==4
with tempfile.TemporaryDirectory(prefix='care-pd-'+os.environ['SLURM_JOB_ID']+'-') as target:
    hashes={}
    for wheel in wheels:
        hashes[wheel.name]=hashlib.sha256(wheel.read_bytes()).hexdigest()
        with zipfile.ZipFile(wheel) as z: z.extractall(target)
    print('LOCAL_DEPENDENCIES_READY',json.dumps(hashes),flush=True)
    env=dict(os.environ,PYTHONPATH=target,PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,'-u','-B',str(here/'run.py')]+sys.argv[1:],cwd=here,env=env,check=True)
