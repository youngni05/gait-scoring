"""Run the already prepared CNN repetitions locally, following a passed pilot."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import hashlib

HERE=Path(__file__).resolve().parent
def main():
    pilot=HERE/'cnn/r00-s20261030/pilot-summary.json'
    assert json.loads(pilot.read_text())['passed']
    shutil.copy2(pilot,HERE/'pilot-summary.json')
    started=time.time();completed=[]
    for name in json.loads((HERE/'cnn-runs.json').read_text()):
        folder=HERE/'cnn'/name
        print('CNN_CPU_RUN_STARTED',name,flush=True)
        with (folder/'cpu-training.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8','-u',str(folder/'cnn_cpu.py'),'train'],stdout=log,stderr=subprocess.STDOUT,check=True)
        s=json.loads((folder/'gpu-summary.json').read_text())
        assert s['folds_finished']==5 and s['models_finished']==1
        completed.append({'run':name,'metrics':s['metrics'],'elapsed_seconds':s['elapsed_seconds'],
                          'code_sha256':hashlib.sha256((folder/'cnn_cpu.py').read_bytes()).hexdigest()})
        print('CNN_CPU_RUN_FINISHED',name,json.dumps(s['metrics']),flush=True)
        (HERE/'cnn-progress.json').write_text(json.dumps(completed,indent=2))
    (HERE/'cnn-completion.json').write_text(json.dumps({'completed':True,'device':'cpu','runs':completed,'elapsed_seconds':time.time()-started},indent=2))
    print('ALL_CNN_CPU_REPEATS_COMPLETE',flush=True)

if __name__=='__main__':main()
