"""Run the unchanged strict audit in its original numeric environment, preserving local failures."""
from pathlib import Path
import json,subprocess,time
import manage as m

START=r"""
from pathlib import Path
import os,json,subprocess,hashlib
p=Path(ROOT)/('seed%d'%SEED);out=p/'recovery';out.mkdir(exist_ok=True);record=out/'source-audit-process.json'
if record.exists():print(record.read_text())
else:
 code='''import os,sys,json,subprocess,datetime
from pathlib import Path
import numpy,torch
p=Path.cwd();out=p/'recovery'
env=dict(os.environ);env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
(out/'source-audit-environment.json').write_text(json.dumps(dict(python=sys.version,numpy=numpy.__version__,torch=torch.__version__,started_at=datetime.datetime.now().isoformat()),indent=2))
with (out/'source-audit.log').open('w') as log:
 r=subprocess.run([sys.executable,'-B','support.py','review','--out',str(out/'source-full-review.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
(out/'source-audit-exit.json').write_text(json.dumps(dict(exit_code=r.returncode,finished_at=datetime.datetime.now().isoformat())))
'''
 env=dict(os.environ);env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 with (out/'source-audit-launch.log').open('w') as log:process=subprocess.Popen(['/usr/bin/python3.8','-u','-B','-c',code],cwd=str(p),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 info=dict(pid=process.pid,seed=SEED,purpose='unchanged full strict audit and CPU64 replay; no optimization',manifest_sha256=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest());record.write_text(json.dumps(info,indent=2));print(json.dumps(info))
"""
STATUS=r"""
from pathlib import Path
import json,hashlib
p=Path(ROOT)/('seed%d/recovery'%SEED);exit=p/'source-audit-exit.json';review=p/'source-full-review.json'
out={'exit':json.loads(exit.read_text()) if exit.exists() else None,'review':json.loads(review.read_text()) if review.exists() else None}
if out['exit']:
 out['files']={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['source-audit-environment.json','source-audit-exit.json','source-audit.log']+(['source-full-review.json'] if review.exists() else [])}
 if out['exit']['exit_code']!=0:out['log_tail']=(p/'source-audit.log').read_text()[-4000:]
print(json.dumps(out))
"""

def collect(seed):
    process=json.loads(m.remote('SEED='+repr(seed)+'\n'+START));m.save(m.H/('checks/source-audit-seed%d-process.json'%seed),process)
    deadline=time.monotonic()+600
    while True:
        result=json.loads(m.remote('SEED='+repr(seed)+'\n'+STATUS))
        if result['exit']:break
        if time.monotonic()>deadline:raise TimeoutError('Original-environment audit still running; reuse its saved PID next time, do not resubmit')
        time.sleep(5)
    for name,digest in result['files'].items():
        dest=m.H/'checks'/('seed%d-'%seed+name)
        m.command(m.SCP+['RESEARCH_USER@PRIVATE_HOST:'+m.REMOTE+'/seed%d/recovery/'%seed+name,str(dest)])
        assert m.sha(dest)==digest,'Source audit receipt mismatch'
    m.save(m.H/('checks/source-audit-seed%d-status.json'%seed),result)
    assert result['exit']['exit_code']==0,result.get('log_tail')
    proof=result['review'];assert proof['passed'] and proof['fits']==25 and proof['manifest_sha256']==m.sha(m.H/('seed%d/manifest.json'%seed))
    assert len(proof['final_cpu64_replays'])==5 and all(v['maximum_prediction_error']<1e-4 for v in proof['final_cpu64_replays'])
    return m.H/'checks'/('seed%d-source-full-review.json'%seed)

def finish_local_replay(seed,extracted):
    out=m.H/('checks/local-cpu64-replay-seed%d.json'%seed)
    result=subprocess.run([str(m.PY),'-B',str(m.H/'replay_received.py'),'--seed',str(seed),'--received',str(extracted),'--out',str(out)],cwd=str(m.H),env=m.local_env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (m.H/('checks/local-cpu64-replay-seed%d.log'%seed)).write_text(result.stdout,encoding='utf-8')
    assert result.returncode==0,result.stdout[-4000:]
    assert m.read(out)['passed'];return out
