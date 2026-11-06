from pathlib import Path
import sys,time,subprocess,json,hashlib
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=H/'round04/queued-plan.json';frozen={name:sha(H/name) for name in ['round4.py','review4.py','features4.py']}
    if plan.exists():assert read(plan)['hashes']==frozen
    else:plan.write_text(json.dumps(dict(status='waiting_for_previous_verified_results',hashes=frozen,design='16 frozen candidates: Ridge alpha1/10/100/1000 x tailweight1/4 x raw/nominal6m-speed-residual targets; original and multi-cycle inputs, each with matching speed-only baseline. Inner3 selection and original outer5. No new GPU jobs. Stop without fitting if prior verified feature model reaches all3goals.',goal=dict(mae=.9,p95=3,slow_mae=3)),indent=2),encoding='utf8')
    start=time.monotonic()
    while not all((H/r/'verification.json').exists() for r in ['round02','round03']):
        if time.monotonic()-start>3600:raise TimeoutError('Prior rounds did not verify; no round04 training launched')
        time.sleep(5)
    for rd in ['round02','round03']:
        assert read(H/rd/'verification.json')['status']=='passed'
        if any(v['goal_met'] for k,v in read(H/rd/'comparison.json').items() if k.startswith('expanded/')):
            print('Prior round reached goal; no additional fitting',rd,flush=True);return
    assert read(H/'round04/feature-verification.json')['status']=='passed' and read(H/'round04/solver-verification.json')['status']=='passed'
    assert frozen=={name:sha(H/name) for name in frozen}
    for script,log in [('round4.py','round4-log.txt'),('review4.py','round4-review-log.txt'),('uncertainty.py','uncertainty-log.txt')]:
        with (H/log).open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(H/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('Round04 completed and independently verified',flush=True)
if __name__=='__main__':main()
