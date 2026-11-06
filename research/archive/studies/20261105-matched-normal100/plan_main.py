"""Size one GPU worker per matched arm from six independently replayed pilots."""
import datetime,hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/matched-normal100-20261105')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
workers=[]
for arm in ['adult','child']:
    with np.load(BIG/arm/'data.npz') as z:fold=z['fold'];owner=z['window_record']
    reference=int(np.isin(owner,np.flatnonzero((fold!=0)&(fold!=1))).sum());families={}
    for family in ['CNN','STT','bilateral']:
        audit=read(HERE/f'pilot-{arm}-{family}-verified.json')
        assert audit['passed'] and audit['child_training_only'] and audit['training_only_scaling_and_calibration_verified']
        assert audit['independent_cpu_replay_max_error']<.001 and audit['independent_train_cpu_replay_max_error']<.001
        curve=read(BIG/'received/data'/arm/'runs/pilot'/family/'curve.json');steady=np.mean([h['seconds'] for h in curve[-2:]])
        config=read(HERE/'bundle'/family/'protocol.json');fits=[]
        for f in range(5):
            for j in [None]+[j for j in range(5) if j!=f]:
                tr=(fold!=f) if j is None else (fold!=f)&(fold!=j);n=int(np.isin(owner,np.flatnonzero(tr)).sum())
                fits.append(dict(outer=f,inner=j,train_windows=n,max_epochs=config['max_epochs'],estimated_seconds=float(steady*config['max_epochs']*n/reference+30)))
        families[family]=dict(steady_epoch_seconds=float(steady),estimated_seconds=sum(v['estimated_seconds'] for v in fits),fits=fits)
    seconds=sum(v['estimated_seconds'] for v in families.values());minutes=math.ceil(1.5*seconds/60)
    workers.append(dict(arm=arm,folds=list(range(5)),expected_fits=75,estimated_hours=seconds/3600,time_limit='%02d:%02d:00'%(minutes//60,minutes%60),families=families))
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bundle_sha256=sha(HERE/'bundle/manifest.json'),
    authorization='User requested matched adult/child school-GPU re-comparison on2026-11-05; prior school feature transfer/training authorization persists.',
    expected_main_fits=150,workers=workers,estimate_basis='Mean final2 actual pilot epochs, scaled by training windows, fixed maximum epochs including refit and30seconds per fit. Time limit1.5times estimate. No GPU holding after completion.',
    monitoring='No scheduled job tracking or hidden polling; collect_once.py returns immediately after one snapshot.')
(HERE/'main-approved.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:w[k] for k in ['arm','expected_fits','estimated_hours','time_limit']} for w in workers],indent=2))
