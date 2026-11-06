"""Compute two worker budgets from verified pilot timings and fixed training counts."""
import datetime,hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/child100-train-only-20261105')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(BIG/'data.npz') as z:fold=z['fold'];owner=z['window_record']
pilots={f:read(HERE/f'pilot-{f}-verified.json') for f in ['CNN','STT','bilateral']}
assert all(p['passed'] and p['child_training_only'] and p['training_only_scaling_and_calibration_verified'] for p in pilots.values())
workers=[]
reference=int(np.isin(owner,np.flatnonzero((fold!=0)&(fold!=1))).sum())
for folds in [[0,2,4],[1,3]]:
    estimates={}
    for family,p in pilots.items():
        cfg=read(HERE/'bundle'/family/'protocol.json')
        curve=read(BIG/'received/data/runs/pilot'/family/'curve.json')
        steady=float(np.mean([e['seconds'] for e in curve[-2:]]))
        # Refit epoch is not known yet, so reserve the already-fixed maximum.
        fits=[]
        for f in folds:
            for j in [None]+[j for j in range(5) if j!=f]:
                tr=(fold!=f) if j is None else (fold!=f)&(fold!=j)
                windows=int(np.isin(owner,np.flatnonzero(tr)).sum())
                seconds=steady*cfg['max_epochs']*windows/reference+30
                fits.append(dict(outer=f,inner=j,train_windows=windows,maximum_epochs=cfg['max_epochs'],estimated_seconds=seconds))
        estimates[family]=dict(steady_epoch_seconds=steady,estimated_seconds=sum(r['estimated_seconds'] for r in fits),fits=fits)
    total=sum(v['estimated_seconds'] for v in estimates.values())
    limit_minutes=math.ceil(1.5*total/60)
    workers.append(dict(folds=','.join(map(str,folds)),estimated_hours=total/3600,
        time_limit='%02d:%02d:00'%(limit_minutes//60,limit_minutes%60),family_estimates=estimates))
out=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pilot_job='35882',
    bundle_sha256=sha(HERE/'bundle/manifest.json'),data_sha256=sha(BIG/'data.npz'),pilot_cpu_replays=pilots,
    authorization='User explicitly approved child bundle transfer and school GPU training on 2026-11-05.',
    workers=workers,expected_main_fits=75,
    estimate_basis='Mean last two pilot epochs scaled by training-window count; fixed max epochs for all inner fits and refit, plus 30 seconds per fit for setup/replay. Slurm limit is 1.5 times estimate rounded to a minute. No holding after training.')
(HERE/'main-approved.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'workers':[{k:w[k] for k in ['folds','estimated_hours','time_limit']} for w in workers],'fits':75},indent=2))
