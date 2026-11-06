"""Independent OOF/baseline metrics; full checkpoint audit is a separate requirement."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=read(H/'checks/full-receipt.json');assert receipt['passed']
report=read(H.parent/'inner-bagging692/analysis/comparison.json')
assert sha(report['predictions_path'])==report['predictions_sha256']
with np.load(report['predictions_path']) as z:base={k:z[k].copy() for k in ['y','person_id','video_id','fold','bagged']}
models={'best_blend':base['bagged']};raw={}
for v in ['mse','ranksim']:
    p=Path(receipt['extracted'])/'runs/main'/v/'oof_predictions.npz'
    with np.load(p) as z:
        for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(base[k],z[k])
        models[v]=z['predictions'][:,0].copy();raw[v]=z['predictions_raw'][:,0].copy()
ids=base['person_id'];y=base['y'];people=np.unique(ids)
high=np.array([y[ids==p].mean()>=90 for p in people]);assert high.sum()==68
def metrics(pred):
    e=pred-y
    mae=np.array([np.abs(e[ids==p]).mean() for p in people]);bias=np.array([e[ids==p].mean() for p in people])
    out=dict(mae=float(mae.mean()),bias=float(bias.mean()),high_person90=dict(people=68,mae=float(mae[high].mean()),bias=float(bias[high].mean())),observations={})
    for t in [90,100]:
        ae=[];be=[];n=0
        for p in people:
            v=e[ids==p][y[ids==p]>=t]
            if len(v):ae.append(abs(v).mean());be.append(v.mean());n+=len(v)
        out['observations'][str(t)]=dict(people=len(ae),side_observations=n,mae=float(np.mean(ae)),bias=float(np.mean(be)))
    return out,mae
results={};errors={}
for k,v in models.items():results[k],errors[k]=metrics(v)
assert abs(results['best_blend']['mae']-report['mae']['bagged'])<1e-10
rng=np.random.default_rng(20261101);paired={}
for k in ['mse','ranksim']:
    delta=errors[k]-errors['best_blend'];boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)])
    paired[k]=dict(delta=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
out=dict(oof_metadata_and_baseline_reproduction_passed=True,full_checkpoint_audit='separate checks/full-verification.json; required before final acceptance',
    results=results,raw_results={k:metrics(v)[0] for k,v in raw.items()},paired_vs_best=paired,baseline_prediction_sha256=report['predictions_sha256'])
(H/'checks/comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
