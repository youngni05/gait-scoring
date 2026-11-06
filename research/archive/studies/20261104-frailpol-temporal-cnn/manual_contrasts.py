"""Descriptive fixed manual-input/loss ablations on the same 594 heldout people."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'manual-controls'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(R/'verification.json')['status']=='passed';rows=read(B/'manual-controls/heldout.json');d=np.load(B/'data.npz');ids=d['ids'];y=d['y'];pred={}
for c in sorted(set(r['condition'] for r in rows)):
    a=[r for r in rows if r['condition']==c];lookup={r['id']:r for r in a};assert len(a)==len(lookup)==594 and set(lookup)==set(ids) and all(lookup[int(i)]['actual']==v for i,v in zip(ids,y));pred[c]=np.array([lookup[int(i)]['prediction'] for i in ids])
pairs=[('speed_duration_tail4','speed_tail4'),('speed_duration_aux_tail4','speed_aux_tail4'),('speed_aux_tail4','speed_tail4'),('speed_duration_aux_tail4','speed_duration_tail4'),('speed_tail16','speed_tail4')];rng=np.random.default_rng(20261106);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))];out={}
for a,b in pairs:
    e=np.abs(pred[a]-y);f=np.abs(pred[b]-y);delta=e-f;out[a+' vs '+b]=dict(n=594,slow_n=9,mae_a=float(e.mean()),mae_b=float(f.mean()),mae_difference=float(delta.mean()),mae_difference_ci95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),slow_mae_a=float(e[slow].mean()),slow_mae_b=float(f[slow].mean()),slow_difference_ci95=np.quantile(delta[sb].mean(1),[.025,.975]).tolist())
record=dict(status='descriptive_paired_contrasts_rebuilt',source_verification_sha256=sha(R/'verification.json'),comparisons=out,limits='Exploratory/descriptive aftersourcepointresults;conditional5000personbootstrap,notselection-adjusted/external/clinical. Durationisautomaticallyselectedshortcoresupport,notwholeTUGtimeornecessarilyonecycle. Noinputwaveform,noage/clinicalcovariates. All5fixedmanualcontrastpairsreported.');(R/'contrasts.json').write_text(json.dumps(record,indent=2));print(json.dumps(out,indent=2))
