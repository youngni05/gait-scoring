"""Additional descriptive comparison against the previously best-MAE speed model."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path('D:/Projects/D_P/studies/20261104-pose-compatible-tug');B=Path('D:/gait-research-data/D_P/pose-compatible-tug-20261104');OLD=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=read(H/'verification.json');assert v['status']=='passed' and v['heldout_sha256']==sha(B/'heldout.json');d=np.load('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104/data.npz');ids=d['ids'];y=d['y'];score=read(OLD/'scoreboard.json');key='round07/speed/rbf_ridge/tail4';reference=score['conditions'][key];path=Path(reference['source_heldout']);assert sha(path)==reference['source_heldout_sha256'];original=[]
for r in read(path):
    name=r['condition']+('/'+r['kind'] if 'kind' in r else '')
    if name==reference['condition']:original.append(r)
lookup={int(r['id']):r for r in original};assert set(lookup)==set(ids) and len(original)==594 and all(lookup[int(i)]['actual']==a for i,a in zip(ids,y));baseline=np.array([lookup[int(i)]['prediction'] for i in ids]);base_error=np.abs(baseline-y);assert np.isclose(base_error.mean(),reference['mae'],atol=1e-12,rtol=1e-12);rng=np.random.default_rng(20261104);boot=rng.integers(0,594,(5000,594));rows=read(B/'heldout.json');out={}
for c in ['timing','calibrated_spatiotemporal','timing_core','metric_core','all_sensor']:
    lookup={int(r['id']):r for r in rows if r['condition']==c and r['kind']=='stacked'};assert set(lookup)==set(ids);p=np.array([lookup[int(i)]['prediction'] for i in ids]);difference=np.abs(p-y)-base_error;out[c]=dict(mae=float(np.abs(p-y).mean()),difference_prior_speed_mae=float(difference.mean()),paired_difference_ci95=np.quantile(difference[boot].mean(1),[.025,.975]).tolist())
result=dict(status='descriptive_additional_existing_benchmark_comparison',people=594,reference=key,prior_speed_mae=float(base_error.mean()),prior_reference_heldout_sha256=sha(path),current_heldout_sha256=sha(B/'heldout.json'),conditions=out,limits='The earlier best speed configuration was identified in the existing whole development scoreboard, not selected within a new untouched test set. This extra benchmark does not change primary fixed-R5 same-procedure comparisons or select/retrain any model. Bootstrap is conditional and not adjusted for prior model search or multiple comparisons.',code_sha256=sha(Path(__file__)))
(H/'prior-speed-benchmark.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(result,indent=2))
