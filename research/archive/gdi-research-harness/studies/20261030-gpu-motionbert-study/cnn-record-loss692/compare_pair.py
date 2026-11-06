"""Matched record/window loss pair and historical references; no model selection."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
v=read(H/'analysis/record_mean/verification.json');assert v['passed']
z=arrays(Path('D:/gait-research-data/D_P')/H.relative_to(H.parents[3])/'review-record-mean-20261031/unpacked/results/oof_predictions.npz')
models={'record_mean':z['predictions'][:,0],'record_mean_raw':z['predictions_raw'][:,0]};proofs={}
sources={'window_control':H/'analysis/window_control/unpacked/results/oof_predictions.npz','multiwindow_CNN':S/'cnn-multiwindow695/analysis/cnn_oof.npz','multiwindow_STT':S/'stt692/multiwindow/analysis/unpacked/results/oof_predictions.npz'}
for k,p in sources.items():
    a=arrays(p);lookup={str(x):i for i,x in enumerate(a['video_id'])};ix=[lookup[str(x)] for x in z['video_id']]
    for field in ['y','person_id','fold']:np.testing.assert_array_equal(a[field][ix],z[field])
    models[k]=a['predictions'][ix,0];proofs[k]=dict(path=str(p),sha256=sha(p))
report=read(S/'robust-aggregation692/analysis/comparison.json');p=Path(report['predictions_path']);assert sha(p)==report['predictions_sha256'];a=arrays(p)
for field in ['y','person_id','fold','video_id']:np.testing.assert_array_equal(a[field],z[field])
for name in ['existing_CNN_STT','existing_mean_difference','inner_selected']:models[name]=a[name]
proofs['combinations']=dict(path=str(p),sha256=sha(p));ids=z['person_id'];y=z['y'];people=np.unique(ids)
errors={name:np.array([abs(pred[ids==i]-y[ids==i]).mean() for i in people]) for name,pred in models.items()}
high=np.array([y[ids==i].mean()>=90 for i in people]);metrics={}
for name,pred in models.items():
    metrics[name]=dict(mae=float(errors[name].mean()),high90_mae=float(errors[name][high].mean()),high90_people=int(high.sum()),
        exact_band=float(np.mean([(np.floor(pred[ids==i]/10)==np.floor(y[ids==i]/10)).mean() for i in people])))
paired={};rng=np.random.default_rng(20261031)
for name in ['window_control','multiwindow_CNN','inner_selected']:
    delta=errors['record_mean']-errors[name];boot=[delta[rng.integers(0,692,692)].mean() for _ in range(10000)]
    paired[name]=dict(delta=float(delta.mean()),descriptive_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
case=dict(record_row=93,actual=float(y[93,0]),predictions={k:float(p[93,0]) for k,p in models.items()})
result=dict(passed=True,people=692,records=1061,metrics=metrics,paired=paired,previous_explosion_case=case,sources=proofs,
    verification_sha256=sha(H/'analysis/record_mean/verification.json'),protected_final_opened=False)
save(H/'analysis/pair_comparison.json',result);print(json.dumps(result),flush=True)
