"""Person-equal paired comparison of verified, unchanged held-out predictions."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent.parent;O=H/'analysis'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
v=read(O/'verification.json');assert v['passed']
source=read(O/'source_audit.json');assert source['passed'] and len(source['fits'])==25
received=Path(read(O/'transfer_verification.json')['received'])
assert source['summary_sha256']==sha(received/'unpacked/results/summary.json')
assert source['run_sha256']==sha(H/'bundle/run.py') and source['pretrain_sha256']==sha(H/'bundle/pretrain.py')
for fit in source['fits']:
    folder=received/'unpacked/results'/('fold%d'%fit['fold'])/('refit' if fit['inner'] is None else 'inner%d'%fit['inner'])
    assert sha(folder/'latest.pt')==fit['checkpoint_sha256'] and sha(folder/'curve.json')==fit['curve_sha256']
z=arrays(received/'unpacked/results/oof_predictions.npz');y=z['y'];ids=z['person_id'];people=np.unique(ids)
models={'SSL_STT':z['predictions'][:,0],'SSL_STT_raw':z['predictions_raw'][:,0]};proofs={}
paths={'multiwindow_CNN':S/'cnn-multiwindow695/analysis/cnn_oof.npz',
       'multiwindow_STT':S/'stt692/multiwindow/analysis/unpacked/results/oof_predictions.npz'}
for name,path in paths.items():
    a=arrays(path);lookup={str(q):i for i,q in enumerate(a['video_id'])};assert len(lookup)==1061
    ix=np.array([lookup[str(q)] for q in z['video_id']])
    for k in ['y','person_id','fold']:np.testing.assert_array_equal(a[k][ix],z[k])
    models[name]=a['predictions'][ix,0];proofs[name]=dict(path=str(path),sha256=sha(path))
ra=read(S/'robust-aggregation692/analysis/comparison.json');path=Path(ra['predictions_path']);assert sha(path)==ra['predictions_sha256']
a=arrays(path)
for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(a[k],z[k])
for key in ['existing_CNN_STT','existing_mean_difference','inner_selected']:models[key]=a[key]
proofs['robust_combinations']=dict(path=str(path),sha256=sha(path))
def errors(p):return np.array([abs(y[ids==q]-p[ids==q]).mean() for q in people])
es={k:errors(p) for k,p in models.items()};high=np.array([y[ids==q].mean()>=90 for q in people])
metrics={}
for k,p in models.items():
    metrics[k]=dict(mae=float(es[k].mean()),high90_people=int(high.sum()),high90_mae=float(es[k][high].mean()),
       high90_bias=float(np.mean([(p[ids==q]-y[ids==q]).mean() for q in people[high]])),
       common_mean_mae=float(np.mean([abs(p[ids==q].mean(1)-y[ids==q].mean(1)).mean() for q in people])),
       left_right_difference_mae=float(np.mean([abs((p[ids==q,0]-p[ids==q,1])-(y[ids==q,0]-y[ids==q,1])).mean() for q in people])),
       exact_band=float(np.mean([(np.floor(p[ids==q]/10)==np.floor(y[ids==q]/10)).mean() for q in people])))
paired={};rng=np.random.default_rng(20261031)
for name in ['multiwindow_STT','multiwindow_CNN','inner_selected']:
    delta=es['SSL_STT']-es[name]
    boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(10000)])
    paired[name]=dict(ssl_minus_reference=float(delta.mean()),descriptive_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
       better_people=int((delta<0).sum()),worse_people=int((delta>0).sum()))
assert abs(metrics['SSL_STT']['mae']-6.960038975864489)<1e-10
diagnostics=read(O/'diagnostics.json')
overfit=[]
for f in range(5):
    dest=received/'unpacked/results'/('fold%d'%f);sel=read(dest/'selection.json')['selected_epoch']
    rows=[]
    for j in range(5):
        if j==f:continue
        h=read(dest/('inner%d'%j)/'curve.json');rows.append((h[sel-1],h[-1]))
    weights=[a['validation_people'] for a,b in rows]
    overfit.append(dict(fold=f,selected_epoch=sel,
        selected_validation=float(np.average([a['validation_person_mae'] for a,b in rows],weights=weights)),
        terminal_validation=float(np.average([b['validation_person_mae'] for a,b in rows],weights=weights)),
        selected_train=float(np.average([a['train_person_mae'] for a,b in rows],weights=weights)),
        terminal_train=float(np.average([b['train_person_mae'] for a,b in rows],weights=weights))))
report=dict(people=692,records=1061,metrics=metrics,paired=paired,overfit=overfit,
    final_slopes=[r['slope'] for r in diagnostics['replay']],final_intercepts=[r['intercept'] for r in diagnostics['replay']],
    sources=proofs,verification_sha256=sha(O/'verification.json'),source_audit_sha256=sha(O/'source_audit.json'),
    protected_final_opened=False,limitations=['Repeated development exploration; bootstrap is descriptive.',
    'One training seed; no universal claim about self-supervision.', 'No new ensemble weights selected from outer predictions.'])
save(O/'comparison.json',report);print(json.dumps(report),flush=True)
