"""Reconstruct selection and outputs from saved RF artifacts, independent of fit execution."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/child100-train-only-20261105')
sys.path.insert(0,'D:/Projects/D_P/outputs/gdi-ridge-experiment-20261016/deps')
import joblib
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files}
x=d['x'];w=np.concatenate([np.mean(x,axis=2),np.std(x,axis=2),np.max(x,axis=2)-np.min(x,axis=2),np.mean(np.abs(x[:,:,1:]-x[:,:,:-1]),axis=2)],axis=2)
features=np.stack([w[d['window_record']==i].mean(axis=0,dtype=np.float64) for i in range(len(d['y']))])
ids=d['person_id'];fold=d['fold'];child=d['source']=='LIVE-TD';report=[]
for f in range(5):
    folder=BIG/'runs/mixed/RF'/('fold%d'%f);done=read(folder/'complete.json')
    assert done['protocol_sha256']==sha(HERE/'rf-protocol.json')
    for file,key in [('model.joblib','model_sha256'),('predictions.npz','prediction_sha256'),('inner.npz','inner_sha256')]:assert sha(folder/file)==done[key]
    tr=fold!=f;te=fold==f;scored=tr&~child
    assert set(done['train_people'])==set(ids[tr]) and set(done['evaluation_people'])==set(ids[te])
    for b in done['inner_splits']:
        j=b['inner_fold'];it=tr&(fold!=j);iv=fold==j
        assert set(b['train_people'])==set(ids[it]) and set(b['validation_people'])==set(ids[iv])
        assert it[child].all() and not iv[child].any()
    with np.load(folder/'inner.npz') as z:
        assert np.array_equal(z['row_index'],np.flatnonzero(scored))
        errors=np.abs(z['predictions']-d['y'][scored][None])
        scores=np.array([np.mean([e[ids[scored]==p].mean() for p in np.unique(ids[scored])]) for e in errors])
    assert np.max(np.abs(scores-done['candidate_person_mae']))<1e-12
    chosen=np.flatnonzero(scores<=scores.min()+1e-12)[0];assert chosen==done['selected_candidate']
    model=joblib.load(folder/'model.joblib')
    expected=read(HERE/'rf-protocol.json')['candidates'][int(chosen)]
    assert all(model.get_params()[k]==v for k,v in expected.items())
    assert model.random_state==20261030+100*f+50 and model.n_estimators==128
    replay=model.predict(features[te].reshape(-1,48)).reshape(-1,2)
    with np.load(folder/'predictions.npz') as z:
        assert np.array_equal(z['row_index'],np.flatnonzero(te))
        error=float(np.max(np.abs(replay-z['refit'])))
    assert error<1e-10
    report.append(dict(fold=f,child_training_only=True,selection_reconstructed=True,replay_max_error=error))
out=dict(passed=True,folds=report,protocol_sha256=sha(HERE/'rf-protocol.json'),
    protocol_text_corrections={
        'training':'The initial RF protocol accidentally inherited the neural OLS description. Executed RF code has no scaling or OLS; predictions replay directly from the saved forest.',
        'metric':'The initial protocol retains an LJMU source sentence. This experiment evaluates only original Stanford outer folds; no LJMU or child validation rows exist.',
        'provenance':'Initial protocol bytes are retained for the completed-fit hash chain. These are explanatory text corrections; estimator settings, predictions and model selection are unchanged.'})
(HERE/'rf-audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
