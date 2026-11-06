"""Nested person-split RF comparison on the frozen multiwindow input."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'D:/Projects/D_P/outputs/gdi-ridge-experiment-20261016/deps')
import joblib,sklearn
from sklearn.ensemble import RandomForestRegressor

HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/child100-train-only-20261105')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mae(y,p,ids):return float(np.mean([np.abs(y[ids==q]-p[ids==q]).mean() for q in np.unique(ids)]))
def features(d):
    x=d['x'];w=np.concatenate([x.mean(axis=2),x.std(axis=2),np.ptp(x,axis=2),np.abs(np.diff(x,axis=2)).mean(axis=2)],axis=2)
    out=np.zeros((len(d['y']),2,48),dtype=np.float64)
    np.add.at(out,d['window_record'],w)
    out/=np.bincount(d['window_record'])[:,None,None]
    assert np.isfinite(out).all();return out

def main():
    start=time.monotonic()
    protocol=read(HERE/'rf-protocol.json');fingerprint=sha(HERE/'rf-protocol.json')
    assert sha(BIG/'data.npz')==protocol['data_sha256']
    with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files}
    x=features(d);y=d['y'];ids=d['person_id'];fold=d['fold']
    def model(candidate,seed):return RandomForestRegressor(n_estimators=128,**candidate,random_state=seed,n_jobs=2)
    def fit(m,ix):return m.fit(x[ix].reshape(-1,48),y[ix].reshape(-1))
    def predict(m,ix):return m.predict(x[ix].reshape(-1,48)).reshape(-1,2)
    summary=[]
    for arm in ['mixed']:
        for f in range(5):
            folder=BIG/'runs'/arm/'RF'/f'fold{f}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                done=read(folder/'complete.json');assert done['protocol_sha256']==fingerprint
                assert done['prediction_sha256']==sha(folder/'predictions.npz');summary.append(done);continue
            tr=(fold!=f)&((d['source']=='Stanford') if arm=='baseline' else True)
            te=fold==f
            # Children are never inner-validation targets, even though always trained on.
            scored=tr&(d['source']=='Stanford');train_rows=np.flatnonzero(scored)
            child=d['source']=='LIVE-TD'
            assert tr[child].all() and not te[child].any()
            assert not set(ids[tr])&set(ids[te])
            inner=np.full((4,scored.sum(),2),np.nan);boundaries=[]
            for j in range(5):
                if j==f:continue
                it=tr&(fold!=j);iv=tr&(fold==j)
                assert it[child].all() and not iv[child].any()
                assert not set(ids[it])&(set(ids[iv])|set(ids[te]))
                boundaries.append({'inner_fold':j,'train_people':np.unique(ids[it]).tolist(),'validation_people':np.unique(ids[iv]).tolist()})
                for ci,c in enumerate(protocol['candidates']):
                    m=fit(model(c,20261030+f*100+j),it)
                    inner[ci,np.searchsorted(train_rows,np.flatnonzero(iv))]=predict(m,iv)
            assert np.isfinite(inner).all()
            scores=[mae(y[scored],p,ids[scored]) for p in inner]
            chosen=min(i for i,v in enumerate(scores) if v<=min(scores)+1e-12)
            m=fit(model(protocol['candidates'][chosen],20261030+f*100+50),tr)
            pred=predict(m,te);joblib.dump(m,folder/'model.joblib',compress=3)
            replay=predict(joblib.load(folder/'model.joblib'),te)
            assert np.max(np.abs(pred-replay))<1e-10
            np.savez_compressed(folder/'inner.npz',predictions=inner,row_index=train_rows)
            np.savez_compressed(folder/'predictions.npz',refit=pred,row_index=np.flatnonzero(te))
            done={'arm':arm,'family':'RF','fold':f,'fits':17,'protocol_sha256':fingerprint,
                  'source_sha256':sha(Path(__file__)),'prediction_sha256':sha(folder/'predictions.npz'),
                  'inner_sha256':sha(folder/'inner.npz'),'model_sha256':sha(folder/'model.joblib'),
                  'train_people':np.unique(ids[tr]).tolist(),'evaluation_people':np.unique(ids[te]).tolist(),
                  'inner_splits':boundaries,'candidate_person_mae':scores,'selected_candidate':chosen,
                  'checkpoint_replay_max_error':float(np.max(np.abs(pred-replay))),
                  'source_mae':{'Stanford':mae(y[te],pred,ids[te])}}
            save(folder/'complete.json',done);summary.append(done)
            print(json.dumps({'arm':arm,'fold':f,'selected':chosen,'mae':done['source_mae'],'elapsed_seconds':time.monotonic()-start}),flush=True)
    save(HERE/'rf-run-summary.json',{'complete':True,'fits':85,'folds':summary,'sklearn':sklearn.__version__,'numpy':np.__version__,'elapsed_seconds':time.monotonic()-start})

if __name__=='__main__':main()
