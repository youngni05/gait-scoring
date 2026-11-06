"""Independent baseline metric replay and CPU replay of the GPU pilot checkpoint."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/child100-train-only-20261105')
RUNS=BIG/'received/data/runs'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v): p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def mae(y,p,ids): return float(np.mean([np.abs(y[ids==q]-p[ids==q]).mean() for q in np.unique(ids)]))

ap=argparse.ArgumentParser();ap.add_argument('--family',required=True,choices=['CNN','STT','bilateral']);a=ap.parse_args()
with np.load(BIG/'data.npz') as z: d={k:z[k] for k in z.files}
if a.family:
    sys.path[:0]=[str(HERE/'bundle'/a.family),str(HERE/'bundle'/a.family/'vendor')]
    spec=importlib.util.spec_from_file_location('original_training',HERE/'bundle'/a.family/'run.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.setup(20261028,'cpu');c=read(HERE/'bundle'/a.family/'protocol.json')
    x=d['x'] if a.family=='CNN' else np.ascontiguousarray(d['x'][...,:8].reshape(-1,2,124,4,2))
    x=r.WindowSet(x,d['window_record'],len(d['y']),d['window_start'],d['canonical_fallback'])
    folder=RUNS/'pilot'/a.family
    ck=r.torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
    tr,va,te=r.split(d,0,1);child=d['source']=='LIVE-TD'
    assert tr[child].all() and not va[child].any() and not te[child].any()
    assert set(ck['identity']['train_people'])==set(d['person_id'][tr])
    assert set(ck['identity']['validation_people'])==set(d['person_id'][va])
    assert ck['identity']['epochs']==3
    assert float(ck['lo'])==float(np.float32(d['y'][tr].min()))
    assert float(ck['span'])==float(np.float32(d['y'][tr].max()-d['y'][tr].min()))
    model=r.GdiCNN(c['dropout']) if a.family=='CNN' else r.new_model()
    model.load_state_dict(ck['model'],strict=True)
    actual=r.predict(model,x[d['fold']==1],ck['lo'],ck['span'],'cpu')
    with np.load(folder/'pilot_predictions.npz') as z: expected=z['validation_raw']
    error=float(np.max(np.abs(actual-expected)));assert error<.001,error
    with np.load(folder/'pilot_predictions.npz') as z:train_raw=z['train_raw']
    train_replay=r.predict(model,x[tr],ck['lo'],ck['span'],'cpu')
    train_error=float(np.max(np.abs(train_replay-train_raw)));assert train_error<.001
    coefficient=np.linalg.lstsq(np.column_stack([train_raw.reshape(-1).astype(float),np.ones(train_raw.size)]),d['y'][tr].reshape(-1).astype(float),rcond=None)[0]
    assert np.max(np.abs(coefficient-np.array(ck['history'][-1]['calibration'])))<1e-8
    v=read(folder/'verification.json');assert v['passed'] and v['checkpoint_sha256']==hashlib.sha256((folder/'latest.pt').read_bytes()).hexdigest()
    v['independent_cpu_replay_max_error']=error
    v['independent_train_cpu_replay_max_error']=train_error
    v['child_training_only']=True
    v['training_only_scaling_and_calibration_verified']=True
    save(HERE/f'pilot-{a.family}-verified.json',v);print(json.dumps(v))
else:
    preds={}
    for family in ['CNN','STT','bilateral']:
        for kind in ['refit','bag']:
            p=np.full_like(d['y'],np.nan,dtype=float)
            for f in range(5):
                path=RUNS/'baseline'/family/f'fold{f}'/'predictions.npz'
                assert hashlib.sha256(path.read_bytes()).hexdigest()==read(path.parent/'complete.json')['prediction_sha256']
                with np.load(path) as z:
                    assert np.array_equal(z['row_index'],np.flatnonzero(d['fold']==f))
                    p[z['row_index']]=z[kind]
            assert np.isfinite(p).all();preds[family+'_'+kind]=p
    fusion=np.empty_like(d['y'],dtype=float)
    for v in read(HERE/'bundle/ensemble.json'):
        m=d['fold']==v['outer'];w=v['cnn_weight'];t=v['fraction']
        b=w*preds['CNN_bag'][m]+(1-w)*preds['STT_bag'][m]
        common=b.mean(axis=1);bil=preds['bilateral_refit'][m]
        diff=(1-t)*(b[:,0]-b[:,1])+t*(bil[:,0]-bil[:,1])
        fusion[m]=np.stack([common+diff/2,common-diff/2],axis=1)
    preds['fusion']=fusion
    expected={'CNN_refit':6.866655321854305,'STT_refit':6.944988193492919,'CNN_bag':6.590389070965508,'STT_bag':6.752031334574583,'bilateral_refit':7.003698215738529,'fusion':6.515816569178735}
    out={}
    for name,old in expected.items():
        out[name]={s:mae(d['y'][d['source']==s],preds[name][d['source']==s],d['person_id'][d['source']==s]) for s in ['Stanford','LJMU']}
        out[name]['old_Stanford_mae']=old;out[name]['reproduction_error']=abs(out[name]['Stanford']-old)
        assert out[name]['reproduction_error']<.001,(name,out[name])
    save(HERE/'baseline-verified.json',out);print(json.dumps(out,indent=2))
