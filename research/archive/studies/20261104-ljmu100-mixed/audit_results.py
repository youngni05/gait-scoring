"""Independently audit completed folds, serialized models and numerical summaries."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def mae(y,p,ids):return float(np.mean([np.abs(y[ids==q]-p[ids==q]).mean() for q in np.unique(ids)]))
def close(a,b,tol=1e-8):np.testing.assert_allclose(a,b,atol=tol,rtol=0)
def coefficient(raw,y):return np.linalg.lstsq(np.column_stack([raw.reshape(-1).astype(float),np.ones(raw.size)]),y.reshape(-1).astype(float),rcond=None)[0]

ap=argparse.ArgumentParser();ap.add_argument('--family',required=True,choices=['CNN','STT','bilateral','RF']);ap.add_argument('--folds',default='0,1,2,3,4');a=ap.parse_args()
with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files}
assert sha(BIG/'data.npz')==read(HERE/'bundle/study-protocol.json')['data_sha256']
ids=d['person_id'];y=d['y'];fold=d['fold'];audits=[]
if a.family=='RF':
    sys.path.insert(0,str(HERE));import train_rf as rf
    x=rf.features(d);protocol=read(HERE/'rf-protocol.json')
    for arm in ['baseline','mixed']:
        for f in map(int,a.folds.split(',')):
            root=BIG/'runs'/arm/'RF'/f'fold{f}';done=read(root/'complete.json')
            assert done['protocol_sha256']==sha(HERE/'rf-protocol.json')
            for name,key in [('predictions.npz','prediction_sha256'),('inner.npz','inner_sha256'),('model.joblib','model_sha256')]:assert sha(root/name)==done[key]
            tr=(fold!=f)&((d['source']=='Stanford') if arm=='baseline' else True);te=fold==f
            assert done['train_people']==np.unique(ids[tr]).tolist() and done['evaluation_people']==np.unique(ids[te]).tolist()
            for b in done['inner_splits']:
                j=b['inner_fold'];assert b['train_people']==np.unique(ids[tr&(fold!=j)]).tolist()
                assert b['validation_people']==np.unique(ids[tr&(fold==j)]).tolist()
                assert not set(b['train_people'])&(set(b['validation_people'])|set(ids[te]))
            with np.load(root/'inner.npz') as z:
                assert np.array_equal(z['row_index'],np.flatnonzero(tr))
                scores=[mae(y[tr],p,ids[tr]) for p in z['predictions']]
            close(scores,done['candidate_person_mae']);chosen=min(i for i,v in enumerate(scores) if v<=min(scores)+1e-12)
            assert chosen==done['selected_candidate']
            model=rf.joblib.load(root/'model.joblib')
            for k,v in protocol['candidates'][chosen].items():assert model.get_params()[k]==v
            actual=model.predict(x[te].reshape(-1,48)).reshape(-1,2)
            with np.load(root/'predictions.npz') as z:
                assert np.array_equal(z['row_index'],np.flatnonzero(te));close(actual,z['refit'],1e-10)
            audits.append({'arm':arm,'fold':f,'passed':True,'selected_candidate':chosen})
else:
    sys.path[:0]=[str(HERE/'bundle'/a.family),str(HERE/'bundle'/a.family/'vendor')]
    spec=importlib.util.spec_from_file_location('original_training',HERE/'bundle'/a.family/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.setup(20261028,'cpu');c=read(HERE/'bundle'/a.family/'protocol.json')
    x=d['x'] if a.family=='CNN' else np.ascontiguousarray(d['x'][...,:8].reshape(-1,2,124,4,2))
    x=r.WindowSet(x,d['window_record'],len(y),d['window_start'],d['canonical_fallback'])
    for f in map(int,a.folds.split(',')):
        root=BIG/'received/data/runs/mixed'/a.family/f'fold{f}';done=read(root/'complete.json')
        assert done['bundle_sha256']==sha(HERE/'bundle/manifest.json')
        assert done['fits']==5 and sha(root/'predictions.npz')==done['prediction_sha256']
        histories=[];bag=[];maxerror=0.;epoch_raw_files=0
        for member in done['members']:
            tag=member['tag'];folder=root/tag;j=None if tag=='refit' else int(tag[5:])
            tr=(fold!=f) if j is None else ((fold!=f)&(fold!=j));va=np.zeros(len(y),bool) if j is None else fold==j;te=fold==f
            ck=r.torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
            ident=read(folder/'identity.json');assert ident==ck['identity']
            assert ident['train_people']==np.unique(ids[tr]).tolist();assert ident['validation_people']==np.unique(ids[va]).tolist();assert ident['outer_evaluation_people']==np.unique(ids[te]).tolist()
            assert not set(ids[tr])&(set(ids[va])|set(ids[te]))
            assert ident['bundle_sha256']==sha(HERE/'bundle/manifest.json')
            assert ident['fit_seed' if a.family=='CNN' else 'seed']==20261028+f*100+(50 if j is None else j)
            close(float(ck['lo']),float(np.float32(y[tr].min())),0)
            close(float(ck['span']),float(np.float32(max(float(y[tr].max()-y[tr].min()),1e-6))),0)
            fit=read(folder/'fit_complete.json');history=read(folder/'curve.json')
            assert history==ck['history'];assert len(history)==ident['epochs']==fit['epochs']
            assert sha(folder/'latest.pt')==fit['checkpoint_sha256']==member['checkpoint_sha256']
            assert sha(folder/'curve.json')==fit['curve_sha256']
            model=r.GdiCNN(c['dropout']) if a.family=='CNN' else r.new_model();model.load_state_dict(ck['model'],strict=True)
            with np.load(folder/'replay.npz') as z:
                assert np.array_equal(z['train_row'],np.flatnonzero(tr));assert np.array_equal(z['evaluation_row'],np.flatnonzero(te))
                train=z['train_raw'];outer=z['evaluation_raw'];coef=coefficient(train,y[tr])
                close(coef,history[-1]['calibration']);close(coef,member['calibration'])
                close(mae(y[tr],train*coef[0]+coef[1],ids[tr]),history[-1]['train_person_mae'])
                for mask,expected in [(tr,train),(te,outer)]:
                    actual=r.predict(model,x[mask],ck['lo'],ck['span'],'cpu');err=float(np.max(np.abs(actual-expected)));assert err<.001,err;maxerror=max(maxerror,err)
                if j is not None:
                    assert np.array_equal(z['validation_row'],np.flatnonzero(va))
                    close(mae(y[va],z['validation_raw']*coef[0]+coef[1],ids[va]),history[-1]['validation_person_mae'])
            collapsed=bool((train.std(axis=0)<.01).any());assert collapsed==member['excluded_from_bag'];close(train.std(axis=0),member['train_raw_sd'])
            pred=outer*coef[0]+coef[1]
            if not collapsed:bag.append(pred)
            if j is None:refit=pred;refit_raw=outer
            else:histories.append(history)
            if a.family!='CNN':
                for h in history:
                    path=folder/'raw_predictions'/('epoch%04d.npz'%h['epoch']);assert sha(path)==h['raw_predictions_sha256']
                    with np.load(path) as z:
                        cr=coefficient(z['train_raw'],y[tr]);close(cr,h['calibration'])
                        close(mae(y[tr],z['train_raw']*cr[0]+cr[1],ids[tr]),h['train_person_mae'])
                        if j is not None:close(mae(y[va],z['validation_raw']*cr[0]+cr[1],ids[va]),h['validation_person_mae'])
                    epoch_raw_files+=1
        selection=read(root/'selection.json')
        curves=np.array([[h['validation_person_mae'] for h in hist] for hist in histories])
        means=np.average(curves,axis=0,weights=[h[0]['validation_people'] for h in histories]);selected=int(np.argmin(means))+1
        close(means,selection['inner_mean_person_mae_by_epoch']);assert selected==selection['selected_epoch']==done['selected_epoch']
        assert len(read(root/'refit/curve.json'))==selected
        with np.load(root/'predictions.npz') as z:
            assert np.array_equal(z['row_index'],np.flatnonzero(te));close(z['refit'],refit);close(z['refit_raw'],refit_raw);close(z['bag'],np.mean(bag,axis=0))
        audits.append({'fold':f,'passed':True,'fits':5,'cpu_replay_max_error':maxerror,'epoch_raw_files_recomputed':epoch_raw_files,'selected_epoch':selected})
        print(json.dumps(audits[-1]),flush=True)
save(HERE/'audits'/f'{a.family}-{a.folds.replace(",","_")}.json',{'passed':True,'family':a.family,'folds':audits})
print('AUDIT_PASSED',a.family,flush=True)
