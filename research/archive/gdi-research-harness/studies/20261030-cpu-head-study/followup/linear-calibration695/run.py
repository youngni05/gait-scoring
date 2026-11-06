"""Train-only affine calibration on existing fixed-model predictions."""
import sys,json,csv,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'mean-leg-followup695'
spec=importlib.util.spec_from_file_location('mean_study',PREV/'run.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
p,np,m=prior.p,prior.np,prior.m
CNN=p.GPU/'cnn-followup695'
CFG=m.load_json(HERE/'protocol.json')
METHODS=[f'{model}__{variant}' for model in CFG['models'] for variant in CFG['variants']]+['cnn__unchanged']

def js(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def table(name,rows):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sources_and_inputs():
    sources={}
    for folder in [PREV,p.PLS]:
        for name,digest in m.load_json(folder/'artifact_manifest.json').items():
            path=folder/name;assert m.digest(path)==digest,path
            sources[path.relative_to(m.WORKSPACE).as_posix()]=digest
    for path in [p.PARENT/'development695.npz',CNN/'analysis/verification.json',CNN/'bundle/run.py',CNN/'bundle/model.py',CNN/'bundle/protocol.json',CNN/'bundle/data.npz',HERE/'protocol.json',HERE/'run.py']:
        sources[path.relative_to(m.WORKSPACE).as_posix()]=m.digest(path)
    d=p.load(p.PARENT/'development695.npz');prev=p.load(PREV/'fold_predictions.npz')
    cube={model:prev['predictions'][list(prev['methods']).index(method)] for model,method in [('pls32','joint_pls32'),('temporal_combined','joint_combined_temporal')]}
    cnn=np.full_like(d['y'],np.nan,dtype=float)
    for f in range(5):
        folder=CNN/f'received/results/fold{f}';z=p.load(folder/'predictions.npz');done=m.load_json(folder/'complete.json')
        assert m.digest(folder/'predictions.npz')==done['prediction_sha256']
        idx=np.flatnonzero(d['fold']==f);np.testing.assert_array_equal(idx,z['row_index'])
        for key in ['person_id','video_id','fold','y']:np.testing.assert_array_equal(d[key][idx],z[key])
        cnn[idx]=z['predictions'][:,0]
        for path in [folder/'predictions.npz',folder/'complete.json',folder/'refit/identity.json',folder/'refit/fit_complete.json']:
            sources[path.relative_to(m.WORKSPACE).as_posix()]=m.digest(path)
    assert np.isfinite(cnn).all()
    inner={model:[] for model in CFG['models']}
    selections=m.load_json(PREV/'summary.json')['selections']
    for f in range(5):
        tr=np.flatnonzero(d['fold']!=f)
        z=p.load(PREV/f'models/fold{f}_inner.npz');np.testing.assert_array_equal(tr,z['train_indices'])
        inner['pls32'].append(z['base_predictions'])
        keys={q['method']:q['candidate'] for q in selections if q['fold']==f}
        mean=z['mean_combined_temporal_ridge__'+keys['mean_combined_temporal_ridge']]
        gap=z['gap_combined_temporal_ridge__'+keys['gap_combined_temporal_ridge']]
        inner['temporal_combined'].append(p.combine(mean,gap))
    return sources,d,cube,inner,cnn

def fit_affine(pred,y):
    x=pred.ravel();assert np.isfinite(x).all() and np.std(x)>1e-12
    return np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y.ravel(),rcond=None)[0]

def main():
    assert not (HERE/'summary.json').exists(),'preserve completed study'
    sources,d,cube,inner,cnn=sources_and_inputs();js('source_hashes.json',sources)
    folds=np.empty((6,5,1064,2));oof=np.empty((7,1064,2));coeffs=[];trainheld=[];calibration_inputs={}
    for model in CFG['models']:
        for f in range(5):
            tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f)
            assert len(set(d['person_id'][tr]))==556 and len(set(d['person_id'][te]))==139
            assert not set(d['person_id'][tr])&set(d['person_id'][te])
            for variant in CFG['variants']:
                name=model+'__'+variant;mi=METHODS.index(name)
                if variant=='unchanged':a,b=1.,0.
                else:
                    x=cube[model][f,tr] if variant=='train_affine' else inner[model][f]
                    assert x.shape==d['y'][tr].shape
                    a,b=fit_affine(x,d['y'][tr]);calibration_inputs[name+f'__fold{f}']=x
                    coeffs.append(dict(model=model,variant=variant,fold=f,slope=float(a),intercept=float(b),train_people=556,held_people=139,overlap=0))
                folds[mi,f]=a*cube[model][f]+b;oof[mi,te]=folds[mi,f,te]
                for split,idx in [('train',tr),('held',te)]:trainheld.append(dict(method=name,fold=f,split=split,**p.metrics(d['y'][idx],folds[mi,f,idx],d['person_id'][idx])))
    oof[-1]=cnn
    np.savez_compressed(HERE/'fold_predictions.npz',predictions=folds,methods=np.array(METHODS[:6]))
    np.savez_compressed(HERE/'oof_predictions.npz',predictions=oof,methods=np.array(METHODS),**{k:d[k] for k in ['person_id','video_id','fold','y','source_group']})
    np.savez_compressed(HERE/'calibration_inputs.npz',**calibration_inputs)
    rows=[];bands=[];comparisons=[];draws=np.random.default_rng(20261029).integers(0,695,(10000,695))
    for name,pred in zip(METHODS,oof):
        for group in ['all695','original175','additional520']:
            mask=np.ones(1064,dtype=bool) if group=='all695' else d['source_group']==group
            rows.append(dict(method=name,group=group,**{**p.metrics(d['y'][mask],pred[mask],d['person_id'][mask]),**p.parent.evaluate(d,pred,mask)}))
        for band in np.unique(np.floor(d['y']/10).astype(int)):bands.append(dict(method=name,actual_bin_low=int(band*10),**p.parent.evaluate(d,pred,band=band)))
        model,variant=name.split('__')
        if variant!='unchanged':
            base=oof[METHODS.index(model+'__unchanged')]
            delta=p.person_mean(abs(base-d['y']),d['person_id'])-p.person_mean(abs(pred-d['y']),d['person_id'])
            comparisons.append(dict(method=name,reduction=float(delta.mean()),bootstrap95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    for name,data in [('metrics.csv',rows),('score_bands.csv',bands),('coefficients.csv',coeffs),('train_held.csv',trainheld)]:table(name,data)
    for path,digest in sources.items():assert m.digest(m.WORKSPACE/path)==digest,path
    js('summary.json',dict(metrics=rows,bands=bands,coefficients=coeffs,train_held=trainheld,comparisons=comparisons,
        cnn_calibration_status='pending_train_predictions_not_available',GPU_used=False,independent_final_evaluation=False))
    print(json.dumps([dict(method=q['method'],MAE=q['overall_MAE'],sd=q['prediction_person_sd']) for q in rows if q['group']=='all695'],indent=2))

if __name__=='__main__':main()
