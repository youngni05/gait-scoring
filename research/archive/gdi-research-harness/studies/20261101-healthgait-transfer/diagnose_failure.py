"""Read-only model/data diagnostics; never changes running jobs or original weights."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BUNDLE=Path('D:/gait-research-data/D_P/healthgait-transfer-20261101/bundle')
OUT=HERE/'diagnostics'


def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,o):
    OUT.mkdir(exist_ok=True)
    (OUT/n).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n',encoding='utf8')
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def person_values(ids,y,pred):
    people=np.unique(ids)
    return people,np.array([y[ids==p].mean() for p in people]),np.array([pred[ids==p].mean() for p in people])
def metrics(ids,y,pred):
    _,yy,pp=person_values(ids,y,pred);err=pp-yy
    return {'MAE':float(np.abs(err).mean()),'RMSE':float(np.sqrt((err**2).mean())),
            'low5_8_MAE':float(np.abs(err[yy<=8]).mean()),'high9_12_MAE':float(np.abs(err[yy>=9]).mean()),
            'prediction_std':float(pp.std()),'prediction_min':float(pp.min()),'prediction_max':float(pp.max()),
            'correlation':float(np.corrcoef(yy,pp)[0,1]) if pp.std()>1e-10 else None}


def distribution():
    import torch
    from model import TransferCNN
    d=arrays(BUNDLE/'target.npz');y=d['y'][:,0];ids=d['ids'];people,yy,_=person_values(ids,y,y)
    folds=[]
    for f in range(5):
        tr=d['folds']!=f;te=~tr;unique=np.unique(ids[tr]);train_y=np.array([y[ids==p].mean() for p in unique])
        folds.append({'fold':f,'train_people':len(unique),'test_people':len(set(ids[te])),
                      'train_median':float(np.median(train_y)),'train_mean':float(train_y.mean()),
                      'test_scores':dict(collections.Counter(map(str,y[np.array([np.flatnonzero(ids==p)[0] for p in np.unique(ids[te])])])) )})
    result={'people':len(people),'trials':len(y),'score_counts':{str(int(k)):int(v) for k,v in sorted(collections.Counter(yy).items())},
            'mean':float(yy.mean()),'median':float(np.median(yy)),'score_std':float(yy.std()),'score9_12_people':int((yy>=9).sum()),
            'constant10_metrics':metrics(ids,y,np.full_like(y,10.)),'folds':folds,
            'trainable_parameters':sum(p.numel() for p in TransferCNN().parameters()),
            'input_range':[float(d['x'].min()),float(d['x'].max())],
            'all_input_finite':bool(np.isfinite(d['x']).all()),'inner_train_people_range':[23,24],
            'source_condition_records':load(HERE/'data-readiness.json')['source']['conditions']}
    save('distribution.json',result);print(json.dumps(result),flush=True)


def features(x):return np.concatenate([x.mean(-1),x.std(-1),np.abs(np.diff(x,axis=-1)).mean(-1)],axis=1)


def baselines():
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from threadpoolctl import threadpool_limits
    d=arrays(BUNDLE/'target.npz');old=arrays(HERE.parent/'20261030-older44-sppb-timeseries/input.npz')
    for k in ('ids','folds'):np.testing.assert_array_equal(d[k],old[k])
    np.testing.assert_array_equal(d['y'][:,0],old['y'])
    ids=d['ids'];y=d['y'][:,0];folds=d['folds'];speed=old['x'][:,37,:].mean(-1)[:,None]
    common=features(d['x'].reshape(len(ids),12,124))
    variants={'common_2d':common,'common_2d_plus_speed':np.column_stack([common,speed]),
              'speed_only':speed,'original_3d_without_velocity':features(old['x'][:,:36]),'original_3d_plus_velocity':features(old['x'])}
    result={};predictions={};choices=[]
    def fit(name,c,x,tr,te):
        weights=np.array([1./np.sum(ids[tr]==p) for p in ids[tr]])
        scaler=StandardScaler().fit(x[tr],sample_weight=weights)
        model=Ridge(alpha=c) if name=='Ridge' else RandomForestRegressor(n_estimators=128,min_samples_leaf=c,random_state=20261030,n_jobs=2,max_features=1.)
        model.fit(scaler.transform(x[tr]),y[tr],sample_weight=weights/weights.mean())
        return model.predict(scaler.transform(x[te]))
    with threadpool_limits(limits=2):
        for variant,x in variants.items():
            result[variant]={}
            for name,grid in [('Ridge',[1.,100.]),('RF',[2,5])]:
                pp=np.full(len(y),np.nan)
                for f in range(5):
                    tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f);people=np.unique(ids[tr]);splits=[]
                    for a,b in KFold(3,shuffle=True,random_state=20261031).split(people):
                        it=tr[np.isin(ids[tr],people[a])];iv=tr[np.isin(ids[tr],people[b])]
                        assert not(set(ids[it])&set(ids[iv]) or set(ids[it])&set(ids[te]) or set(ids[iv])&set(ids[te]))
                        splits.append((it,iv))
                    scores=[]
                    for c in grid:
                        val=np.full(len(y),np.nan)
                        for it,iv in splits:val[iv]=fit(name,c,x,it,iv)
                        scores.append(metrics(ids[tr],y[tr],val[tr])['MAE'])
                    selected=grid[int(np.argmin(scores))];pp[te]=fit(name,selected,x,tr,te)
                    choices.append({'variant':variant,'model':name,'fold':f,'chosen':selected,'inner_MAE':scores})
                result[variant][name]=metrics(ids,y,pp);predictions[variant+'_'+name]=pp
            print(json.dumps({'variant':variant,'metrics':result[variant]}),flush=True)
    speed_person=np.array([speed[ids==p].mean() for p in np.unique(ids)])
    yy=np.array([y[ids==p].mean() for p in np.unique(ids)])
    result['scope']='Same44people,220trials and outer5folds; each candidate selected in training-person3fold only; diagnostic development comparisons, not independent validation.'
    result['person_speed_range_m_s']=[float(speed_person.min()),float(speed_person.max())]
    result['person_speed_SPPB_correlation']=float(np.corrcoef(speed_person,yy)[0,1])
    save('baseline-comparisons.json',result);save('baseline-selections.json',choices)
    with (OUT/'baseline-person-predictions.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['id','score',*predictions])
        for p in np.unique(ids):w.writerow([p,float(y[ids==p].mean()),*[float(v[ids==p].mean()) for v in predictions.values()]])


def batchnorm():
    import torch
    from model import TransferCNN
    torch.set_num_threads(2)
    d=arrays(BUNDLE/'target.npz');root=Path(load(HERE/'analysis/round0/receipt.json')['originals'])
    assert hashlib.sha256((HERE/'model.py').read_bytes()).hexdigest()==hashlib.sha256((root/'model.py').read_bytes()).hexdigest()
    rows=[];summary={}
    def predict(model,people):
        mask=np.isin(d['ids'],list(people));wi=np.flatnonzero(mask[d['owner']]);pieces=[]
        with torch.no_grad():
            for a in range(0,len(wi),128):pieces.append(model(torch.from_numpy(d['x'][wi[a:a+128]])).numpy()[:,0])
        wp=np.concatenate(pieces)*float(ck['sd'][0])+float(ck['mu'][0]);owners=d['owner'][wi]
        pp={r:wp[owners==r].astype(np.float64).mean() for r in np.flatnonzero(mask)}
        errors=[];preds=[]
        for person in sorted(people):
            indices=np.flatnonzero(d['ids']==person);pred=float(np.mean([pp[r] for r in indices]));truth=float(d['y'][indices].mean())
            errors.append(abs(pred-truth));preds.append(pred)
        return float(np.mean(errors)),np.array(preds)
    for arm in ('scratch','gdi','health'):
        for path in sorted((root/'runs/round0').glob('fold*/'+arm+'-s*/candidate0/inner*/model.pt')):
            identity=load(path.parent/'identity.json');complete=load(path.parent/'fit_complete.json');tr=set(identity['train_people']);va=set(identity['validation_people'])
            assert not(tr&va or tr&set(identity['test_people']) or va&set(identity['test_people']))
            ck=torch.load(path,map_location='cpu',weights_only=True);m=TransferCNN(1,.5);m.load_state_dict(ck['model']);m.eval()
            before,p0=predict(m,va);train_before,_=predict(m,tr);assert abs(before-complete['best_validation_MAE'])<1e-4
            params={k:v.detach().clone() for k,v in m.named_parameters()}
            wi=np.flatnonzero(np.isin(d['ids'][d['owner']],list(tr)))
            # One whole-training-batch pass, no labels, no dropout, no optimizer.
            # Only BN running mean/variance/counters are replaced in memory.
            for bn in m.norms:bn.reset_running_stats();bn.momentum=1.;bn.train()
            with torch.no_grad():m(torch.from_numpy(d['x'][wi]))
            m.eval();assert all(torch.equal(params[k],v) for k,v in m.named_parameters())
            after,p1=predict(m,va);train_after,_=predict(m,tr)
            epoch=complete['best_epoch'];steps=math.ceil(len(wi)/32)*epoch
            rows.append({'arm':arm,'fit':str(path.parent.relative_to(root)),'valid_people':len(va),'train_people':len(tr),
                         'selected_epoch':epoch,'estimated_BN_initial_stat_fraction':.99**steps,
                         'validation_MAE_original':before,'validation_MAE_recalibrated':after,
                         'training_MAE_original':train_before,'training_MAE_recalibrated':train_after,
                         'prediction_shift_mean':float(np.abs(p1-p0).mean()),'weights_unchanged':True})
        selected=[r for r in rows if r['arm']==arm];weights=[r['valid_people'] for r in selected]
        summary[arm]={'inner_models':len(selected),
                      'validation_MAE_original':float(np.average([r['validation_MAE_original'] for r in selected],weights=weights)),
                      'validation_MAE_recalibrated':float(np.average([r['validation_MAE_recalibrated'] for r in selected],weights=weights)),
                      'training_MAE_original_median':float(np.median([r['training_MAE_original'] for r in selected])),
                      'training_MAE_recalibrated_median':float(np.median([r['training_MAE_recalibrated'] for r in selected])),
                      'mean_prediction_shift':float(np.average([r['prediction_shift_mean'] for r in selected],weights=weights)),
                      'models_with_lower_validation_MAE':sum(r['validation_MAE_recalibrated']<r['validation_MAE_original'] for r in selected)}
        print(json.dumps({'arm':arm,'batchnorm_diagnostic':summary[arm]}),flush=True)
    summary['scope']='135 round0 inner checkpoints; no outer-test labels evaluated. Training-only BN buffer recalibration, all learned weights unchanged, no optimizer. Repeated selected inner validation comparisons are mechanistic diagnostics, not a newly validated final model.'
    save('batchnorm-recalibration.json',summary);save('batchnorm-per-fit.json',rows)


def batchnorm_outer():
    import torch
    from model import TransferCNN
    torch.set_num_threads(2)
    d=arrays(BUNDLE/'target.npz');root=Path(load(HERE/'analysis/round1/receipt.json')['originals'])
    rows=[]
    for arm in ('scratch','gdi','health'):
        for path in sorted((root/'runs/round1').glob('fold*/'+arm+'-s*/refit/model.pt')):
            identity=load(path.parent/'identity.json');tr=set(identity['train_people']);test=set(identity['test_people'])
            assert not(tr&test) and not identity['validation_people']
            ck=torch.load(path,map_location='cpu',weights_only=True)
            m=TransferCNN(1,identity['config']['dropout']);m.load_state_dict(ck['model']);m.eval()
            wi=np.flatnonzero(np.isin(d['ids'][d['owner']],list(test)))
            def predict():
                parts=[]
                with torch.no_grad():
                    for a in range(0,len(wi),128):parts.append(m(torch.from_numpy(d['x'][wi[a:a+128]])).numpy()[:,0])
                wp=np.concatenate(parts)*float(ck['sd'][0])+float(ck['mu'][0]);owner=d['owner'][wi]
                return {p:float(np.mean([wp[owner==r].mean() for r in np.flatnonzero(d['ids']==p)])) for p in test}
            before=predict();params={k:v.detach().clone() for k,v in m.named_parameters()}
            tw=np.flatnonzero(np.isin(d['ids'][d['owner']],list(tr)))
            for bn in m.norms:bn.reset_running_stats();bn.momentum=1.;bn.train()
            with torch.no_grad():m(torch.from_numpy(d['x'][tw]))
            m.eval();assert all(torch.equal(params[k],v) for k,v in m.named_parameters())
            after=predict()
            for p in sorted(test):rows.append({'arm':arm,'fit':str(path.parent.relative_to(root)),'id':p,'y':float(d['y'][d['ids']==p].mean()),'original':before[p],'recalibrated':after[p]})
    result={}
    for arm in ('scratch','gdi','health'):
        subset=[r for r in rows if r['arm']==arm];people=sorted({r['id'] for r in subset});assert len(people)==44
        yy=np.array([next(r['y'] for r in subset if r['id']==p) for p in people]);orig=np.array([np.mean([r['original'] for r in subset if r['id']==p]) for p in people]);post=np.array([np.mean([r['recalibrated'] for r in subset if r['id']==p]) for p in people])
        baseline=load(HERE/'analysis/round1/verification.json')['metrics'][arm]['ensemble_MAE'];assert abs(np.abs(orig-yy).mean()-baseline)<1e-4
        result[arm]={'original':metrics(np.array(people),yy,orig),'recalibrated':metrics(np.array(people),yy,post),'all_parameter_tensors_unchanged':True,'final_models':15}
    result['scope']='Post-hoc mechanistic diagnostic on the already inspected development folds. Each BN recalibration uses its outer training people only, no labels or optimizer, no selected hyperparameters. Does not replace original results or establish independent clinical/video performance.'
    save('batchnorm-outer-diagnostic.json',result)
    with (OUT/'batchnorm-outer-person-predictions.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['distribution','baselines','batchnorm','batchnorm_outer']);args=ap.parse_args()
    {'distribution':distribution,'baselines':baselines,'batchnorm':batchnorm,'batchnorm_outer':batchnorm_outer}[args.mode]()
