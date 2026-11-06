"""Fixed, fully reported exploratory comparisons; reuse existing reference fits."""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;H=HERE.parents[1]
sys.path.insert(0,str(H))
from representation_comparison import fit_one
from weighted_ridge_job import metrics
from prepare import DEV,SHORT,OLD,VARIANTS,require,load,save,sha,ROOT

def effects(y,a,b,people):
    _,ea=metrics(y,a,people);_,eb=metrics(y,b,people)
    delta=ea-eb
    choices=np.random.default_rng(20261026).integers(0,len(delta),size=(10000,len(delta)))
    low,high=np.percentile(delta[choices].mean(axis=1),[2.5,97.5])
    return {'person_mae_reduction':float(delta.mean()),'people_improved':int((delta>0).sum()),
            'people_worsened':int((delta<0).sum()),'people_equal':int((delta==0).sum()),
            'paired_bootstrap95':[float(low),float(high)],'contains_zero':bool(low<=0<=high),
            'independent_final_evaluation':False,'multiple_comparison_adjusted':False}

def main():
    require(not (HERE/'comparison.json').exists(),'Comparison already exists; no repeated fits')
    prep=load(HERE/'raw_preparation.json')
    require(prep['verified'] and prep['records']==243 and sha(HERE/'raw_features.npz')==prep['consolidated_sha256'],'Incomplete raw features')
    require(all(sha(ROOT/p)==digest for p,digest in load(HERE/'sources.json')['sha256'].items()),'Preparation source changed')
    require(all(sha(ROOT/p)==digest for p,digest in load(HERE/'raw_sources.json')['sha256'].items()),'Raw recovery source changed')
    baseline=H/'control/desktop-sessions/20261026T041807Z-f050df'
    paths=[Path(__file__),HERE/'prepare.py',HERE/'raw_prepare.py',HERE/'plan.md',HERE/'sources.json',HERE/'raw_sources.json',HERE/'raw_features.npz',
           DEV/'development_packet.npz',OLD/'length_models.npz',OLD/'length_comparison.json',OLD/'length_consistency.json',
           OLD/'length_internal_oof.csv',baseline/'internal_cv_protocol.json',H/'representation_comparison.py',H/'weighted_ridge_job.py']
    before={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    save(HERE/'analysis_sources.json',{'sha256':before,'label_scope':'existing development packet only after complete input gate'})
    with np.load(DEV/'development_packet.npz',allow_pickle=False) as dev,np.load(HERE/'raw_features.npz',allow_pickle=False) as f:
        for k in ('patient_ids','video_ids','split'): np.testing.assert_array_equal(dev[k],f[k])
        x={name:f[name].copy() for name in VARIANTS+('front_local',)}
        x['reference243']=dev['X'].copy()
        y=dev['Y'].copy();people=dev['patient_ids'].copy();videos=dev['video_ids'].copy();split=dev['split'].copy()
    train,val=split=='train',split=='validation'
    require((sum(train),sum(val),len(set(people[train])),len(set(people[val])))==(203,40,175,37),'Frozen split drift')
    require(not set(people[train])&set(people[val]),'Person leakage')
    for matrix in x.values(): require(matrix.shape==(243,512) and np.isfinite(matrix).all(),'Feature shape/value')
    models={};validation={};internal={};diagnostics={}
    with np.load(OLD/'length_models.npz',allow_pickle=False) as saved:
        for method,prefix in (('reference243','reference243'),('front_local','short124')):
            pars={k:saved[prefix+'__'+k].copy() for k in ('scaler_mean','scaler_scale','coef','intercept')}
            validation[method]=((x[method][val]-pars['scaler_mean'])/pars['scaler_scale'])@pars['coef'].T+pars['intercept']
    expected=load(OLD/'length_comparison.json')
    for method,key in (('reference243','reference243'),('front_local','short124')):
        require(abs(metrics(y[val],validation[method],people[val])[0]['person_mae']-expected[key]['person_mae'])<1e-9,'Reference replay mismatch')
    with (OLD/'length_internal_oof.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    require([r['video_id'] for r in rows]==[str(v) for v in videos[train]],'OOF row mismatch')
    for method,prefix in (('reference243','reference243'),('front_local','short124')):
        internal[method]=np.array([[float(r[prefix+'_L']),float(r[prefix+'_R'])] for r in rows])
    groups=load(baseline/'internal_cv_protocol.json')['fold_people']
    train_people=people[train];train_y=y[train]
    require(len(groups)==5 and all(len(g)==35 for g in groups) and set(p for g in groups for p in g)==set(train_people),'Internal fold drift')
    for method in VARIANTS:
        pred,pars,detail=fit_one(x[method],y,train,val)
        validation[method]=pred;diagnostics[method]=detail
        models.update({method+'__'+k:v for k,v in pars.items()})
        result=np.full((203,2),np.nan);visits=np.zeros(203,dtype=int)
        xt=x[method][train]
        for g in groups:
            held=np.isin(train_people,g);fit=~held
            require(not set(train_people[held])&set(train_people[fit]),'Internal overlap')
            p,_,_=fit_one(xt,train_y,fit,held)
            result[held]=p;visits[held]+=1
        require((visits==1).all() and np.isfinite(result).all(),'Incomplete OOF')
        internal[method]=result
        print(json.dumps({'completed_new_method':method,'new_fixed_fits':6}),flush=True)
    # The original reference and front-local fits are reused, including their five internal folds.
    methods=('reference243','front_global','front_local','middle_global','middle_local','last_global','last_local')
    val_metrics={m:metrics(y[val],validation[m],people[val])[0] for m in methods}
    internal_metrics={m:metrics(train_y,internal[m],train_people)[0] for m in methods}
    pairs={'length_with_global_coordinates':('reference243','front_global'),
           'front_normalization':('front_global','front_local'),
           'middle_normalization':('middle_global','middle_local'),
           'last_normalization':('last_global','last_local'),
           'front_vs_middle_global':('front_global','middle_global'),
           'front_vs_last_global':('front_global','last_global'),
           'front_vs_middle_local':('front_local','middle_local'),
           'front_vs_last_local':('front_local','last_local')}
    comparison={name:{'reference':a,'candidate':b,
                'validation37':effects(y[val],validation[a],validation[b],people[val]),
                'internal175':effects(train_y,internal[a],internal[b],train_people)} for name,(a,b) in pairs.items()}
    filename=HERE/'predictions.csv'
    with filename.open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['scope','method','patient_id','video_id','actual_L','actual_R','prediction_L','prediction_R'])
        for scope,mask,preds in (('validation37',val,validation),('internal175',train,internal)):
            for m in methods:
                for n,i in enumerate(np.flatnonzero(mask)):w.writerow([scope,m,people[i],videos[i],*y[i],*preds[m][n]])
    # Independent standard-library calculation on exported predictions, no extra training.
    grouped=defaultdict(lambda:defaultdict(list))
    with filename.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            error=statistics.mean(abs(float(r['actual_'+s])-float(r['prediction_'+s])) for s in ('L','R'))
            grouped[(r['scope'],r['method'])][r['patient_id']].append(error)
    for (scope,m),by_person in grouped.items():
        independent=statistics.mean(statistics.mean(v) for v in by_person.values())
        expected_mae=(val_metrics if scope=='validation37' else internal_metrics)[m]['person_mae']
        require(abs(independent-expected_mae)<1e-10,'Independent CSV metric mismatch')
    np.savez_compressed(HERE/'new_validation_models.npz',**models)
    with np.load(HERE/'new_validation_models.npz',allow_pickle=False) as z:
        for m in VARIANTS:
            replay=((x[m][val]-z[m+'__scaler_mean'])/z[m+'__scaler_scale'])@z[m+'__coef'].T+z[m+'__intercept']
            np.testing.assert_allclose(replay,validation[m],rtol=0,atol=1e-9)
    require(all(sha(ROOT/p)==digest for p,digest in before.items()),'Analysis source changed')
    result={'verified':True,'validation37':val_metrics,'internal175':internal_metrics,'comparisons':comparison,
            'new_methods':5,'new_validation_fits':5,'new_internal_fits':25,'reused_methods':2,
            'original_evaluation38_used':False,'new_people':0,'new_GDI_labels_opened':False,
            'same_frozen_encoder':True,'hyperparameter_search':False,'independent_final_evaluation':False,
            'independent_csv_check':True,'saved_model_replay_check':True,'independent_ridge_equation_check':True,
            'diagnostics':diagnostics,'original_sources_unchanged':True,
            'caveat':'Fixed243 frame domain; position windows overlap; all local windows normalized from raw coordinates after rejecting the chained-normalization route; no new participant sample or final causal proof.'}
    save(HERE/'comparison.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ('verified','validation37','internal175','new_validation_fits','new_internal_fits')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
