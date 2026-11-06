"""Evaluate every declared arm; complete primary cohorts only, no failed-row selection."""
import csv
import json
from pathlib import Path
import sys
import time
from study import HERE, TUG, TRAITS, load, bins, sha, save, payload
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def model(family):
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=10.) if family=='Ridge' else
                         RandomForestRegressor(n_estimators=128,min_samples_leaf=5,random_state=20261031,n_jobs=1))


def oof(X,y,fold,family):
    pred=np.full(len(y),np.nan)
    for f in range(5):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        assert not set(tr)&set(te)
        pred[te]=model(family).fit(X[tr],y[tr]).predict(X[te])
    assert np.isfinite(pred).all()
    return pred


def metrics(y,p):
    e=p-y
    return {'n':len(y),'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e**2))),
            'bias':float(e.mean()),'predicted_range':[float(p.min()),float(p.max())],
            'underprediction':float(np.maximum(-e,0).mean()),'overprediction':float(np.maximum(e,0).mean())}


def paired(y,p,baseline):
    delta=np.abs(p-y)-np.abs(baseline-y)
    rng=np.random.default_rng(20261031)
    samples=rng.integers(0,len(y),size=(10000,len(y)))
    return {'MAE_difference_new_minus_baseline':float(delta.mean()),
            'conditional_bootstrap95':np.quantile(delta[samples].mean(1),[.025,.975]).tolist(),
            'people_improved_equal_worse':[int((delta< -1e-10).sum()),int((np.abs(delta)<=1e-10).sum()),int((delta>1e-10).sum())]}


def trait_vector(answer):
    return [answer[t]['probabilities'][c] for t in TRAITS for c in ['low','middle','high','unknown']]


def direct(dataset,answers,fallback):
    _,mids,_=bins(dataset);choice=answers['target_band']['choice']
    return (fallback,False) if choice=='unknown' else (float(mids[int(choice[1:])]),True)


def evaluate():
    started=time.monotonic();data=load()
    rows=[json.loads(line) for line in (HERE/'responses.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['job'] for r in rows})
    lookup={(r['dataset'],r['index'],r['arm']):r for r in rows if r['status']=='ok'}
    results={}; predictions={}; controls={}; skipped=[]
    for name,d in data.items():
        n=len(d['y']);X=np.asarray(d['X'],float);y=np.asarray(d['y'],float);fold=np.asarray(d['fold'])
        if any((name,i,a) not in lookup for i in range(n) for a in ['numeric','narrative']):
            skipped.append(name);continue
        median=np.array([np.median(y[fold!=f]) for f in fold])
        pred={'Median':median,'RF_numeric':oof(X,y,fold,'RF'),'Ridge_numeric':oof(X,y,fold,'Ridge')}
        coverage={};direct_metrics={}
        for arm in ['numeric','narrative']:
            answers=[lookup[name,i,arm]['response']['answers'] for i in range(n)]
            J=np.array([trait_vector(a) for a in answers])
            for family in ['RF','Ridge']:
                pred[f'{family}_Jev_{arm}_only']=oof(J,y,fold,family)
                pred[f'{family}_numeric_plus_Jev_{arm}']=oof(np.column_stack([X,J]),y,fold,family)
            proxy=np.array([direct(name,a,b)[0] for a,b in zip(answers,median)])
            accepted=np.array([direct(name,a,b)[1] for a,b in zip(answers,median)])
            pred[f'Jev_{arm}_band_midpoint_or_train_median']=proxy
            coverage[arm]={'accepted':int(accepted.sum()),'total':n,'fraction':float(accepted.mean())}
            edges,_,_=bins(name);truth=np.clip(np.searchsorted(edges,y,side='right')-1,0,len(edges)-2)
            direct_metrics[arm]={'coverage':coverage[arm],
                'accepted_only':metrics(y[accepted],proxy[accepted]) if accepted.any() else None,
                'accepted_bin_accuracy':float(np.mean([a['target_band']['choice']==f'b{k}' for a,k,ok in zip(answers,truth,accepted) if ok])) if accepted.any() else None,
                'trait_unknown_rate':{t:float(np.mean([a[t]['choice']=='unknown' for a in answers])) for t in TRAITS}}
        if name=='GDI':pred['existing_PLS_person_mean']=np.asarray(d['existing_pls'])
        else:
            old={}
            with (TUG/'predictions.csv').open() as f:
                for row in csv.DictReader(f):
                    if row['repeat']=='0' and row['model']=='RF' and row['seed']=='20261030' and row['view']=='pattern':
                        i=int(row['sample']);assert abs(float(row['target'])-y[i])<1e-4
                        assert int(row['fold'])==fold[i];old[i]=float(row['prediction'])
            assert len(old)==n
            pred['existing_tuned_RF_pattern']=np.array([old[i] for i in range(n)])
        groups={'all':np.ones(n,bool)}
        if name=='TUG':groups.update(TUG_ge20=y>=20,TUG_lt10=y<10)
        else:groups.update(GDI_lt60=y<60,GDI_ge90=y>=90)
        summary={'metrics':{k:metrics(y,p) for k,p in pred.items()},'direct':direct_metrics,
                 'groups':{g:{k:metrics(y[mask],p[mask]) for k,p in pred.items()} for g,mask in groups.items() if mask.any()},
                 'paired':{},'fold_MAE':{k:[float(np.abs(p[fold==f]-y[fold==f]).mean()) for f in range(5)] for k,p in pred.items()}}
        for family in ['RF','Ridge']:
            for arm in ['numeric','narrative']:
                key=f'{family}_numeric_plus_Jev_{arm}'
                summary['paired'][key]=paired(y,pred[key],pred[f'{family}_numeric'])
        results[name]=summary
        predictions[name]={'ids':d['ids'],'target':y.tolist(),'fold':fold.tolist(),'predictions':{k:v.tolist() for k,v in pred.items()}}
        variants={}
        for arm in ['sequence','examples','repeat','reverse','old_model','missing']:
            indices=[i for i in range(n) if (name,i,arm) in lookup]
            if not indices:continue
            answers=[lookup[name,i,arm]['response']['answers'] for i in indices]
            base=[lookup[name,i,'numeric']['response']['answers'] for i in indices]
            values=np.array([direct(name,a,median[i])[0] for i,a in zip(indices,answers)])
            accepted=np.array([direct(name,a,median[i])[1] for i,a in zip(indices,answers)])
            variants[arm]={'n':len(indices),'accepted':int(accepted.sum()),'indices':indices,
                'band_midpoint_or_median':metrics(y[indices],values),'numeric_same_people':metrics(y[indices],pred['Jev_numeric_band_midpoint_or_train_median'][indices]),
                'RF_numeric_same_people':metrics(y[indices],pred['RF_numeric'][indices]),
                'question_choice_agreement':{t:float(np.mean([a[t]['choice']==b[t]['choice'] for a,b in zip(answers,base)])) for t in [*TRAITS,'target_band']},
                'mean_absolute_trait_probability_change':float(np.mean(np.abs(np.array([trait_vector(a) for a in answers])-np.array([trait_vector(a) for a in base])))),
                'unknown_choices':{t:sum(a[t]['choice']=='unknown' for a in answers) for t in [*TRAITS,'target_band']}}
            if arm=='examples':
                # Diagnostic control: exactly the six already-disclosed examples,
                # without Jev. No neighbor count or weighting is tuned here.
                near_median=np.array([np.median([r['target'] for r in payload(data,name,i,'examples')['state']['training_examples']]) for i in indices])
                variants[arm]['same_six_examples_median']=metrics(y[indices],near_median)
                variants[arm]['comparison_scope']='Post-request diagnostic control, not a prespecified primary model.'
        controls[name]=variants
        print(name,'RF',summary['metrics']['RF_numeric']['MAE'],'RF+Jev',summary['metrics']['RF_numeric_plus_Jev_numeric']['MAE'],flush=True)
    save('results.json',results);save('predictions.json',predictions);save('controls.json',controls)
    receipt=json.loads((HERE/'preparation.json').read_text())
    unchanged={p:sha(p)==h for p,h in receipt['source_hashes'].items()}
    assert all(unchanged.values())
    save('evaluation.json',{'complete_primary_datasets':list(results),'skipped_incomplete':skipped,
                            'source_files_unchanged':unchanged,'elapsed_seconds':time.monotonic()-started,
                            'responses_sha256':sha(HERE/'responses.jsonl'),'results_sha256':sha(HERE/'results.json'),
                            'code_sha256':{f:sha(HERE/f) for f in ['study.py','jev_client.py','evaluate.py']},
                            'data_scope':'TUG195 + GDI695 reused development people; no independent validation; no method selected for deployment'})


if __name__=='__main__':
    with threadpool_limits(limits=2):evaluate()
