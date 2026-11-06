from study import *
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import collections

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());counts={}
    for name,h in protocol['source_hashes'].items():assert sha(HERE.parent/name)==h
    with threadpool_limits(limits=1):
        for cohort,info in protocol['cohorts'].items():
            out=HERE/cohort;d=dict(np.load(out/'input.npz'));p=pd.read_csv(out/'predictions.csv');plans=json.loads((out/'splits.json').read_text())
            lookup={(r['repeat'],f['fold']):f for r in plans for f in r['splits']}
            selections=json.loads((out/'selections.json').read_text());inner=json.loads((out/'inner.json').read_text());inner_map=collections.defaultdict(list)
            for q in inner:
                fold=lookup[q['repeat'],q['fold']];assert q['train_order']==fold['train']
                err=np.mean(abs(np.array(q['predictions'])-d['y'][q['train_order']]))
                assert abs(err-q['MAE'])<1e-10
                inner_map[(q['repeat'],q['fold'],q['family'],q['view'],q['seed'])].append(q)
            biggest=0.;refit_difference=0.;seen=[]
            for s in selections:
                key=(s['repeat'],s['fold'],s['family'],s['view'],s['seed'])
                grid=inner_map[key];best=min(grid,key=lambda q:q['MAE'])
                assert best['setting']==s['setting'] and best['MAE']==s['inner_MAE']
                fold=lookup[s['repeat'],s['fold']];tr=np.array(fold['train']);te=np.array(fold['test']);x=d[s['view']];y=d['y']
                assert REMOVED not in d['ids'] and not set(tr)&set(te)
                file=HERE/s['file'];assert sha(file)==s['sha256'];scale,model=joblib.load(file)
                assert np.allclose(scale.mean_,np.mean(x[tr],axis=0),atol=1e-12)
                rows=p[(p['repeat']==s['repeat'])&(p.fold==s['fold'])&(p.family==s['family'])&(p.view==s['view'])&(p.seed==s['seed'])].set_index('sample').loc[te]
                assert np.array_equal(rows.target,y[te]) and list(rows.id)==list(d['ids'][te])
                pred=model.predict(scale.transform(x[te]));pred=np.maximum(pred,0) if s['family']=='Ridge' else pred
                diff=float(max(abs(pred-rows.prediction)));biggest=max(biggest,diff);assert diff<1e-9
                # Independent refit uses sklearn objects directly, not the training helper.
                sc=clone(scale);mo=clone(model)
                if s['family']=='RF':
                    sc.fit(x[tr],sample_weight=np.ones(len(tr)));mo.fit(sc.transform(x[tr]),y[tr],sample_weight=np.ones(len(tr)))
                    assert model.min_samples_leaf==s['setting'] and model.random_state==s['seed']
                else:
                    sc.fit(x[tr]);mo.fit(sc.transform(x[tr]),y[tr]);assert model.alpha==s['setting']
                again=mo.predict(sc.transform(x[te]));again=np.maximum(again,0) if s['family']=='Ridge' else again
                diff=float(max(abs(again-pred)));refit_difference=max(refit_difference,diff);assert diff<1e-9
                seen.extend(rows.reset_index().index.tolist())
            expected=25*(len(info['ridge_views'])+3*len(info['rf_views']))
            assert len(selections)==expected and len(p)==5*info['people']*(len(info['ridge_views'])+3*len(info['rf_views']))
            assert not p.duplicated(['repeat','family','view','seed','id']).any()
            assert (p.groupby(['family','view','seed','id']).size()==5).all()
            # Independent arithmetic from raw rows, then average seeds and partitions per person.
            from assess import old_predictions
            p['scenario']='TrainExcluded';allp=pd.concat([p,old_predictions(cohort,info)],ignore_index=True)
            accum=collections.defaultdict(list)
            for row in allp.itertuples():
                e=row.prediction-row.target
                accum[(row.scenario,row.family,row.view,row.id)].append([abs(e),max(-e,0),max(e,0),e])
            person=pd.read_csv(out/'person-results.csv')
            for r in person.itertuples():
                actual=np.mean(accum[(r.scenario,r.family,r.view,r.id)],axis=0)
                assert np.allclose(actual,[r.MAE,r.underprediction,r.overprediction,r.bias],atol=1e-10,rtol=0)
            summary=json.loads((out/'summary.json').read_text())
            def selected(scenario,family,view,band):
                g=person[(person.scenario==scenario)&(person.family==family)&(person.view==view)].set_index('id').sort_index()
                return g[g.target>=20] if band=='ge20' else (g[g.target<20] if band=='lt20' else g)
            for r in summary['metrics']:
                g=selected(r['scenario'],r['family'],r['view'],r['band']);assert len(g)==r['people']
                for metric in ['MAE','underprediction','overprediction','bias']:assert abs(g[metric].mean()-r[metric])<1e-10
            for r in summary['contrasts']:
                a=selected(*r['a'],r['band']);b=selected(*r['b'],r['band']).loc[a.index]
                ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
                for metric in ['MAE','underprediction','overprediction','bias']:
                    delta=(a[metric]-b[metric]).to_numpy()
                    assert abs(delta.mean()-r[metric]['difference'])<1e-10
                    assert np.allclose(np.quantile(delta[ix].mean(1),[.025,.975]),r[metric]['conditional_bootstrap95'],atol=1e-10,rtol=0)
            counts[cohort]={'models_replayed_and_independently_refit':len(selections),'largest_replay_difference':biggest,'largest_refit_difference':refit_difference,
                'training_only_scaling_fitting_selection':True,'prediction_coverage_verified':True,'metrics_and_conditional_intervals_verified':True}
            print('VERIFIED',cohort,len(selections),flush=True)
    save('results-verification.json',{'passed':True,'cohorts':counts,'models':sum(v['models_replayed_and_independently_refit'] for v in counts.values()),
        'verifier_sha256':sha(HERE/'check_results.py')})
if __name__=='__main__':main()
