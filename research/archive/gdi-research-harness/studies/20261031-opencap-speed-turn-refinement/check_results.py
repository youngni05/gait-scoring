from study import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from collections import defaultdict

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());results={}
    for n,h in protocol['source_hashes'].items():assert sha(HERE.parent/n)==h
    with threadpool_limits(limits=1):
        for cohort,info in protocol['cohorts'].items():
            out=HERE/cohort;d=dict(np.load(out/'input.npz'));y=d['y'];p=pd.read_csv(out/'predictions.csv')
            plans={(r['repeat'],f['fold']):f for r in json.loads((out/'splits.json').read_text()) for f in r['splits']}
            inner=json.loads((out/'inner.json').read_text());selections=json.loads((out/'selections.json').read_text());keyed=defaultdict(list)
            maxreplay=0.;maxrefit=0.;maxinner=0.
            for q in inner:
                f=plans[q['repeat'],q['fold']];x=d[q['view']];pred=np.full(len(y),np.nan);assert q['train_order']==f['train']
                for v in f['inner']:
                    tr=np.array(v['train']);te=np.array(v['validation']);sc=StandardScaler().fit(x[tr]);m=Ridge(alpha=q['alpha'],solver='svd').fit(sc.transform(x[tr]),y[tr])
                    pred[te]=np.maximum(m.predict(sc.transform(x[te])),0)
                difference=float(max(abs(pred[f['train']]-q['predictions'])));assert difference<1e-10;maxinner=max(maxinner,difference)
                assert abs(np.mean(abs(pred[f['train']]-y[f['train']]))-q['MAE'])<1e-10;keyed[q['repeat'],q['fold'],q['view']].append(q)
            for s in selections:
                f=plans[s['repeat'],s['fold']];tr=np.array(f['train']);te=np.array(f['test']);x=d[s['view']]
                best=min(keyed[s['repeat'],s['fold'],s['view']],key=lambda q:q['MAE']);assert best['alpha']==s['alpha'] and best['MAE']==s['inner_MAE']
                assert sha(HERE/s['file'])==s['sha256'];scale,model=joblib.load(HERE/s['file'])
                rows=p[(p['repeat']==s['repeat'])&(p.fold==s['fold'])&(p.view==s['view'])].set_index('sample').loc[te]
                assert np.array_equal(rows.target,y[te]) and list(rows.id)==list(d['ids'][te])
                raw=model.predict(scale.transform(x[te]));assert max(abs(raw-rows.raw_prediction))<1e-10
                pred=np.maximum(raw,0);diff=float(max(abs(pred-rows.prediction)));assert diff<1e-10;maxreplay=max(maxreplay,diff)
                sc=StandardScaler().fit(x[tr]);m=Ridge(alpha=s['alpha'],solver='svd').fit(sc.transform(x[tr]),y[tr]);again=np.maximum(m.predict(sc.transform(x[te])),0)
                diff=float(max(abs(again-pred)));assert diff<1e-10;maxrefit=max(maxrefit,diff);assert np.array_equal(sc.mean_,scale.mean_)
            assert len(selections)==25*len(info['views']) and len(inner)==100*len(info['views'])
            assert len(p)==5*info['people']*len(info['views']) and not p.duplicated(['view','repeat','id']).any()
            assert (p.groupby(['view','id']).size()==5).all() and 'sub-283' not in p.id.values
            person=pd.read_csv(out/'person-results.csv');summary=json.loads((out/'summary.json').read_text())
            def select(view,band):
                z=person[person.view==view].set_index('id').sort_index()
                return z[z.target>=20] if band=='ge20' else (z[z.target<20] if band=='lt20' else z)
            for r in person.itertuples():
                g=p[(p.view==r.view)&(p.id==r.id)];e=(g.prediction-g.target).to_numpy()
                assert np.allclose([abs(e).mean(),np.maximum(-e,0).mean(),np.maximum(e,0).mean(),e.mean()],[r.MAE,r.underprediction,r.overprediction,r.bias],atol=1e-10,rtol=0)
            for r in summary['metrics']:
                g=select(r['view'],r['band']);assert len(g)==r['people']
                for metric in ['MAE','underprediction','overprediction','bias']:assert abs(g[metric].mean()-r[metric])<1e-10
            for r in summary['contrasts']:
                a=select(r['a'][2],r['band']);b=select(r['b'][2],r['band']).loc[a.index];ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
                for metric in ['MAE','underprediction','overprediction','bias']:
                    delta=(a[metric]-b[metric]).to_numpy();assert abs(delta.mean()-r[metric]['difference'])<1e-10
                    assert np.allclose(np.quantile(delta[ix].mean(1),[.025,.975]),r[metric]['conditional_bootstrap95'],atol=1e-10,rtol=0)
                    expected=[]
                    for rep in range(5):
                        metrics=[]
                        for view in [r['a'][2],r['b'][2]]:
                            g=p[(p.view==view)&(p['repeat']==rep)];g=g[g.target>=20] if r['band']=='ge20' else (g[g.target<20] if r['band']=='lt20' else g)
                            e=(g.prediction-g.target).to_numpy();metrics.append({'MAE':abs(e).mean(),'underprediction':np.maximum(-e,0).mean(),'overprediction':np.maximum(e,0).mean(),'bias':e.mean()}[metric])
                        expected.append(metrics[0]-metrics[1])
                    assert np.allclose(expected,r[metric]['partition_deltas'],atol=1e-10,rtol=0)
            if cohort=='all237':
                old=pd.read_csv(PRIOR/'predictions.csv');old=old[(old.family=='Ridge')&(old.view=='TurnTimeOnly')]
                joined=p[p.view=='Current'].merge(old,on=['repeat','fold','id'],suffixes=('_new','_old'),validate='one_to_one')
                assert len(joined)==237*5 and max(abs(joined.prediction_new-joined.prediction_old))<1e-10
            results[cohort]={'final_models_replayed_and_refit':len(selections),'inner_fits_recomputed':len(inner)*3,'max_replay_difference':maxreplay,'max_refit_difference':maxrefit,'max_inner_difference':maxinner,
                'training_only_selection':True,'coverage_metrics_and_intervals':True,'prior_baseline_exactly_reproduced':cohort=='all237'}
            print('VERIFIED',cohort,len(selections),flush=True)
    save('results-verification.json',{'passed':True,'cohorts':results,'models':sum(x['final_models_replayed_and_refit'] for x in results.values()),'inner_fits':sum(x['inner_fits_recomputed'] for x in results.values())})
if __name__=='__main__':main()
