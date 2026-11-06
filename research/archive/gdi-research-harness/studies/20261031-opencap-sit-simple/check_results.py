from study import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from collections import defaultdict

def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    for n,h in protocol['source_hashes'].items():assert sha(HERE.parent/n)==h
    for n,h in protocol['input_hashes'].items():assert sha(HERE/n)==h
    d=dict(np.load(HERE/'input.npz'));y=d['y'];p=pd.read_csv(HERE/'predictions.csv')
    plans={(r['repeat'],f['fold']):f for r in json.loads((HERE/'splits.json').read_text()) for f in r['splits']}
    evaluations=json.loads((HERE/'inner.json').read_text());selections=json.loads((HERE/'selections.json').read_text())
    keyed=defaultdict(list);max_difference=0.;max_inner=0.;max_refit=0.
    with threadpool_limits(limits=1):
        # Recompute every inner prediction independently, not only the selected model.
        for q in evaluations:
            f=plans[q['repeat'],q['fold']];x=d[q['view']];pred=np.full(len(y),np.nan)
            assert q['train_order']==f['train']
            for v in f['inner']:
                tr=np.array(v['train']);te=np.array(v['validation']);sc=StandardScaler().fit(x[tr]);m=Ridge(alpha=q['alpha'],solver='svd').fit(sc.transform(x[tr]),y[tr])
                pred[te]=np.maximum(m.predict(sc.transform(x[te])),0)
            difference=float(max(abs(pred[f['train']]-q['predictions'])));assert difference<1e-10;max_inner=max(max_inner,difference)
            assert abs(np.mean(abs(pred[f['train']]-y[f['train']]))-q['MAE'])<1e-10
            keyed[q['repeat'],q['fold'],q['view']].append(q)
        for s in selections:
            f=plans[s['repeat'],s['fold']];tr=np.array(f['train']);te=np.array(f['test']);x=d[s['view']]
            best=min(keyed[s['repeat'],s['fold'],s['view']],key=lambda q:q['MAE']);assert s['alpha']==best['alpha'] and s['inner_MAE']==best['MAE']
            assert sha(HERE/s['file'])==s['sha256'];sc,m=joblib.load(HERE/s['file']);assert m.alpha==s['alpha']
            rows=p[(p['repeat']==s['repeat'])&(p.fold==s['fold'])&(p.view==s['view'])].set_index('sample').loc[te]
            assert list(rows.id)==list(d['ids'][te]) and np.array_equal(rows.target,y[te])
            raw=m.predict(sc.transform(x[te]));assert max(abs(raw-rows.raw_prediction))<1e-10
            pred=np.maximum(raw,0);diff=float(max(abs(pred-rows.prediction)));assert diff<1e-10;max_difference=max(max_difference,diff)
            scl=StandardScaler().fit(x[tr]);model=Ridge(alpha=s['alpha'],solver='svd').fit(scl.transform(x[tr]),y[tr])
            again=np.maximum(model.predict(scl.transform(x[te])),0);diff=float(max(abs(again-pred)));assert diff<1e-10;max_refit=max(max_refit,diff)
            assert np.allclose(sc.mean_,scl.mean_,atol=1e-12,rtol=0)
    assert len(selections)==125 and len(evaluations)==500 and len(p)==2500
    assert not p.duplicated(['view','repeat','id']).any() and (p.groupby(['view','id']).size()==5).all()
    assert 'sub-283' not in p.id.values
    persons=pd.read_csv(HERE/'person-results.csv');runs=pd.read_csv(HERE/'partition-metrics.csv');summary=json.loads((HERE/'summary.json').read_text())
    def select(view,band):
        z=persons[persons.view==view].set_index('id').sort_index()
        return z[z.target>=20] if band=='ge20' else (z[z.target<20] if band=='lt20' else z)
    for r in persons.itertuples():
        g=p[(p.view==r.view)&(p.id==r.id)];e=(g.prediction-g.target).to_numpy()
        assert np.allclose([abs(e).mean(),np.maximum(-e,0).mean(),np.maximum(e,0).mean(),e.mean()],[r.MAE,r.underprediction,r.overprediction,r.bias],atol=1e-10,rtol=0)
    for r in summary['metrics']:
        g=select(r['view'],r['band']);assert len(g)==r['people']
        for metric in ['MAE','underprediction','overprediction','bias']:assert abs(g[metric].mean()-r[metric])<1e-10
    for r in summary['contrasts']:
        a=select(r['a'][2],r['band']);b=select(r['b'][2],r['band']).loc[a.index]
        ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
        for metric in ['MAE','underprediction','overprediction','bias']:
            delta=(a[metric]-b[metric]).to_numpy()
            assert abs(delta.mean()-r[metric]['difference'])<1e-10
            assert np.allclose(np.quantile(delta[ix].mean(1),[.025,.975]),r[metric]['conditional_bootstrap95'],atol=1e-10,rtol=0)
            expected=[]
            for rep in range(5):
                raw=[]
                for view in [r['a'][2],r['b'][2]]:
                    g=p[(p.view==view)&(p['repeat']==rep)];g=g[g.target>=20] if r['band']=='ge20' else (g[g.target<20] if r['band']=='lt20' else g)
                    e=(g.prediction-g.target).to_numpy();raw.append({'MAE':abs(e).mean(),'underprediction':np.maximum(-e,0).mean(),'overprediction':np.maximum(e,0).mean(),'bias':e.mean()}[metric])
                expected.append(raw[0]-raw[1])
            assert np.allclose(expected,r[metric]['partition_deltas'],atol=1e-10,rtol=0)
    save('results-verification.json',{'passed':True,'models_replayed_and_independently_refit':125,'inner_fits_independently_recomputed':1500,
        'largest_replay_difference':max_difference,'largest_refit_difference':max_refit,'largest_inner_prediction_difference':max_inner,
        'selection_metrics_person_partition_bootstrap_verified':True,'excluded_subject_absent':True,'verifier_sha256':sha(HERE/'check_results.py')})
    print('VERIFIED:125 final models +1500 inner fits; metrics and paired intervals',flush=True)
if __name__=='__main__':main()
