from study import *
from threadpoolctl import threadpool_limits
from collections import defaultdict

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());results={};stability=json.loads((HERE/'stability.json').read_text())
    assert json.loads((HERE/'input-verification.json').read_text())['passed']
    for n,h in protocol['source_hashes'].items():assert sha(HERE.parent/n)==h
    with threadpool_limits(limits=1):
        for cohort,info in protocol['cohorts'].items():
            out=HERE/cohort;d=dict(np.load(out/'input.npz'));y=d['y'];p=pd.read_csv(out/'predictions.csv')
            plans={(r['repeat'],f['fold']):f for r in json.loads((out/'splits.json').read_text()) for f in r['splits']}
            inner=json.loads((out/'inner.json').read_text());selections=json.loads((out/'selections.json').read_text());keyed=defaultdict(list);maxdiff=0.
            for q in inner:
                f=plans[q['repeat'],q['fold']];xx=d[q['view']];x=xx[:,:-1];off=xx[:,-1];target=y-off;pred=np.full(len(y),np.nan);assert q['train_order']==f['train']
                for v in f['inner']:
                    tr=np.array(v['train']);te=np.array(v['validation']);sc=StandardScaler().fit(x[tr]);m=Ridge(alpha=q['alpha'],solver='svd').fit(sc.transform(x[tr]),target[tr]);pred[te]=np.maximum(m.predict(sc.transform(x[te])),0)+off[te]
                diff=float(max(abs(pred[f['train']]-q['predictions'])));assert diff<1e-10;maxdiff=max(maxdiff,diff)
                assert abs(np.mean(abs(pred[f['train']]-y[f['train']]))-q['MAE'])<1e-10;keyed[q['repeat'],q['fold'],q['view']].append(q)
            for s in selections:
                f=plans[s['repeat'],s['fold']];tr=np.array(f['train']);te=np.array(f['test']);xx=d[s['view']];x=xx[:,:-1];off=xx[:,-1];target=y-off
                best=min(keyed[s['repeat'],s['fold'],s['view']],key=lambda q:q['MAE']);assert best['alpha']==s['alpha'] and best['MAE']==s['inner_MAE']
                assert sha(HERE/s['file'])==s['sha256'];scale,model=joblib.load(HERE/s['file']);rows=p[(p['repeat']==s['repeat'])&(p.fold==s['fold'])&(p.view==s['view'])].set_index('sample').loc[te]
                assert np.array_equal(rows.target,y[te]) and list(rows.id)==list(d['ids'][te])
                part=model.predict(scale.transform(x[te]));assert max(abs(part+off[te]-rows.raw_prediction))<1e-10
                pred=np.maximum(part,0)+off[te];assert max(abs(pred-rows.prediction))<1e-10
                sc=StandardScaler().fit(x[tr]);m=Ridge(alpha=s['alpha'],solver='svd').fit(sc.transform(x[tr]),target[tr]);again=np.maximum(m.predict(sc.transform(x[te])),0)+off[te]
                diff=float(max(abs(again-pred)));assert diff<1e-10;maxdiff=max(maxdiff,diff)
            assert len(selections)==25*len(info['views']) and len(inner)==100*len(info['views'])
            assert len(p)==5*info['people']*len(info['views']) and not p.duplicated(['view','repeat','id']).any()
            assert (p.groupby(['view','id']).size()==5).all()
            person=pd.read_csv(out/'person-results.csv');summary=json.loads((out/'summary.json').read_text())
            def select(view,band):
                z=person[person.view==view].set_index('id').sort_index()
                return z[z.target>=20] if band=='ge20' else (z[z.target<20] if band=='lt20' else z)
            for r in person.itertuples():
                g=p[(p.view==r.view)&(p.id==r.id)];e=(g.prediction-g.target).to_numpy()
                assert np.allclose([abs(e).mean(),np.maximum(-e,0).mean(),np.maximum(e,0).mean(),e.mean()],[r.MAE,r.underprediction,r.overprediction,r.bias],atol=1e-10,rtol=0)
            for r in summary['metrics']:
                g=select(r['view'],r['band']);assert len(g)==r['people']
                for k in ['MAE','underprediction','overprediction','bias']:assert abs(g[k].mean()-r[k])<1e-10
            for r in summary['contrasts']:
                a=select(r['a'][2],r['band']);b=select(r['b'][2],r['band']).loc[a.index];ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
                for k in ['MAE','underprediction','overprediction','bias']:
                    delta=(a[k]-b[k]).to_numpy();assert abs(delta.mean()-r[k]['difference'])<1e-10
                    assert np.allclose(np.quantile(delta[ix].mean(1),[.025,.975]),r[k]['conditional_bootstrap95'],atol=1e-10,rtol=0)
                    assert sum(delta<0)==r[k]['people_better'] and sum(delta>0)==r[k]['people_worse']
                    partition=[]
                    for rep in range(5):
                        vals=[]
                        for view in [r['a'][2],r['b'][2]]:
                            g=p[(p.view==view)&(p['repeat']==rep)];g=g[g.target>=20] if r['band']=='ge20' else (g[g.target<20] if r['band']=='lt20' else g);e=(g.prediction-g.target).to_numpy()
                            vals.append({'MAE':abs(e).mean(),'underprediction':np.maximum(-e,0).mean(),'overprediction':np.maximum(e,0).mean(),'bias':e.mean()}[k])
                        partition.append(vals[0]-vals[1])
                    assert np.allclose(partition,r[k]['partition_deltas'],atol=1e-10,rtol=0)
            for r in stability:
                if r['cohort']!=cohort:continue
                g=p[(p.view==r['view'])&(p.id==r['id'])];q=g.prediction.to_numpy();assert np.allclose(q,r['predictions'],atol=1e-10,rtol=0)
                assert np.allclose([q.mean(),q.std(),q.min(),q.max()],[r[k] for k in ['mean','sd','min','max']],atol=1e-10,rtol=0)
            for r in json.loads((out/'case-exclusion-sensitivity.json').read_text()):
                a=select(f'Prep_{r["budget"]}','ge20').drop(r['omit_evaluation_only'],errors='ignore');b=select(f'Gait_{r["budget"]}','ge20').loc[a.index];assert len(a)==r['people']
                for k in ['MAE','underprediction','overprediction']:assert abs((a[k]-b[k]).mean()-r[k])<1e-10
            results[cohort]={'models':len(selections),'inner_fits':len(inner)*3,'maximum_prediction_difference':maxdiff,'metrics_intervals_stability_verified':True}
            print('VERIFIED',cohort,len(selections),flush=True)
    save('results-verification.json',{'passed':True,'cohorts':results,'models':sum(r['models'] for r in results.values()),'inner_fits':sum(r['inner_fits'] for r in results.values())})
if __name__=='__main__':main()
