from study import *
from threadpoolctl import threadpool_limits
LARGE=['sub-246','sub-266','sub-310','sub-406']
def band(g,name):return g[g.id.isin(LARGE)] if name=='large4' else g[g.target>=20] if name=='ge20' else g
def main():
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for f,h in protocol['source_hashes'].items():assert sha(HERE.parent/f)==h
    out=HERE/'cohort176';d=np.load(out/'input.npz');y=d['y'];p=pd.read_csv(out/'predictions.csv');plans={(r['repeat'],f['fold']):f for r in json.loads((out/'splits.json').read_text()) for f in r['splits']}
    inner=json.loads((out/'inner.json').read_text());selections=json.loads((out/'selections.json').read_text());scores={};maxdiff=0.
    with threadpool_limits(limits=1):
        for q in inner:
            f=plans[q['repeat'],q['fold']];full=d[q['view']];x=full[:,:-1];offset=full[:,-1];pred=np.full(len(y),np.nan);assert q['train_order']==f['train']
            for v in f['inner']:
                tr=np.array(v['train']);te=np.array(v['validation']);sc=StandardScaler().fit(x[tr]);model=Ridge(alpha=q['alpha'],solver='svd').fit(sc.transform(x[tr]),y[tr]-offset[tr]);pred[te]=np.maximum(model.predict(sc.transform(x[te])),0)+offset[te]
            assert max(abs(pred[f['train']]-q['predictions']))<1e-10
            mae=float(abs(pred[f['train']]-y[f['train']]).mean());assert abs(mae-q['MAE'])<1e-10;scores.setdefault((q['repeat'],q['fold'],q['view']),[]).append((mae,q['alpha']))
        for q in selections:
            f=plans[q['repeat'],q['fold']];tr=np.array(f['train']);te=np.array(f['test']);full=d[q['view']];x=full[:,:-1];offset=full[:,-1]
            assert min(scores[q['repeat'],q['fold'],q['view']],key=lambda z:z[0])==(q['inner_MAE'],q['alpha'])
            assert sha(HERE/q['file'])==q['sha256'];sc,model=joblib.load(HERE/q['file']);part=model.predict(sc.transform(x[te]));pred=np.maximum(part,0)+offset[te]
            rows=p[(p['repeat']==q['repeat'])&(p.fold==q['fold'])&(p.view==q['view'])].set_index('sample').loc[te]
            assert np.array_equal(rows.target,y[te]) and list(rows.id)==list(d['ids'][te]);assert max(abs(rows.raw_prediction-(part+offset[te])))<1e-10
            diff=float(max(abs(rows.prediction-pred)));maxdiff=max(maxdiff,diff);assert diff<1e-10
            sc2=StandardScaler().fit(x[tr]);m2=Ridge(alpha=q['alpha'],solver='svd').fit(sc2.transform(x[tr]),y[tr]-offset[tr]);again=np.maximum(m2.predict(sc2.transform(x[te])),0)+offset[te];assert max(abs(again-pred))<1e-10
    assert len(selections)==200 and len(inner)*3==2400
    assert len(p)==176*5*8 and not p.duplicated(['view','id','repeat']).any() and 'sub-283' not in p.id.values
    oldp=pd.read_csv(PRIOR/'exclude283/predictions.csv');oldp=oldp[oldp.view=='Baseline_20'];joined=p[p.view=='Original'].merge(oldp,on=['repeat','fold','id'],validate='one_to_one',suffixes=('_new','_old'));assert len(joined)==880 and max(abs(joined.prediction_new-joined.prediction_old))<1e-10
    e=p.prediction-p.target;p['MAE']=abs(e);p['underprediction']=np.maximum(-e,0);p['overprediction']=np.maximum(e,0);p['bias']=e
    summary=json.loads((HERE/'summary.json').read_text());persons=pd.read_csv(out/'person-results.csv');keys=['MAE','underprediction','overprediction','bias']
    for r in persons.itertuples():
        g=p[(p.view==r.view)&(p.id==r.id)]
        for k in keys:assert abs(g[k].mean()-getattr(r,k))<1e-10
    for r in summary['metrics']:
        g=band(persons[persons.view==r['view']],r['band']);assert len(g)==r['people']
        for k in keys:assert abs(g[k].mean()-r[k])<1e-10
    for r in summary['contrasts']:
        a=band(persons[persons.view==r['a']],r['band']).set_index('id').sort_index();b=band(persons[persons.view==r['b']],r['band']).set_index('id').loc[a.index];ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
        for k in keys:
            delta=(a[k]-b[k]).to_numpy();assert abs(delta.mean()-r[k]['difference'])<1e-10 and np.allclose(np.quantile(delta[ix].mean(1),[.025,.975]),r[k]['conditional95'],atol=1e-10,rtol=0)
            ds=[band(p[(p.view==r['a'])&(p['repeat']==rep)],r['band'])[k].mean()-band(p[(p.view==r['b'])&(p['repeat']==rep)],r['band'])[k].mean() for rep in range(5)];assert np.allclose(ds,r[k]['partition_deltas'],atol=1e-10,rtol=0)
    for r in pd.read_csv(HERE/'case-results.csv').itertuples():
        g=p[(p.view==r.view)&(p.id==r.id)];assert np.allclose([g.prediction.mean(),g.prediction.std(ddof=0),g.prediction.min(),g.prediction.max()],[r.prediction,r.sd,r.minimum,r.maximum],atol=1e-10,rtol=0)
    save('results-verification.json',{'passed':True,'final_models_refit':200,'inner_fits_refit':2400,'max_prediction_difference':maxdiff,'original_predictions_reproduced':True,'person_equal_metrics_and_intervals':True,'case_stability_checked':True})
    print('VERIFIED 200 final models and 2400 inner fits',flush=True)
if __name__=='__main__':main()
