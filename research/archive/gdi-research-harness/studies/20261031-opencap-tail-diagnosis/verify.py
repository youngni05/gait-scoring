from analyze import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

def main():
    proto=json.loads((HERE/'learning-protocol.json').read_text());assert sha(HERE/'learning.py')==proto['code_sha256'] and sha(HERE/'learning-plans.json')==proto['plans_sha256']
    for file,h in (json.loads((HERE/'sources.json').read_text())|proto['source_hashes']).items():assert sha(HERE.parent/file)==h
    data=np.load(PRIOR/'exclude283/input.npz');y=data['y'];ids=data['ids'];plans={(r['repeat'],r['fold'],r['condition']):r for r in json.loads((HERE/'learning-plans.json').read_text())}
    p=pd.read_csv(HERE/'learning-predictions.csv');inner=json.loads((HERE/'learning-inner.json').read_text());selected=json.loads((HERE/'learning-selections.json').read_text());scores={};maxdiff=0.
    with threadpool_limits(limits=1):
        for q in inner:
            f=plans[q['repeat'],q['fold'],q['condition']];full=data[q['view']];x=full[:,:-1];offset=full[:,-1];target=y-offset;pp=np.full(len(y),np.nan)
            assert not set(f['train'])&set(f['test'])
            for v in f['inner']:
                tr=np.array(v['train']);te=np.array(v['validation']);assert not set(tr)&set(te) and set(tr)|set(te)==set(f['train'])
                sc=StandardScaler().fit(x[tr]);model=Ridge(alpha=q['alpha'],solver='svd').fit(sc.transform(x[tr]),target[tr]);pp[te]=np.maximum(model.predict(sc.transform(x[te])),0)+offset[te]
            assert np.max(abs(pp[f['train']]-q['predictions']))<1e-10
            mae=float(abs(pp[f['train']]-y[f['train']]).mean());assert abs(mae-q['MAE'])<1e-10
            scores.setdefault((q['repeat'],q['fold'],q['condition'],q['view']),[]).append((mae,q['alpha']))
        for q in selected:
            f=plans[q['repeat'],q['fold'],q['condition']];tr=np.array(f['train']);te=np.array(f['test']);full=data[q['view']];x=full[:,:-1];offset=full[:,-1]
            assert min(scores[q['repeat'],q['fold'],q['condition'],q['view']],key=lambda z:z[0])==(q['inner_MAE'],q['alpha'])
            sc=StandardScaler().fit(x[tr]);model=Ridge(alpha=q['alpha'],solver='svd').fit(sc.transform(x[tr]),y[tr]-offset[tr]);pred=np.maximum(model.predict(sc.transform(x[te])),0)+offset[te]
            assert sha(HERE/q['file'])==q['sha256'];savedsc,savedmodel=joblib.load(HERE/q['file']);replay=np.maximum(savedmodel.predict(savedsc.transform(x[te])),0)+offset[te]
            assert np.max(abs(pred-replay))<1e-10
            r=p[(p['repeat']==q['repeat'])&(p.fold==q['fold'])&(p.condition==q['condition'])&(p.view==q['view'])].set_index('sample').loc[te]
            assert np.array_equal(r.target,y[te]) and list(r.id)==list(ids[te]);diff=float(max(abs(r.prediction-pred)));assert diff<1e-10;maxdiff=max(diff,maxdiff)
    assert len(selected)==550 and len(inner)*3==6600
    for (condition,view),g in p.groupby(['condition','view']):assert len(g)==880 and not g.duplicated(['repeat','id']).any()
    old=pd.read_csv(PRIOR/'exclude283/predictions.csv');j=p[p.condition=='Full'].merge(old,on=['repeat','fold','view','id'],suffixes=('_new','_old'),validate='one_to_one');assert len(j)==1760 and max(abs(j.prediction_new-j.prediction_old))<1e-10
    # Metrics recomputed from raw errors, with equal weight per real person.
    p['setting']=p.condition.str.replace(r'_s\d+$','',regex=True);e=p.prediction-p.target;p['MAE']=abs(e);p['under']=np.maximum(-e,0);p['over']=np.maximum(e,0)
    large={r['id'] for r in json.loads((HERE/'person-audit.json').read_text()) if r['large_under']}
    agg=p.groupby(['setting','view','id'])[['target','MAE','under','over']].mean().reset_index()
    for r in pd.read_csv(HERE/'learning-summary.csv').itertuples():
        g=agg[(agg.setting==r.setting)&(agg.view==r.view)];g=g if r.band=='all' else g[g.target>=20] if r.band=='long8' else g[g.id.isin(large)]
        assert len(g)==r.people
        for k in ['MAE','under','over']:assert abs(g[k].mean()-getattr(r,k))<1e-10
    for q in json.loads((HERE/'learning-contrasts.json').read_text()):
        g=agg[agg.view==q['view']];g=g if q['band']=='all' else g[g.target>=20] if q['band']=='long8' else g[g.id.isin(large)]
        a=g[g.setting=='Full'].set_index('id').sort_index();b=g[g.setting==q['b']].set_index('id').loc[a.index];ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)))
        for k in ['MAE','under','over']:
            delta=(a[k]-b[k]).to_numpy();assert abs(delta.mean()-q[k]['difference'])<1e-10 and np.allclose(np.quantile(delta[ix].mean(1),[.025,.975]),q[k]['conditional95'],atol=1e-10,rtol=0)
    # Check the temporal decomposition independently as a union of native time spans.
    audit=json.loads((HERE/'person-audit.json').read_text());labels=pd.read_csv(LABELS).set_index(['subid','visit'])
    for r in audit:
        assert r['target']==float(labels.loc[(r['id'],r['visit']),'tft_tug'])
        if 'core_rise_to_sit_span' not in r:continue
        rise=r['phases']['rise']['window'];sit=r['phases']['sit']['window'];assert abs(r['clinical_minus_core_span']-(r['target']-sit[1]+rise[0]))<1e-10
        assert abs(r['core_rise_to_sit_span']-(r['rise_seconds']+r['sit_seconds']+r['full_eligible_gait_turn_seconds']+r['between_outside_gait_mask']))<1e-8
    save('verification.json',{'passed':True,'final_models_independently_refit':550,'inner_fits_independently_refit':6600,'max_prediction_difference':maxdiff,'full_baselines_reproduced':True,'same_test_people_windows_and_targets':True,'person_equal_metrics_and_conditional_intervals':True,'label_join_and_temporal_decomposition':True})
    print('VERIFIED 550 final models, 6600 inner fits',flush=True)
if __name__=='__main__':main()
