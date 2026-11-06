from study import *
import joblib
from sklearn.base import clone
from threadpoolctl import threadpool_limits

def main():
 protocol=json.loads((HERE/'protocol.json').read_text());verified={};new_total=rf_total=0
 with threadpool_limits(limits=1):
  for cohort,info in protocol['cohorts'].items():
   src=source(info);out=HERE/cohort;d=dict(np.load(src/'input.npz'));y=d['y'];plan=json.loads((src/'splits.json').read_text());p=pd.read_csv(out/'predictions.csv');sel=json.loads((out/'selections.json').read_text());inner=json.loads((out/'inner.json').read_text());choices=json.loads((out/'family-choice.json').read_text());completion=json.loads((out/'completion.json').read_text());assert completion['complete'] and completion['protocol_sha256']==sha(HERE/'protocol.json') and completion['driver_sha256']==sha(HERE/'fit_models.py') and completion['fit_code_sha256']==sha(HERE/'study.py')
   for n,h in info['hashes'].items():assert sha(src/n)==h
   assert len(sel)==150 and len(choices)==50 and not p.duplicated(['repeat','view','family','id']).any()
   for _,g in p.groupby(['repeat','view','family']):assert len(g)==len(y) and set(g.id)==set(d['ids'])
   best={};maxdiff=refitdiff=0.
   for r in sel:
    path=HERE/r['file'];assert sha(path)==r['sha256'];scale,model=joblib.load(path);fold=plan[r['repeat']]['splits'][r['fold']];tr=np.array(fold['train']);te=np.array(fold['test']);x=d[r['view']];expected=StandardScaler().fit(x[tr]);assert np.allclose(scale.mean_,expected.mean_,atol=1e-10) and np.allclose(scale.scale_,expected.scale_,atol=1e-10);assert model.alpha==r['alpha'];assert r['train_target_max']==max(y[tr])
    candidates=[z for z in inner if all(z[k]==r[k] for k in ['repeat','fold','view','family'])];assert [q['alpha'] for q in candidates]==GRIDS[r['family']];scores=[]
    for q in candidates:
     assert q['train_order']==tr.tolist()
     if q['error'] is None:
      v=float(np.mean(abs(np.array(q['predictions'])-y[tr])));assert abs(v-q['MAE'])<1e-8;scores.append(v)
     else:assert q['MAE'] is None;scores.append(float('inf'))
    j=int(np.argmin(scores));assert r['alpha']==GRIDS[r['family']][j] and abs(r['inner_MAE']-scores[j])<1e-8;best[(r['repeat'],r['fold'],r['view'],r['family'])]=scores[j]
    z=scale.transform(x[te]);manual=z@model.coef_+model.intercept_
    if r['family']=='Gamma':manual=np.exp(manual)
    raw=model.predict(z);assert np.allclose(raw,manual,rtol=1e-10,atol=1e-8)
    # Independent refit uses only the recorded training rows and saved hyperparameters.
    fresh=clone(model).fit(expected.transform(x[tr]),y[tr]);re=fresh.predict(expected.transform(x[te]));refitdiff=max(refitdiff,float(max(abs(re-raw))));assert np.allclose(re,raw,rtol=1e-8,atol=1e-8)
    g=p[(p.repeat==r['repeat'])&(p.fold==r['fold'])&(p.view==r['view'])&(p.family==r['family'])].set_index('sample').loc[te];assert np.array_equal(g.target,y[te]);diff=float(max(abs(np.maximum(raw,0)-g.prediction)));maxdiff=max(maxdiff,diff);assert np.allclose(raw,g.raw_prediction,rtol=1e-10,atol=1e-8) and np.allclose(np.maximum(raw,0),g.prediction,rtol=1e-10,atol=1e-8)
   for r in choices:
    picked=min(GRIDS,key=lambda family:best[(r['repeat'],r['fold'],r['view'],family)]);assert picked==r['family'];g=p[(p.repeat==r['repeat'])&(p.fold==r['fold'])&(p.view==r['view'])];a=g[g.family=='CVSelect'].set_index('id').sort_index();b=g[g.family==picked].set_index('id').loc[a.index];assert np.array_equal(a.prediction.to_numpy(),b.prediction.to_numpy())
   rf=pd.read_csv(src/'predictions.csv');rf=rf[rf.method.isin(info['views'])].rename(columns={'method':'view'});rf['family']='RF';rfsel=[r for r in json.loads((src/'selections.json').read_text()) if r['method'] in info['views']];assert len(rfsel)==150
   for r in rfsel:
    file=src.parent/r['file'];assert sha(file)==r['sha256'];scale,model=joblib.load(file);f=plan[r['repeat']]['splits'][r['fold']];te=np.array(f['test']);pred=model.predict(scale.transform(d[r['method']][te]));g=rf[(rf.repeat==r['repeat'])&(rf.fold==r['fold'])&(rf.seed==r['seed'])&(rf.view==r['method'])].set_index('sample').loc[te];assert np.allclose(pred,g.prediction,rtol=1e-10,atol=1e-8);assert np.all(pred<=max(y[f['train']])+1e-8)
   raw=pd.concat([p,rf],ignore_index=True);raw['MAE']=abs(raw.prediction-raw.target);raw['underprediction']=np.maximum(raw.target-raw.prediction,0);raw['overprediction']=np.maximum(raw.prediction-raw.target,0);raw['bias']=raw.prediction-raw.target
   runs=pd.read_csv(out/'partition-metrics.csv');summary=json.loads((out/'summary.json').read_text())
   for r in runs.to_dict('records'):
    g=raw[(raw.repeat==r['repeat'])&(raw.view==r['view'])&(raw.family==r['family'])]
    if r['band']=='ge20':g=g[g.target>=20]
    elif r['band']=='lt20':g=g[g.target<20]
    for k in ['MAE','underprediction','overprediction','bias']:assert np.isclose(g[k].mean(),r[k],rtol=1e-10,atol=1e-8)
    assert np.isclose(np.sqrt(np.mean((g.prediction-g.target)**2)),r['RMSE'],rtol=1e-10,atol=1e-8)
   for key,q in summary['comparisons'].items():
    pair,band=key.split(' / ');aa,bb=pair.split(' minus ');va,fa=aa.split(':');vb,fb=bb.split(':');a=raw[(raw.view==va)&(raw.family==fa)];b=raw[(raw.view==vb)&(raw.family==fb)]
    if band=='ge20':a=a[a.target>=20];b=b[b.target>=20]
    elif band=='lt20':a=a[a.target<20];b=b[b.target<20]
    for metric in ['MAE','underprediction','overprediction']:
     x=a.groupby('id')[metric].mean().sort_index();z=b.groupby('id')[metric].mean().loc[x.index];delta=(x-z).to_numpy();rng=np.random.default_rng(20261031);boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1);assert np.isclose(delta.mean(),q[metric]['difference'],rtol=1e-10,atol=1e-8) and np.allclose(np.quantile(boot,[.025,.975]),q[metric]['conditional_person_bootstrap95'],rtol=1e-10,atol=1e-8)
   verified[cohort]={'new_models_replayed_and_refit':len(sel),'RF_models_replayed':len(rfsel),'largest_prediction_difference':maxdiff,'largest_refit_difference':refitdiff,'train_only_scaling_and_fitting':True,'inner_alpha_and_family_selection_verified':True,'metrics_and_intervals_recomputed':True};new_total+=len(sel);rf_total+=len(rfsel);print('VERIFIED',cohort,len(sel),len(rfsel),flush=True)
 save('results-verification.json',{'passed':True,'new_models':new_total,'RF_models':rf_total,'cohorts':verified,'code_sha256':sha(HERE/'check_results.py')});print('ALL_VERIFIED',new_total,rf_total,flush=True)
if __name__=='__main__':main()
