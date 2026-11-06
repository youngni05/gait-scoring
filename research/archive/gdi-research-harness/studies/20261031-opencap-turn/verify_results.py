from common import *
import joblib
from sklearn.preprocessing import StandardScaler
def main():
 verified={};total=0
 cohorts=['event_matched','fixed6','event_flagged']+(['geometry'] if (HERE/'geometry/completion.json').exists() else [])
 for cohort in cohorts:
  out=HERE/cohort;d=dict(np.load(out/'input.npz'));y=d['y'];plan=json.loads((out/'splits.json').read_text());p=pd.read_csv(out/'predictions.csv');sel=json.loads((out/'selections.json').read_text());inner=json.loads((out/'inner-evaluations.json').read_text());runs=pd.read_csv(out/'run-metrics.csv');summary=json.loads((out/'summary.json').read_text());completion=json.loads((out/'completion.json').read_text())
  assert completion['complete'] and completion['input_sha256']==sha(out/'input.npz') and completion['code_sha256']==sha(HERE/'train.py')
  assert not p.duplicated(['repeat','seed','method','id']).any()
  for rep in plan:
   for f in rep['splits']:
    baseline=p[(p.repeat==rep['repeat'])&(p.fold==f['fold'])&(p.method=='Median')]
    assert np.allclose(baseline.prediction,np.sort(y[f['train']])[(len(f['train'])-1)//2])
  for (rep,seed,method),g in p.groupby(['repeat','seed','method']):
   assert set(g.id)==set(d['ids']) and len(g)==len(y)
  ie={(r['repeat'],r['fold'],r['seed'],r['method'],r['leaf']):r for r in inner}
  maxdiff=0.
  for r in sel:
   file=HERE/r['file'];assert sha(file)==r['sha256'];scale,model=joblib.load(file)
   fold=plan[r['repeat']]['splits'][r['fold']];tr=np.array(fold['train']);te=np.array(fold['test']);z=d[r['method']]
   expectedscale=StandardScaler().fit(z[tr],sample_weight=np.ones(len(tr)))
   assert np.allclose(scale.mean_,expectedscale.mean_,atol=1e-10) and np.allclose(scale.scale_,expectedscale.scale_,atol=1e-10)
   assert model.min_samples_leaf==r['leaf'] and model.n_estimators==128 and model._n_samples==len(tr)
   for k in [0,127]:
    ids=model.estimators_samples_[k];expected=float(np.mean(y[tr][ids]));assert abs(float(model.estimators_[k].tree_.value[0,0,0])-expected)<1e-8
   scores=[]
   for leaf in [2,5]:
    q=ie[(r['repeat'],r['fold'],r['seed'],r['method'],leaf)];assert q['train_order']==tr.tolist();v=float(np.mean(abs(y[tr]-np.array(q['predictions']))));assert abs(v-q['MAE'])<1e-9;scores.append(v)
   assert r['leaf']==[2,5][int(np.argmin(scores))]
   replay=model.predict(scale.transform(z[te]));g=p[(p.repeat==r['repeat'])&(p.fold==r['fold'])&(p.seed==r['seed'])&(p.method==r['method'])].set_index('sample').loc[te]
   assert np.array_equal(g.target.to_numpy(),y[te]);diff=float(max(abs(replay-g.prediction.to_numpy())));maxdiff=max(maxdiff,diff);assert diff<1e-8
  for row in runs.to_dict('records'):
   g=p[(p.repeat==row['repeat'])&(p.seed==row['seed'])&(p.method==row['method'])]
   if row['band']=='lt10':g=g[g.target<10]
   elif row['band']=='10to20':g=g[(g.target>=10)&(g.target<20)]
   elif row['band']=='ge20':g=g[g.target>=20]
   e=g.prediction.to_numpy()-g.target.to_numpy()
   for k,v in {'MAE':np.mean(abs(e)),'bias':np.mean(e),'RMSE':np.sqrt(np.mean(e**2)),'underprediction':np.mean(np.maximum(-e,0))}.items():assert abs(row[k]-v)<1e-8
  for name,q in summary['comparisons'].items():
   models,band=name.split(' / ');a,b=models.split(' minus ');pa=p[p.method==a].copy();pb=p[p.method==b].copy()
   if band=='ge20':pa=pa[pa.target>=20];pb=pb[pb.target>=20]
   pa['ae']=abs(pa.prediction-pa.target);pb['ae']=abs(pb.prediction-pb.target);aa=pa.groupby('id').ae.mean().sort_index();bb=pb.groupby('id').ae.mean().loc[aa.index];delta=(aa-bb).to_numpy()
   rng=np.random.default_rng(20261031);boot=np.mean(delta[rng.integers(len(delta),size=(10000,len(delta)))],axis=1)
   assert abs(delta.mean()-q['MAE_difference'])<1e-8 and np.allclose(np.quantile(boot,[.025,.975]),q['conditional_person_bootstrap95'],atol=1e-8)
  total+=len(sel);verified[cohort]={'models_replayed':len(sel),'max_prediction_difference':maxdiff,'all_people_all15runs':True,'train_only_scaling_and_bootstrap_label_means':True,'inner_selection_recomputed':True,'metrics_and_bootstrap_independently_recomputed':True}
 save('results-verification.json',{'passed':True,'models_replayed':total,'cohorts':verified,'code_sha256':sha(HERE/'verify_results.py')});print(json.dumps(verified),flush=True)
if __name__=='__main__':main()
