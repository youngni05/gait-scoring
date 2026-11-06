from context import *
import joblib
from sklearn.preprocessing import StandardScaler

def main():
 verified={};prep=json.loads((STUDY/'preparation.json').read_text());total=0
 if (STUDY/'prep_event/completion.json').exists():prep['cohorts'].update(json.loads((STUDY/'preparation-extension.json').read_text())['cohorts'])
 for cohort,info in prep['cohorts'].items():
  out=STUDY/cohort;d=dict(np.load(out/'input.npz'));y=d['y'];plan=json.loads((out/'splits.json').read_text());p=pd.read_csv(out/'predictions.csv');sel=json.loads((out/'selections.json').read_text());inner=json.loads((out/'inner-evaluations.json').read_text());runs=pd.read_csv(out/'run-metrics.csv');summary=json.loads((out/'summary.json').read_text());completion=json.loads((out/'completion.json').read_text())
  assert completion['complete'] and completion['input_sha256']==sha(out/'input.npz') and completion['driver_sha256']==sha(STUDY/'fit_study.py') and completion['learner_sha256']==sha(PRIOR/'train.py')
  assert len(sel)==len(info['views'])*75 and not p.duplicated(['repeat','seed','method','id']).any()
  for rep in plan:
   for f in rep['splits']:
    baseline=p[(p.repeat==rep['repeat'])&(p.fold==f['fold'])&(p.method=='Median')];assert np.allclose(baseline.prediction,np.sort(y[f['train']])[(len(f['train'])-1)//2])
  for _,g in p.groupby(['repeat','seed','method']):assert set(g.id)==set(d['ids']) and len(g)==len(y)
  ie={(r['repeat'],r['fold'],r['seed'],r['method'],r['leaf']):r for r in inner};worst=0.
  for r in sel:
   file=STUDY/r['file'];assert sha(file)==r['sha256'];scale,model=joblib.load(file);f=plan[r['repeat']]['splits'][r['fold']];tr=np.array(f['train']);te=np.array(f['test']);z=d[r['method']]
   expected=StandardScaler().fit(z[tr],sample_weight=np.ones(len(tr)));assert np.allclose(scale.mean_,expected.mean_,atol=1e-10) and np.allclose(scale.scale_,expected.scale_,atol=1e-10)
   assert model.min_samples_leaf==r['leaf'] and model.n_estimators==128 and model._n_samples==len(tr) and model.random_state==r['seed']
   for k in [0,127]:assert abs(float(model.estimators_[k].tree_.value[0,0,0])-np.mean(y[tr][model.estimators_samples_[k]]))<1e-8
   scores=[]
   for leaf in [2,5]:
    q=ie[(r['repeat'],r['fold'],r['seed'],r['method'],leaf)];assert q['train_order']==tr.tolist();v=float(np.mean(abs(y[tr]-np.array(q['predictions']))));assert abs(v-q['MAE'])<1e-9;scores.append(v)
   assert r['leaf']==[2,5][int(np.argmin(scores))]
   replay=model.predict(scale.transform(z[te]));g=p[(p.repeat==r['repeat'])&(p.fold==r['fold'])&(p.seed==r['seed'])&(p.method==r['method'])].set_index('sample').loc[te];assert np.array_equal(g.target,y[te]);diff=float(max(abs(replay-g.prediction.to_numpy())));worst=max(worst,diff);assert diff<1e-8
  for row in runs.to_dict('records'):
   g=p[(p.repeat==row['repeat'])&(p.seed==row['seed'])&(p.method==row['method'])]
   if row['band']=='lt20':g=g[g.target<20]
   elif row['band']=='ge20':g=g[g.target>=20]
   e=g.prediction.to_numpy()-g.target.to_numpy()
   for k,v in {'MAE':np.mean(abs(e)),'bias':np.mean(e),'RMSE':np.sqrt(np.mean(e**2)),'underprediction':np.mean(np.maximum(-e,0))}.items():assert abs(row[k]-v)<1e-8
  for name,q in summary['comparisons'].items():
   methods,band=name.split(' / ');a,b=methods.split(' minus ');pa=p[p.method==a].copy();pb=p[p.method==b].copy()
   if band=='ge20':pa=pa[pa.target>=20];pb=pb[pb.target>=20]
   for metric in ['MAE','underprediction']:
    for g in [pa,pb]:g['value']=abs(g.prediction-g.target) if metric=='MAE' else np.maximum(g.target-g.prediction,0)
    aa=pa.groupby('id').value.mean().sort_index();bb=pb.groupby('id').value.mean().loc[aa.index];delta=(aa-bb).to_numpy();rng=np.random.default_rng(20261031);boot=np.mean(delta[rng.integers(len(delta),size=(10000,len(delta)))],axis=1)
    assert abs(delta.mean()-q[metric]['difference'])<1e-8 and np.allclose(np.quantile(boot,[.025,.975]),q[metric]['conditional_person_bootstrap95'],atol=1e-8)
  total+=len(sel);verified[cohort]={'models_replayed':len(sel),'maximum_prediction_difference':worst,'all15runs':True,'train_only_scaling_and_labels':True,'inner_selection_and_metrics_and_intervals_recomputed':True};print('VERIFIED',cohort,len(sel),flush=True)
 save('results-verification.json',{'passed':True,'models_replayed':total,'cohorts':verified,'code_sha256':sha(STUDY/'check_results.py')});print('ALL_VERIFIED',total,flush=True)
if __name__=='__main__':main()
