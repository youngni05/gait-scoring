from study import *

def combined(info,cohort):
 src=source(info);d=dict(np.load(src/'input.npz'));rf=pd.read_csv(src/'predictions.csv');rf=rf[rf.method.isin(info['views'])].rename(columns={'method':'view'});rf['family']='RF';rf['raw_prediction']=rf.prediction
 maxima={(r['repeat'],f['fold']):float(max(d['y'][f['train']])) for r in json.loads((src/'splits.json').read_text()) for f in r['splits']};rf['train_max']=[maxima[(r,f)] for r,f in zip(rf.repeat,rf.fold)]
 p=pd.concat([pd.read_csv(HERE/cohort/'predictions.csv'),rf],ignore_index=True);e=p.prediction-p.target;p['MAE']=abs(e);p['underprediction']=np.maximum(-e,0);p['overprediction']=np.maximum(e,0);p['bias']=e;p['MSE']=e**2;p['above_train_max']=(p.prediction>p.train_max+1e-8).astype(float)
 return p

def main():
 protocol=json.loads((HERE/'protocol.json').read_text());all_results={}
 for cohort,info in protocol['cohorts'].items():
  assert json.loads((HERE/cohort/'completion.json').read_text())['complete'];p=combined(info,cohort);out=HERE/cohort
  # RF errors are averaged across seeds, not errors of an averaged prediction.
  grouped=p.groupby(['view','family','repeat','id']).agg(target=('target','first'),prediction=('prediction','mean'),MAE=('MAE','mean'),underprediction=('underprediction','mean'),overprediction=('overprediction','mean'),bias=('bias','mean'),MSE=('MSE','mean'),above_train_max=('above_train_max','mean')).reset_index();grouped.to_csv(out/'partition-person.csv',index=False)
  persons=grouped.groupby(['view','family','id']).agg(target=('target','first'),prediction=('prediction','mean'),MAE=('MAE','mean'),underprediction=('underprediction','mean'),overprediction=('overprediction','mean'),bias=('bias','mean'),above_train_max=('above_train_max','mean')).reset_index();persons.to_csv(out/'person-results.csv',index=False);persons[persons.target>=20].to_csv(out/'long-people.csv',index=False)
  runs=[]
  for (view,family,rep),g in grouped.groupby(['view','family','repeat']):
   for band,mask in [('all',np.ones(len(g),bool)),('lt20',g.target<20),('ge20',g.target>=20)]:
    z=g.loc[mask];runs.append({'view':view,'family':family,'repeat':rep,'band':band,'people':len(z),**{m:float(z[m].mean()) for m in ['MAE','underprediction','overprediction','bias','above_train_max']},'RMSE':float(np.sqrt(z.MSE.mean()))})
  runs=pd.DataFrame(runs);runs.to_csv(out/'partition-metrics.csv',index=False);summary={}
  for (view,family,band),g in runs.groupby(['view','family','band']):summary.setdefault(view,{}).setdefault(family,{})[band]={'people':int(g.people.iloc[0]),**{m:float(g[m].mean()) for m in ['MAE','underprediction','overprediction','bias','RMSE','above_train_max']},'MAE_range':[float(g.MAE.min()),float(g.MAE.max())]}
  comparisons={};pairs=[(v,f,v,'RF') for v in info['views'] for f in list(GRIDS)+['CVSelect']]+[(info['views'][1],f,info['views'][0],f) for f in ['RF']+list(GRIDS)+['CVSelect']]
  for va,fa,vb,fb in pairs:
   for band in ['all','lt20','ge20']:
    z=persons[(persons.view==va)&(persons.family==fa)];z=z[z.target>=20] if band=='ge20' else (z[z.target<20] if band=='lt20' else z);z=z.set_index('id').sort_index();b=persons[(persons.view==vb)&(persons.family==fb)].set_index('id').loc[z.index];a_run=runs[(runs.view==va)&(runs.family==fa)&(runs.band==band)].set_index('repeat');b_run=runs[(runs.view==vb)&(runs.family==fb)&(runs.band==band)].set_index('repeat').loc[a_run.index];rng=np.random.default_rng(20261031);index=rng.integers(len(z),size=(10000,len(z)));q={}
    for metric in ['MAE','underprediction','overprediction']:
     delta=(z[metric]-b[metric]).to_numpy();boot=delta[index].mean(1);q[metric]={'difference':float(delta.mean()),'conditional_person_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'better_partitions':int(sum(a_run[metric]<b_run[metric])),'partitions':5,'people_improved':int(sum(delta<0)),'people_worse':int(sum(delta>0))}
    comparisons[f'{va}:{fa} minus {vb}:{fb} / {band}']=q
  extremes=[]
  for (view,family),g in p.groupby(['view','family']):extremes.append({'view':view,'family':family,'raw_min':float(g.raw_prediction.min()),'prediction_max':float(g.prediction.max()),'negative_raw_count':int(sum(g.raw_prediction<0)),'above_train_max_count':int(sum(g.above_train_max)),'predictions':len(g),'maximum_abs_error':float(g.MAE.max())})
  failures=[r for r in json.loads((out/'inner.json').read_text()) if r['error'] is not None];families=pd.DataFrame(json.loads((out/'family-choice.json').read_text()));choices={view:g.family.value_counts().to_dict() for view,g in families.groupby('view')}
  all_results[cohort]={'models':summary,'comparisons':comparisons,'extremes':extremes,'candidate_failures':len(failures),'selected_family_counts':choices};save(cohort+'/summary.json',all_results[cohort]);print(cohort,json.dumps({v:{f:{'all':r['all']['MAE'],'tail_MAE':r['ge20']['MAE'],'tail_under':r['ge20']['underprediction'],'tail_over':r['ge20']['overprediction']} for f,r in model.items()} for v,model in summary.items()}),flush=True)
 save('summary.json',all_results)
if __name__=='__main__':main()
