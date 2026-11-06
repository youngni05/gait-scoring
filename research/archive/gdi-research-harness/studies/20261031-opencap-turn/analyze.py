from common import *
def metrics(g):
 e=g.prediction.to_numpy()-g.target.to_numpy()
 return {'people':int(g.id.nunique()),'MAE':float(np.mean(abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'bias':float(np.mean(e)),'underprediction':float(np.mean(np.maximum(-e,0)))}
def main():
 results={}
 cohorts=['event_matched','fixed6','event_flagged']+(['geometry'] if (HERE/'geometry/completion.json').exists() else [])
 for cohort in cohorts:
  out=HERE/cohort;p=pd.read_csv(out/'predictions.csv');p['error']=p.prediction-p.target;p['abs_error']=abs(p.error);p['under']=np.maximum(-p.error,0)
  rr=[]
  for (r,s,m),g in p.groupby(['repeat','seed','method']):
   for name,mask in [('all',np.ones(len(g),bool)),('lt10',g.target<10),('10to20',(g.target>=10)&(g.target<20)),('ge20',g.target>=20)]:
    h=g.loc[mask]
    if len(h):rr.append({'repeat':r,'seed':s,'method':m,'band':name,**metrics(h)})
  runs=pd.DataFrame(rr);runs.to_csv(out/'run-metrics.csv',index=False)
  persons=p.groupby(['id','method']).agg(target=('target','first'),prediction=('prediction','mean'),MAE=('abs_error','mean'),bias=('error','mean'),underprediction=('under','mean')).reset_index();persons.to_csv(out/'person-results.csv',index=False)
  summary={}
  for (m,b),g in runs.groupby(['method','band']):
   summary.setdefault(m,{})[b]={'people':int(g.people.iloc[0]),'MAE':float(g.MAE.mean()),'MAE_range':[float(g.MAE.min()),float(g.MAE.max())],'bias':float(g.bias.mean()),'underprediction':float(g.underprediction.mean()),'RMSE':float(g.RMSE.mean())}
  comparisons={};pairs=[('Walk3Turn','Walk3Speed'),('Walk3Turn','WalkMatchedSpeed'),('Walk3Turn','WalkMatchedFull'),('WalkMatchedSpeed','Walk3Speed'),('Walk3Turn','Walk3TurnTime'),('Walk3TurnShape','WalkMatchedSpeed'),('Walk3Speed','OldWalk3Speed'),('Walk3Turn','OldWalk3Speed')]
  for a,b in pairs:
   if b not in summary or a not in summary:continue
   for band in ['all','ge20']:
    q=persons[(persons.method==a)&((persons.target>=20) if band=='ge20' else True)].set_index('id').sort_index();z=persons[persons.method==b].set_index('id').loc[q.index]
    delta=q.MAE.to_numpy()-z.MAE.to_numpy();rng=np.random.default_rng(20261031);means=[]
    for batch in range(10):means.extend(delta[rng.integers(0,len(delta),(1000,len(delta)))].mean(1).tolist())
    ra=runs[(runs.method==a)&(runs.band==band)].set_index(['repeat','seed']);rb=runs[(runs.method==b)&(runs.band==band)].set_index(['repeat','seed']).loc[ra.index]
    comparisons[a+' minus '+b+' / '+band]={'MAE_difference':float(delta.mean()),'conditional_person_bootstrap95':np.quantile(means,[.025,.975]).tolist(),'better_runs':int(np.sum(ra.MAE<rb.MAE)),'runs':len(ra),'people_improved':int(np.sum(delta<0)),'people_worse':int(np.sum(delta>0))}
  long=persons[persons.target>=20];long.to_csv(out/'long-people.csv',index=False)
  manifest=pd.DataFrame(json.loads((out/'manifest.json').read_text()));p2=p.merge(manifest[['id','task']],on='id',validate='many_to_one')
  task_summary={task:{m:metrics(g) for m,g in pg.groupby('method')} for task,pg in p2.groupby('task')}
  results[cohort]={'models':summary,'comparisons':comparisons,'task_subgroups_descriptive':task_summary}
  save(cohort+'/summary.json',results[cohort])
  print(cohort,json.dumps({m:{'all_MAE':v['all']['MAE'],'long_MAE':v['ge20']['MAE']} for m,v in summary.items()}),flush=True)
 save('summary.json',results)
if __name__=='__main__':main()
