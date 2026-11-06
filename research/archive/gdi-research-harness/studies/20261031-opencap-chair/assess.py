from context import *
from analyze import metrics

def main():
 results={};prep=json.loads((STUDY/'preparation.json').read_text())
 if (STUDY/'prep_event/completion.json').exists():prep['cohorts'].update(json.loads((STUDY/'preparation-extension.json').read_text())['cohorts'])
 for cohort,info in prep['cohorts'].items():
  out=STUDY/cohort;assert json.loads((out/'completion.json').read_text())['complete'];p=pd.read_csv(out/'predictions.csv');p['error']=p.prediction-p.target;p['abs_error']=abs(p.error);p['under']=np.maximum(-p.error,0);rr=[]
  for (rep,seed,method),g in p.groupby(['repeat','seed','method']):
   for band,mask in [('all',np.ones(len(g),bool)),('lt20',g.target<20),('ge20',g.target>=20)]:
    h=g.loc[mask]
    if len(h):rr.append({'repeat':rep,'seed':seed,'method':method,'band':band,**metrics(h)})
  runs=pd.DataFrame(rr);runs.to_csv(out/'run-metrics.csv',index=False)
  persons=p.groupby(['id','method']).agg(target=('target','first'),prediction=('prediction','mean'),MAE=('abs_error','mean'),bias=('error','mean'),underprediction=('under','mean')).reset_index();persons.to_csv(out/'person-results.csv',index=False);persons[persons.target>=20].to_csv(out/'long-people.csv',index=False)
  summary={}
  for (method,band),g in runs.groupby(['method','band']):summary.setdefault(method,{})[band]={'people':int(g.people.iloc[0]),**{k:float(g[k].mean()) for k in ['MAE','bias','underprediction','RMSE']},'MAE_range':[float(g.MAE.min()),float(g.MAE.max())]}
  pairs=[('TurnContext','TurnBase')]+[(m,b) for m in ['ChairTime','ChairMotion','ChairAll','RiseMotion','PrepTime','PrepAll'] for b in ['TurnBase','TurnContext'] if m in summary]+([('ChairAll','ChairTime'),('ChairAll','ChairMotion')] if 'ChairAll' in summary else [])+([('PrepAll','PrepTime')] if 'PrepAll' in summary else [])
  comparisons={}
  for left,right in pairs:
   for band in ['all','ge20']:
    a=persons[(persons.method==left)&((persons.target>=20) if band=='ge20' else True)].set_index('id').sort_index();b=persons[persons.method==right].set_index('id').loc[a.index]
    ra=runs[(runs.method==left)&(runs.band==band)].set_index(['repeat','seed']);rb=runs[(runs.method==right)&(runs.band==band)].set_index(['repeat','seed']).loc[ra.index]
    q={}
    for metric in ['MAE','underprediction']:
     delta=(a[metric]-b[metric]).to_numpy();rng=np.random.default_rng(20261031);boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
     q[metric]={'difference':float(delta.mean()),'conditional_person_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'better_runs':int(sum(ra[metric]<rb[metric])),'runs':len(ra),'people_improved':int(sum(delta<0)),'people_worse':int(sum(delta>0))}
    comparisons[left+' minus '+right+' / '+band]=q
  results[cohort]={'models':summary,'comparisons':comparisons};save(cohort+'/summary.json',results[cohort]);print(cohort,json.dumps({m:{'all_MAE':v['all']['MAE'],'long_MAE':v['ge20']['MAE'],'long_under':v['ge20']['underprediction']} for m,v in summary.items()}),flush=True)
 save('summary.json',results)
if __name__=='__main__':main()
