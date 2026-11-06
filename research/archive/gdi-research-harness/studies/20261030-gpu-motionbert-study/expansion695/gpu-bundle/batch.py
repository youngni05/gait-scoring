"""Sequential queue and paired person-level aggregation; no automatic hyperparameter search."""
import argparse,subprocess,sys
from pathlib import Path
import numpy as np
import run

def job_name(variant,seed,fold): return f'{variant}_seed{seed}_fold{fold}'
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['list','run','summarize']);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
 cfg=run.protocol(a.config);d=run.data();root=Path(a.output).resolve();jobs=[(v,s,f) for v in cfg['variants'] for s in cfg['seeds'] for f in cfg['folds']]
 if a.action in ('list','run'):
  for v,s,f in jobs:
   cmd=[sys.executable,'-B',str(run.ROOT/'run.py'),'train','--config',str(Path(a.config).resolve()),'--variant',v,'--seed',str(s),'--fold',str(f),'--device',a.device,'--output',str(root/job_name(v,s,f))]
   if a.action=='list': print(subprocess.list2cmdline(cmd))
   else: subprocess.run(cmd,check=True)
  return
 ids=d['person_id'];people=np.unique(ids)
 def person(a): return np.array([a[ids==i].mean(axis=0) for i in people])
 def person_metrics(pred):
  return {'front_mae':person(abs(pred[:,0]-d['y']).mean(1))}
 with np.load(run.ROOT/'baseline_predictions.npz',allow_pickle=False) as z:baseline=z['predictions'].copy()
 values={'baseline':{k:x[None] for k,x in person_metrics(baseline).items()}};metrics={};rng=np.random.default_rng(20261029);draw=rng.integers(0,len(people),(10000,len(people)))
 for v in cfg['variants']:
  allp=[]
  for seed in cfg['seeds']:
   pred=np.full((1064,1,2),np.nan);covered=[]
   for fold in cfg['folds']:
    out=root/job_name(v,seed,fold);e=run.load(out/'complete.json');identity=e['identity']
    if identity['config']!=run.sha(a.config) or identity['manifest']!=run.sha(run.ROOT/'manifest.json') or identity['variant']!=v or identity['fold']!=fold or identity['seed']!=seed or e['steps']!=cfg['steps']: raise ValueError('mismatched job')
    if run.sha(out/'predictions.npz')!=e['predictions_sha256']: raise ValueError('prediction file changed')
    with np.load(out/'predictions.npz',allow_pickle=False) as z:
     ix=z['indices'];assert np.array_equal(ix,np.flatnonzero(d['fold']==fold));pred[ix]=z['predictions'];covered.extend(ix.tolist())
   if sorted(covered)!=list(range(1064)) or not np.isfinite(pred).all(): raise ValueError('incomplete folds')
   allp.append(pred)
  per=[person_metrics(pred) for pred in allp];values[v]={k:np.stack([x[k] for x in per]) for k in per[0]}
  metrics[v]={'means':{k:float(x.mean()) for k,x in values[v].items()},'seed_means':{k:x.mean(1).tolist() for k,x in values[v].items()},'person_prediction_sd_per_seed':[float(person(pred.mean(1)).mean(1).std(ddof=1)) for pred in allp],'result_provision_rate':1.0}
 metrics['baseline']={'means':{k:float(x.mean()) for k,x in values['baseline'].items()},'person_prediction_sd':float(person(baseline.mean(1)).mean(1).std(ddof=1))}
 comparisons={}
 for ref,v in [('baseline','head_only'),('head_only','tail'),('head_only','last_stage'),('tail','last_stage')]:
  comparisons[f'{ref}_minus_{v}']={}
  for k in values[ref]:
   delta=values[ref][k].mean(0)-values[v][k].mean(0)
   comparisons[f'{ref}_minus_{v}'][k]={'reduction':float(delta.mean()),'person_bootstrap95':np.percentile(delta[draw].mean(1),[2.5,97.5]).tolist()}
 run.save(root/'summary.json',{'people':695,'records':1064,'seeds':cfg['seeds'],'config_sha256':run.sha(a.config),'metrics':metrics,'comparisons':comparisons,'interpretation':'Expanded development only; front124 only; no back-window or final-test claim.'})
 print('Saved '+str(root/'summary.json'))
if __name__=='__main__':main()
