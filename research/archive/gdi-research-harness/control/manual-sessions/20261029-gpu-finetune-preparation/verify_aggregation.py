from pathlib import Path
import sys,json,types
R=Path('C:/Users/jun/Desktop/D_P');S=R/'gdi-research-harness/control/manual-sessions/20261029-gpu-finetune-preparation';B=S/'bundle'
for p in [R/'outputs/motionbert-smoke-20261015/deps',R/'outputs/gdi-ridge-experiment-20261016/deps',B]:sys.path.insert(0,str(p))
import numpy as np
import run,batch
root=S/'FAKE_METRIC_FIXTURE_DO_NOT_USE';root.mkdir(exist_ok=False)
d=run.data();c=run.load(S/'cpu_integration_v2/SYNTHETIC_ONLY.json');config=root/'SYNTHETIC_ONLY.json';run.save(config,c)
with np.load(B/'baseline_predictions.npz') as z:pred=z['predictions'].copy()
for v in c['variants']:
 for seed in c['seeds']:
  for f in c['folds']:
   out=root/batch.job_name(v,seed,f);out.mkdir();ix=np.flatnonzero(d['fold']==f);np.savez_compressed(out/'predictions.npz',indices=ix,predictions=pred[ix])
   identity={'config':run.sha(config),'manifest':run.sha(B/'manifest.json'),'variant':v,'fold':f,'seed':seed}
   run.save(out/'complete.json',{'SYNTHETIC_TEST_FIXTURE':True,'identity':identity,'steps':c['steps'],'predictions_sha256':run.sha(out/'predictions.npz')})
sys.argv=['batch.py','summarize','--config',str(config),'--output',str(root)];batch.main();res=run.load(root/'summary.json')
for vals in res['comparisons'].values():
 for value in vals.values():assert value['reduction']==0. and value['person_bootstrap95']==[0.,0.]
for v in c['variants']:
 assert abs(res['metrics'][v]['means']['front_mae']-8.275834265385386)<1e-10
 assert abs(res['metrics'][v]['means']['front_back_gap']-3.960482771640348)<1e-10
run.save(S/'aggregation_checks.json',{'status':'passed','fixture_only':True,'new_model_performance_claim':False,'all_folds_required':True,'macro_person_not_macro_record':True,'identical_predictions_zero_delta_and_interval':True})
print('Aggregation fixture checks passed; no new model evaluated')
