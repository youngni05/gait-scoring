"""Synthetic integration test of training loop, stopped job, resume and final predictions."""
import argparse,types
from pathlib import Path
import numpy as np
import torch
import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(123);fake={'x':rng.normal(size=(5,2,4,17,3)).astype('float32'),'features':rng.normal(size=(5,2,512)).astype('float32'),'y':rng.normal(size=(5,2)),'fold':np.array([1,1,1,0,0])}
 original_data=run.data;run.data=lambda:fake
 cfg={'status':'ready','variants':list(run.VARIANTS),'folds':list(range(5)),'seeds':[123],'steps':2,'batch_size':2,'checkpoint_every':1,'head_lr':1e-6,'encoder_lr':1e-7,'weight_decay':0.,'loss':'record_equal_MSE_left_right','train_window':'front','evaluate_windows':['front','back'],'optimizer':'AdamW','dropout':'disabled_all_arms'}
 config=root/'SYNTHETIC_ONLY.json';run.save(config,cfg);rows=[]
 for variant in run.VARIANTS:
  args=types.SimpleNamespace(config=config,variant=variant,fold=0,seed=123,output=root/(variant+'_continuous'))
  run.train(args,torch.device('cpu'))
  original_checkpoint=run.checkpoint
  def interrupted(p,m,opt,done,identity):
   original_checkpoint(p,m,opt,done,identity)
   if done==1: raise InterruptedError('intentional synthetic interruption')
  args.output=root/(variant+'_resumed');run.checkpoint=interrupted
  try:run.train(args,torch.device('cpu'))
  except InterruptedError: pass
  else: raise AssertionError('interruption not exercised')
  finally:run.checkpoint=original_checkpoint
  assert not (args.output/'complete.json').exists()
  run.train(args,torch.device('cpu'))
  c=torch.load(root/(variant+'_continuous')/'checkpoint.pt',map_location='cpu',weights_only=True);r=torch.load(args.output/'checkpoint.pt',map_location='cpu',weights_only=True)
  assert c['done']==r['done']==2
  assert all(torch.equal(c['model'][k],r['model'][k]) for k in c['model'])
  with np.load(root/(variant+'_continuous')/'predictions.npz') as z:cp=z['predictions'].copy()
  with np.load(args.output/'predictions.npz') as z:rp=z['predictions'].copy()
  assert np.array_equal(cp,rp)
  digest=run.sha(args.output/'checkpoint.pt');run.train(args,torch.device('cpu'));assert run.sha(args.output/'checkpoint.pt')==digest
  rows.append({'variant':variant,'loop_resume_equal':True,'predictions_equal':True,'complete_job_skips_retraining':True})
 run.data=original_data
 # A final summary must reject missing fold jobs, not quietly report a partial run.
 import batch,sys
 old_argv=sys.argv;sys.argv=['batch.py','summarize','--config',str(config),'--output',str(root/'incomplete')]
 try:batch.main()
 except FileNotFoundError: pass
 else:raise AssertionError('missing folds accepted')
 finally:sys.argv=old_argv
 run.save(root/'integration_checks.json',{'synthetic_only':True,'real_data_optimizer_steps':0,'missing_fold_summary_rejected':True,'results':rows})
 print('Synthetic integration checks passed')
if __name__=='__main__':main()
