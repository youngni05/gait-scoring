"""Replay fixed CP ensemble then infer camera simulations, never optimize it."""
from prepare import *
import time
prior=module('old_predict',ROOT/'gdi-research-harness/studies/20261101-self-video-gdi/predict.py')
torch=prior.torch
def relocated(s):return Path(str(s).replace('C:\\Users\\jun\\Desktop\\D_P','D:\\Projects\\D_P'))
def main():
 audit=read(HERE/'input-audit.json');assert sha(HERE/'protocol.json')==audit['protocol_sha256'];assert sha(BIG/'features.npz')==audit['features_sha256'];assert (HERE/'operational-note.json').exists()
 manifest_path=ROOT/'gdi-research-harness/studies/20261102-final223-frozen/models.json';manifest=read(manifest_path)
 torch.set_num_threads(4);torch.manual_seed(20261103);torch.use_deterministic_algorithms(True)
 z=np.load(BIG/'features.npz');x=z['x']
 replay_x=np.load('D:/gait-research-data/D_P/self-video-gdi-20261101/features.npz')['x'][:3]
 expected=np.load('D:/gait-research-data/D_P/self-video-gdi-20261101/predictions.npz')
 results={};checks=[];start=time.monotonic();(BIG/'members').mkdir(exist_ok=True)
 for i,m in enumerate(manifest['members']):
  family=m['family'];f=m['fold'];key=f'bilateral_{f}' if family=='bilateral' else f'{family}_{f}_{m["tag"]}'
  p=relocated(m['checkpoint']);assert sha(p)==m['checkpoint_sha256'];out=BIG/'members'/f'{key}.npz';record=out.with_suffix('.json')
  if out.exists() and record.exists():
   check=read(record);assert check['features_sha256']==audit['features_sha256'] and check['prediction_sha256']==sha(out) and check['checkpoint_sha256']==m['checkpoint_sha256'];pred=np.load(out)['prediction']
  else:
   ck=torch.load(p,map_location='cpu',weights_only=True);model=prior.new_model(family);model.load_state_dict(ck['model'],strict=True)
   a,b=m['calibration'];replay=prior.forward(model,family,replay_x,ck['lo'],ck['span'])*a+b
   error=float(np.max(np.abs(replay-expected[key][:3])));assert error<1e-4,(key,error)
   pred=prior.forward(model,family,x,ck['lo'],ck['span'])*a+b;assert np.isfinite(pred).all()
   np.savez_compressed(out,prediction=pred)
   check=dict(key=key,checkpoint=str(p),checkpoint_sha256=m['checkpoint_sha256'],features_sha256=audit['features_sha256'],prediction_sha256=sha(out),prior_replay_max=error,optimizer_steps=0);save(record,check)
  results[key]=pred;checks.append(check);print(i+1,len(manifest['members']),key,round(time.monotonic()-start,1),flush=True)
 for f,s in enumerate(manifest['selections']):
  for family in ['CNN','STT']:
   results[f'{family}_{f}_bag']=np.mean([results[f'{family}_{f}_{m["tag"]}'] for m in manifest['members'] if m['family']==family and m['fold']==f],axis=0)
  base=s['cnn_weight']*results[f'CNN_{f}_bag']+(1-s['cnn_weight'])*results[f'STT_{f}_bag'];avg=base.mean(1)
  diff=(1-s['fraction'])*(base[:,0]-base[:,1])+s['fraction']*(results[f'bilateral_{f}'][:,0]-results[f'bilateral_{f}'][:,1])
  results[f'fusion_{f}']=np.column_stack([avg+diff/2,avg-diff/2])
 results['fusion']=np.mean([results[f'fusion_{f}'] for f in range(5)],axis=0)
 np.savez_compressed(BIG/'frozen-predictions.npz',**results)
 rows=read(HERE/'trial-audit.json');people=read(HERE/'people.json');person_scores=[]
 for person in people:
  values={};trials={}
  for variant in VARIANTS:
   vals=[]
   for ri,r in enumerate(rows):
    ix=(z['owner']==ri)&(z['variant']==variant)
    if r['id']==person['id'] and ix.any():vals.append(float(results['fusion'][ix].mean()))
   if vals:values[variant]=float(np.mean(vals));trials[variant]=vals
  if values:person_scores.append(dict(id=person['id'],scores=values,trial_scores=trials))
 save(HERE/'gdi-scores.json',person_scores)
 save(HERE/'frozen-verification.json',dict(checks=checks,models_manifest_sha256=sha(manifest_path),protocol_sha256=sha(HERE/'protocol.json'),features_sha256=audit['features_sha256'],predictions_sha256=sha(BIG/'frozen-predictions.npz'),windows=len(x),optimizer_steps=0,SPPB_loaded_for_inference=False))
if __name__=='__main__':main()
