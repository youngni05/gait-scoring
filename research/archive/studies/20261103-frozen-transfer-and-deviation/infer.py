from prepare import *
import time
prior=module('prior_frozen',ROOT/'gdi-research-harness/studies/20261101-self-video-gdi/predict.py')
torch=prior.torch
def relocated(s):
 p=Path(s);s=str(p).replace('C:\\Users\\jun\\Desktop\\D_P','D:\\Projects\\D_P');return Path(s)
def main():
 proof=read(HERE/'preparation.json');assert sha(HERE/'protocol.json')==proof['protocol_sha256'] and sha(BIG/'features.npz')==proof['features_sha256']
 manifest_path=ROOT/'gdi-research-harness/studies/20261102-final223-frozen/models.json';manifest=read(manifest_path)
 torch.set_num_threads(4);torch.manual_seed(20261103);torch.use_deterministic_algorithms(True)
 with np.load(BIG/'features.npz') as z:x=z['x'];owner=z['owner']
 replay_x=np.load('D:/gait-research-data/D_P/self-video-gdi-20261101/features.npz')['x'][:3];expected=np.load('D:/gait-research-data/D_P/self-video-gdi-20261101/predictions.npz')
 result={};checks=[];start=time.monotonic();(BIG/'members').mkdir(exist_ok=True)
 for i,m in enumerate(manifest['members']):
  family=m['family'];fold=m['fold'];key=f'bilateral_{fold}' if family=='bilateral' else f'{family}_{fold}_{m["tag"]}'
  p=relocated(m['checkpoint']);assert sha(p)==m['checkpoint_sha256']
  out=BIG/'members'/f'{key}.npz';record=out.with_suffix('.json')
  if out.exists() and record.exists():
   check=read(record);assert check['features_sha256']==proof['features_sha256'] and check['prediction_sha256']==sha(out) and check['checkpoint_sha256']==m['checkpoint_sha256'];pred=np.load(out)['prediction']
  else:
   ck=torch.load(p,map_location='cpu',weights_only=True);model=prior.new_model(family);model.load_state_dict(ck['model'],strict=True)
   a,b=m['calibration'];replay=prior.forward(model,family,replay_x,ck['lo'],ck['span'])*a+b
   error=float(np.max(np.abs(replay-expected[key][:3])));assert error<1e-4,(key,error)
   pred=prior.forward(model,family,x,ck['lo'],ck['span'])*a+b;assert np.isfinite(pred).all()
   np.savez_compressed(out,prediction=pred)
   check=dict(key=key,checkpoint=str(p),checkpoint_sha256=m['checkpoint_sha256'],features_sha256=proof['features_sha256'],prediction_sha256=sha(out),prior_replay_max=error,optimizer_steps=0);save(record,check)
  result[key]=pred;checks.append(check);print('member',i+1,len(manifest['members']),key,'elapsed',round(time.monotonic()-start,1),flush=True)
 for f,s in enumerate(manifest['selections']):
  for family in ['CNN','STT']:
   result[f'{family}_{f}_bag']=np.mean([result[f'{family}_{f}_{m["tag"]}'] for m in manifest['members'] if m['family']==family and m['fold']==f],axis=0)
  base=s['cnn_weight']*result[f'CNN_{f}_bag']+(1-s['cnn_weight'])*result[f'STT_{f}_bag'];avg=base.mean(1);diff=(1-s['fraction'])*(base[:,0]-base[:,1])+s['fraction']*(result[f'bilateral_{f}'][:,0]-result[f'bilateral_{f}'][:,1]);result[f'fusion_{f}']=np.column_stack([avg+diff/2,avg-diff/2])
 for name,pattern in [('CNN_refit','CNN_{}_refit'),('CNN_bag','CNN_{}_bag'),('fusion','fusion_{}')]:result[name]=np.mean([result[pattern.format(f)] for f in range(5)],axis=0)
 np.savez_compressed(BIG/'frozen-predictions.npz',owner=owner,**result)
 save(HERE/'frozen-verification.json',dict(checks=checks,models_manifest_sha256=sha(manifest_path),protocol_sha256=sha(HERE/'protocol.json'),features_sha256=proof['features_sha256'],predictions_sha256=sha(BIG/'frozen-predictions.npz'),windows=len(x),optimizer_steps=0,labels_loaded=False))
if __name__=='__main__':main()
