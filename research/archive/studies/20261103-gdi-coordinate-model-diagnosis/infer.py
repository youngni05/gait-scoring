"""Held-out replay of unchanged neural checkpoints on every frozen coordinate condition."""
from prepare import *
import time
frozen=module('frozen_gdi',ROOT/'gdi-research-harness/studies/20261101-self-video-gdi/predict.py')
torch=frozen.torch

def forward(model,family,x,lo,span):
 out=[];model.eval()
 with torch.inference_mode():
  for start in range(0,len(x),64):
   v=torch.from_numpy(np.array(x[start:start+64],copy=True))
   if family=='CNN':value=model(frozen.CNN.channels(v).reshape(-1,12,124))[:,0]
   else:value=model(v[...,:8].reshape(-1,124,4,2))
   out.append((value*float(span)+float(lo)).reshape(-1,2).numpy())
 return np.concatenate(out).astype(float)

def main():
 started=time.monotonic();prep=read(H/'preparation.json');assert sha(H/'protocol.json')==prep['protocol_sha256'];d=np.load(B/'summaries.npz');owner=d['window_record'];fold=d['fold'];conditions=read(H/'protocol.json')['conditions']
 for name in conditions:assert sha(B/'inputs'/f'{name}.npy')==prep['input_sha256'][name]
 torch.set_num_threads(4);torch.manual_seed(20261103);torch.use_deterministic_algorithms(True)
 manifest=read(ROOT/'gdi-research-harness/studies/20261102-final223-frozen/models.json');members=manifest['members'];saved={};audit=[]
 for family in ['CNN','STT']:
  for condition in conditions:saved[family+'__'+condition]=np.full((len(owner),2),np.nan)
  for f in range(5):
   member=next(m for m in members if m['family']==family and m['fold']==f and m.get('tag')=='refit');path=Path(member['checkpoint'].replace('C:\\Users\\jun\\Desktop\\D_P','D:\\Projects\\D_P'));assert sha(path)==member['checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=True)
   tr=fold!=f;te=fold==f;assert set(ck['identity']['train_people'])==set(d['person_id'][tr]);assert set(ck['identity']['outer_evaluation_people'])==set(d['person_id'][te]);assert set(d['person_id'][tr]).isdisjoint(d['person_id'][te])
   cache=np.load(member['cache']);assert sha(member['cache'])==member['cache_sha256'];assert np.array_equal(cache['y'],d['y']) and np.array_equal(cache['video_id'],d['video_id']);assert np.array_equal(cache['train_mask'],tr) and np.array_equal(cache['validation_mask'],te)
   coef=np.array(member['calibration']);assert np.allclose(coef,cache['calibration'][0]);window_ix=np.flatnonzero(te[owner]);model=frozen.new_model(family);model.load_state_dict(ck['model'],strict=True)
   record=dict(family=family,fold=f,checkpoint=str(path),sha256=sha(path),train_people=sorted(set(d['person_id'][tr])),test_people=sorted(set(d['person_id'][te])),calibration=coef.tolist(),conditions={})
   for condition in conditions:
    x=np.load(B/'inputs'/f'{condition}.npy',mmap_mode='r');raw=forward(model,family,x[window_ix],ck['lo'],ck['span']);saved[family+'__'+condition][window_ix]=raw*coef[0]+coef[1]
    if condition=='baseline':
     err=float(np.max(np.abs(raw-cache['window_raw'][window_ix])));assert err<2e-4,(family,f,err);record['baseline_replay_max_error']=err
    record['conditions'][condition]=dict(windows=len(window_ix),first_indices=window_ix[:3].tolist(),first_calibrated=saved[family+'__'+condition][window_ix[:3]].tolist())
    print(family,'fold',f,condition,'elapsed',round(time.monotonic()-started),flush=True)
   audit.append(record)
  np.savez_compressed(B/f'{family}-window-predictions.npz',**{k:v for k,v in saved.items() if k.startswith(family+'__')})
 aggregate={k:np.stack([v[owner==i].mean(0) for i in range(len(fold))]) for k,v in saved.items()}
 assert all(np.isfinite(p).all() for p in aggregate.values());np.savez_compressed(B/'neural-record-predictions.npz',**aggregate)
 save(H/'neural-audit.json',audit);save(H/'neural-completion.json',dict(protocol_sha256=sha(H/'protocol.json'),predictions_sha256=sha(B/'neural-record-predictions.npz'),window_sha256={family:sha(B/f'{family}-window-predictions.npz') for family in ['CNN','STT']},elapsed_seconds=time.monotonic()-started,new_neural_training_steps=0))
 print('Neural inference completed',flush=True)
if __name__=='__main__':main()
