"""Evaluate existing inner base models on their still-excluded outer people."""
import argparse,time
from core import *
import torch
p=argparse.ArgumentParser();p.add_argument('family',choices=['CNN','STT']);a=p.parse_args()
assert read(O/'training.json')['complete']
bundle=S/('stt692/multiwindow/bundle' if a.family=='STT' else 'cnn-multiwindow695/bundle')
sys.path.insert(0,str(bundle));sys.path.insert(0,str(bundle/'vendor'))
import run as r
r.setup(20261031,'cpu');d=arrays(bundle/'data.npz')
for name,digest in read(bundle/'manifest.json').items():assert sha(bundle/name)==digest
for f in range(5):
 rows=np.flatnonzero(d['fold']==f);keep=np.isin(d['window_record'],rows);x=d['x'][keep]
 for j in range(5):
  if j==f:continue
  src=OLD/a.family/('fold%d_inner%d.npz'%(f,j));proof=read(src.with_suffix('.json'));assert sha(src)==proof['cache_sha256']
  ckpath=Path(proof['checkpoint']);assert sha(ckpath)==proof['checkpoint_sha256'];ck=torch.load(ckpath,map_location='cpu',weights_only=True)
  disjoint(ck['identity']['train_people'],ck['identity']['validation_people'],np.unique(d['person_id'][rows]))
  assert ck['identity']['outer_evaluation_people']==np.unique(d['person_id'][rows]).tolist()
  dest=CACHE/'outer'/a.family/('fold%d_inner%d.npz'%(f,j));receipt=dest.with_suffix('.json')
  if receipt.exists():
   old=read(receipt);assert old['checkpoint_sha256']==proof['checkpoint_sha256'] and sha(dest)==old['sha256'];continue
  m=r.new_model() if a.family=='STT' else r.GdiCNN();m.load_state_dict(ck['model'],strict=True);m.eval();batch=64 if a.family=='STT' else 128
  def infer(values):
   out=[]
   with torch.inference_mode():
    for start in range(0,len(values),batch):
     v=torch.from_numpy(values[start:start+batch])
     q=m(v.reshape(-1,124,4,2)) if a.family=='STT' else m(r.channels(v).reshape(-1,12,124))[:,0]
     out.append((q*float(ck['span'])+float(ck['lo'])).reshape(-1,2).numpy())
   return np.concatenate(out)
  start=time.monotonic();raw=infer(x)
  # Verify a changed batching/order gives the same first/last window predictions.
  indices=np.r_[np.arange(min(13,len(x))),np.arange(max(13,len(x)-13),len(x))][::-1]
  err=float(np.abs(infer(x[indices])-raw[indices]).max());assert err<1e-4
  dest.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(dest,window_raw=raw,window_record=d['window_record'][keep],window_start=d['window_start'][keep],row_index=rows,
      person_id=d['person_id'][rows],video_id=d['video_id'][rows],fold=d['fold'][rows])
  save(receipt,dict(sha256=sha(dest),checkpoint_sha256=proof['checkpoint_sha256'],checkpoint=str(ckpath),source_data_sha256=sha(bundle/'data.npz'),
      protocol_sha256=sha(H/'protocol.json'),batch_order_replay_max=err,seconds=time.monotonic()-start))
  print(a.family,f,j,'seconds',time.monotonic()-start,flush=True)
