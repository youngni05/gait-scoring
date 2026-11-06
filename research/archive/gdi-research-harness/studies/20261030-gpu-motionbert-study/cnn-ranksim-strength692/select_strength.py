"""Select regularizer strength and epoch using only outer-training inner curves."""
from pathlib import Path
import numpy as np
import run as r
from controls import verify_reuse
def choose(candidates):
 assert candidates and all(np.isfinite(q['score']) for q in candidates)
 score=min(q['score'] for q in candidates)
 return min((q for q in candidates if q['score']<=score+1e-12),key=lambda q:(q['strength'],q['epoch']))
def main():
 verify_reuse(r);d=r.data();old=Path(r.config()['reuse_control_remote']);rows=[]
 p=np.full_like(d['y'],np.nan);seen=np.zeros(len(p),int)
 arms=[(0.,old/'runs/main/mse'),(.1,r.ROOT/'runs/main/rank01'),(.3,r.ROOT/'runs/main/rank03'),(1.,old/'runs/main/ranksim')]
 for f in range(5):
  candidates=[]
  for strength,root in arms:
   folder=root/('fold%d'%f);hist=[r.load(folder/('inner%d'%j)/'curve.json') for j in range(5) if j!=f]
   epoch,means=r.select_epoch(hist);score=float(means[epoch-1])
   done=r.load(folder/'complete.json');assert done['selected_epoch']==epoch
   assert r.sha(folder/'predictions.npz')==done['prediction_sha256']
   candidates.append(dict(strength=strength,epoch=epoch,score=score,folder=str(folder),prediction_sha256=done['prediction_sha256']))
  best=choose(candidates)
  with np.load(Path(best['folder'])/'predictions.npz') as z:
   ix=z['row_index'];np.testing.assert_array_equal(ix,np.flatnonzero(d['fold']==f))
   for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(z[k],d[k][ix])
   p[ix]=z['predictions'][:,0];seen[ix]+=1
  rows.append(dict(outer=f,selected=best,candidates=candidates,outer_labels_used=False))
 assert (seen==1).all() and np.isfinite(p).all()
 np.savez_compressed(r.ROOT/'runs/selected_oof.npz',predictions=p[:,None,:],**{k:d[k] for k in ['y','person_id','video_id','fold','cohort']})
 r.save(r.ROOT/'runs/strength_selection.json',dict(folds=rows,OOF_sha256=r.sha(r.ROOT/'runs/selected_oof.npz'),metrics=r.metrics(d['y'],p,d['person_id']),selection_uses_outer_labels=False,control_manifest_sha256=r.config()['reuse_control_manifest_sha256']))
if __name__=='__main__':main()
