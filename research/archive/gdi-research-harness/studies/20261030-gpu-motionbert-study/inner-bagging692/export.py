"""CPU inference only: inner model (f,j) predicts outer fold f people, whom it never trained on."""
import argparse,sys,time
import numpy as np
import torch
from common import *
p=argparse.ArgumentParser();p.add_argument('family',choices=['CNN','STT']);a=p.parse_args()
bundle=T/'bundle' if a.family=='STT' else C/'bundle'
sys.path.insert(0,str(bundle));sys.path.insert(0,str(bundle/'vendor'))
import run as r
r.setup(20261031,'cpu')
data=arrays(bundle/'data.npz');assert len(data['y'])==1061 and len(np.unique(data['person_id']))==692
for name,digest in read(bundle/'manifest.json').items():assert sha(bundle/name)==digest
def paths(f,j):
    if a.family=='CNN':folder=C/('received/results/fold%d/inner%d'%(f,j));ck=E/('received/checkpoints/fold%d/inner%d/latest.pt'%(f,j))
    else:folder=T/('analysis/unpacked/results/fold%d/inner%d'%(f,j));ck=folder/'latest.pt'
    return ck,folder
for f in range(5):
    for j in [j for j in range(5) if j!=f]:
        ckpath,folder=paths(f,j);cksha=sha(ckpath);dest=CACHE/a.family/('fold%d_inner%d.npz'%(f,j));proofpath=dest.with_suffix('.json')
        if proofpath.exists():
            proof=read(proofpath);assert proof['checkpoint_sha256']==cksha and proof['cache_sha256']==sha(dest) and proof['protocol_sha256']==sha(H/'protocol.json')
            print('REUSE',a.family,f,j,flush=True);continue
        started=time.monotonic();ck=torch.load(ckpath,map_location='cpu',weights_only=True);done=read(folder/'fit_complete.json')
        assert cksha==done['checkpoint_sha256'] and ck['identity']==done['identity']==read(folder/'identity.json')
        tr=(data['fold']!=f)&(data['fold']!=j);va=data['fold']==j;te=data['fold']==f
        assert ck['identity']['train_people']==np.unique(data['person_id'][tr]).tolist()
        assert ck['identity']['validation_people']==np.unique(data['person_id'][va]).tolist()
        assert not set(data['person_id'][tr])&set(data['person_id'][te])
        ref,refpath=racache(a.family,f,j);np.testing.assert_array_equal(ref['row_index'],np.flatnonzero(tr|va))
        rows=np.flatnonzero(va|te);keep=np.isin(data['window_record'],rows);owner=data['window_record'][keep];x=data['x'][keep]
        m=r.new_model() if a.family=='STT' else r.GdiCNN();m.load_state_dict(ck['model'],strict=True);m.eval();out=[]
        step=64 if a.family=='STT' else 128
        with torch.inference_mode():
            for start in range(0,len(x),step):
                v=torch.from_numpy(x[start:start+step]).to(dtype=torch.float32)
                value=m(v.reshape(-1,124,4,2)) if a.family=='STT' else m(r.channels(v).reshape(-1,12,124))[:,0]
                out.append((value*float(ck['span'])+float(ck['lo'])).reshape(-1,2).numpy())
        raw=np.concatenate(out).astype(np.float64);agg=np.stack([raw[owner==i].mean(0) for i in rows])
        coef=ref['calibration'][0];corrected=agg*coef[0]+coef[1]
        # Replaying the inner validation people must reproduce the stored inner predictions.
        vpos=va[rows];refv=ref['corrected'][0][ref['validation_mask']]
        replay=float(np.abs(corrected[vpos]-refv).max());assert replay<1e-4,('inner replay',a.family,f,j,replay)
        train_std=ref['aggregate_raw'][0][ref['train_mask']].std(0)
        tpos=te[rows];dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,row_index=rows[tpos],prediction=corrected[tpos],**{k:data[k][rows[tpos]] for k in ('y','person_id','video_id','fold')})
        proof=dict(family=a.family,outer=f,inner=j,checkpoint=str(ckpath),checkpoint_sha256=cksha,robust_cache=str(refpath),robust_cache_sha256=sha(refpath),
            cache_sha256=sha(dest),protocol_sha256=sha(H/'protocol.json'),inner_replay_max=replay,calibration=coef.tolist(),
            train_prediction_std=train_std.tolist(),excluded=bool((train_std<0.01).any()),outer_people=len(np.unique(data['person_id'][te])),
            seconds=time.monotonic()-started,optimizer_steps=0,precision='torch.float32')
        save(proofpath,proof);print(json.dumps({k:proof[k] for k in ('family','outer','inner','seconds','inner_replay_max','excluded')}),flush=True)
