"""CPU inference only; each inner export excludes its outer people entirely."""
import argparse,sys,time
import numpy as np
import torch
from common import *
p=argparse.ArgumentParser();p.add_argument('family',choices=['CNN','STT','control']);p.add_argument('stage',choices=['inner','outer']);a=p.parse_args()
assert a.family!='control' or a.stage=='outer'
if a.family!='control' and a.stage=='outer':assert read(O/'selection.json')['protocol_sha256']==sha(H/'protocol.json')
bundle=T/'bundle' if a.family=='STT' else C/'bundle'
sys.path.insert(0,str(bundle));sys.path.insert(0,str(bundle/'vendor'))
import run as r
r.setup(20261031,'cpu')
data=arrays(bundle/'data.npz');n=len(data['y']);assert n==1061 and len(np.unique(data['person_id']))==692
for name,digest in read(bundle/'manifest.json').items():assert sha(bundle/name)==digest
def paths(f,j):
    tag='refit' if j is None else 'inner%d'%j
    if a.family=='CNN':
        folder=C/('received/results/fold%d/%s'%(f,tag))
        ck=C/('received/checkpoints/fold%d.pt'%f) if j is None else E/('received/checkpoints/fold%d/inner%d/latest.pt'%(f,j))
    elif a.family=='STT':folder=T/('analysis/unpacked/results/fold%d/%s'%(f,tag));ck=folder/'latest.pt'
    else:folder=S/('cnn-record-loss692/analysis/window_control/unpacked/results/fold%d/%s'%(f,tag));ck=folder/'latest.pt'
    return ck,folder
for f in range(5):
    for j in ([None] if a.stage=='outer' else [j for j in range(5) if j!=f]):
        ckpath,folder=paths(f,j);cksha=sha(ckpath);dest=CACHE/a.family/('fold%d_%s.npz'%(f,'refit' if j is None else 'inner%d'%j));proofpath=dest.with_suffix('.json')
        if proofpath.exists():
            proof=read(proofpath);assert proof['checkpoint_sha256']==cksha and proof['cache_sha256']==sha(dest) and proof['protocol_sha256']==sha(H/'protocol.json')
            print('REUSE',a.family,f,j,flush=True);continue
        started=time.monotonic();ck=torch.load(ckpath,map_location='cpu',weights_only=True);done=read(folder/'fit_complete.json')
        assert cksha==done['checkpoint_sha256'] and ck['identity']==done['identity']==read(folder/'identity.json')
        tr=(data['fold']!=f)&(data['fold']!=j) if j is not None else data['fold']!=f
        va=data['fold']==j if j is not None else data['fold']==f
        for key,mask in [('train_people',tr),('outer_evaluation_people',data['fold']==f)]:assert ck['identity'][key]==np.unique(data['person_id'][mask]).tolist()
        if j is not None:assert ck['identity']['validation_people']==np.unique(data['person_id'][va]).tolist()
        assert not set(data['person_id'][tr])&set(data['person_id'][va])
        if j is not None:assert not (set(data['person_id'][tr])|set(data['person_id'][va]))&set(data['person_id'][data['fold']==f])
        expected_lo=float(np.float32(data['y'][tr].min()));expected_span=float(np.float32(data['y'][tr].max()-data['y'][tr].min()))
        assert float(ck['lo'])==expected_lo and abs(float(ck['span'])-expected_span)<1e-5
        rows=np.flatnonzero(tr|va);keep=np.isin(data['window_record'],rows);owner=data['window_record'][keep];x=data['x'][keep]
        m=r.new_model() if a.family=='STT' else r.GdiCNN();m.load_state_dict(ck['model'],strict=True);m.eval()
        # The established CPU float64 reference is needed for packed control's extreme prediction.
        dtype=torch.float64 if a.family=='control' else torch.float32
        m=m.to(dtype=dtype);out=[]
        with torch.inference_mode():
            for start in range(0,len(x),64 if a.family=='STT' else 128):
                v=torch.from_numpy(x[start:start+(64 if a.family=='STT' else 128)]).to(dtype=dtype)
                if a.family=='STT':value=m(v.reshape(-1,124,4,2))
                else:value=m(r.channels(v).reshape(-1,12,124))[:,0]
                out.append((value*float(ck['span'])+float(ck['lo'])).reshape(-1,2).numpy())
        raw=np.concatenate(out);agg=aggregate(raw,owner,rows);trainpos=tr[rows];valpos=va[rows]
        coef=np.array([fit_calibration(pred[trainpos],data['y'][tr]) for pred in agg]);corrected=agg*coef[:,None,0,None]+coef[:,None,1,None]
        mean_stored=agg[0]*ck['history'][-1]['calibration'][0]+ck['history'][-1]['calibration'][1]
        calibration_delta=float(np.abs(corrected[0]-mean_stored).max());assert calibration_delta<1e-4,('mean calibration replay',a.family,f,j,calibration_delta)
        if j is not None:
            delta=abs(errors(data['y'][va],mean_stored[valpos],data['person_id'][va]).mean()-ck['history'][-1]['validation_person_mae'])
            assert delta<1e-4
        else:
            ref=arrays(folder.parent/'predictions.npz');np.testing.assert_array_equal(ref['row_index'],np.flatnonzero(va))
            delta=float(np.abs(mean_stored[valpos]-ref['predictions'][:,0]).max());assert delta<1e-4,('mean replay',a.family,f,delta)
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,window_raw=raw,window_record=owner,window_start=data['window_start'][keep],row_index=rows,
            aggregate_raw=agg,corrected=corrected,calibration=coef,train_mask=trainpos,validation_mask=valpos,
            **{k:data[k][rows] for k in ('y','person_id','video_id','fold')})
        proof=dict(family=a.family,outer=f,inner=j,checkpoint=str(ckpath),checkpoint_sha256=cksha,
            source_files={str(q):sha(q) for q in (folder/'identity.json',folder/'fit_complete.json',bundle/'data.npz')},
            cache_sha256=sha(dest),protocol_sha256=sha(H/'protocol.json'),mean_replay_delta=float(delta),mean_calibration_delta=calibration_delta,
            train_people=len(np.unique(data['person_id'][tr])),validation_people=len(np.unique(data['person_id'][va])),
            outer_predictions_opened=j is None,precision=str(dtype),seconds=time.monotonic()-started,optimizer_steps=0)
        save(proofpath,proof);print(json.dumps({k:proof[k] for k in ('family','outer','inner','seconds','mean_replay_delta')}),flush=True)
