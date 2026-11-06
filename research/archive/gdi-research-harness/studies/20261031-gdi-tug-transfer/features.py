"""Frozen GDI inference and penultimate features; never fit to TUG labels."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

ROOT=Path('D:/Projects/D_P')
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
BUNDLE=ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
sys.path.insert(0,str(BUNDLE))
from preprocess import extract_windows
from model import GdiCNN,channels
from infer import verify_bundle,predict_windows,configure

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def encode(x):
    verify_bundle();configure()
    meta=json.loads((BUNDLE/'models.json').read_text())
    all_features=[];all_scores=[];replays=[]
    for item in meta['folds']:
        ck=torch.load(BUNDLE/item['file'],map_location='cpu',weights_only=True)
        model=GdiCNN(meta['dropout']);model.load_state_dict(ck['model']);model.eval()
        seen=[]
        handle=model.output.register_forward_pre_hook(lambda layer,args:seen.append(args[0].detach().clone()))
        tx=channels(torch.from_numpy(x)).reshape(-1,12,124)
        with torch.inference_mode():
            prediction=(model(tx)[:,0]*ck['span']+ck['lo']).reshape(-1,2).numpy()
        handle.remove()
        features=seen[0].reshape(-1,2,10).numpy()
        # Reconstruct the output directly from exported features and final layer.
        with torch.inference_mode():
            replay=(model.output(torch.from_numpy(features).reshape(-1,10))[:,0]*ck['span']+ck['lo']).reshape(-1,2).numpy()
        np.testing.assert_allclose(prediction,replay,atol=1e-5,rtol=0)
        original=predict_windows(x,item['fold'],meta)
        np.testing.assert_allclose(prediction,original,atol=1e-5,rtol=0)
        a,b=item['calibration_slope_intercept']
        all_scores.append(prediction.astype(np.float64)*a+b);all_features.append(features)
        replays.append({'fold':item['fold'],'checkpoint_sha256':sha(BUNDLE/item['file']),
                        'max_replay_difference':float(np.max(np.abs(replay-prediction))),
                        'max_frozen_api_difference':float(np.max(np.abs(original-prediction)))})
    # Fold-specific neurons are not aligned; concatenate rather than average them.
    learned=np.concatenate([f.reshape(len(x),-1) for f in all_features],axis=1)
    score=np.stack(all_scores).mean(0)
    base=np.concatenate([x.mean(2),x.std(2),np.abs(np.diff(x,axis=2)).mean(2)],axis=2).reshape(len(x),-1)
    return base,score,learned,replays

def pilot():
    receipt=json.loads((OUT/'pilot-pose-receipt.json').read_text())
    folder=OUT/'body25/OAW06'
    chunks=json.loads((OUT/'decoded-chunks.json').read_text())
    hashes={}
    for name,item in chunks.items():
        if name.startswith('OAW06_'):hashes.update(item['json_sha256'])
    assert len(hashes)==124
    reference_errors=[]
    for frame in range(250,255):
        a=np.asarray(json.loads((OUT/'pilot_pose'/('%06d.json'%frame)).read_text())['people'])
        b=np.asarray(json.loads((folder/('%06d.json'%frame)).read_text())['people'])
        assert a.shape==b.shape
        err=float(np.abs(a-b).max());assert err<.01;reference_errors.append(err)
    raw=[];selected=[];previous=None
    for index in range(250,374):
        p=folder/('%06d.json'%index)
        assert sha(p)==hashes[p.name]
        people=json.loads(p.read_text())['people']
        candidates=[(i,np.asarray(a)) for i,a in enumerate(people) if a[8][2]>.1]
        if not candidates:
            raw.append(np.full(75,np.nan));selected.append(-1);continue
        # Visually identified participant is on the right in frame250; then
        # preserve trajectory continuity instead of trusting OpenPose ordering.
        if previous is None:i,a=max(candidates,key=lambda item:item[1][8,0])
        else:i,a=min(candidates,key=lambda item:np.linalg.norm(item[1][8,:2]-previous))
        assert previous is None or np.linalg.norm(a[8,:2]-previous)<192,'Tracking discontinuity needs review'
        previous=a[8,:2];selected.append(i);raw.append(a.reshape(-1))
    raw=np.stack(raw);d=extract_windows(raw)
    assert d['x'].shape==(1,2,124,12) and d['window_start'].tolist()==[0]
    base,score,learned,replays=encode(d['x'])
    np.savez_compressed(OUT/'pilot-features.npz',x=d['x'],raw=raw,selected_index=selected,base=base,gdi_score=score,gdi_features=learned)
    result={'passed':True,'people':1,'windows':1,'frames':124,'native_fps':30,
            'input_shape':list(d['x'].shape),'base_shape':list(base.shape),'gdi_score_shape':list(score.shape),'learned_shape':list(learned.shape),
            'selected_indices':selected,'missing_fraction':d['window_missing_fraction'].tolist(),'replays':replays,
            'official_CPU_reference_max_differences':reference_errors,
            'input_feature_sha256':sha(OUT/'pilot-features.npz'),
            'scope':'Input and feature-extraction feasibility only; frontal source, no TUG model fitting and no performance conclusion'}
    (OUT/'pilot-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('selected_indices','replays')},indent=2))

if __name__=='__main__':pilot()
