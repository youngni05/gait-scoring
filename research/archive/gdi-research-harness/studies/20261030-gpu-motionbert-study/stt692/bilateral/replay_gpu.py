"""Read pilot checkpoint, replay on the same allocated GPU, write evidence only."""
import os,sys,json
from pathlib import Path
root=Path('/abr/RESEARCH_USER/gdi-stt692-bilateral-20261031');sys.path.insert(0,str(root))
import run as r
import torch,numpy as np
r.setup(20261029,'cuda');d=r.data();tr,va,_=r.split(d,0,1);p=root/'runs/pilot'
ck=torch.load(p/'fit/latest.pt',map_location='cuda',weights_only=True)
m=r.new_model().cuda();m.load_state_dict(ck['model'],strict=True)
train=r.predict(m,d['x'][tr],ck['lo'],ck['span'],'cuda');coef=r.calibration(train,d['y'][tr])
raw=r.predict(m,d['x'][va],ck['lo'],ck['span'],'cuda');pred=r.corrected(raw,coef)
with np.load(p/'fit/raw_predictions/epoch0003.npz') as z:train_error=float(abs(train-z['train_raw']).max())
with np.load(p/'validation_predictions.npz') as z:
    raw_error=float(abs(raw-z['raw']).max());corrected_error=float(abs(pred-z['predictions']).max())
assert train_error==raw_error==corrected_error==0
result=dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],checkpoint_sha256=r.sha(p/'fit/latest.pt'),train_raw_max_error=train_error,raw_max_error=raw_error,corrected_max_error=corrected_error,real_optimizer_steps=0)
r.save(root/'control/gpu_checkpoint_replay.json',result);print(json.dumps(result))
