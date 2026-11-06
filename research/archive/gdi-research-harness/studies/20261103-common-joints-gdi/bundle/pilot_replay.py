import os,sys,json,argparse
from pathlib import Path
if os.name=='nt':sys.path[:0]=['D:/Projects/D_P/outputs/motionbert-smoke-20261015/deps']
import numpy as np,torch
import run
from model import GdiCNN,channels
from settings import NF,NT,VARIANT
ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args();d=run.data();torch.set_num_threads(4)
ck=torch.load(a.folder/'pilot/latest.pt',map_location='cpu',weights_only=True);m=GdiCNN().double();m.load_state_dict(ck['model']);m.eval()
# Fixed rows from training people, never outer evaluation.
allowed=np.flatnonzero((d['fold']!=0)&(d['fold']!=1));rows=allowed[:3];ix=np.array([np.flatnonzero(d['x'].owner==r)[0] for r in rows]);x=channels(torch.from_numpy(d['x'].values[ix])).reshape(-1,NF,NT).double()
coef=np.array(ck['history'][-1]['calibration']);lo=float(ck['lo']);span=float(ck['span'])
with torch.inference_mode():cpu=(m(x)[:,0]*span+lo).numpy().reshape(-1,2)*coef[0]+coef[1]
if (a.folder/'replay.npz').exists():
 with np.load(a.folder/'replay.npz') as z:
  assert np.array_equal(z['window_indices'],ix);error=float(np.max(np.abs(cpu-z['gpu64'])));assert error<1e-4,(VARIANT,error)
 run.save(a.folder/'local-replay.json',dict(passed=True,max_error=error,checkpoint_sha256=run.sha(a.folder/'pilot/latest.pt'),variant=VARIANT))
 print(VARIANT,'local replay',error)
else:
 m=m.cuda()
 with torch.inference_mode():gpu=(m(x.cuda())[:,0]*span+lo).cpu().numpy().reshape(-1,2)*coef[0]+coef[1]
 error=float(np.max(np.abs(cpu-gpu)));assert error<1e-4
 np.savez_compressed(a.folder/'replay.npz',gpu64=gpu,cpu64=cpu,window_indices=ix,rows=rows)
 run.save(a.folder/'replay.json',dict(passed=True,max_error=error,checkpoint_sha256=run.sha(a.folder/'pilot/latest.pt'),variant=VARIANT))
