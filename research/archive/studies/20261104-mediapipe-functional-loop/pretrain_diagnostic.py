import sys,json,numpy as np,torch
from pathlib import Path
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');sys.path.insert(0,str(H));from mp_compare14 import Net,inputs,scaled
ids,ix,y,inp=inputs();root=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104/motion-pretrain/pelvis_stable_xy/masked_pretrain/fold0/inner0');ctx=json.loads((root/'context.json').read_text());rec=json.loads((root/'pretrain.json').read_text());x,m=inp['pelvis_stable_xy'];a=scaled(x,m,ctx['prep']);tr=np.array(ctx['train']);C=x.shape[-1];corruption=np.load(rec['probe'])['corruption'];noisy=a[tr].copy();noisy[:,:C]*=~corruption;noisy[:,C:]*=~corruption;state=torch.load(rec['model'],weights_only=True);old=np.load(rec['replay'])['prediction'];torch.set_num_threads(2);results=[]
for dt in [torch.float32,torch.float64]:
    model=Net(2*C).to(dt);model.conv.load_state_dict(state['encoder']);dec=torch.nn.Conv1d(16,C,1).to(dt);dec.load_state_dict(state['decoder']);model.eval();dec.eval()
    with torch.no_grad():out=dec(model.conv(torch.tensor(noisy,dtype=dt))).numpy()
    diff=np.abs(out-old);bad=diff>1e-6+1e-6*np.abs(old);results.append(dict(dtype=str(dt),max_delta=float(diff.max()),failed_points=int(bad.sum()),total=old.size,relative_normalized_max=float(np.max(diff/(1e-6+1e-6*np.abs(old))))))
(H/'motion-pretrain/numerical-diagnostic.json').write_text(json.dumps(results,indent=2));print(results)
