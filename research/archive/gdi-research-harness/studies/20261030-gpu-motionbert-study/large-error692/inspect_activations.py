"""Non-mutating forward hooks: localize the excessive response, not causal proof."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import torch
H=Path(__file__).resolve().parent;S=H.parent;C=S/'cnn-multiwindow695';R=S/'cnn-record-loss692'
sys.path.insert(0,str(C/'bundle'))
from model import GdiCNN,channels
torch.set_num_threads(4)
report=json.loads((H/'analysis/diagnostics.json').read_text(encoding='utf-8'))
out=[]
for case in report['cases']:
    f=case['fold'];orig=case['original_row'];models={}
    with np.load(C/'records'/('%04d.npz'%orig)) as z:x=z['multi_x'].copy()
    for name,path in [('control',R/('analysis/window_control/unpacked/results/fold%d/refit/latest.pt'%f)),('historical',C/('received/checkpoints/fold%d.pt'%f))]:
        ck=torch.load(path,map_location='cpu',weights_only=True);m=GdiCNN().double().eval();m.load_state_dict(ck['model'])
        layers={};handles=[]
        def hook(key):
            def collect(mod,args,value):layers[key]=value.detach().abs().reshape(len(x),2,-1).amax(-1).numpy().tolist()
            return collect
        for key,mod in m.named_modules():
            if key.startswith(('convs.','norms.')) or key in ('hidden','output'):handles.append(mod.register_forward_hook(hook(key)))
        with torch.inference_mode():m(channels(torch.from_numpy(x).double()).reshape(-1,12,124))
        for h in handles:h.remove()
        bn=[]
        for norm in m.norms:
            variance=norm.running_var.numpy();gain=np.abs(norm.weight.detach().numpy())/np.sqrt(variance+norm.eps)
            bn.append(dict(running_var_min=float(variance.min()),running_var_median=float(np.median(variance)),gain_max=float(gain.max()),gain_median=float(np.median(gain))))
        models[name]=dict(batchnorm=bn,absolute_activation_max_by_window_and_side=layers)
    out.append(dict(rank=case['rank'],row=case['row'],starts0=case['starts0'],models=models))
(H/'analysis/activations.json').write_text(json.dumps(dict(cases=out,diagnostics_sha256=hashlib.sha256((H/'analysis/diagnostics.json').read_bytes()).hexdigest(),
    interpretation='Descriptive forward trace; no layer intervention and no claim of isolated causality'),indent=2)+'\n',encoding='utf-8')
print(json.dumps(out[0]),flush=True)
