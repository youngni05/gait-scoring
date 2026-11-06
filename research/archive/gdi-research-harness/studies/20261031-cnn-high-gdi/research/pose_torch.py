"""Weight-preserving BODY25 forward pass, checked against official Caffe outputs."""
from pathlib import Path
import json,time,argparse
import numpy as np
import torch
import torch.nn.functional as F

class Body25(torch.nn.Module):
    def __init__(self,folder):
        super().__init__();folder=Path(folder)
        self.graph=json.loads((folder/'network.json').read_text())
        with np.load(folder/'network_weights.npz') as z:
            for k in z.files:self.register_buffer(k,torch.from_numpy(z[k].copy()))
    def forward(self,x):
        values={'image':x}
        for r in self.graph:
            xs=[values[k] for k in r['bottom']];weights=[getattr(self,k) for k in r['weights']]
            kind=r['type']
            if kind=='Convolution': y=F.conv2d(xs[0],weights[0],weights[1].flatten(),r['stride'],r['pad'])
            elif kind=='PReLU': y=F.prelu(xs[0],weights[0].flatten())
            elif kind=='ReLU': y=F.relu(xs[0])
            elif kind=='Pooling': y=F.max_pool2d(xs[0],r['kernel'],r['stride'],r['pad'],ceil_mode=True)
            elif kind=='Concat': y=torch.cat(xs,dim=r['axis'])
            else: raise ValueError(kind)
            values[r['top']]=y
        return values['net_output']

def probe(folder,device):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net=Body25(folder).to(device).eval()
    z=np.load(Path(folder)/'native_arrays.npz')
    x=torch.from_numpy(z['blob']).to(device)
    with torch.inference_mode():
        for i in range(2):
            t=time.perf_counter();y=net(x).cpu().numpy()[0]
            print('seconds',time.perf_counter()-t,flush=True)
    error=np.abs(y-z['heatmaps'])
    result=dict(device=device,max_error=float(error.max()),mean_error=float(error.mean()))
    print(result,flush=True)
    assert error.max()<1e-4,'Official Caffe equivalence failed'
    np.savez_compressed(Path(folder)/('torch_'+device+'_arrays.npz'),heatmaps=y)
    (Path(folder)/('torch_'+device+'_check.json')).write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('folder');a.add_argument('--device',default='cpu')
    args=a.parse_args();probe(args.folder,args.device)
