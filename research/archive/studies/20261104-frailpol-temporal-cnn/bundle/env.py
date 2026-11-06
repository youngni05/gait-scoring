"""SYNTHETIC data only. No participant signals, IDs, clinical labels or scores."""
import os,time,json,hashlib,tarfile
from pathlib import Path
import numpy as np
import torch
from train import Model,CONDS,gpu_environment,seed,save,sha
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4);seed(20261104);env=gpu_environment();out=ROOT/'checks';out.mkdir(exist_ok=True);runs=[]
for c in CONDS:
    nf=1 if c.endswith('speed') or c=='speed_mlp' else 213
    # Generated here, on the GPU server. No local human-data file is sent.
    x=torch.randn(64,24,128,device='cuda');f=torch.randn(64,nf,device='cuda');target=torch.randn(64,device='cuda');m=Model(c,nf).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(100):
        m.train();opt.zero_grad(set_to_none=True);p=m(x,f);loss=torch.nn.functional.smooth_l1_loss(p,target,beta=.5);assert torch.isfinite(loss);loss.backward();opt.step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;m.eval()
    with torch.no_grad():pred=m(x,f).cpu().numpy()
    state=out/(c+'.pt');torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},state);np.savez_compressed(out/(c+'.npz'),x=x.cpu().numpy(),f=f.cpu().numpy(),prediction=pred)
    cpu=Model(c,nf).double();cpu.load_state_dict(torch.load(state,map_location='cpu',weights_only=True));cpu.eval()
    with torch.no_grad():replay=cpu(x.cpu().double(),f.cpu().double()).numpy()
    assert np.allclose(pred,replay,atol=1e-6,rtol=1e-6);runs.append(dict(condition=c,nfeatures=nf,steps=100,seconds=elapsed,cpu64_max_delta=float(np.max(np.abs(pred-replay))),checkpoint_sha256=sha(state)));del m,cpu,opt,x,f,target;torch.cuda.empty_cache()
summary=dict(status='synthetic_environment_passed_pending_local_replay',data_origin='All input tensors and targets randomly generated on this GPU server. No real-person data, no clinical performance.',environment=env,runs=runs,training_library_sha256=sha(ROOT/'train.py'),env_code_sha256=sha(Path(__file__)))
save(out/'summary.json',summary)
with tarfile.open(ROOT/'environment-review.tar.gz','w:gz') as t:
    for name in ['checks','train.py','env.py']:t.add(ROOT/name,arcname=name)
save(ROOT/'environment-review.sha256.json',dict(sha256=sha(ROOT/'environment-review.tar.gz')));print(json.dumps(summary))
