"""Mamba-1 official v2.2.4 class with official reference scan on CPU.

Only the class/function AST nodes are loaded: package-level CUDA/Triton imports
are not executed. This is a reference-backend feasibility test, not a GPU test.
"""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, math, platform, sys, time, urllib.request
from typing import Optional
H=Path(__file__).resolve().parent;ROOT=H.parents[2];S=H.parent/'20261030-gpu-motionbert-study'
sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
sys.path.insert(0,str(S/'stt692/multiwindow/bundle/vendor'))
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange,repeat

SRC=H/'mamba-sources';TAG='v2.2.4'
FILES={'mamba_simple.py':'mamba_ssm/modules/mamba_simple.py',
       'selective_scan_interface.py':'mamba_ssm/ops/selective_scan_interface.py',
       'README.md':'README.md','setup.py':'setup.py','LICENSE':'LICENSE'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def acquire():
    SRC.mkdir(exist_ok=True);rows=[]
    for name,relative in FILES.items():
        url=f'https://raw.githubusercontent.com/state-spaces/mamba/{TAG}/{relative}'
        request=urllib.request.Request(url,headers={'User-Agent':'D_P-research-feasibility'})
        with urllib.request.urlopen(request,timeout=25) as response:raw=response.read(2_000_000)
        assert 0<len(raw)<2_000_000
        p=SRC/name;p.write_bytes(raw);rows.append(dict(file=name,url=url,sha256=sha(p),bytes=len(raw)))
    save(SRC/'provenance.json',dict(tag=TAG,sources=rows))
    print(json.dumps(dict(downloaded_files=len(rows),bytes=sum(r['bytes'] for r in rows))))

def node(name,filename,kind,scope):
    p=SRC/filename;tree=ast.parse(p.read_text(encoding='utf8'))
    found=[n for n in tree.body if isinstance(n,kind) and n.name==name];assert len(found)==1
    exec(compile(ast.Module(body=found,type_ignores=[]),str(p),'exec'),scope)
    return scope[name]

def main():
    torch.set_num_threads(2);torch.manual_seed(20261101);start=time.perf_counter()
    proof=json.loads((SRC/'provenance.json').read_text(encoding='utf8'))
    for r in proof['sources']:assert sha(SRC/r['file'])==r['sha256']
    scope=dict(torch=torch,nn=nn,F=F,math=math,rearrange=rearrange,repeat=repeat,Optional=Optional,Tensor=Tensor,
               causal_conv1d_fn=None,causal_conv1d_update=None,selective_state_update=None,
               RMSNorm=None,layer_norm_fn=None,rms_norm_fn=None)
    scan=node('selective_scan_ref','selective_scan_interface.py',ast.FunctionDef,scope)
    scope['selective_scan_fn']=scan
    Mamba=node('Mamba','mamba_simple.py',ast.ClassDef,scope)
    # Independent loop against upstream real-valued selective scan, including all input gradients.
    u=torch.randn(2,3,9,requires_grad=True);dt=torch.randn(2,3,9,requires_grad=True)
    A=-torch.rand(3,4);B=torch.randn(2,4,9,requires_grad=True);C=torch.randn(2,4,9,requires_grad=True)
    D=torch.randn(3);z=torch.randn(2,3,9,requires_grad=True)
    actual=scan(u,dt,A,B,C,D,z=z,delta_softplus=True)
    state=torch.zeros(2,3,4);ys=[]
    for t in range(9):
        delta=F.softplus(dt[:,:,t]);state=torch.exp(delta[:,:,None]*A[None])*state+delta[:,:,None]*B[:,None,:,t]*u[:,:,t,None]
        ys.append(((state*C[:,None,:,t]).sum(-1)+D[None]*u[:,:,t])*F.silu(z[:,:,t]))
    expected=torch.stack(ys,dim=-1);scan_error=float((actual-expected).abs().max().detach())
    assert scan_error<1e-5
    inputs=(u,dt,B,C,z);ga=torch.autograd.grad(actual.square().mean(),inputs,retain_graph=True)
    gb=torch.autograd.grad(expected.square().mean(),inputs)
    grad_error=max(float((a-b).abs().max()) for a,b in zip(ga,gb));assert grad_error<1e-5
    model=Mamba(d_model=16,d_state=8,d_conv=4,expand=2,use_fast_path=False)
    x=torch.randn(2,124,16);out=model(x);assert out.shape==x.shape and torch.isfinite(out).all()
    # Verify the block can process a sequence incrementally with the same state.
    conv,ssm=model.allocate_inference_cache(2,124);steps=[]
    with torch.no_grad():
        for t in range(124):
            value,conv,ssm=model.step(x[:,t:t+1],conv,ssm);steps.append(value)
    step_error=float((out-torch.cat(steps,dim=1)).abs().max().detach());assert step_error<1e-5
    # Actual GDI training people only. Use native 124-frame coordinates and window-ordered summaries.
    source=S/'cnn-multiwindow695/bundle/data.npz'
    with np.load(source,allow_pickle=False) as d:
        train_rows=np.flatnonzero((d['fold']!=0)&(d['fold']!=1))
        rows=train_rows[:4];records=[];counts=[]
        for row in rows:
            keep=np.flatnonzero(d['window_record']==row);starts=d['window_start'][keep]
            assert len(keep) and np.all(np.diff(starts)>0)
            records.append(d['x'][keep]);counts.append(len(keep))
        coordinates=torch.from_numpy(records[0][0,:,:,:8].reshape(2,124,8).copy())
        # Dataset shape is [window,side,time,feature]. First eight are four joint xy values.
        tokens=[torch.from_numpy(r.mean(axis=(1,2)).copy()) for r in records]
        selected_people=d['person_id'][rows].tolist()
    coordinate_proj=nn.Linear(8,16);token_proj=nn.Linear(12,16);head=nn.Linear(16,2)
    params=list(model.parameters())+list(coordinate_proj.parameters())+list(token_proj.parameters())+list(head.parameters())
    optimizer=torch.optim.AdamW(params,lr=1e-3)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()};timings=[]
    for _ in range(3):
        t=time.perf_counter();optimizer.zero_grad()
        p=head(model(coordinate_proj(coordinates)).mean(1))
        # Numerical smoke objective, not GDI learning and not a reported prediction score.
        loss=p.square().mean()+sum(model(token_proj(tok)[None]).square().mean() for tok in tokens)/len(tokens)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        optimizer.step();timings.append(time.perf_counter()-t)
    assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
    with torch.no_grad():reference=model(coordinate_proj(coordinates)).clone()
    ck=H/'analysis/mamba-reference-smoke.pt';ck.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(),ck);other=Mamba(d_model=16,d_state=8,d_conv=4,expand=2,use_fast_path=False)
    other.load_state_dict(torch.load(ck,weights_only=True));other.eval()
    with torch.no_grad():replay=float((reference-other(coordinate_proj(coordinates))).abs().max())
    assert replay==0
    # Representative, small CPU reference timings: these are not CUDA performance estimates.
    benchmark=[]
    for length in [12,124,500]:
        v=torch.randn(8,length,16);runs=[]
        for k in range(4):
            t=time.perf_counter();model(v).square().mean().backward();model.zero_grad(set_to_none=True)
            if k:runs.append(time.perf_counter()-t)
        benchmark.append(dict(batch=8,length=length,dim=16,median_forward_backward_seconds=float(np.median(runs))))
    result=dict(passed=True,backend='official Mamba v2.2.4 class + official selective_scan_ref; CPU; fast path disabled',
        python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda_available=torch.cuda.is_available(),
        installed_modules={m:bool(importlib.util.find_spec(m)) for m in ['mamba_ssm','triton','causal_conv1d']},
        parameters=sum(p.numel() for p in model.parameters()),scan_forward_max_error=scan_error,
        scan_gradient_max_error=grad_error,streaming_full_max_error=step_error,saved_model_replay_error=replay,
        actual_coordinate_shape=list(coordinates.shape),window_token_shapes=[list(t.shape) for t in tokens],
        actual_training_people=selected_people,window_counts=counts,smoke_updates=3,
        smoke_objective='squared numerical outputs only; clinical labels not used',smoke_step_seconds=timings,
        cpu_reference_benchmark=benchmark,source_data_sha256=sha(source),sources=proof,
        seconds=time.perf_counter()-start,official_package_import_tested=False,official_cuda_tested=False,
        new_packages_installed=False,new_gpu_job_submitted=False,full_gdi_training_performed=False)
    save(H/'analysis/mamba-feasibility.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['acquire','check']);args=p.parse_args()
    acquire() if args.stage=='acquire' else main()
