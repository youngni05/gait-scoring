"""One-owner Slurm pilot/main trainer; raw IMU sequence regression, not RGB."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,time,hashlib,argparse,random,subprocess,tarfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
GRID=[10,25,50,100]
CONDS=['speed_mlp','features_mlp','motion_speed','motion_features']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def score(y,p):
    e=np.abs(y-p);m=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()));m['score']=max(m['mae']/.9,m['p95']/3,m['slow_mae']/3);return m
class Block(nn.Module):
    def __init__(self,c,d):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(c,c,5,padding=2*d,dilation=d),nn.GroupNorm(4,c),nn.SiLU(),nn.Conv1d(c,c,5,padding=2*d,dilation=d),nn.GroupNorm(4,c))
    def forward(self,x):return torch.nn.functional.silu(x+self.net(x))
class Model(nn.Module):
    def __init__(self,condition,nfeatures):
        super().__init__();self.motion=condition.startswith('motion');self.manual=nn.Sequential(nn.Linear(nfeatures,32),nn.SiLU(),nn.Dropout(.15))
        if self.motion:self.conv=nn.Sequential(nn.Conv1d(24,32,1),Block(32,1),Block(32,2),Block(32,4))
        self.head=nn.Sequential(nn.Linear(96 if self.motion else 32,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x,f):
        a=self.manual(f)
        if self.motion:
            z=self.conv(x);a=torch.cat([a,z.mean(dim=-1),z.std(dim=-1,unbiased=False)],dim=1)
        return self.head(a).squeeze(-1)
def seed(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
def preprocess(data,tr,condition):
    F=data['F'][:,:1] if condition.endswith('speed') or condition=='speed_mlp' else data['F'];raw=F[tr];med=np.zeros(F.shape[1])
    for c in range(F.shape[1]):
        good=raw[:,c][np.isfinite(raw[:,c])];med[c]=float(np.median(good)) if len(good) else 0.
    filled=np.where(np.isfinite(raw),raw,med);mean=filled.mean(axis=0);std=filled.std(axis=0);std[std<1e-8]=1.;ym=float(data['y'][tr].mean());ys=max(float(data['y'][tr].std()),1e-8)
    return dict(median=med.tolist(),mean=mean.tolist(),std=std.tolist(),target_mean=ym,target_std=ys,condition=condition,nfeatures=F.shape[1])
def arrays(data,idx,prep,device,dtype=torch.float32):
    F=data['F'][:,:1] if prep['nfeatures']==1 else data['F'];a=np.where(np.isfinite(F[idx]),F[idx],np.array(prep['median']));a=(a-np.array(prep['mean']))/np.array(prep['std'])
    return torch.as_tensor(data['X'][idx],dtype=dtype,device=device),torch.as_tensor(a,dtype=dtype,device=device)
def infer(model,data,idx,prep,device,dtype=torch.float32):
    model.eval();out=[]
    with torch.no_grad():
        for lo in range(0,len(idx),64):
            x,f=arrays(data,idx[lo:lo+64],prep,device,dtype);out.extend((model(x,f).cpu().numpy()*prep['target_std']+prep['target_mean']).tolist())
    return np.array(out,float)
def fit(data,tr,ev,condition,epochs,out,seed_value,checkpoints):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'fit_complete.json').exists():return read(out/'fit_complete.json')
    seed(seed_value);prep=preprocess(data,tr,condition);model=Model(condition,prep['nfeatures']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);X,F=arrays(data,tr,prep,'cuda');yy=data['y'][tr];target=torch.as_tensor((yy-prep['target_mean'])/prep['target_std'],dtype=torch.float32,device='cuda');weight=torch.as_tensor(np.where(yy>=20,4.,np.where(yy>=12,2.,1.)),dtype=torch.float32,device='cuda');start=time.perf_counter();rows=[]
    meta=dict(preprocessing=prep,condition=condition,seed=seed_value,train_indices=tr.tolist(),eval_indices=ev.tolist(),train_ids=data['ids'][tr].tolist(),eval_ids=data['ids'][ev].tolist(),protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)));save(out/'context.json',meta)
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(tr),device='cuda');loss_sum=0.
        for lo in range(0,len(tr),64):
            ii=order[lo:lo+64];opt.zero_grad(set_to_none=True);p=model(X[ii],F[ii]);loss=(torch.nn.functional.smooth_l1_loss(p,target[ii],beta=.5,reduction='none')*weight[ii]).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();loss_sum+=float(loss.detach())*len(ii)
        if epoch in checkpoints:
            p=infer(model,data,ev,prep,'cuda');pt=out/('epoch-%03d.pt'%epoch);torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},pt);reload_model=Model(condition,prep['nfeatures']).cuda();reload_model.load_state_dict(torch.load(pt,map_location='cuda',weights_only=True));rp=infer(reload_model,data,ev,prep,'cuda');assert np.array_equal(p,rp);del reload_model
            rows.append(dict(epoch=epoch,prediction=p.tolist(),metrics=score(data['y'][ev],p),checkpoint=str(pt.relative_to(ROOT)),checkpoint_sha256=sha(pt),elapsed_seconds=time.perf_counter()-start,training_loss=loss_sum/len(tr)));save(out/'curve.json',rows);print(condition,out.name,epoch,'elapsed',round(time.perf_counter()-start,2),flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;result=dict(context=str((out/'context.json').relative_to(ROOT)),curve=str((out/'curve.json').relative_to(ROOT)),epochs=epochs,elapsed_seconds=elapsed,prediction=rows[-1]['prediction'],checkpoint=rows[-1]['checkpoint'],checkpoint_sha256=rows[-1]['checkpoint_sha256']);save(out/'fit_complete.json',result);del model,opt,X,F,target,weight;torch.cuda.empty_cache();return result
def gpu_environment():
    assert torch.cuda.is_available();a=torch.ones(16,device='cuda');assert float((a@a).cpu())==16.;torch.cuda.synchronize();text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);uuids=[line.split(',')[1].strip() for line in text.splitlines() if line.split(',')[0].strip()==str(os.getpid())];assert len(set(uuids))==1
    return dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),gpu_uuid=uuids[0],pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),hostname=os.uname().nodename)
def pilot(data,proto):
    env=gpu_environment();torch.set_num_threads(4);split=proto['splits']['0']['inner'][0];tr=np.array(split['train']);ev=np.array(split['valid']);runs={}
    for ci,c in enumerate(CONDS):
        r=fit(data,tr,ev,c,10,ROOT/'runs/pilot'/c,20261104+ci,[10]);ctx=read(ROOT/r['context']);prep=ctx['preprocessing'];m=Model(c,prep['nfeatures']).double();m.load_state_dict(torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True));rp=infer(m,data,ev,prep,'cpu',torch.float64);saved=np.array(r['prediction']);assert np.allclose(rp,saved,atol=1e-6,rtol=1e-6);runs[c]=dict(fit=r,cpu64_max_delta=float(np.max(np.abs(rp-saved))));del m
    save(ROOT/'runs/pilot/summary.json',dict(status='passed_pending_local_review',environment=env,conditions=runs,protocol_sha256=sha(ROOT/'protocol.json'),data_sha256=sha(ROOT/'data.npz'),code_sha256=sha(Path(__file__))))
    archive(ROOT/'runs/pilot-review.tar.gz',['train.py','protocol.json','runs/pilot'])
def archive(path,names):
    with tarfile.open(path,'w:gz') as tar:
        for name in names:tar.add(ROOT/name,arcname=name)
    save(Path(str(path)+'.sha256.json'),dict(path=str(path),sha256=sha(path)))
def main(data,proto,folds):
    env=gpu_environment();torch.set_num_threads(4);save(ROOT/('control/worker-'+os.environ.get('SLURM_JOB_ID','local')+'.json'),dict(job_id=os.environ.get('SLURM_JOB_ID'),phase='training',folds=folds,environment=env))
    for f in folds:
      split=proto['splits'][str(f)];tr=np.array(split['train']);te=np.array(split['test'])
      for ci,c in enumerate(CONDS):
        out=ROOT/'runs/main'/('fold%d'%f)/c
        if(out/'complete.json').exists():continue
        records=[]
        for k,ss in enumerate(split['inner']):records.append(fit(data,np.array(ss['train']),np.array(ss['valid']),c,100,out/('inner%d'%k),20261104+f*100+ci+k*1000,GRID))
        scores=[]
        for epoch in GRID:
            oof=np.zeros(len(tr));seen=[]
            for rr,ss in zip(records,split['inner']):
                row=[r for r in read(ROOT/rr['curve']) if r['epoch']==epoch][0];ix=[int(np.flatnonzero(tr==i)[0]) for i in ss['valid']];oof[ix]=row['prediction'];seen+=ss['valid']
            assert len(seen)==len(tr) and set(seen)==set(tr);scores.append(score(data['y'][tr],oof))
        selected=GRID[int(np.argmin([s['score'] for s in scores]))];final=fit(data,tr,te,c,selected,out/'final',20261104+f*100+ci+5000,[selected]);save(out/'complete.json',dict(condition=c,fold=f,inner_scores=scores,epoch_grid=GRID,selected_epoch=selected,inner=records,final=final,test_ids=data['ids'][te].tolist(),prediction=final['prediction']))
      save(ROOT/'runs/main'/('fold%d'%f)/'complete.json',dict(fold=f,conditions=CONDS))
    save(ROOT/('control/worker-'+os.environ.get('SLURM_JOB_ID','local')+'.json'),dict(job_id=os.environ.get('SLURM_JOB_ID'),phase='assigned_folds_complete',folds=folds,environment=env))
    # Each worker exports only its own disjoint results; no shared aggregator mutation.
    job=os.environ.get('SLURM_JOB_ID','local');archive(ROOT/('runs/worker-'+job+'-review.tar.gz'),['train.py','protocol.json']+['runs/main/fold%d'%f for f in folds])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);ap.add_argument('--folds',default='0,1,2,3,4');args=ap.parse_args();proto=read(ROOT/'protocol.json');assert sha(ROOT/'data.npz')==proto['data_sha256'];data=np.load(ROOT/'data.npz');assert len(data['y'])==594 and sum(data['y']>=20)==9
    if args.mode=='pilot':pilot(data,proto)
    else:main(data,proto,[int(x) for x in args.folds.split(',')])
