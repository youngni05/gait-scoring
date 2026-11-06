"""Pinned STT check and GPU pilot only. No outer evaluation or full-training command."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,json,math,socket,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
import torch
from model.Transformer import SpatioTemporalTransformer

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(str(temp),str(p))
def require(ok,message):
    if not ok:raise ValueError(message)
def setup(seed,device):
    torch.set_num_threads(4);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device=='cuda':
        require(torch.cuda.is_available(),'CUDA unavailable; no CPU fallback')
        print('CUDA allocation probe',flush=True);torch.empty(1,device=device);torch.cuda.synchronize();torch.cuda.manual_seed_all(seed)
def new_model():
    return SpatioTemporalTransformer(num_frame=124,num_joints=4,in_chans=2,joint_embed_dim=12,num_heads=2,drop_rate=0.)
def split(d,f,j):
    require(f in range(5) and j in range(5) and f!=j,'Invalid folds')
    tr=(d['fold']!=f)&(d['fold']!=j);va=d['fold']==j;te=d['fold']==f
    groups=[set(d['person_id'][m]) for m in (tr,va,te)]
    require(not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]),'Person overlap')
    return tr,va,te
def data():
    for name,digest in load(ROOT/'manifest.json').items():require(sha(ROOT/name)==digest,'Bundle changed: '+name)
    with np.load(ROOT/'data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    require(d['x'].shape==(1061,2,124,4,2) and d['x'].dtype==np.float32,'Input shape/dtype')
    require(d['y'].shape==(1061,2) and np.isfinite(d['x']).all() and np.isfinite(d['y']).all(),'Invalid data')
    require(len(np.unique(d['person_id']))==692 and len(np.unique(d['video_id']))==1061,'Wrong cohort')
    require(len(np.unique(d['original_row_index']))==1061,'Duplicate row')
    for f in range(5):
        require(len(np.unique(d['person_id'][d['fold']==f]))==[138,138,138,139,139][f],'Wrong fold sizes')
        for j in range(5):
            if f!=j:split(d,f,j)
    return d
def aligned_mse(pred,target):
    require(pred.ndim==target.ndim==1 and pred.shape==target.shape,'Loss axes must be aligned 1D')
    return (pred-target).square().mean()
def predict(model,x,lo,span,device):
    model.eval();out=[]
    with torch.inference_mode():
        for k in range(0,len(x),64):
            value=torch.from_numpy(x[k:k+64].reshape(-1,124,4,2)).to(device)
            out.append((model(value)*span+lo).reshape(-1,2).cpu().numpy())
    return np.concatenate(out)
def calibration(p,y):
    return np.linalg.lstsq(np.column_stack([p.reshape(-1).astype(float),np.ones(p.size)]),y.reshape(-1),rcond=None)[0]
def metric(y,p,ids):return float(np.mean([np.abs(y[ids==q]-p[ids==q]).mean() for q in np.unique(ids)]))
def digest_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fit(x,y,ids,vx,vy,vids,c,epochs,out,identity,device,stop_after=None):
    require(not(set(ids)&set(vids)),'Training/validation people overlap')
    require(x.shape==(len(y),2,124,4,2) and y.shape==(len(ids),2),'Training shape')
    require(vx.shape==(len(vy),2,124,4,2) and vy.shape==(len(vids),2),'Validation shape')
    identity=dict(identity,train_x_sha=digest_array(x),train_y_sha=digest_array(y),train_ids_sha=digest_array(ids))
    setup(identity['seed'],device);out.mkdir(parents=True,exist_ok=True)
    model=new_model().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=40,T_mult=1,eta_min=c['eta_min'])
    lo=float(np.float32(y.min()));span=float(np.float32(max(float(y.max()-y.min()),1e-6)))
    xt=x.reshape(-1,124,4,2);yt=(y.astype(np.float32).reshape(-1)-lo)/span
    history=[];path=out/'latest.pt'
    if path.exists():
        ck=torch.load(path,map_location=device,weights_only=True)
        require(ck['identity']==identity,'Checkpoint identity changed')
        require(ck['lo']==lo and ck['span']==span,'Training scale changed')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        history=ck['history'];require(len(history)<=epochs,'Checkpoint exceeds epochs')
        torch.set_rng_state(ck['rng_cpu'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng_cuda']])
    for epoch in range(len(history),epochs):
        started=time.monotonic();model.train();loss_sum=0.;lr=float(opt.param_groups[0]['lr'])
        order=np.random.default_rng(identity['seed']+epoch).permutation(len(yt))
        for start in range(0,len(order),c['batch_size']):
            ix=order[start:start+c['batch_size']];opt.zero_grad(set_to_none=True)
            pred=model(torch.from_numpy(xt[ix]).to(device));target=torch.from_numpy(yt[ix]).to(device)
            loss=aligned_mse(pred,target);require(bool(torch.isfinite(loss)),'Nonfinite loss');loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Invalid gradient')
            opt.step();require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Invalid weight')
            loss_sum+=float(loss.detach())*len(ix)
        scheduler.step()
        raw=predict(model,x,lo,span,device);coef=calibration(raw,y)
        valraw=predict(model,vx,lo,span,device)
        row=dict(epoch=epoch+1,lr=lr,loss=loss_sum/len(yt),calibration=coef.tolist(),
            train_person_mae=metric(y,raw*coef[0]+coef[1],ids),
            validation_raw_person_mae=metric(vy,valraw,vids),validation_person_mae=metric(vy,valraw*coef[0]+coef[1],vids),
            seconds=time.monotonic()-started)
        history.append(row)
        ck=dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
            lo=lo,span=span,history=history,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        tmp=out/'latest.tmp';torch.save(ck,tmp);os.replace(str(tmp),str(path));save(out/'curve.json',history)
        print(json.dumps(row),flush=True)
        if stop_after is not None and len(history)>=stop_after:break
    return model,lo,span,history
def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['check','pilot'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();c=load(ROOT/'protocol.json');setup(c['seed'],a.device);d=data();a.output.mkdir(parents=True,exist_ok=True)
    env=dict(python=sys.version,torch=str(torch.__version__),device=a.device,hostname=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None)
    save(a.output/'environment.json',env)
    if a.command=='check':
        m=new_model().to(a.device).eval()
        with torch.inference_mode():p=m(torch.from_numpy(d['x'][:2].reshape(-1,124,4,2)).to(a.device))
        require(p.shape==(4,) and bool(torch.isfinite(p).all()),'Real input forward failed')
        result=dict(passed=True,people=692,records=1061,input_shape=list(d['x'].shape),parameters=sum(p.numel() for p in m.parameters()),all20_inner_splits_checked=True,real_optimizer_steps=0)
        save(a.output/'check.json',result);print(json.dumps(result));return
    require(a.device=='cuda','Real pilot requires explicit CUDA; CPU verification uses synthetic data only')
    tr,va,te=split(d,0,1);torch.cuda.reset_peak_memory_stats()
    ident=dict(bundle_sha256=sha(ROOT/'manifest.json'),seed=c['seed']+1,epochs=c['pilot_epochs'],outer_fold=0,inner_fold=1,
        train_people=np.unique(d['person_id'][tr]).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),outer_evaluation_people=np.unique(d['person_id'][te]).tolist())
    model,lo,span,h=fit(d['x'][tr],d['y'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],c,3,a.output/'fit',ident,a.device)
    raw=predict(model,d['x'][va],lo,span,a.device);coef=h[-1]['calibration'];pred=raw*coef[0]+coef[1]
    np.savez_compressed(a.output/'validation_predictions.npz',row_index=np.flatnonzero(va),predictions=pred,raw=raw,y=d['y'][va],person_id=d['person_id'][va])
    result=dict(passed=True,pilot_only=True,epochs=3,train_people=int(len(set(d['person_id'][tr]))),validation_people=int(len(set(d['person_id'][va]))),
        outer_evaluation_used=False,optimizer_steps=3*math.ceil(2*int(tr.sum())/c['batch_size']),validation_person_mae=metric(d['y'][va],pred,d['person_id'][va]),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),epoch_seconds=[v['seconds'] for v in h],parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(a.output/'fit/latest.pt'))
    save(a.output/'pilot_summary.json',result);print(json.dumps(result));print('STT_PILOT_FINISHED')

if __name__=='__main__':main()
