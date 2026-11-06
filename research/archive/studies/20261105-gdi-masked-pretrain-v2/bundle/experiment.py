"""Frozen-budget, person-disjoint masked pretraining comparison; Python3.8+."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cnn import GdiCNN
from model.Transformer import SpatioTemporalTransformer

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(str(tmp),str(p))
def checkpoint(p,v):
    tmp=p.with_suffix('.tmp');torch.save(v,tmp);os.replace(str(tmp),str(p))
def setup(seed):
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def rngstate():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def rngrestore(v):
    torch.set_rng_state(v['cpu'].cpu())
    if v['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([s.cpu() for s in v['cuda']])
def newmodel(family):
    return GdiCNN(.5) if family=='CNN' else SpatioTemporalTransformer(num_frame=124,num_joints=4,in_chans=2,joint_embed_dim=12,num_heads=2,drop_rate=0.)
def forward(model,family,x):
    return model(x.transpose(1,2))[:,0] if family=='CNN' else model(x[:,:,:8].reshape(-1,124,4,2))
def backbone_keys(model,family,protocol):
    prefixes=protocol['pretrained_keys'][family]
    return [k for k in model.state_dict() if any(k.startswith(p) for p in prefixes)]
def load_data(path,protocol):
    assert sha(path)==protocol['data_sha256']
    with np.load(path) as z:d={k:z[k].copy() for k in z.files}
    assert d['x'].shape==(11285,2,124,12) and d['observed_xy'].shape==(11285,2,124,8)
    assert np.isfinite(d['y'][:1061]).all() and np.isnan(d['y'][1061:]).all()
    assert np.all(d['fold'][1061:]==-1) and len(np.unique(d['person_id'][:1061]))==692
    return d
def rows_for(d,fold,condition,pilot=False):
    tr=(d['fold']>=0)&(d['fold']!=fold)
    evaluation=d['fold']==fold
    if pilot:
        inner=(fold+1)%5;tr&=d['fold']!=inner;evaluation=d['fold']==inner
    ext=np.zeros(len(tr),bool)
    if condition.endswith('adult'):ext[1061:1073]=True
    elif condition.endswith('child'):ext[1073:1085]=True
    assert len(set(d['person_id'][ext])) in [0,9]
    assert not set(d['person_id'][tr|ext])&set(d['person_id'][evaluation])
    return np.flatnonzero(tr),np.flatnonzero(ext),np.flatnonzero(evaluation)
def example_pool(d,rows):
    chosen=np.isin(d['window_record'],rows);wi=np.flatnonzero(chosen)
    owners=d['window_record'][wi];counts=np.bincount(owners,minlength=len(d['person_id']))
    weight=len(wi)*d['record_weight'][owners]/(d['record_weight'][rows].sum()*counts[owners])
    return wi,np.repeat(weight,2).astype(np.float32)
def predict(model,family,d,rows,lo,span,device):
    wi,_=example_pool(d,rows);model.eval();chunks=[];batch=128 if family=='CNN' else 64
    with torch.inference_mode():
        for start in range(0,len(wi),batch):
            x=torch.from_numpy(d['x'][wi[start:start+batch]].reshape(-1,124,12)).to(device)
            chunks.append((forward(model,family,x)*span+lo).reshape(-1,2).cpu().numpy())
    out=np.concatenate(chunks);owner=d['window_record'][wi];mapping=np.full(len(d['y']),-1);mapping[rows]=np.arange(len(rows))
    result=np.zeros((len(rows),2),np.float64);np.add.at(result,mapping[owner],out)
    result/=np.bincount(mapping[owner],minlength=len(rows))[:,None]
    return result
def ols(raw,y):return np.linalg.lstsq(np.column_stack([raw.reshape(-1).astype(float),np.ones(raw.size)]),y.reshape(-1),rcond=None)[0]
def person_mae(y,p,ids):return float(np.mean([np.abs(y[ids==i]-p[ids==i]).mean() for i in np.unique(ids)]))

class Reconstruction(nn.Module):
    def __init__(self,family,protocol):
        super().__init__();self.family=family;self.base=newmodel(family)
        keys=backbone_keys(self.base,family,protocol)
        for name,p in self.base.named_parameters():p.requires_grad_(name in keys)
        self.mask_token=nn.Parameter(torch.zeros(12 if family=='CNN' else 8))
        self.decoder=nn.Sequential(nn.Conv1d(32,32,3,padding=1),nn.GELU(),nn.Conv1d(32,8,1)) if family=='CNN' else nn.Sequential(nn.Linear(48,48),nn.GELU(),nn.Linear(48,8))
        if family=='CNN':
            # Build the fixed linear operator on CPU using the original float32
            # coordinate convention; CUDA backward then uses deterministic matmul.
            weights=F.interpolate(torch.eye(10).unsqueeze(0),size=124,mode='linear',align_corners=False)[0]
            self.register_buffer('upsample_weights',weights)
    def forward(self,x,mask):
        x=x[:,:,:len(self.mask_token)];x=torch.where(mask[:,:,None],self.mask_token[None,None,:],x)
        if self.family=='CNN':
            x=x.transpose(1,2)
            for i,(conv,norm) in enumerate(zip(self.base.convs,self.base.norms)):
                x=norm(F.relu(conv(x)))
                if i%2:x=self.base.dropout(F.max_pool1d(x,3 if i==5 else 2))
            x=x@self.upsample_weights
            return self.decoder(x).transpose(1,2)
        x=x.reshape(-1,124,4,2).permute(0,3,1,2)
        x=self.base.spatial_forward(x)
        x=self.base.pos_drop(x+self.base.temporal_positional_enc)
        x=self.base.temporal_norm(self.base.temporal_block(x))
        return self.decoder(x)

def time_mask(seed,step,pairs):
    rng=np.random.default_rng(seed+step);mask=np.zeros((pairs,124),bool)
    starts=rng.integers(0,24,size=(pairs,4))+np.arange(4)*31
    for b in range(pairs):
        for start in starts[b]:mask[b,start:start+8]=True
    assert (mask.sum(1)==32).all()
    return np.repeat(mask,2,axis=0)
def sampling_plan(d,stanford_rows,external_rows,steps,pairs,seed):
    rng=np.random.default_rng(seed);people=np.unique(d['person_id'][stanford_rows])
    # Draw common Stanford examples first. External replacement never changes that stream.
    groups=[stanford_rows[d['person_id'][stanford_rows]==p] for p in people]
    win=[np.flatnonzero(d['window_record']==r) for r in range(len(d['person_id']))]
    plan=[]
    for _ in range(steps*pairs):
        rows=groups[int(rng.integers(len(groups)))];r=int(rows[int(rng.integers(len(rows)))]);w=win[r];plan.append(int(w[int(rng.integers(len(w)))]))
    plan=np.array(plan,dtype=np.int64).reshape(steps,pairs)
    flags=np.zeros_like(plan,bool)
    if len(external_rows):
        rng=np.random.default_rng(seed+1);extra_people=np.unique(d['person_id'][external_rows]);assert len(extra_people)==9
        flags=rng.random(plan.shape)<len(extra_people)/(len(people)+len(extra_people))
        groups=[external_rows[d['person_id'][external_rows]==p] for p in extra_people]
        for i,j in np.argwhere(flags):
            rows=groups[int(rng.integers(len(groups)))];r=int(rows[int(rng.integers(len(rows)))]);w=win[r];plan[i,j]=int(w[int(rng.integers(len(w)))] )
    assert np.isin(d['window_record'][plan],np.concatenate([stanford_rows,external_rows])).all()
    return plan,flags

def pretrain(d,protocol,family,train_rows,external_rows,out,seed,device,steps,identity):
    setup(seed);model=Reconstruction(family,protocol).to(device)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=protocol['pretrain_lr'])
    pairs=protocol['pretrain_batch_pairs'];plan,flags=sampling_plan(d,train_rows,external_rows,steps,pairs,seed+2000)
    np.savez_compressed(out/'pretrain-plan.npz',windows=plan,external_flags=flags)
    # Shared target scaling: observed coordinates from Stanford training rows only.
    wi,_=example_pool(d,train_rows);x=d['x'][wi,:,:,:8];valid=d['observed_xy'][wi]
    n=valid.sum(axis=(0,1,2));mean=(x*valid).sum(axis=(0,1,2),dtype=float)/n
    scale=np.sqrt((((x-mean)**2)*valid).sum(axis=(0,1,2))/n);scale=np.maximum(scale,.05).astype(np.float32)
    scale_t=torch.from_numpy(scale).to(device);start=0;history=[];path=out/'pretrain.pt'
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['identity']==identity and ck['plan_sha256']==sha(out/'pretrain-plan.npz')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);rngrestore(ck['rng']);history=ck['history'];start=ck['steps']
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
    started=time.monotonic();losses=[]
    for step in range(start,steps):
        model.train();w=plan[step];xx=torch.from_numpy(d['x'][w].reshape(-1,124,12)).to(device)
        mask=torch.from_numpy(time_mask(seed+3000,step,pairs)).to(device)
        valid_t=torch.from_numpy(d['observed_xy'][w].reshape(-1,124,8)).to(device)&mask[:,:,None]
        target=xx[:,:,:8];opt.zero_grad(set_to_none=True);pred=model(xx,mask)
        denom=valid_t.sum((1,2));assert bool((denom>0).all())
        loss=((((pred-target)/scale_t)**2)*valid_t).sum((1,2)).div(denom).mean()
        assert bool(torch.isfinite(loss));loss.backward();opt.step();losses.append(float(loss.detach()))
        if (step+1)%100==0 or step+1==steps:
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            history.append(dict(step=step+1,loss=float(np.mean(losses)),seconds=time.monotonic()-started));losses=[];started=time.monotonic()
            checkpoint(path,dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),rng=rngstate(),steps=step+1,history=history,target_scale=scale.tolist(),plan_sha256=sha(out/'pretrain-plan.npz')))
            save(out/'pretrain-curve.json',history);print('PRETRAIN',family,identity['condition'],identity['fold'],step+1,flush=True)
    model.eval();w=plan[-1];xx=torch.from_numpy(d['x'][w].reshape(-1,124,12)).to(device);mask=time_mask(seed+3000,steps-1,pairs)
    with torch.inference_mode():reconstruction=model(xx,torch.from_numpy(mask).to(device)).cpu().numpy()
    np.savez_compressed(out/'pretrain-replay.npz',windows=w,mask=mask,reconstruction=reconstruction)
    keys=backbone_keys(model.base,family,protocol)
    save(out/'pretrain-complete.json',dict(passed=True,steps=steps,side_examples=steps*pairs*2,external_pair_examples=int(flags.sum()),sampled_people=np.unique(d['person_id'][d['window_record'][plan]]).tolist(),checkpoint_sha256=sha(path),plan_sha256=sha(out/'pretrain-plan.npz'),replay_sha256=sha(out/'pretrain-replay.npz')))
    return {k:model.base.state_dict()[k].detach().cpu().clone() for k in keys}

def run_fit(d,protocol,family,condition,fold,root,device,pilot=False):
    config=read(HERE/(family+'-original.json'));seed=protocol['seed']+fold*100+50
    tr,ext,ev=rows_for(d,fold,condition,pilot);epochs=2 if pilot else protocol['gdi_epochs'][family][str(fold)]
    sslsteps=40 if pilot else protocol['pretrain_steps']
    stage='pilot' if pilot else 'main';out=root/stage/family/condition/('fold%d'%fold);out.mkdir(parents=True,exist_ok=True)
    identity=dict(manifest_sha256=sha(HERE/'manifest.json'),family=family,condition=condition,fold=fold,pilot=pilot,seed=seed,epochs=epochs,
                  train_people=np.unique(d['person_id'][tr]).tolist(),external_people=np.unique(d['person_id'][ext]).tolist(),evaluation_people=np.unique(d['person_id'][ev]).tolist(),
                  evaluation_role='inner_validation' if pilot else 'outer_evaluation')
    if (out/'complete.json').exists():
        done=read(out/'complete.json');assert done['identity']==identity and done['prediction_sha256']==sha(out/'predictions.npz');return
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity
    save(out/'identity.json',identity)
    pretrained=None
    if condition=='C' or condition.startswith('D_'):
        # Pretraining function receives coordinates and metadata only, no score array.
        ssl={k:d[k] for k in ['x','observed_xy','window_record','record_weight','person_id']}
        pretrained=pretrain(ssl,protocol,family,tr,ext if condition.startswith('D_') else np.array([],int),out,seed+500000,device,sslsteps,identity)
    setup(seed);model=newmodel(family)
    initial=model.state_dict();headkeys=[k for k in initial if k not in backbone_keys(model,family,protocol)]
    head_digest=hashlib.sha256(b''.join(initial[k].cpu().numpy().tobytes() for k in sorted(headkeys))).hexdigest()
    if pretrained is not None:
        initial.update(pretrained);model.load_state_dict(initial,strict=True)
    model.to(device)
    if family=='CNN':
        opt=torch.optim.RMSprop(model.parameters(),lr=config['learning_rate'],alpha=config['rmsprop_alpha'],eps=config['rmsprop_eps'],weight_decay=0,momentum=0,centered=False);scheduler=None
    else:
        opt=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=0)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=40,T_mult=1,eta_min=config['eta_min'])
    rows=np.concatenate([tr,ext]) if condition.startswith('B_') else tr
    wi,weights=example_pool(d,rows);base_wi,_=example_pool(d,tr);samples=2*len(base_wi);batch=config['batch_size']
    owners=d['window_record'][wi];yy=d['y'][owners].astype(np.float32).copy()
    if condition.startswith('B_'):yy[owners>=1061]=100
    assert np.isfinite(yy).all()
    lo=float(np.float32(d['y'][tr].min()));span=float(np.float32(max(float(d['y'][tr].max()-d['y'][tr].min()),1e-6)))
    xt=torch.from_numpy(d['x'][wi].reshape(-1,124,12)).to(device)
    yt=torch.from_numpy(((yy-lo)/span).reshape(-1)).to(device);wt=torch.from_numpy(weights).to(device)
    history=[];path=out/'model.pt'
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['identity']==identity and ck['lo']==lo and ck['span']==span
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);rngrestore(ck['rng']);history=ck['history']
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        if scheduler is not None:scheduler.load_state_dict(ck['scheduler'])
    for epoch in range(len(history),epochs):
        started=time.monotonic();model.train();loss_sum=0.;observed_external=0
        if family=='CNN':
            for group in opt.param_groups:group['lr']=config['learning_rate']*config['lr_drop_factor']**((epoch+1)//config['lr_drop_every'])
        lr=float(opt.param_groups[0]['lr']);order=np.random.default_rng(seed+epoch).permutation(len(yt))[:samples]
        for start in range(0,samples,batch):
            ix=order[start:start+batch];opt.zero_grad(set_to_none=True)
            pred=forward(model,family,xt[ix]);loss=((pred-yt[ix])**2*wt[ix]).mean()
            if family=='CNN':loss=loss+config['kernel_l2']*model.penalty()
            assert bool(torch.isfinite(loss));loss.backward();opt.step();loss_sum+=float(loss.detach())*len(ix)
            observed_external+=int((owners[ix//2]>=1061).sum())
        if scheduler is not None:scheduler.step()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        raw=predict(model,family,d,tr,lo,span,device);coef=ols(raw,d['y'][tr])
        rawdir=out/'epoch-train';rawdir.mkdir(exist_ok=True);rawpath=rawdir/('epoch%04d.npz'%(epoch+1))
        np.savez_compressed(rawpath,rows=tr,raw=raw)
        history.append(dict(epoch=epoch+1,lr=lr,loss=loss_sum/samples,calibration=coef.tolist(),train_person_mae=person_mae(d['y'][tr],raw*coef[0]+coef[1],d['person_id'][tr]),optimizer_steps=math.ceil(samples/batch),examples=samples,external_examples=observed_external,seconds=time.monotonic()-started,train_raw_sha256=sha(rawpath)))
        checkpoint(path,dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=None if scheduler is None else scheduler.state_dict(),rng=rngstate(),history=history,lo=lo,span=span,head_initial_sha256=head_digest))
        save(out/'curve.json',history)
        if (epoch+1)%10==0 or epoch+1==epochs:print('GDI',family,condition,fold,epoch+1,epochs,flush=True)
    raw=predict(model,family,d,tr,lo,span,device);evaluation=predict(model,family,d,ev,lo,span,device);coef=ols(raw,d['y'][tr])
    np.savez_compressed(out/'predictions.npz',train_rows=tr,train_raw=raw,evaluation_rows=ev,evaluation_raw=evaluation,calibration=coef,prediction=evaluation*coef[0]+coef[1])
    with torch.inference_mode():
        sample=wi[:16];model.eval();probe=forward(model,family,torch.from_numpy(d['x'][sample].reshape(-1,124,12)).to(device)).cpu().numpy()
    np.savez_compressed(out/'model-replay.npz',windows=sample,scaled_prediction=probe)
    save(out/'complete.json',dict(identity=identity,epochs=epochs,optimizer_steps=epochs*math.ceil(samples/batch),examples=epochs*samples,pretraining_steps=sslsteps if pretrained is not None else 0,head_initial_sha256=head_digest,checkpoint_sha256=sha(path),prediction_sha256=sha(out/'predictions.npz'),curve_sha256=sha(out/'curve.json'),replay_sha256=sha(out/'model-replay.npz'),pretrain_checkpoint_sha256=sha(out/'pretrain.pt') if pretrained is not None else None))
    print('FIT_COMPLETE',family,condition,fold,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['pilot','train']);ap.add_argument('--family',choices=['CNN','STT'],required=True)
    ap.add_argument('--conditions',default='A,B_adult,B_child,C,D_adult,D_child');ap.add_argument('--folds',default='0,1,2,3,4')
    ap.add_argument('--data',type=Path,default=HERE/'data.npz');ap.add_argument('--output',type=Path,default=HERE/'runs');ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=ap.parse_args()
    for name,digest in read(HERE/'manifest.json').items():assert sha(HERE/name)==digest,name
    if a.device=='cuda':assert torch.cuda.is_available(),'No CPU fallback for GPU experiment'
    protocol=read(HERE/'protocol.json');d=load_data(a.data,protocol)
    conditions=a.conditions.split(',');assert set(conditions)<=set(protocol['conditions'])
    for fold in map(int,a.folds.split(',')):
        assert fold in protocol['folds']
        for condition in conditions:run_fit(d,protocol,a.family,condition,fold,a.output,a.device,a.action=='pilot')

if __name__=='__main__':main()
