"""Training-subset-only reconstruction/forecasting. No GDI-label argument."""
from pathlib import Path
import hashlib,json,math,os,time
import numpy as np
import torch
from torch import nn

EXCLUDED=('weighted_mean.','regression_head.')
def encoder_key(k):return not k.startswith(EXCLUDED)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(str(tmp),str(p))
def load(p):return json.loads(Path(p).read_text())
def identity(x,ids,c,seed):
    return dict(config=c['ssl'],seed=int(seed),train_people=np.unique(ids).tolist(),train_ids_sha=array_sha(ids),
        train_values_sha=array_sha(x.values),window_record_sha=array_sha(x.owner),window_start_sha=array_sha(x.starts),
        fallback_sha=array_sha(x.fallback),records=len(x),windows=len(x.values),labels_used=False)
def masked_inputs(target,seed,mask_frames=31,prefix_frames=93):
    if target.ndim!=4 or tuple(target.shape[1:])!=(124,4,2):raise ValueError('Invalid SSL input')
    if not(0<mask_frames<124 and 0<prefix_frames<124):raise ValueError('Invalid mask')
    starts=np.random.default_rng(seed).integers(0,125-mask_frames,size=len(target))
    t=torch.arange(124,device=target.device)[None,:]
    start=torch.as_tensor(starts,device=target.device)[:,None]
    local=(t>=start)&(t<start+mask_frames)
    future=(t>=prefix_frames).expand(len(target),-1)
    return target.masked_fill(local[:,:,None,None],0),target.masked_fill(future[:,:,None,None],0),local,future
class SSL(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        for name,p in encoder.named_parameters():p.requires_grad_(encoder_key(name))
        self.local_decoder=nn.Linear(48,8);self.future_decoder=nn.Linear(48,8)
    def features(self,x):
        e=self.encoder;h=e.spatial_forward(x.permute(0,3,1,2))
        h=e.pos_drop(h+e.temporal_positional_enc)
        return e.temporal_norm(e.temporal_block(h))
    def forward(self,x,kind):
        head=self.local_decoder if kind=='local' else self.future_decoder
        return head(self.features(x)).reshape(-1,124,4,2)
def losses(model,target,seed,weights,c):
    local,future,lmask,fmask=masked_inputs(target,seed,c['mask_frames'],c['prefix_frames'])
    lp=model(local,'local');fp=model(future,'future')
    def error(p,m):
        per=((p-target).square().mean(dim=(2,3))*m).sum(dim=1)/m.sum(dim=1)
        return (per*weights).mean()
    return error(lp,lmask),error(fp,fmask)
def audit(x,ids,c,out,seed):
    ident=identity(x,ids,c,seed);done=load(out/'complete.json');curve=load(out/'curve.json')
    if done['identity']!=ident or load(out/'identity.json')!=ident:raise ValueError('SSL training identity mismatch')
    if done['checkpoint_sha256']!=sha(out/'latest.pt') or done['curve_sha256']!=sha(out/'curve.json') or done['probe_sha256']!=sha(out/'probe.npz'):raise ValueError('SSL hash mismatch')
    expected=c['ssl']['epochs']*math.ceil(2*len(x.values)/c['ssl']['batch_size'])
    if len(curve)!=c['ssl']['epochs'] or done['optimizer_steps']!=expected:raise ValueError('SSL incomplete')
    if [v['epoch'] for v in curve]!=list(range(1,len(curve)+1)):raise ValueError('SSL epoch mismatch')
    if not all(math.isfinite(v[k]) for v in curve for k in ('local_mse','future_mse','loss')):raise ValueError('SSL nonfinite')
    ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
    if ck['identity']!=ident or ck['history']!=curve:raise ValueError('SSL checkpoint identity/curve mismatch')
    if not all(int(v['step'])==expected for v in ck['optimizer']['state'].values()):raise ValueError('SSL optimizer steps')
    return done
def initialize(model,x,ids,c,out,device,seed,stop_after=None):
    import run as r
    out=Path(out);out.mkdir(parents=True,exist_ok=True);x.validate()
    if len(x)!=len(ids):raise ValueError('SSL person alignment')
    ident=identity(x,ids,c,seed);path=out/'latest.pt'
    if (out/'identity.json').exists():
        if load(out/'identity.json')!=ident:raise ValueError('SSL identity changed')
    else:save(out/'identity.json',ident)
    if (out/'complete.json').exists():audit(x,ids,c,out,seed)
    r.setup(ident['seed'],device);net=SSL(r.new_model()).to(device)
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=c['ssl']['learning_rate'])
    hist=[]
    if path.exists():
        ck=torch.load(path,map_location=device,weights_only=True)
        if ck['identity']!=ident:raise ValueError('SSL checkpoint mismatch')
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);hist=ck['history']
        torch.set_rng_state(ck['rng_cpu'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng_cuda']])
    xt=x.values.reshape(-1,124,4,2);weights=x.side_weights();cfg=c['ssl'];epochs=cfg['epochs']
    if len(hist)>epochs:raise ValueError('SSL too many epochs')
    for epoch in range(len(hist),epochs):
        started=time.monotonic();net.train();totals=np.zeros(2);order=np.random.default_rng(ident['seed']+epoch).permutation(len(xt))
        for start in range(0,len(xt),cfg['batch_size']):
            ix=order[start:start+cfg['batch_size']];opt.zero_grad(set_to_none=True)
            target=torch.from_numpy(xt[ix]).to(device);w=torch.from_numpy(weights[ix]).to(device)
            lm,fm=losses(net,target,ident['seed']+epoch*100000+start,w,cfg);loss=.5*(lm+fm)
            if not bool(torch.isfinite(loss)):raise ValueError('SSL nonfinite loss')
            loss.backward()
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.requires_grad):raise ValueError('SSL invalid gradient')
            opt.step()
            if not all(bool(torch.isfinite(p).all()) for p in net.parameters()):raise ValueError('SSL invalid weight')
            totals+=np.array([float(lm.detach()),float(fm.detach())])*len(ix)
        vals=totals/len(xt);row=dict(epoch=epoch+1,local_mse=float(vals[0]),future_mse=float(vals[1]),loss=float(vals.mean()),seconds=time.monotonic()-started)
        hist.append(row)
        ck=dict(identity=ident,model=net.state_dict(),optimizer=opt.state_dict(),history=hist,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        torch.save(ck,out/'latest.tmp');os.replace(str(out/'latest.tmp'),str(path));save(out/'curve.json',hist)
        print(json.dumps(dict(pretrain=str(out),**row)),flush=True)
        if stop_after is not None and len(hist)>=stop_after:break
    if len(hist)!=epochs:return None
    if not (out/'probe.npz').exists():
        net.eval()
        with torch.inference_mode():
            target=torch.from_numpy(x.values[:4].reshape(-1,124,4,2)).to(device)
            local,future,_,_=masked_inputs(target,42,cfg['mask_frames'],cfg['prefix_frames'])
            lp=net(local,'local').cpu().numpy();fp=net(future,'future').cpu().numpy()
        np.savez_compressed(out/'probe.npz',local=lp,future=fp)
    save(out/'complete.json',dict(identity=ident,epochs=epochs,optimizer_steps=epochs*math.ceil(len(xt)/cfg['batch_size']),checkpoint_sha256=sha(path),curve_sha256=sha(out/'curve.json'),probe_sha256=sha(out/'probe.npz')))
    audit(x,ids,c,out,seed)
    state=model.state_dict();transferred=[]
    for k,v in net.encoder.state_dict().items():
        if encoder_key(k):state[k]=v;transferred.append(k)
    model.load_state_dict(state,strict=True)
    save(out/'transfer.json',dict(checkpoint_sha256=sha(path),transferred_keys=transferred,excluded_prefixes=list(EXCLUDED),decoder_transferred=False))
    return load(out/'complete.json')
