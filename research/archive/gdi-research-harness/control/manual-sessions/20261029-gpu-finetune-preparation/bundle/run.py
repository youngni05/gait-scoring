"""MotionBERT adaptation preparation. Training requires an explicit complete protocol.
Only synthetic inputs are used by smoke/profile. Never edits the source checkpoint.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from lib.model.DSTformer import DSTformer
ROOT = Path(__file__).resolve().parent
VARIANTS = ('head_only', 'tail', 'last_stage')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def environment(seed):
    torch.set_num_threads(4);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,'cuda'): torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends,'cudnn'): torch.backends.cudnn.allow_tf32=False

def selected(n, variant):
    if variant=='head_only' or n.startswith('head.'): return False
    if n.startswith(('norm.','pre_logits.')): return True
    return variant=='last_stage' and n.startswith(('blocks_st.4.','blocks_ts.4.','ts_attn.4.'))

class Scorer(torch.nn.Module):
    def __init__(self, variant, fold):
        super().__init__();self.variant=variant
        self.encoder=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
        ck=torch.load(ROOT/'MB_lite.bin',map_location='cpu',weights_only=True)
        self.encoder.load_state_dict({k.removeprefix('module.'):v for k,v in ck['model_pos'].items()},strict=True)
        for n,p in self.encoder.named_parameters(): p.requires_grad_(selected(n,variant))
        with np.load(ROOT/'baseline_heads.npz',allow_pickle=False) as z:
            pre=f'fold{fold}__'
            self.register_buffer('mu',torch.from_numpy(z[pre+'scaler_mean'].copy()).double())
            self.register_buffer('scale',torch.from_numpy(z[pre+'scaler_scale'].copy()).double())
            self.output=torch.nn.Linear(512,2).double()
            with torch.no_grad():
                self.output.weight.copy_(torch.from_numpy(z[pre+'coef']))
                self.output.bias.copy_(torch.from_numpy(z[pre+'intercept']))
        self.eval() # Preserve frozen baseline: no stochastic dropout in any arm.
    def forward(self,x):
        if x.ndim==2:
            if self.variant!='head_only': raise ValueError('512-feature cache is invalid for encoder adaptation')
            f=x
        else: f=self.encoder.get_representation(x).mean((1,2))
        return self.output((f.double()-self.mu)/self.scale)

def optimizer(m,cfg):
    groups=[{'params':list(m.output.parameters()),'lr':cfg['head_lr']}]
    ps=[p for p in m.encoder.parameters() if p.requires_grad]
    if ps: groups.append({'params':ps,'lr':cfg['encoder_lr']})
    return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])

def step(m,opt,x,y):
    opt.zero_grad(set_to_none=True);pred=m(x);loss=(pred-y).square().mean()
    if not torch.isfinite(loss): raise ValueError('nonfinite loss')
    loss.backward()
    for n,p in m.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all(): raise ValueError('invalid gradient '+n)
        elif p.grad is not None: raise ValueError('frozen parameter received gradient '+n)
    opt.step()
    if not all(torch.isfinite(p).all() for p in m.parameters()): raise ValueError('nonfinite weights')
    return float(loss.detach())

def checkpoint(p,m,opt,done,identity):
    p=Path(p);tmp=p.with_suffix('.tmp')
    torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'done':done,'identity':identity,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},tmp)
    os.replace(tmp,p)

def restore(p,m,opt,identity,device):
    c=torch.load(p,map_location=device,weights_only=True)
    if c['identity']!=identity: raise ValueError('checkpoint identity mismatch')
    m.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer']);torch.set_rng_state(c['rng'].cpu())
    if c['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in c['cuda_rng']])
    return int(c['done'])

def validate_data(d):
    if d['x'].shape!=(203,2,124,17,3) or d['y'].shape!=(203,2): raise ValueError('unexpected dataset shape')
    if d['features'].shape!=(203,2,512): raise ValueError('unexpected feature shape')
    if len(set(d['video_id']))!=203 or len(set(d['person_id']))!=175: raise ValueError('identity count mismatch')
    if set(d['fold'])!=set(range(5)): raise ValueError('fold mismatch')
    for k in range(5):
        te=set(d['person_id'][d['fold']==k]);tr=set(d['person_id'][d['fold']!=k])
        if len(te)!=35 or len(tr)!=140 or te&tr: raise ValueError('person leakage')
    if not all(np.isfinite(d[k]).all() for k in ('x','y','features')): raise ValueError('nonfinite data')
    if d['x'].dtype!=np.float32: raise ValueError('unexpected input dtype')

def data():
    manifest=load(ROOT/'manifest.json')
    for name,digest in manifest.items():
        if sha(ROOT/name)!=digest: raise ValueError('bundle changed: '+name)
    with np.load(ROOT/'train_only.npz',allow_pickle=False) as z: d={k:z[k].copy() for k in z.files}
    validate_data(d);return d

def protocol(p):
    c=load(p)
    if c.get('status')!='ready': raise ValueError('draft protocol: real-data training is disabled')
    for n in ('steps','batch_size','checkpoint_every'):
        if type(c.get(n)) is not int or c[n]<1: raise ValueError('set positive '+n)
    for n in ('head_lr','encoder_lr','weight_decay'):
        if not isinstance(c.get(n),(int,float)) or not math.isfinite(c[n]) or c[n]<0: raise ValueError('set finite '+n)
    if c['head_lr']==0 or c['encoder_lr']==0: raise ValueError('learning rates must be positive')
    if c['variants']!=list(VARIANTS) or c['folds']!=list(range(5)) or not c['seeds'] or len(set(c['seeds']))!=len(c['seeds']): raise ValueError('incomplete comparison')
    if not all(type(s) is int and s>=0 for s in c['seeds']): raise ValueError('invalid seeds')
    expected={'loss':'record_equal_MSE_left_right','train_window':'front','evaluate_windows':['front','back'],'optimizer':'AdamW','dropout':'disabled_all_arms'}
    if any(c.get(k)!=v for k,v in expected.items()): raise ValueError('protocol contract changed')
    return c

def smoke(out,device,profile=False,batch=1):
    out.mkdir(parents=True,exist_ok=True);environment(1234)
    cfg={'head_lr':1e-6,'encoder_lr':1e-7,'weight_decay':0.} # synthetic engineering test only
    results=[]
    for variant in VARIANTS:
        environment(1234);m=Scorer(variant,0).to(device);opt=optimizer(m,cfg)
        frames=124 if profile else 4
        x=torch.randn(batch,frames,17,3,device=device);y=torch.randn(batch,2,device=device,dtype=torch.float64)
        identity={'kind':'synthetic','variant':variant,'frames':frames,'batch':batch,'device':str(device)}
        before={n:p.detach().cpu().clone() for n,p in m.named_parameters()}
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.perf_counter();l1=step(m,opt,x,y);checkpoint(out/(variant+'.pt'),m,opt,1,identity);l2=step(m,opt,x,y)
        if device.type=='cuda': torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
        after={n:p.detach().cpu().clone() for n,p in m.named_parameters()}
        changed=[n for n,p in after.items() if not torch.equal(p,before[n])]
        if not changed or any(not dict(m.named_parameters())[n].requires_grad for n in changed): raise ValueError('training scope violated')
        if variant!='head_only' and not any(n.startswith('encoder.') for n in changed): raise ValueError('encoder did not change')
        scope=sum(p.numel() for p in m.parameters() if p.requires_grad)
        del opt,m
        m=Scorer(variant,0).to(device);opt=optimizer(m,cfg)
        try: restore(out/(variant+'.pt'),m,opt,{'wrong':'job'},device)
        except ValueError: pass
        else: raise ValueError('wrong checkpoint accepted')
        assert restore(out/(variant+'.pt'),m,opt,identity,device)==1
        resumed_loss=step(m,opt,x,y)
        diff=max(float(abs(p.detach().cpu()-after[n]).max()) for n,p in m.named_parameters())
        if diff!=0 or resumed_loss!=l2: raise ValueError('resume differs from uninterrupted run')
        results.append({'variant':variant,'trainable_parameters':scope,'changed_tensors':len(changed),'resume_weight_max_error':diff,'two_steps_seconds_including_checkpoint':elapsed,'peak_allocated_bytes':peak,'loss1':l1,'loss2':l2})
        del m,opt,before,after,x,y
    save(out/'checks.json',{'synthetic_only':True,'real_data_optimizer_steps':0,'torch':str(torch.__version__),'device':str(device),'frames':frames,'batch':batch,'results':results})
    print(json.dumps(results))

def replay(out,device):
    d=data();rows=[]
    for fold in range(5):
        i=int(np.flatnonzero(d['fold']==fold)[0]);m=Scorer('head_only',fold).to(device)
        for w in range(2):
            with torch.inference_mode():
                actual=m(torch.from_numpy(d['x'][i:i+1,w]).to(device)).cpu().numpy()
                expected=m(torch.from_numpy(d['features'][i:i+1,w]).to(device)).cpu().numpy()
            err=float(abs(actual-expected).max())
            if err>1e-4: raise ValueError('baseline startup score differs '+str(err))
            rows.append({'fold':fold,'window':w,'max_score_error':err})
        del m
    bad={k:v.copy() for k,v in d.items()};p=bad['person_id'][0];idx=np.flatnonzero(bad['person_id']!=p)[0];bad['person_id'][idx]=p
    try: validate_data(bad)
    except ValueError: pass
    else: raise ValueError('corrupted identity accepted')
    try: protocol(ROOT/'protocol.draft.json')
    except ValueError: pass
    else: raise ValueError('draft training accepted')
    save(out/'replay.json',{'rows':rows,'identity_corruption_rejected':True,'draft_rejected':True,'real_data_optimizer_steps':0,'people':175,'records':203})
    print(json.dumps({'replay_max':max(r['max_score_error'] for r in rows),'records':203,'real_data_optimizer_steps':0}))

def train(args,device):
    cfg=protocol(args.config);d=data()
    if args.variant not in cfg['variants'] or args.fold not in cfg['folds'] or args.seed not in cfg['seeds']: raise ValueError('job not in protocol')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    identity={'config':sha(args.config),'manifest':sha(ROOT/'manifest.json'),'variant':args.variant,'fold':args.fold,'seed':args.seed,'torch':str(torch.__version__),'device_type':device.type}
    if (out/'job.json').exists() and load(out/'job.json')!=identity: raise ValueError('output belongs to another job')
    save(out/'job.json',identity);environment(args.seed)
    if (out/'complete.json').exists():
        done=load(out/'complete.json')
        if done['identity']!=identity or sha(out/'predictions.npz')!=done['predictions_sha256']: raise ValueError('completion evidence mismatch')
        print('already complete');return
    m=Scorer(args.variant,args.fold).to(device);opt=optimizer(m,cfg)
    tr=np.flatnonzero(d['fold']!=args.fold);te=np.flatnonzero(d['fold']==args.fold)
    if cfg['batch_size']>len(tr): raise ValueError('batch size exceeds training data')
    before={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad}
    ck=out/'checkpoint.pt';start=restore(ck,m,opt,identity,device) if ck.exists() else 0
    if start>cfg['steps']: raise ValueError('checkpoint exceeds declared budget')
    batches=math.ceil(len(tr)/cfg['batch_size']);loss=None
    for j in range(start,cfg['steps']):
        epoch,part=divmod(j,batches);g=torch.Generator().manual_seed(args.seed+epoch)
        order=torch.randperm(len(tr),generator=g).numpy();ix=tr[order[part*cfg['batch_size']:(part+1)*cfg['batch_size']]]
        arr=d['features'][ix,0] if args.variant=='head_only' else d['x'][ix,0]
        loss=step(m,opt,torch.from_numpy(arr).to(device),torch.from_numpy(d['y'][ix]).to(device))
        if (j+1)%cfg['checkpoint_every']==0 or j+1==cfg['steps']: checkpoint(ck,m,opt,j+1,identity)
    for n,p in m.named_parameters():
        if n in before and not torch.equal(before[n],p.detach().cpu()): raise ValueError('frozen weights changed')
    predictions=np.empty((len(te),2,2))
    with torch.inference_mode():
        for a,i in enumerate(te):
            for w in range(2):
                arr=d['features'][i:i+1,w] if args.variant=='head_only' else d['x'][i:i+1,w]
                predictions[a,w]=m(torch.from_numpy(arr).to(device)).cpu().numpy()[0]
    if not np.isfinite(predictions).all(): raise ValueError('nonfinite prediction')
    np.savez_compressed(out/'predictions.npz',indices=te,predictions=predictions)
    save(out/'complete.json',{'identity':identity,'steps':cfg['steps'],'last_loss_this_process':loss,'predictions_sha256':sha(out/'predictions.npz'),'frozen_unchanged':True})
    print(json.dumps({'complete':str(out),'steps':cfg['steps']}))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','smoke','profile','train']);p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--variant',choices=VARIANTS);p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);p.add_argument('--batch-size',type=int,default=1);a=p.parse_args()
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    if device.type not in ('cpu','cuda'): raise ValueError('unsupported device')
    environment(20261029);data();out=Path(a.output)
    if a.action=='check': replay(out,device)
    elif a.action in ('smoke','profile'): smoke(out,device,a.action=='profile',a.batch_size)
    else: train(a,device)
if __name__=='__main__': main()
