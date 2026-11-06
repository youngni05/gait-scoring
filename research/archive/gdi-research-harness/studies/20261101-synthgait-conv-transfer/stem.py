"""Native-length cadence pretraining and convolution-only transfer. Python3.8+."""
from pathlib import Path
import hashlib,json,time,os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import GdiCNN

class CadenceNet(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        base=GdiCNN(dropout)
        self.convs,self.norms,self.dropout=base.convs,base.norms,base.dropout
        self.head=nn.Linear(32,1)
        nn.init.xavier_uniform_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1]!=12 or x.shape[2]<12:raise ValueError('Native B,12,T; T>=12, no fabricated frames')
        for i,(conv,norm) in enumerate(zip(self.convs,self.norms)):
            x=norm(F.relu(conv(x)))
            if i%2:x=self.dropout(F.max_pool1d(x,3 if i==5 else 2))
        return self.head(x.mean(dim=2))[:,0]
    def penalty(self):return sum(c.weight.square().sum() for c in self.convs[2:])

def state_digest(state):
    h=hashlib.sha256()
    for k in sorted(state):
        v=state[k].detach().cpu().contiguous();h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()
def convolution_state(model):return {k:v for k,v in model.state_dict().items() if k.startswith('convs.')}
def checkpoint_path(root,c,seed):return Path(root)/'runs/pretrain'/c['pretraining_stage']/('seed%d'%seed)/'selected.pt'
def source_sha(root,c,seed):
    if c['initialization']=='scratch':return None
    p=checkpoint_path(root,c,seed)
    return hashlib.sha256(p.read_bytes()).hexdigest()

def initialize(model,c,seed,root):
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    proof={'initialization':c['initialization'],'fit_seed':seed,'initial_conv_sha256':state_digest(convolution_state(model)),
           'other_state_sha256':state_digest({k:v for k,v in before.items() if not k.startswith('convs.')})}
    if c['initialization']=='synth':
        path=checkpoint_path(root,c,seed);ck=torch.load(path,map_location='cpu',weights_only=True)
        assert ck['seed']==seed and ck['initial_conv_sha256']==proof['initial_conv_sha256'],'Unmatched scratch initialization'
        assert ck['external_data_sha256']==c['external_data_sha256']
        state=model.state_dict();changed=[]
        for k,v in ck['model'].items():
            if k.startswith('convs.'):
                assert k in state and v.shape==state[k].shape
                state[k].copy_(v.to(state[k].device));changed.append(k)
        assert len(changed)==12
        proof.update(pretrain_checkpoint=str(path.relative_to(root)),pretrain_checkpoint_sha256=source_sha(root,c,seed),
                     selected_pretrain_epoch=ck['selected_epoch'],transferred_keys=changed)
        assert any(not torch.equal(before[k],model.state_dict()[k]) for k in changed),'No convolution learned'
    else:assert c['initialization']=='scratch'
    for k,v in model.state_dict().items():
        if not k.startswith('convs.'):assert torch.equal(v,before[k]),'Non-convolution state changed: '+k
    assert torch.equal(cpu_rng,torch.get_rng_state())
    if cuda_rng:assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    for norm in model.norms:
        assert int(norm.num_batches_tracked)==0 and torch.equal(norm.running_mean,torch.zeros_like(norm.running_mean))
        assert torch.equal(norm.running_var,torch.ones_like(norm.running_var))
    proof.update(after_conv_sha256=state_digest(convolution_state(model)),all_other_state_unchanged=True,batchnorm_reset=True,rng_unchanged=True)
    return proof

def load_external(root,c):
    p=Path(os.environ.get('EXTERNAL_DATA_PATH',str(Path(root)/'external.npz')));assert hashlib.sha256(p.read_bytes()).hexdigest()==c['external_data_sha256']
    with np.load(p,allow_pickle=False) as z:
        xs=[z['x%02d'%i].copy() for i in range(len(z['walk_id']))]
        d={k:z[k].copy() for k in ['walk_id','split','cadence','missing_fraction','fps']};d['x']=xs
    assert len(xs)==18 and np.all(d['fps']==16) and np.isfinite(d['cadence']).all()
    tr=np.flatnonzero(d['split']=='train');va=np.flatnonzero(d['split']=='validation')
    assert len(tr)==14 and len(va)==4 and not set(d['walk_id'][tr])&set(d['walk_id'][va])
    for x in xs:assert x.shape[0]==2 and x.shape[2]==12 and 25<=x.shape[1]<=81 and np.isfinite(x).all()
    assert (d['missing_fraction']<=.25).all()
    return d,tr,va

def predict_native(model,d,indices,mean,std,device):
    model.eval();result=[];dtype=next(model.parameters()).dtype
    with torch.no_grad():
        for i in indices:
            x=torch.from_numpy(d['x'][i]).transpose(1,2).contiguous().to(device=device,dtype=dtype)
            result.append(model(x).cpu().numpy().astype(np.float64)*std+mean)
    return np.stack(result)

def fit_external(d,tr,va,c,seed,epochs,folder,device,r,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    r.setup(seed,device);model=CadenceNet(c['dropout']).to(device)
    initial=state_digest(convolution_state(model));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    mean=float(d['cadence'][tr].mean());std=float(d['cadence'][tr].std());assert std>0
    identity={'seed':seed,'epochs':epochs,'external_data_sha256':c['external_data_sha256'],
        'train_walk_ids':d['walk_id'][tr].tolist(),'validation_walk_ids':d['walk_id'][va].tolist(),
        'time_policy':'native16fps_no_resampling','label':'clip cadence, no heel-strike frame supervision',
        'normalization_mean':mean,'normalization_std':std,'initial_conv_sha256':initial,
        'manifest_sha256':r.sha(r.ROOT/'manifest.json')}
    if (folder/'identity.json').exists():assert r.load(folder/'identity.json')==identity
    else:r.save(folder/'identity.json',identity)
    history=[];best=float('inf');latest=folder/'latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location=device,weights_only=True);assert ck['identity']==identity
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);r.restore_rng(ck['rng'])
        history=ck['history'];best=ck['best_validation_mae']
    for epoch in range(len(history),epochs):
        if stop_after is not None and epoch>=stop_after:break
        t=time.perf_counter();order=np.random.default_rng(seed+epoch).permutation(tr);losses=[]
        for i in order:
            # A minibatch is exactly both sides of one native-length walk. No padding.
            x=torch.from_numpy(d['x'][i]).transpose(1,2).contiguous().to(device)
            target=x.new_full((2,),float((d['cadence'][i]-mean)/std))
            model.train();opt.zero_grad(set_to_none=True);p=model(x)
            loss=(p-target).square().mean()+c['kernel_l2']*model.penalty();assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step();losses.append(float(loss.detach()))
        a=predict_native(model,d,tr,mean,std,device);b=predict_native(model,d,va,mean,std,device)
        train=float(np.abs(a-d['cadence'][tr,None]).mean());valid=float(np.abs(b-d['cadence'][va,None]).mean())
        path=folder/('epoch%03d.npz'%(epoch+1));np.savez_compressed(path,train_predictions=a,validation_predictions=b)
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'train_cadence_mae':train,'validation_cadence_mae':valid,
            'prediction_sha256':r.sha(path),'order_sha256':hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),
            'seconds':time.perf_counter()-t};history.append(row)
        if valid<best:
            best=valid;r.save_torch(folder/'selected.pt',{'seed':seed,'selected_epoch':epoch+1,'model':model.state_dict(),
                'initial_conv_sha256':initial,'external_data_sha256':c['external_data_sha256'],'identity':identity,
                'mean':mean,'std':std,'validation_cadence_mae':valid})
        r.save_torch(latest,{'identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),
            'rng':r.rng_state(),'history':history,'best_validation_mae':best})
        r.save(folder/'curve.json',history);print(json.dumps({'pretraining_seed':seed,**row}),flush=True)
    selected=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    assert selected['selected_epoch']==1+int(np.argmin([row['validation_cadence_mae'] for row in history]))
    if len(history)==epochs:
        r.save(folder/'complete.json',{'passed':True,'epochs':epochs,'optimizer_steps':epochs*len(tr),'selected_epoch':selected['selected_epoch'],
            'checkpoint_sha256':r.sha(folder/'selected.pt'),'latest_sha256':r.sha(latest),'identity':identity})
    return folder/'selected.pt'

def audit_external(folder,c,r,device='cpu'):
    folder=Path(folder);d,tr,va=load_external(r.ROOT,c);done=r.load(folder/'complete.json');ident=r.load(folder/'identity.json')
    ck=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True);assert ck['identity']==ident==done['identity']
    assert r.sha(folder/'selected.pt')==done['checkpoint_sha256'] and r.sha(folder/'latest.pt')==done['latest_sha256']
    assert ident['train_walk_ids']==d['walk_id'][tr].tolist() and ident['validation_walk_ids']==d['walk_id'][va].tolist()
    assert ident['normalization_mean']==float(d['cadence'][tr].mean()) and ident['normalization_std']==float(d['cadence'][tr].std())
    hist=r.load(folder/'curve.json');assert len(hist)==done['epochs'] and done['optimizer_steps']==len(tr)*len(hist)
    for row in hist:
        path=folder/('epoch%03d.npz'%row['epoch']);assert r.sha(path)==row['prediction_sha256']
        order=np.random.default_rng(ident['seed']+row['epoch']-1).permutation(tr)
        assert hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest()==row['order_sha256']
        with np.load(path) as z:
            assert abs(float(np.abs(z['train_predictions']-d['cadence'][tr,None]).mean())-row['train_cadence_mae'])<1e-10
            assert abs(float(np.abs(z['validation_predictions']-d['cadence'][va,None]).mean())-row['validation_cadence_mae'])<1e-10
    assert ck['selected_epoch']==1+int(np.argmin([x['validation_cadence_mae'] for x in hist]))
    model=CadenceNet(c['dropout']);model.load_state_dict(ck['model']);model.to(device)
    if device=='cpu':model.double()
    error=0.
    with np.load(folder/('epoch%03d.npz'%ck['selected_epoch'])) as z:
        for ix,key in [(tr,'train_predictions'),(va,'validation_predictions')]:
            got=predict_native(model,d,ix,ck['mean'],ck['std'],device)
            error=max(error,float(abs(got-z[key]).max()))
    assert error<1e-4,error
    return {'passed':True,'device':device,'reference_dtype':'float64' if device=='cpu' else 'float32',
        'max_cadence_bpm_error':error,'tolerance':1e-4,'checkpoint_sha256':r.sha(folder/'selected.pt'),
        'epochs':len(hist),'train_walks':len(tr),'validation_walks':len(va),'GDI_used':False}
