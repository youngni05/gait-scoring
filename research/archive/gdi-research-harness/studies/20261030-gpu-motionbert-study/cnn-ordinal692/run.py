"""Nested person-disjoint ordinal CNN experiments; no protected cohort access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,json,math,time
import numpy as np
import torch
from ordinal import OrdinalCNN,objective,probabilities,targets,boundaries
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(str(t),str(p))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def setup(seed,device):
    torch.set_num_threads(4);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device=='cuda':
        assert torch.cuda.is_available();torch.cuda.manual_seed_all(seed);torch.empty(1,device='cuda');torch.cuda.synchronize()
def data():
    for name,dig in read(H/'manifest.json').items():assert sha(H/name)==dig,name
    path=Path(os.environ.get('GDI_ORDINAL_DATA',str(H/'data.npz')));c=read(H/'protocol.json');assert sha(path)==c['data_sha256']
    d=arrays(path);assert d['x'].shape==(11261,2,124,12) and len(d['y'])==1061
    assert len(np.unique(d['person_id']))==692 and len(np.unique(d['video_id']))==1061
    assert np.isfinite(d['x']).all() and np.isfinite(d['y']).all()
    d['_windows']=[np.flatnonzero(d['window_record']==i) for i in range(len(d['y']))]
    assert all(len(ix)>0 and len(set(d['window_start'][ix]))==len(ix) for ix in d['_windows'])
    for f in range(5):
        for j in range(5):
            if j!=f:split(d,f,j)
    return d
def split(d,f,j=None):
    te=np.flatnonzero(d['fold']==f);va=np.flatnonzero(d['fold']==j) if j is not None else np.array([],dtype=int)
    tr=np.flatnonzero((d['fold']!=f)&(d['fold']!=j)) if j is not None else np.flatnonzero(d['fold']!=f)
    sets=[set(d['person_id'][ix]) for ix in (tr,va,te)]
    assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]);return tr,va,te
def spec(group):return (10 if group=='regression' else int(group[-2:]),'regression' if group=='regression' else group[:-2])
def packed(d,indices,device,dtype=torch.float32):
    pieces=[d['_windows'][int(i)] for i in indices];ix=np.concatenate(pieces)
    x=torch.from_numpy(d['x'][ix]).transpose(2,3).contiguous().reshape(-1,12,124).to(device=device,dtype=dtype)
    return x,[len(a) for a in pieces]
def weights(ids):
    _,inv,count=np.unique(ids,return_inverse=True,return_counts=True)
    return (len(ids)/(len(count)*count[inv])).astype(np.float32)
def macro(v,ids):return float(np.mean([v[ids==i].mean() for i in np.unique(ids)]))
def metrics(y,raw,prob,ids,width):
    y=np.asarray(y,dtype=np.float64);raw=np.asarray(raw,dtype=np.float64);prob=np.asarray(prob,dtype=np.float64)
    result={}
    if raw.size:result['mae']=macro(abs(y-raw),ids)
    if prob.size:
        assert prob.shape==(*y.shape,len(boundaries(width))+1) and np.isfinite(prob).all()
        assert prob.min()>=-1e-12 and abs(prob.sum(-1)-1).max()<1e-6
        cls=targets(y,width);pred=prob.argmax(-1);one=np.eye(prob.shape[-1])[cls]
        result.update(accuracy=macro(pred==cls,ids),two_or_more=macro(abs(pred-cls)>=2,ids),
            nll=macro(-np.log(np.maximum(np.take_along_axis(prob,cls[...,None],axis=-1)[...,0],1e-12)),ids),
            brier=macro(((prob-one)**2).sum(-1),ids),
            rps=macro(((prob.cumsum(-1)-one.cumsum(-1))[...,:-1]**2).mean(-1),ids))
    return result
def predict(m,d,indices,lo,span,device,dtype=torch.float32):
    m.to(device=device,dtype=dtype).eval();raw=[];ps=[]
    with torch.inference_mode():
        for start in range(0,len(indices),2):
            x,counts=packed(d,indices[start:start+2],device,dtype);r,l=m(x,counts)
            if r is not None:raw.append((r*span+lo).cpu().numpy())
            if l is not None:ps.append(probabilities(l).cpu().numpy())
    return np.concatenate(raw) if raw else np.empty((0,2)),np.concatenate(ps) if ps else np.empty((0,2,0))
def identity(d,c,group,f,j,lam,epochs):
    tr,va,te=split(d,f,j)
    return dict(group=group,outer=f,inner=j,lam=lam,epochs=epochs,seed=c['seed']+100*f+(50 if j is None else j),
        manifest_sha256=sha(H/'manifest.json'),data_sha256=c['data_sha256'],
        train_people=np.unique(d['person_id'][tr]).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),outer_people=np.unique(d['person_id'][te]).tolist(),
        train_y_sha256=digest(d['y'][tr]),train_rows=tr.tolist(),validation_rows=va.tolist())
def fit(d,c,group,f,j,lam,epochs,out,device,stop_after=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);ident=identity(d,c,group,f,j,lam,epochs)
    if (out/'fit_complete.json').exists():audit(d,c,group,f,j,lam,epochs,out);return read(out/'curve.json')
    if (out/'identity.json').exists():assert read(out/'identity.json')==ident
    save(out/'identity.json',ident);tr,va,_=split(d,f,j);width,mode=spec(group);setup(ident['seed'],device)
    lo=float(np.float32(d['y'][tr].min()));span=float(np.float32(max(np.ptp(d['y'][tr]),1e-6)))
    m=OrdinalCNN(width,mode).to(device);opt=torch.optim.RMSprop(m.parameters(),lr=c['learning_rate'],alpha=c['rmsprop_alpha'],eps=c['rmsprop_eps'])
    history=[];w=weights(d['person_id'][tr]);steps=0
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location=device,weights_only=True);assert ck['identity']==ident
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];steps=ck['steps']
        torch.set_rng_state(ck['rng_cpu'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng_cuda']])
    for epoch in range(len(history),epochs):
        started=time.monotonic();m.train();lr=c['learning_rate']*c['lr_drop_factor']**(epoch//c['lr_drop_every'])
        for param in opt.param_groups:param['lr']=lr
        order=np.random.default_rng(ident['seed']+epoch).permutation(len(tr));loss_sum=0.
        for start in range(0,len(tr),c['batch_records']):
            ix=order[start:start+c['batch_records']];rows=tr[ix];x,counts=packed(d,rows,device)
            opt.zero_grad(set_to_none=True);score,logits=m(x,counts)
            y=torch.as_tensor(d['y'][rows],device=device,dtype=torch.float32);ww=torch.from_numpy(w[ix]).to(device)
            loss,_=objective(m,score,logits,y,ww,lo,span,lam);assert bool(torch.isfinite(loss));loss.backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
            opt.step();steps+=1;assert all(bool(torch.isfinite(p).all()) for p in m.parameters());loss_sum+=float(loss.detach())*len(rows)
        raw,pr=predict(m,d,tr,lo,span,device);vr,vp=predict(m,d,va,lo,span,device) if len(va) else (np.empty((0,2)),np.empty((0,2,0)))
        path=out/'raw_predictions'/('epoch%04d.npz'%(epoch+1));path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,train_raw=raw,train_prob=pr,validation_raw=vr,validation_prob=vp)
        row=dict(epoch=epoch+1,lr=lr,loss=loss_sum/len(tr),train=metrics(d['y'][tr],raw,pr,d['person_id'][tr],width),
            validation=metrics(d['y'][va],vr,vp,d['person_id'][va],width) if len(va) else {},
            validation_people=len(np.unique(d['person_id'][va])),predictions_sha256=sha(path),seconds=time.monotonic()-started)
        history.append(row)
        ck=dict(model=m.state_dict(),optimizer=opt.state_dict(),identity=ident,history=history,lo=lo,span=span,steps=steps,
            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        torch.save(ck,out/'latest.tmp');os.replace(str(out/'latest.tmp'),str(out/'latest.pt'));save(out/'curve.json',history)
        print(json.dumps(dict(fit=str(out),**row)),flush=True)
        if stop_after is not None and len(history)>=stop_after and len(history)<epochs:return history
    save(out/'fit_complete.json',dict(identity=ident,epochs=epochs,steps=steps,checkpoint_sha256=sha(out/'latest.pt'),curve_sha256=sha(out/'curve.json')))
    return history
def audit(d,c,group,f,j,lam,epochs,out):
    tr,va,_=split(d,f,j);ident=identity(d,c,group,f,j,lam,epochs);done=read(out/'fit_complete.json');h=read(out/'curve.json')
    assert done['identity']==read(out/'identity.json')==ident and len(h)==epochs
    assert sha(out/'latest.pt')==done['checkpoint_sha256'] and sha(out/'curve.json')==done['curve_sha256']
    assert done['steps']==epochs*math.ceil(len(tr)/c['batch_records'])
    ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=True);assert ck['identity']==ident and ck['history']==h
    assert ck['lo']==float(np.float32(d['y'][tr].min())) and ck['span']==float(np.float32(max(np.ptp(d['y'][tr]),1e-6)))
    width,_=spec(group)
    for e,row in enumerate(h,1):
        p=out/'raw_predictions'/('epoch%04d.npz'%e);assert sha(p)==row['predictions_sha256'] and row['epoch']==e
        z=arrays(p)
        for label,ix in [('train',tr),('validation',va)]:
            if not len(ix):assert row[label]=={} and len(z[label+'_raw'])==len(z[label+'_prob'])==0;continue
            got=metrics(d['y'][ix],z[label+'_raw'],z[label+'_prob'],d['person_id'][ix],width)
            for k,v in got.items():assert abs(v-row[label][k])<1e-10,(label,k,v,row[label][k])
    return h
def replay(d,c,group,f,j,lam,epochs,out,device='cpu'):
    audit(d,c,group,f,j,lam,epochs,out);ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
    setup(ck['identity']['seed'],device);width,mode=spec(group);m=OrdinalCNN(width,mode);m.load_state_dict(ck['model'])
    dtype=torch.float64 if device=='cpu' else torch.float32;tr,va,_=split(d,f,j);z=arrays(out/'raw_predictions'/('epoch%04d.npz'%epochs));errs={}
    for label,ix in [('train',tr),('validation',va)]:
        if not len(ix):continue
        raw,p=predict(m,d,ix,ck['lo'],ck['span'],device,dtype)
        for key,v in [('raw',raw),('prob',p)]:
            if v.size:errs[label+'_'+key]=float(abs(v-z[label+'_'+key]).max())
    result=dict(passed=max(errs.values())<1e-4,device=device,arithmetic=str(dtype),errors=errs,tolerance=1e-4,checkpoint_sha256=sha(out/'latest.pt'))
    save(out/('replay-'+device+'.json'),result);assert result['passed'],result;return result
def choose(histories,group,lambdas):
    candidates=[]
    for lam in lambdas:
        hs=histories[str(lam)];weights_=[h[0]['validation_people'] for h in hs]
        for e in range(len(hs[0])):
            vals={k:float(np.average([h[e]['validation'][k] for h in hs],weights=weights_)) for k in hs[0][e]['validation']}
            key=(vals['mae'],e,lambdas.index(lam)) if group=='regression' else (-vals['accuracy'],vals['nll'],e,lambdas.index(lam))
            candidates.append((key,dict(lam=lam,epoch=e+1,metrics=vals)))
    return min(candidates,key=lambda x:x[0])[1], [v for k,v in candidates]
def train_outer(d,c,group,f,root,device):
    dest=root/group/('fold%d'%f);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    lambdas=c['joint_lambdas'] if group.startswith('joint') else [1.0];histories={}
    for lam in lambdas:
        hs=[]
        for j in range(5):
            if j==f:continue
            folder=dest/('lambda'+str(lam))/('inner%d'%j)
            fit(d,c,group,f,j,lam,c['max_epochs'],folder,device)
            hs.append(audit(d,c,group,f,j,lam,c['max_epochs'],folder))
        histories[str(lam)]=hs
    chosen,candidates=choose(histories,group,lambdas);save(dest/'selection.json',dict(chosen=chosen,candidates=candidates,outer_used=False))
    lam,epochs=chosen['lam'],chosen['epoch'];folder=dest/'refit'
    fit(d,c,group,f,None,lam,epochs,folder,device);audit(d,c,group,f,None,lam,epochs,folder)
    ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True);width,mode=spec(group);m=OrdinalCNN(width,mode);m.load_state_dict(ck['model'])
    _,_,te=split(d,f);raw,p=predict(m,d,te,ck['lo'],ck['span'],device)
    np.savez_compressed(dest/'predictions.npz',raw=raw,prob=p,row_index=te,**{k:d[k][te] for k in ['y','person_id','video_id','fold']})
    save(dest/'complete.json',dict(group=group,fold=f,chosen=chosen,metrics=metrics(d['y'][te],raw,p,d['person_id'][te],width),
        predictions_sha256=sha(dest/'predictions.npz'),checkpoint_sha256=sha(folder/'latest.pt'),selection_sha256=sha(dest/'selection.json')))
def summarize(d,c,root):
    summaries={}
    for group in c['groups']:
        width,mode=spec(group);raw=np.full(d['y'].shape,np.nan) if mode!='ordinal' else np.empty((0,2))
        p=np.full((*d['y'].shape,len(boundaries(width))+1),np.nan) if mode!='regression' else np.empty((0,2,0));seen=np.zeros(len(d['y']),int)
        for f in range(5):
            dest=root/group/('fold%d'%f);done=read(dest/'complete.json');assert sha(dest/'predictions.npz')==done['predictions_sha256']
            z=arrays(dest/'predictions.npz');ix=z['row_index'];assert np.array_equal(ix,np.flatnonzero(d['fold']==f))
            for key in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(z[key],d[key][ix])
            if raw.size:raw[ix]=z['raw']
            if p.size:p[ix]=z['prob']
            seen[ix]+=1
        assert (seen==1).all();np.savez_compressed(root/group/'oof.npz',raw=raw,prob=p,**{k:d[k] for k in ['y','person_id','video_id','fold']})
        summaries[group]=dict(metrics=metrics(d['y'],raw,p,d['person_id'],width),oof_sha256=sha(root/group/'oof.npz'))
    save(root/'summary.json',dict(groups=summaries,people=len(np.unique(d['person_id'])),records=len(d['y']),main_fits=165,outer_predictions=25,independent_final=False))
    return summaries
