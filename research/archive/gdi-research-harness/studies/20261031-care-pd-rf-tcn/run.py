"""Matched person-held-out RF and small temporal CNN comparison."""
import argparse
import csv
import hashlib
import json
import os
import random
import socket
import time
from collections import Counter
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


def weights(person):
    count = Counter(person)
    w = np.array([1/count[p] for p in person], dtype=np.float64)
    return w / w.mean()


def mae(y, p, person):
    return float(np.average(np.abs(y-p), weights=weights(person)))


def metrics(y, p, person):
    w=weights(person); e=p-y; c=np.clip(np.floor(p+.5),0,3).astype(int)
    recalls=[float(np.average(c[y==k]==k, weights=w[y==k])) for k in range(4) if np.any(y==k)]
    return dict(people=len(set(person)), walks=len(y), MAE=float(np.average(abs(e),weights=w)),
        RMSE=float(np.sqrt(np.average(e**2,weights=w))), bias=float(np.average(e,weights=w)),
        large_error_rate=float(np.average(abs(e)>=2,weights=w)),
        accuracy=float(np.average(c==y,weights=w)), macro_recall=float(np.mean(recalls)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs=nn.ModuleList([nn.Conv1d(138 if j==0 else 32,32,5,padding=2*d,dilation=d)
                                 for j,d in enumerate([1,2,4])])
        self.head=nn.Linear(32,1)
    def forward(self,x,mask):
        for conv in self.convs:
            x=torch.relu(conv(x))*mask
        pooled=x.sum(-1)/mask.sum(-1)
        return 3*torch.sigmoid(self.head(pooled).squeeze(1))


def normalize(d, train):
    w=weights(d['person'][train]); w=w/w.sum()
    means=[]; seconds=[]
    for i in train:
        x=d['x'][d['offsets'][i]:d['offsets'][i+1]].astype(np.float64)
        means.append(x.mean(0)); seconds.append((x*x).mean(0))
    mu=np.average(means,axis=0,weights=w)
    sd=np.sqrt(np.maximum(np.average(seconds,axis=0,weights=w)-mu*mu,1e-6))
    return mu.astype(np.float32),sd.astype(np.float32)


def tensor_batch(d, indices, mu, sd, device):
    lengths=np.diff(d['offsets'])[indices]
    x=np.zeros((len(indices),138,int(lengths.max())),np.float32)
    mask=np.zeros((len(indices),1,int(lengths.max())),np.float32)
    for j,i in enumerate(indices):
        seq=d['x'][d['offsets'][i]:d['offsets'][i+1]]
        x[j,:,:len(seq)]=((seq-mu)/sd).T
        mask[j,:,:len(seq)]=1
    return torch.from_numpy(x).to(device),torch.from_numpy(mask).to(device)


def batches(d, indices, rng=None):
    # Nearby lengths reduce padding without cropping or duplicating walks.
    order=np.array(indices)[np.argsort(np.diff(d['offsets'])[indices],kind='stable')]
    chunks=[order[i:i+32] for i in range(0,len(order),32)]
    if rng is not None:
        rng.shuffle(chunks)
    return chunks


def infer(d, model, indices, mu, sd, device='cuda'):
    model.eval(); p=np.empty(len(d['y']),np.float32)
    with torch.no_grad():
        for ids in batches(d,indices):
            x,m=tensor_batch(d,ids,mu,sd,device)
            p[ids]=model(x,m).cpu().numpy()
    return p[indices]


def fit_cnn(d, train, validation, seed, epochs, output):
    output.mkdir(parents=True,exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    mu,sd=normalize(d,train)
    model=Net().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(seed); w=np.zeros(len(d['y']),np.float32); w[train]=weights(d['person'][train])
    best=float('inf'); selected=epochs; curve=[]; start=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        for ids in batches(d,train,rng):
            x,m=tensor_batch(d,ids,mu,sd,'cuda')
            target=torch.as_tensor(d['y'][ids],device='cuda')
            ww=torch.as_tensor(w[ids],device='cuda')
            opt.zero_grad(set_to_none=True)
            loss=(nn.functional.smooth_l1_loss(model(x,m),target,beta=.5,reduction='none')*ww).mean()
            assert torch.isfinite(loss)
            loss.backward(); opt.step(); total+=float(loss.detach())*len(ids)
        row=dict(epoch=epoch,train_loss=total/len(train),elapsed_seconds=time.monotonic()-start)
        if validation is not None:
            p=infer(d,model,validation,mu,sd)
            value=mae(d['y'][validation],p,d['person'][validation]); row['validation_MAE']=value
            if value<best:
                best=value; selected=epoch
            curve.append(row)
            if epoch-selected>=10:
                break
        else:
            curve.append(row)
        if epoch==1 or epoch%10==0:
            print(str(output.relative_to(HERE)),row,flush=True)
        save(output/'curve.json',curve)
    save(output/'curve.json',curve)
    checkpoint=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},
        mu=torch.from_numpy(mu),sd=torch.from_numpy(sd),train=list(map(int,train)),
        validation=[] if validation is None else list(map(int,validation)), seed=int(seed),epochs=len(curve))
    torch.save(checkpoint,output/'model.pt')
    save(output/'fit_complete.json',dict(epochs=len(curve),selected_epoch=selected,
        seconds=time.monotonic()-start,peak_vram_bytes=torch.cuda.max_memory_allocated()))
    return model,mu,sd,selected


def split(d, fold):
    outer=np.flatnonzero(d['fold']!=fold); test=np.flatnonzero(d['fold']==fold)
    valid=np.flatnonzero(d['fold']==((fold+1)%6))
    train=np.flatnonzero((d['fold']!=fold)&(d['fold']!=((fold+1)%6)))
    sets=[set(d['person'][ids]) for ids in (train,valid,test)]
    assert not any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2)))
    return train,valid,outer,test


def baseline(d, train, test):
    w=weights(d['person'][train]); y=d['y'][train]
    by=np.argsort(y,kind='stable'); median=float(y[by][np.searchsorted(np.cumsum(w[by]),w.sum()/2)])
    counts=[w[y==k].sum() for k in range(4)]
    return dict(mean=np.full(len(test),np.average(y,weights=w)),median=np.full(len(test),median),
        mode=np.full(len(test),np.argmax(counts)))


def rf_fit(d,train,seed,leaf):
    model=RandomForestRegressor(n_estimators=200,max_features=.5,min_samples_leaf=leaf,
                               random_state=seed,n_jobs=2)
    model.fit(d['rf'][train],d['y'][train],sample_weight=weights(d['person'][train]))
    return model


def checks(d):
    assert len(d['y'])==2953 and len(set(d['person']))==110
    assert np.isfinite(d['x']).all() and np.isfinite(d['rf']).all()
    for f in range(6): split(d,f)
    t,v,_,_=split(d,0); mu,sd=normalize(d,t)
    # Alter held-out labels: fitted normalization must remain exactly unchanged.
    changed=dict(d); changed['y']=d['y'].copy(); changed['y'][v]=99
    mu2,sd2=normalize(changed,t); assert np.array_equal(mu,mu2) and np.array_equal(sd,sd2)
    net=Net().cuda().eval(); ids=t[:2]; x,m=tensor_batch(d,ids,mu,sd,'cuda')
    with torch.no_grad():
        a=net(x,m); b=net(nn.functional.pad(x,(0,17)),nn.functional.pad(m,(0,17)))
    assert torch.allclose(a,b,atol=1e-6,rtol=0)
    return dict(person_splits=True,finite_inputs=True,heldout_normalization_invariance=True,
        padding_invariance_max_error=float((a-b).abs().max()),parameters=sum(p.numel() for p in net.parameters()))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['pilot','train','replay'])
    parser.add_argument('--folds',default='0,1,2,3,4,5'); a=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'GPU execution requires Slurm allocation'
    assert torch.cuda.is_available(); torch.cuda.init()
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert hashlib.sha256((HERE/'data.npz').read_bytes()).hexdigest()==protocol['data_sha256']
    d=dict(np.load(HERE/'data.npz',allow_pickle=False))
    if a.mode=='replay':
        out=HERE/'runs/main'; worst=0.; n=0
        for f in range(6):
            if f not in [int(v) for v in a.folds.split(',')]:
                continue
            *_,test=split(d,f)
            for seed in protocol['seeds']:
                folder=out/('fold%d'%f)/('seed%d'%seed)
                ck=torch.load(folder/'refit/model.pt',weights_only=True,map_location='cpu')
                assert ck['train']==list(map(int,np.flatnonzero(d['fold']!=f)))
                mu,sd=normalize(d,np.flatnonzero(d['fold']!=f))
                assert np.array_equal(mu,ck['mu'].numpy()) and np.array_equal(sd,ck['sd'].numpy())
                model=Net().cuda(); model.load_state_dict(ck['state'])
                p=infer(d,model,test,ck['mu'].numpy(),ck['sd'].numpy())
                saved=np.load(folder/'predictions.npz')
                err=float(np.max(abs(p-saved['cnn']))); worst=max(worst,err); assert err<=1e-6
                rf=joblib.load(folder/'rf.joblib'); assert np.max(abs(rf.predict(d['rf'][test])-saved['rf']))<=1e-12
                n+=1
        save(out/('replay-'+a.folds.replace(',','-')+'.json'),dict(passed=True,checkpoints=n,max_cnn_error=worst,rf_models=n,folds=a.folds))
        return
    environment=dict(host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(0),numpy=np.__version__)
    if a.mode=='pilot':
        output=HERE/'runs/pilot'; output.mkdir(parents=True,exist_ok=True)
        pre=checks(d); t,v,_,_=split(d,0)
        model,mu,sd,_=fit_cnn(d,t,v,protocol['seeds'][0],3,output/'fit')
        p=infer(d,model,v,mu,sd)
        ck=torch.load(output/'fit/model.pt',weights_only=True,map_location='cpu'); net=Net().cuda();net.load_state_dict(ck['state'])
        replay=infer(d,net,v,ck['mu'].numpy(),ck['sd'].numpy()); error=float(np.max(abs(replay-p))); assert error<=1e-6
        start=time.monotonic(); rf=rf_fit(d,t,protocol['seeds'][0],8)
        pred=rf.predict(d['rf'][v]); assert np.isfinite(pred).all()
        save(output/'summary.json',dict(environment=environment,checks=pre,checkpoint_replay_error=error,
            rf_fit_seconds=time.monotonic()-start,cnn_fit=json.loads((output/'fit/fit_complete.json').read_text()),passed=True))
        print('CARE_PD_PILOT_COMPLETE',flush=True); return
    out=HERE/'runs/main'; out.mkdir(parents=True,exist_ok=True)
    save(out/('environment-'+os.environ['SLURM_JOB_ID']+'.json'),environment); records=[]
    for f in range(6):
        if f not in [int(v) for v in a.folds.split(',')]:
            continue
        train,valid,outer,test=split(d,f)
        for seed in protocol['seeds']:
            folder=out/('fold%d'%f)/('seed%d'%seed); folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'complete.json').exists():
                choices=[]
                for leaf in (2,8):
                    rf=rf_fit(d,train,seed,leaf)
                    value=mae(d['y'][valid],rf.predict(d['rf'][valid]),d['person'][valid])
                    choices.append(dict(leaf=leaf,inner_MAE=value))
                leaf=min(choices,key=lambda r:r['inner_MAE'])['leaf']
                rf=rf_fit(d,outer,seed,leaf); rp=rf.predict(d['rf'][test]); joblib.dump(rf,folder/'rf.joblib')
                _,_,_,epoch=fit_cnn(d,train,valid,seed,60,folder/'inner')
                model,mu,sd,_=fit_cnn(d,outer,None,seed,epoch,folder/'refit')
                cp=infer(d,model,test,mu,sd)
                np.savez_compressed(folder/'predictions.npz',indices=test,cnn=cp,rf=rp,**baseline(d,outer,test))
                save(folder/'complete.json',dict(seed=seed,fold=f,selected_epoch=epoch,rf_choices=choices,
                    rf_leaf=leaf,inner_train=list(map(int,train)),validation=list(map(int,valid)),
                    outer_train=list(map(int,outer)),test=list(map(int,test))))
                print('COMPLETED',f,seed,flush=True)
        save(out/('fold%d'%f)/'complete.json',dict(seeds=5))
    for f in range(6):
        for seed in protocol['seeds']:
            folder=out/('fold%d'%f)/('seed%d'%seed)
            if not (folder/'complete.json').exists():
                print('ASSIGNED_FOLDS_COMPLETE',a.folds,flush=True); return
            saved=np.load(folder/'predictions.npz',allow_pickle=False)
            for name in ('mean','median','mode','rf','cnn'):
                for i,p in zip(saved['indices'],saved[name]):
                    records.append(dict(index=int(i),person=str(d['person'][i]),cohort=str(d['cohort'][i]),
                        fold=f,seed=seed,model=name,y=float(d['y'][i]),prediction=float(p)))
    with (out/'predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0])); writer.writeheader();writer.writerows(records)
    summary=[]
    for seed in protocol['seeds']:
        for name in ('mean','median','mode','rf','cnn'):
            rs=[r for r in records if r['seed']==seed and r['model']==name]
            y=np.array([r['y'] for r in rs]);p=np.array([r['prediction'] for r in rs]);person=np.array([r['person'] for r in rs])
            for group,keep in [('all',np.ones(len(y),bool)),('high_2_3',y>=2),('score3',y==3)]:
                summary.append(dict(seed=seed,model=name,group=group,**metrics(y[keep],p[keep],person[keep])))
    save(out/'summary.json',dict(complete=True,outer_folds=6,seeds=5,comparisons=30,metrics=summary))
    save(HERE/'runs/summary_complete.json',dict(complete=True,outer_folds=6,seeds=5))
    print('CARE_PD_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    main()
