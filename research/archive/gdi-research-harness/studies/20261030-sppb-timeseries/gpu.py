"""Small walking-only CNN/TCN; nested person CV, Slurm allocated GPU only."""
from common import *
import csv
import os
import time
import socket
import torch
from torch import nn

torch.set_num_threads(2)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class Net(nn.Module):
    def __init__(self, name):
        super().__init__()
        layers=[];cin=54
        for dilation in ([1,1,1] if name=='CNN' else [1,2,4]):
            layers += [nn.Conv1d(cin,24,5,padding=2*dilation,dilation=dilation),nn.ReLU(),nn.Dropout(.2)]
            cin=24
        self.body=nn.Sequential(*layers)
        self.head=nn.Linear(25,1)
    def forward(self,x,framecount):
        z=self.body(x).mean(-1)
        return self.head(torch.cat([z,framecount],dim=1)).squeeze(1)

def context(d,train):
    w=weights(d['ids'][train]); w/=w.sum()
    x=d['x'][train].astype(np.float64)
    mu=np.sum(x.mean(-1)*w[:,None],axis=0)[None,:,None]
    sd=np.sqrt(np.sum(((x-mu)**2).mean(-1)*w[:,None],axis=0))[None,:,None]
    sd=np.maximum(sd,.001)
    f=np.log1p(d['frames']).astype(np.float64)[:,None]
    fm=np.average(f[train],axis=0,weights=w); fs=np.maximum(np.sqrt(np.average((f[train]-fm)**2,axis=0,weights=w)),.001)
    ym=float(np.average(d['y'][train],weights=w));ys=max(float(np.sqrt(np.average((d['y'][train]-ym)**2,weights=w))),.1)
    norm={'xmean':mu,'xstd':sd,'fmean':fm,'fstd':fs,'ymean':ym,'ystd':ys}
    tx=torch.as_tensor((d['x']-mu)/sd,dtype=torch.float32,device='cuda')
    tf=torch.as_tensor((f-fm)/fs,dtype=torch.float32,device='cuda')
    ty=torch.as_tensor((d['y']-ym)/ys,dtype=torch.float32,device='cuda')
    return norm,tx,tf,ty

def fit(d,name,train,validation,epochs,seed,out):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    norm,x,f,y=context(d,train)
    model=Net(name).cuda(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    w=torch.as_tensor(weights(d['ids'][train]),dtype=torch.float32,device='cuda');w/=w.sum()
    best=float('inf');best_epoch=epochs;state=None;history=[]
    for epoch in range(1,epochs+1):
        model.train();opt.zero_grad(set_to_none=True)
        pred=model(x[train],f[train]); loss=((pred-y[train])**2*w).sum()
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
        item={'epoch':epoch,'loss':float(loss.item())}
        if validation is not None:
            model.eval()
            with torch.no_grad():
                vp=model(x[validation],f[validation]).cpu().numpy()*norm['ystd']+norm['ymean']
            score=mae(d['y'][validation],vp,d['ids'][validation]);item['validation_person_MAE']=score
            if score<best:
                best=score;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch-best_epoch>=12:
                history.append(item);break
        history.append(item)
        if epoch==1 or epoch%10==0:
            print(name,out,'epoch',epoch,'loss',float(loss.item()),flush=True)
    if state is not None:
        model.load_state_dict(state)
    else:
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    path=HERE/'runs'/out
    path.mkdir(parents=True,exist_ok=True)
    torch.save({'state':state,'normalization':norm,'name':name,'epoch':best_epoch,'train_indices':train.tolist(),'validation_indices':None if validation is None else validation.tolist(),'input_sha256':sha(HERE/'input.npz')},path/'model.pt')
    save(str(path.relative_to(HERE)/'history.json'),history)
    return model,norm,x,f,best_epoch

def predict(model,norm,x,f,ix):
    model.eval()
    with torch.no_grad():
        return model(x[ix],f[ix]).cpu().numpy()*norm['ystd']+norm['ymean']

def environment():
    assert os.environ.get('SLURM_JOB_ID'), 'Require Slurm allocation'
    assert torch.cuda.is_available()
    torch.cuda.init()
    a=torch.ones(1,device='cuda');a.add_(1);torch.cuda.synchronize();assert a.item()==2
    return {'job':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory}

def main(mode):
    started=time.time();env=environment();print(json.dumps(env),flush=True)
    d=load(); splits=json.loads((HERE/'splits.json').read_text())
    for split in splits:
        te=np.array(split['test']);tr=np.array(split['train'])
        assert not(set(d['ids'][te]) & set(d['ids'][tr]))
        for inner in split['inner']:
            a=np.array(inner['train']);b=np.array(inner['validation'])
            assert set(a)|set(b)==set(tr) and not(set(d['ids'][a])&set(d['ids'][b]))
    torch.cuda.reset_peak_memory_stats()
    if mode=='pilot':
        item=splits[0]['inner'][0];tr=np.array(item['train']);va=np.array(item['validation'])
        checks=[]
        for name in ['CNN','TCN']:
            model,norm,x,f,ep=fit(d,name,tr,None,3,20261030,'pilot/'+name)
            a=predict(model,norm,x,f,va)
            ck=torch.load(HERE/'runs/pilot'/name/'model.pt',map_location='cpu',weights_only=False)
            reload=Net(name).cuda();reload.load_state_dict(ck['state'])
            b=predict(reload,norm,x,f,va)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-6);assert np.isfinite(a).all()
            checks.append({'model':name,'parameters':sum(p.numel() for p in model.parameters()),'finite':True,'reload_equal':True,'epochs':3})
        torch.cuda.synchronize()
        save('pilot-summary.json',{'passed':True,'environment':env,'elapsed_seconds':time.time()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'checks':checks,'test_people_used':False})
        print('SPPB_PILOT_PASSED',flush=True)
    elif mode=='train':
        pilot=json.loads((HERE/'pilot-summary.json').read_text());assert pilot['passed']
        rows=[];selection=[]
        for split in splits:
            fold=split['fold'];tr=np.array(split['train']);te=np.array(split['test'])
            for name in ['CNN','TCN']:
                epochs=[]
                for i,item in enumerate(split['inner']):
                    _,_,_,_,ep=fit(d,name,np.array(item['train']),np.array(item['validation']),80,20261030+fold*10+i,f'main/{name}/fold{fold}-inner{i}')
                    epochs.append(ep)
                selected=max(1,int(np.median(epochs)))
                model,norm,x,f,_=fit(d,name,tr,None,selected,20261030+fold*10+9,f'main/{name}/fold{fold}-refit')
                pp=predict(model,norm,x,f,te);assert np.isfinite(pp).all()
                for index,value in zip(te,pp):
                    rows.append({'sample':int(index),'model':name,'fold':fold,'target':float(d['y'][index]),'prediction':float(value)})
                selection.append({'fold':fold,'model':name,'inner_best_epochs':epochs,'refit_epochs':selected})
                save('gpu-selection.json',selection)
                with (HERE/'gpu-predictions.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                print('FOLD_FINISHED',fold,name,selected,flush=True)
        summary={}
        for name in ['CNN','TCN']:
            pred=np.full(len(d['y']),np.nan)
            for r in rows:
                if r['model']==name:pred[r['sample']]=r['prediction']
            assert np.isfinite(pred).all()
            summary[name]=metrics(d['y'],pred,d['ids'])
        torch.cuda.synchronize()
        save('gpu-summary.json',{'metrics':summary,'environment':env,'elapsed_seconds':time.time()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'folds_finished':5,'models_finished':2,'source_input_sha256':sha(HERE/'input.npz')})
        print('SPPB_TIMESERIES_TRAINING_FINISHED',json.dumps(summary),flush=True)
    else:
        raise ValueError(mode)

if __name__=='__main__':
    main(sys.argv[1])
