"""Small train-only tabular, learned temporal and random-temporal-filter models."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
TAB=[{'kind':'ridge','alpha':a} for a in [1,10,100]]+[{'kind':'svr','C':c} for c in [1,10]]+[{'kind':'lgbm','leaf':leaf,'reg':reg} for leaf in [10,20] for reg in [1,10]]
TCN=[{'width':16,'wd':.01,'noise':0},{'width':32,'wd':.01,'noise':0},{'width':16,'wd':.1,'noise':.01},{'width':32,'wd':.1,'noise':.01}]
RANDOM=[{'alpha':a,'kernels':256} for a in [1,10,100]]
def fit_tab(X,y,cfg):
    sc=StandardScaler().fit(X);xx=sc.transform(X);ym=float(y.mean());ys=float(max(y.std(),1e-6));z=(y-ym)/ys
    if cfg['kind']=='ridge':m=Ridge(alpha=cfg['alpha']).fit(xx,z)
    elif cfg['kind']=='svr':m=SVR(C=cfg['C'],epsilon=.05,gamma='scale').fit(xx,z)
    else:m=LGBMRegressor(objective='regression_l1',n_estimators=180,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=cfg['leaf'],reg_lambda=cfg['reg'],verbosity=-1,n_jobs=1,random_state=20261105,deterministic=True,force_col_wise=True).fit(xx,z)
    return {'kind':'tab','config':cfg,'scaler':sc,'model':m,'ymean':ym,'ystd':ys,'ntrain':len(y)}
def predict_tab(m,X):return np.maximum(0,m['model'].predict(m['scaler'].transform(X))*m['ystd']+m['ymean'])
class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),nn.GELU(),nn.Dropout(.1),nn.Conv1d(width,width,3,padding=dilation,dilation=dilation),nn.GELU())
    def forward(self,x):return x+self.net(x)
class Temporal(nn.Module):
    def __init__(self,width):
        super().__init__();self.enc=nn.Sequential(nn.Conv1d(28,width,1),nn.GELU(),*[Block(width,d) for d in [1,2,4]]);self.head=nn.Sequential(nn.Linear(width*2+1,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x,speed):
        z=self.enc(x);return self.head(torch.cat([z.mean(2),z.amax(2),speed],dim=1)).squeeze(1)
def normalize_sequence(S,speed):
    mean=S.mean((0,2),keepdims=True);std=np.maximum(S.std((0,2),keepdims=True),1e-3);sm=float(speed.mean());sd=float(max(speed.std(),1e-6))
    return {'seqmean':mean,'seqstd':std,'speedmean':sm,'speedstd':sd}
def tensors(norm,S,speed):
    x=((S-norm['seqmean'])/norm['seqstd']).astype(np.float32);v=((speed-norm['speedmean'])/norm['speedstd']).astype(np.float32).reshape(-1,1)
    return torch.from_numpy(x),torch.from_numpy(v)
def fit_tcn(S,speed,y,cfg,seed,validation=None,epochs=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);norm=normalize_sequence(S,speed);xx,vv=tensors(norm,S,speed);ym=float(y.mean());ys=float(max(y.std(),1e-6));yy=torch.tensor((y-ym)/ys,dtype=torch.float32)
    m=Temporal(cfg['width']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=cfg['wd']);lossfn=nn.SmoothL1Loss(beta=.5)
    if validation is not None:vs,vsp,vy=validation;vx,vv2=tensors(norm,vs,vsp)
    best=float('inf');bestepoch=0;state=None;stale=0;history=[]
    for epoch in range(1,(epochs or 120)+1):
        m.train()
        for order in np.array_split(rng.permutation(len(y)),max(1,(len(y)+31)//32)):
            xb=xx[order]
            if cfg['noise']:xb=xb+torch.randn_like(xb)*cfg['noise']
            opt.zero_grad();loss=lossfn(m(xb,vv[order]),yy[order]);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
        if validation is not None:
            m.eval()
            with torch.no_grad():p=np.maximum(0,m(vx,vv2).numpy().astype(float)*ys+ym)
            error=float(abs(p-vy).mean());history.append(error)
            if error<best-1e-6:best=error;bestepoch=epoch;state={k:v.detach().clone() for k,v in m.state_dict().items()};stale=0
            else:stale+=1
            if stale>=20:break
    if validation is not None:m.load_state_dict(state)
    else:bestepoch=epochs
    return {'kind':'tcn','config':cfg,'state':m.state_dict(),'norm':norm,'ymean':ym,'ystd':ys,'ntrain':len(y),'seed':seed,'epochs':int(bestepoch),'history':history}
def predict_tcn(record,S,speed,double=False):
    m=Temporal(record['config']['width']);m.load_state_dict(record['state']);m.eval();x,v=tensors(record['norm'],S,speed)
    if double:m=m.double();x=x.double();v=v.double()
    with torch.no_grad():p=m(x,v).numpy().astype(float)
    return np.maximum(0,p*record['ystd']+record['ymean'])
def random_features(S,kernels):
    columns=[]
    for k in kernels:
        x=S[:,k['channels']];length=len(k['weight'][0]);receptive=(length-1)*k['dilation']+1
        w=np.lib.stride_tricks.sliding_window_view(x,receptive,axis=2)[:,:,:,::k['dilation']];c=np.einsum('nctk,ck->nt',w,k['weight'],optimize=True)
        columns.extend([c.max(1),(c>k['bias']).mean(1)])
    return np.column_stack(columns)
def fit_random(S,speed,y,cfg):
    norm=normalize_sequence(S,speed);z=(S-norm['seqmean'])/norm['seqstd'];rng=np.random.default_rng(20261105);kernels=[]
    for _ in range(cfg['kernels']):
        length=int(rng.choice([3,5,7,9]));dilation=int(rng.choice([d for d in [1,2,4,8] if (length-1)*d<60]));channels=rng.choice(28,size=int(rng.choice([1,2,4,8])),replace=False);weight=rng.normal(size=(len(channels),length));weight-=weight.mean();weight/=max(np.linalg.norm(weight),1e-12)
        kernels.append({'channels':channels,'weight':weight,'dilation':dilation,'bias':float(rng.uniform(-1,1))})
    X=np.column_stack([speed,random_features(z,kernels)]);model=fit_tab(X,y,{'kind':'ridge','alpha':cfg['alpha']})
    return {'kind':'random','config':cfg,'norm':norm,'kernels':kernels,'regressor':model,'ntrain':len(y)}
def predict_random(m,S,speed):
    z=(S-m['norm']['seqmean'])/m['norm']['seqstd'];return predict_tab(m['regressor'],np.column_stack([speed,random_features(z,m['kernels'])]))
