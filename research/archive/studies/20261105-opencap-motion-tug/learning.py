"""Small, deterministic train-only temporal regressors for the fixed TUG comparison."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import copy
import numpy as np,torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
TCN=[{'width':w,'target':target} for w in [16,32] for target in ['raw','residual6overv']]
RANDOM=[{'alpha':a,'kernels':256} for a in [10,100,1000]]

def norm(S,X):
    med=np.nanmedian(X,axis=0);med=np.nan_to_num(med,nan=0.)
    filled=np.where(np.isnan(X),med,X);sc=StandardScaler().fit(filled)
    return {'seqmean':S.mean((0,2),keepdims=True),'seqstd':np.maximum(S.std((0,2),keepdims=True),1e-3),'median':med,'scaler':sc}
def tab(m,X):return m['scaler'].transform(np.where(np.isnan(X),m['median'],X))
def tensors(m,S,X):return torch.tensor(((S-m['seqmean'])/m['seqstd']).astype(np.float32)),torch.tensor(tab(m,X).astype(np.float32))
def anchor(X,cfg):return 6/np.maximum(X[:,0],.1) if cfg['target']=='residual6overv' else np.zeros(len(X))
class Block(nn.Module):
    def __init__(self,w,d):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(w,w,5,padding=2*d,dilation=d),nn.GELU(),nn.Dropout(.1),nn.Conv1d(w,w,3,padding=d,dilation=d),nn.GELU())
    def forward(self,x):return x+self.net(x)
class Temporal(nn.Module):
    def __init__(self,width,nfeatures):
        super().__init__();self.enc=nn.Sequential(nn.Conv1d(34,width,1),nn.GELU(),*[Block(width,d) for d in [1,2,4]])
        self.head=nn.Sequential(nn.Linear(width*2+nfeatures,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x,v):
        z=self.enc(x);return self.head(torch.cat([z.mean(2),z.amax(2),v],1)).squeeze(1)
def fit_tcn(S,X,y,cfg,seed,validation=None,epochs=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);mnorm=norm(S,X);xx,vv=tensors(mnorm,S,X);target=y-anchor(X,cfg);ym=float(target.mean());ys=float(max(target.std(),1e-6));yy=torch.tensor(((target-ym)/ys).astype(np.float32));m=Temporal(cfg['width'],X.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.05);criterion=nn.SmoothL1Loss(beta=.5)
    if validation is not None:vs,vX,vy=validation;vx,vv2=tensors(mnorm,vs,vX);va=anchor(vX,cfg)
    best=float('inf');stale=0;history=[];bestepoch=0;state=None
    for epoch in range(1,(epochs or 100)+1):
        m.train()
        for order in np.array_split(rng.permutation(len(y)),max(1,(len(y)+63)//64)):
            xb=xx[order]+torch.randn_like(xx[order])*.005;opt.zero_grad();loss=criterion(m(xb,vv[order]),yy[order]);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
        if validation is not None:
            m.eval()
            with torch.no_grad():pr=np.maximum(0,m(vx,vv2).numpy().astype(float)*ys+ym+va)
            error=float(abs(pr-vy).mean());history.append(error)
            if error<best-1e-6:best=error;bestepoch=epoch;state=copy.deepcopy(m.state_dict());stale=0
            else:stale+=1
            if stale>=15:break
    if validation is not None:m.load_state_dict(state)
    else:bestepoch=epochs
    return {'kind':'tcn','config':cfg,'nfeatures':X.shape[1],'state':m.state_dict(),'norm':mnorm,'ymean':ym,'ystd':ys,'ntrain':len(y),'epochs':int(bestepoch),'history':history,'seed':seed}
def predict_tcn(m,S,X,double=False):
    model=Temporal(m['config']['width'],m['nfeatures']);model.load_state_dict(m['state']);model.eval();xx,vv=tensors(m['norm'],S,X)
    if double:model=model.double();xx=xx.double();vv=vv.double()
    with torch.no_grad():pr=model(xx,vv).numpy().astype(float)
    return np.maximum(0,pr*m['ystd']+m['ymean']+anchor(X,m['config']))
def random_features(S,kernels):
    out=[]
    for k in kernels:
        length=k['weight'].shape[1];span=(length-1)*k['dilation']+1;windows=np.lib.stride_tricks.sliding_window_view(S[:,k['channels']],span,axis=2)[:,:,:,::k['dilation']]
        z=np.einsum('nctk,ck->nt',windows,k['weight'],optimize=True);out.extend([z.max(1),(z>k['bias']).mean(1)])
    return np.column_stack(out)
def fit_random(S,X,y,cfg):
    mnorm=norm(S,X);z=(S-mnorm['seqmean'])/mnorm['seqstd'];rng=np.random.default_rng(20261105);kernels=[]
    for _ in range(cfg['kernels']):
        length=int(rng.choice([3,5,7,9]));dilation=int(rng.choice([1,2,4,8]));channels=rng.choice(34,size=int(rng.choice([1,2,4,8])),replace=False);weight=rng.normal(size=(len(channels),length));weight-=weight.mean();weight/=max(np.linalg.norm(weight),1e-12)
        kernels.append({'channels':channels,'weight':weight,'dilation':dilation,'bias':float(rng.uniform(-1,1))})
    xx=np.c_[tab(mnorm,X),random_features(z,kernels)];sc=StandardScaler().fit(xx);ym=float(y.mean());ys=float(max(y.std(),1e-6));reg=Ridge(alpha=cfg['alpha']).fit(sc.transform(xx),(y-ym)/ys)
    return {'kind':'random','config':cfg,'norm':mnorm,'kernels':kernels,'scaler':sc,'model':reg,'ymean':ym,'ystd':ys,'ntrain':len(y)}
def predict_random(m,S,X):
    z=(S-m['norm']['seqmean'])/m['norm']['seqstd'];xx=np.c_[tab(m['norm'],X),random_features(z,m['kernels'])]
    return np.maximum(0,m['model'].predict(m['scaler'].transform(xx))*m['ystd']+m['ymean'])
