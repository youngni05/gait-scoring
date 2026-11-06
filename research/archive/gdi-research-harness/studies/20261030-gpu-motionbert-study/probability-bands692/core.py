from pathlib import Path
import hashlib,json,sys
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent;ROOT=S.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
from scipy.optimize import minimize,minimize_scalar
from scipy.special import expit,logsumexp
R=S/'robust-aggregation692';OLD=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/robust-aggregation692/cache')
CACHE=OLD.parent.parent/'probability-bands692/cache';O=H/'analysis'
EDGES=np.arange(0.,151.,10.);K=len(EDGES)+1;EPS=1e-12
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def weights(ids):
 w=np.array([1/np.sum(ids==i) for i in ids],dtype=float);return w/w.sum()
def labels(y):return np.searchsorted(EDGES,y,side='right')
def partition(ids,f,j):
 people=np.unique(ids);order=np.random.default_rng(20261031+100*f+j).permutation(people)
 a=int(len(order)*.6);b=a+int(len(order)*.2)
 return [order[:a],order[a:b],order[b:]]
def disjoint(*groups):
 for i,a in enumerate(groups):
  for b in groups[i+1:]:assert set(a).isdisjoint(b),'Person leakage'
def features(cnn,stt,owner,rows,cnn_coef,stt_coef):
 data=[];spreads=[]
 for raw,coef in [(cnn,cnn_coef),(stt,stt_coef)]:
  m=[];q=[]
  for i in rows:
   v=raw[owner==i]*coef[0]+coef[1];assert len(v)>0
   m.append(v.mean(0));q.append(np.stack([v.std(0),np.quantile(v,.75,axis=0)-np.quantile(v,.25,axis=0)],axis=-1))
  data.append(np.array(m));spreads.append(np.log1p(np.array(q)))
 base=(data[0]+data[1])/2
 x=np.concatenate([np.stack([base,data[0]-data[1],base[:,::-1]-base],axis=-1),spreads[0],spreads[1]],axis=-1)
 return x.reshape(-1,7),base.reshape(-1)
def inner_data(f,j):
 parts=[];hashes={};basepeople=None
 for family in ('CNN','STT'):
  p=OLD/family/('fold%d_inner%d.npz'%(f,j));proof=read(p.with_suffix('.json'));assert sha(p)==proof['cache_sha256']
  assert read(R/'analysis/verification.json')['source_cache_hashes'][str(p)]==sha(p)
  assert sha(proof['checkpoint'])==proof['checkpoint_sha256']
  v=arrays(p);valid=v['validation_mask'];rows=v['row_index'][valid];keep=np.isin(v['window_record'],rows)
  parts.append((v,rows,keep));hashes[str(p)]=sha(p);hashes[str(p.with_suffix('.json'))]=sha(p.with_suffix('.json'))
  ids=v['person_id'][valid];trainpeople=np.unique(v['person_id'][v['train_mask']]);assert (v['fold']!=f).all()
  if basepeople is None:basepeople=trainpeople
  else:np.testing.assert_array_equal(basepeople,trainpeople)
 a,rows,keep=parts[0];b,rows2,keep2=parts[1];np.testing.assert_array_equal(rows,rows2)
 for key in ('window_record','window_start'):np.testing.assert_array_equal(a[key][keep],b[key][keep2])
 for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(a[key],b[key])
 valid=a['validation_mask'];x,base=features(a['window_raw'][keep],b['window_raw'][keep2],a['window_record'][keep],rows,a['calibration'][0],b['calibration'][0])
 return dict(x=x,base=base,y=a['y'][valid].reshape(-1),ids=np.repeat(a['person_id'][valid],2),video=np.repeat(a['video_id'][valid],2),
             rows=rows,base_people=basepeople,hashes=hashes)
def probabilities(mu,scale):
 cdf=expit((EDGES[None,:]-mu[:,None])/scale[:,None]);p=np.diff(np.column_stack([np.zeros(len(mu)),cdf,np.ones(len(mu))]),axis=1)
 p=np.maximum(p,EPS);return p/p.sum(1,keepdims=True)
def nll(p,y,w):return float(-np.sum(w*np.log(p[np.arange(len(y)),labels(y)])))
def temperature(p,t):
 a=np.log(p)/t;return np.exp(a-logsumexp(a,axis=1,keepdims=True))
def fit_temperature(p,y,w):
 opt=minimize_scalar(lambda logt:nll(temperature(p,np.exp(logt)),y,w),bounds=(np.log(.25),np.log(4.)),method='bounded',options={'xatol':1e-10})
 assert opt.success;return float(np.exp(opt.x))
def standardize(x,w):
 m=np.sum(x*w[:,None],axis=0);s=np.sqrt(np.sum((x-m)**2*w[:,None],axis=0));s[s<1e-6]=1.
 return m,s
def design(x,family,mean,sd):
 z=(x-mean)/sd
 nx=3 if family=='score_only' else 7
 return np.column_stack([np.ones(len(x)),z[:,:nx]]),np.ones((len(x),1)) if family=='score_only' else np.column_stack([np.ones(len(x)),z[:,3:7]])
def objective(theta,x,z,base,y,w,lam):
 nb=x.shape[1];beta=theta[:nb];gamma=theta[nb:];mu=base+10*x@beta;logscale=z@gamma;scale=np.exp(logscale)
 target=labels(y);lower=np.r_[-np.inf,EDGES][target];upper=np.r_[EDGES,np.inf][target]
 l=(lower-mu)/scale;u=(upper-mu)/scale;a=expit(l);b=expit(u);p=np.maximum(b-a,EPS)
 # Only finite products contribute at the two open tails.
 al=np.zeros_like(l);bu=np.zeros_like(u);finite=np.isfinite(l);al[finite]=l[finite]*a[finite]*(1-a[finite]);finite=np.isfinite(u);bu[finite]=u[finite]*b[finite]*(1-b[finite])
 dm=-(a*(1-a)-b*(1-b))/scale/p;ds=-(al-bu)/p
 active=b-a>EPS;dm[~active]=0;ds[~active]=0
 penalty=lam*(np.sum(beta**2)+np.sum(gamma[1:]**2))
 value=float(-np.sum(w*np.log(p))+penalty)
 grad=np.r_[10*x.T@(w*dm)+2*lam*beta,z.T@(w*ds)+np.r_[0.,2*lam*gamma[1:]]]
 return value,grad
def fit_head(x,base,y,ids,family,lam):
 w=weights(ids);mean,sd=standardize(x,w);xx,zz=design(x,family,mean,sd)
 theta=np.r_[np.zeros(xx.shape[1]),np.log(5.),np.zeros(zz.shape[1]-1)]
 bounds=[(-3,3)]*xx.shape[1]+[(np.log(.5),np.log(40.))]+[(-.5,.5)]*(zz.shape[1]-1)
 opt=minimize(objective,theta,args=(xx,zz,base,y,w,lam),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':1000,'ftol':1e-12,'gtol':1e-7,'maxls':40})
 assert opt.success,(family,lam,opt.message)
 return dict(family=family,lambda_=lam,mean=mean.tolist(),sd=sd.tolist(),theta=opt.x.tolist(),iterations=int(opt.nit),objective=float(opt.fun))
def predict(model,x,base):
 xx,zz=design(x,model['family'],np.array(model['mean']),np.array(model['sd']));t=np.array(model['theta']);mu=base+10*xx@t[:xx.shape[1]];scale=np.exp(zz@t[xx.shape[1]:])
 return probabilities(mu,scale)
def fit_residual(base,y,ids):
 w=weights(ids);opt=minimize_scalar(lambda s:nll(probabilities(base,np.full(len(base),np.exp(s))),y,w),bounds=(np.log(.5),np.log(40.)),method='bounded',options={'xatol':1e-10})
 assert opt.success;return float(np.exp(opt.x))
