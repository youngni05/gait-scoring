"""Nested person-held-out ridge comparison; no clinical labels enter preprocessing."""
from prepare import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
ALPHAS=[.1,1.,10.,100.,1000.]
CONDITIONS=read(S/'protocol.json')['conditions'];FITS=[]
def metrics(y,p,target):
 y=np.asarray(y);p=np.asarray(p);e=p-y;n=len(y)
 out=dict(n=n,MAE=float(abs(e).mean()),RMSE=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),R2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if np.std(y)>0 else None,spearman=float(spearmanr(y,p).statistic) if np.ptp(y)>0 and np.ptp(p)>1e-12 else None)
 if target=='sppb':
  out.update(within1=float((abs(e)<=1).mean()),errors_gt2=int((abs(e)>2).sum()),prediction_range=[float(p.min()),float(p.max())],bands={})
  for label,mask in [('0-4',y<=4),('5-8',(y>=5)&(y<=8)),('9-10',(y>=9)&(y<=10)),('11-12',y>=11)]:out['bands'][label]=dict(n=int(mask.sum()),MAE=float(abs(e[mask]).mean()) if mask.any() else None,bias=float(e[mask].mean()) if mask.any() else None)
  low=y<=8;pl=p<=8;out.update(TP=int((low&pl).sum()),FN=int((low&~pl).sum()),TN=int((~low&~pl).sum()),FP=int((~low&pl).sum()),recall=float(pl[low].mean()) if low.any() else None,specificity=float((~pl[~low]).mean()) if (~low).any() else None)
 return out

def train(X,y,tr,te,alpha,meta,ids):
 scaler=StandardScaler().fit(X[tr]);model=Ridge(alpha=alpha).fit(scaler.transform(X[tr]),y[tr]);p=model.predict(scaler.transform(X[te]))
 FITS.append(dict(**meta,alpha=alpha,train=[ids[i] for i in tr],evaluation=[ids[i] for i in te],mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=model.coef_.tolist(),intercept=float(model.intercept_),prediction=p.tolist()))
 return p

def select(X,y,tr,meta,ids):
 losses=[]
 for a in ALPHAS:
  err=[]
  for inner,(tt,vv) in enumerate(KFold(3,shuffle=True,random_state=20261031).split(tr)):
   p=train(X,y,tr[tt],tr[vv],a,dict(**meta,inner=inner),ids);err.extend(abs(p-y[tr[vv]]))
  losses.append(float(np.mean(err)))
 return ALPHAS[int(np.argmin(losses))],losses

def finish(ids,y,preds,target,extras):
 out=dict(ids=ids,y=y.tolist(),predictions={k:v.tolist() for k,v in preds.items()},metrics={k:metrics(y,v,target) for k,v in preds.items()},**extras);out['contrasts']={}
 rng=np.random.RandomState(20261103);boot=rng.randint(0,len(y),(10000,len(y)))
 for a,b in [('B','A'),('C','A'),('D','A'),('D','C')]:
  if a not in preds or b not in preds:continue
  delta=abs(preds[a]-y)-abs(preds[b]-y)
  out['contrasts'][a+'-'+b]={'MAE_difference':float(delta.mean()),'CI95':np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),'interpretation':'fixed-prediction paired person bootstrap; negative favours first'}
 return out

def run(name,people,target,conditions):
 pp=sorted([p for p in people if p[target] is not None],key=lambda p:p['id']);ids=[p['id'] for p in pp];y=np.array([p[target] for p in pp]);fold=np.zeros(len(pp),int)
 for k,(_,te) in enumerate(KFold(5,shuffle=True,random_state=20261103).split(pp)):fold[te]=k
 mat={c:np.array([[p['features'][f] for f in CONDITIONS[c]] for p in pp]) for c in conditions};preds={c:np.full(len(pp),np.nan) for c in [*conditions,'Mean','Median']};selection=[]
 for k in range(5):
  tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k);preds['Mean'][te]=np.mean(y[tr]);preds['Median'][te]=np.median(y[tr])
  for c,X in mat.items():
   meta=dict(run=name,target=target,condition=c,outer=k);alpha,losses=select(X,y,tr,meta,ids);preds[c][te]=train(X,y,tr,te,alpha,dict(**meta,inner=None),ids);selection.append(dict(condition=c,outer=k,alpha=alpha,inner_MAE=losses))
 assert all(np.isfinite(p).all() for p in preds.values())
 out=finish(ids,y,preds,target,dict(target=target,folds=fold.tolist(),selection=selection,inputs={c:x.tolist() for c,x in mat.items()}))
 community=np.array([p['community'] for p in pp]);out['community_subgroup']={c:metrics(y[community],v[community],target) for c,v in preds.items()} if community.any() else {}
 print(name,target,len(pp),{c:round(v['MAE'],4) for c,v in out['metrics'].items()},flush=True);return out

def transfer(source,test,target='sppb'):
 pp=sorted(source,key=lambda p:p['id'])+sorted(test,key=lambda p:p['id']);ids=[p['id'] for p in pp];y=np.array([p[target] for p in pp]);tr=np.arange(len(source));te=np.arange(len(source),len(pp));preds={};sel=[];mat={}
 for c in ['A','C']:
  X=np.array([[p['features'][f] for f in CONDITIONS[c]] for p in pp]);meta=dict(run='provisional_gstride_to_gait2care',target=target,condition=c,outer='all_source')
  alpha,losses=select(X,y,tr,meta,ids);preds[c]=train(X,y,tr,te,alpha,dict(**meta,inner=None),ids);sel.append(dict(condition=c,alpha=alpha,inner_MAE=losses));mat[c]=X.tolist()
 preds['Mean']=np.repeat(y[tr].mean(),len(te));preds['Median']=np.repeat(np.median(y[tr]),len(te))
 return finish([ids[i] for i in te],y[te],preds,target,dict(target=target,source_ids=[ids[i] for i in tr],selection=sel,all_ids=ids,all_y=y.tolist(),all_inputs=mat,independence='UNRESOLVED cross-cohort identity overlap; provisional transfer only'))

def main():
 pp=read(S/'people.json');audit=read(S/'input-audit.json');assert audit['protocol_sha256']==sha(S/'protocol.json');assert audit['full_clocks']
 g=[p for p in pp if p['dataset']=='gstride'];a=[p for p in pp if p['dataset']=='gait2care'];common=[p for p in a if p['features']['PCI'] is not None]
 results={}
 for target in ['sppb','tug','balance','chair']:
  results['gstride/'+target]=run('gstride',g,target,['A','C'])
  results['gait2care/'+target]=run('gait2care',common,target,['A','B','C','D'])
 if len(common)!=len(a):results['gait2care_available_AC/sppb']=run('gait2care_available_AC',a,'sppb',['A','C'])
 results['provisional_transfer/sppb']=transfer(g,a)
 save(S/'comparison.json',results);save(B/'fits.json',FITS)
 save(S/'fit-receipt.json',dict(protocol_sha256=sha(S/'protocol.json'),people_sha256=sha(S/'people.json'),comparison_sha256=sha(S/'comparison.json'),fits_sha256=sha(B/'fits.json'),fits=len(FITS),code_sha256=sha(Path(__file__))))
 print('COMPLETED',len(FITS),'saved fits',flush=True)
if __name__=='__main__':main()
