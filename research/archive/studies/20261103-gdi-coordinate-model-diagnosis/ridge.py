"""Matched preprocessing / retraining control with person-nested, weighted ridge models."""
from prepare import *
ALPHAS=np.array([.1,1,10,100,1000,10000.])

def fit_many(x,y,ids,alphas):
 # Each person contributes equally across their recordings and both legs.
 _,inverse,count=np.unique(ids,return_inverse=True,return_counts=True);w=1/count[inverse];w=w/w.mean()
 mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-12]=1
 z=(x-mean)/scale;ym=np.average(y,weights=w);gram=z.T@(z*w[:,None]);rhs=z.T@((y-ym)*w);eigen,vec=np.linalg.eigh(gram)
 return [dict(mean=mean,scale=scale,weight=vec@((vec.T@rhs)/(eigen+alpha)),intercept=np.array(ym),alpha=np.array(alpha)) for alpha in alphas]

def predict(m,x):return (x-m['mean'])/m['scale']@m['weight']+m['intercept']
def perperson(y,p,ids):return np.array([np.abs(y[ids==i]-p[ids==i]).mean() for i in np.unique(ids)])

def main():
 start=time.monotonic();d=np.load(B/'summaries.npz');ids=d['person_id'];fold=d['fold'];y=d['y'];n=len(y);audit={};arrays={};(B/'ridge').mkdir(exist_ok=True)
 for condition in ['baseline','smooth2','spike_repair','stable_scale']:
  x=d[condition];out=np.full(y.shape,np.nan);fa=[]
  for f in range(5):
   tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);inner=[];candidate=[np.full(y.shape,np.nan) for _ in ALPHAS]
   for j in range(5):
    if j==f:continue
    ti=np.flatnonzero((fold!=f)&(fold!=j));vi=np.flatnonzero(fold==j);assert set(ids[ti]).isdisjoint(ids[vi]) and set(ids[ti]).isdisjoint(ids[te])
    models=fit_many(x[ti].reshape(-1,96),y[ti].ravel(),np.repeat(ids[ti],2),ALPHAS)
    for q,m in enumerate(models):candidate[q][vi]=predict(m,x[vi].reshape(-1,96)).reshape(-1,2)
    inner.append(dict(train=ti.tolist(),validation=vi.tolist()))
   error=[float(perperson(y[tr],p[tr],ids[tr]).mean()) for p in candidate];alpha=ALPHAS[np.argmin(error)];m=fit_many(x[tr].reshape(-1,96),y[tr].ravel(),np.repeat(ids[tr],2),[alpha])[0];out[te]=predict(m,x[te].reshape(-1,96)).reshape(-1,2)
   path=B/'ridge'/f'{condition}-fold{f}.npz';np.savez_compressed(path,**m,train=tr,test=te)
   fa.append(dict(fold=f,train=tr.tolist(),test=te.tolist(),inner=inner,inner_mae=error,alpha=float(alpha),checkpoint=str(path),sha256=sha(path)))
  assert np.isfinite(out).all();arrays[condition]=out;audit[condition]=fa;print('ridge',condition,'MAE',round(float(perperson(y,out,ids).mean()),4),flush=True)
 np.savez_compressed(B/'ridge-record-predictions.npz',**arrays);save(H/'ridge-audit.json',audit);save(H/'ridge-completion.json',dict(protocol_sha256=sha(H/'protocol.json'),summaries_sha256=sha(B/'summaries.npz'),prediction_sha256=sha(B/'ridge-record-predictions.npz'),elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':main()
