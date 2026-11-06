from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent
C=S/'cnn-multiwindow695';T=S/'stt692/multiwindow';E=S/'cnn-stt-ensemble692';MD=S/'mean-difference692'
CACHE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/robust-aggregation692/cache')
O=H/'analysis'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def aggregate(raw,owner,rows):
    result=np.empty((6,len(rows),2),dtype=np.float64)
    for j,i in enumerate(rows):
        a=raw[owner==i].astype(np.float64);assert len(a)>0 and np.isfinite(a).all();b=np.sort(a,axis=0);n=len(a)
        result[0,j]=a.mean(0);result[1,j]=np.median(a,axis=0)
        for q,p in enumerate((.1,.2)):
            k=int(n*p);result[2+q,j]=b[k:n-k].mean(0)
            result[4+q,j]=np.clip(a,b[k],b[n-k-1]).mean(0)
    return result
def fit_calibration(raw,y):
    return np.linalg.lstsq(np.column_stack([raw.ravel(),np.ones(raw.size)]),y.ravel(),rcond=None)[0]
def errors(y,p,ids):return np.array([np.abs(y[ids==i]-p[ids==i]).mean() for i in np.unique(ids)])
def combine(cnn,stt,bilateral,w,t):
    base=w*cnn+(1-w)*stt;m=base.mean(1);d=(1-t)*(base[:,0]-base[:,1])+t*(bilateral[:,0]-bilateral[:,1])
    return np.column_stack([m+d/2,m-d/2])
def choose(cnn,stt,bilateral,y,ids,w,t):
    scores=np.array([[errors(y,combine(c,st,bilateral,w,t),ids).mean() for st in stt] for c in cnn])
    return tuple(map(int,np.unravel_index(np.argmin(scores),scores.shape))),scores
