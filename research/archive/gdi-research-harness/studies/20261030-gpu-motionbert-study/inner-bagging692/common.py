from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent
C=S/'cnn-multiwindow695';T=S/'stt692/multiwindow';E=S/'cnn-stt-ensemble692';MD=S/'mean-difference692';RA=S/'robust-aggregation692'
RACACHE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/robust-aggregation692/cache')
CACHE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/inner-bagging692/cache')
O=H/'analysis'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def errors(y,p,ids):return np.array([np.abs(y[ids==i]-p[ids==i]).mean() for i in np.unique(ids)])
def combine(cnn,stt,bilateral,w,t):
    base=w*cnn+(1-w)*stt;m=base.mean(1);d=(1-t)*(base[:,0]-base[:,1])+t*(bilateral[:,0]-bilateral[:,1])
    return np.column_stack([m+d/2,m-d/2])
def racache(family,f,j):
    p=RACACHE/family/('fold%d_%s.npz'%(f,'refit' if j is None else 'inner%d'%j));proof=read(p.with_suffix('.json'))
    assert proof['cache_sha256']==sha(p) and proof['protocol_sha256']==sha(RA/'protocol.json') and proof['outer']==f and proof['inner']==j
    return arrays(p),p
