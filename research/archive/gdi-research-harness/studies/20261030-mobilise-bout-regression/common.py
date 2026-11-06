import hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'outputs/sppb-tug-candidates-20261030'
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
FEATURES=['averagecadence','averagestrideduration','averagestridespeed','averagestridelength']
KEY=['participantid','visit_number']
DAY=KEY+['wbday']
SEEDS=[20261030,20261031,20261032]
def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def stable_int(text): return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')
def weights(ids):
    _,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
    w=1/counts[inv]; return w/w.mean()
def person_mae(y,pred,ids):
    return float(pd.DataFrame({'id':ids,'e':np.abs(y-pred)}).groupby('id').e.mean().mean())
def weighted_median(y,w):
    order=np.argsort(y); return float(y[order][np.searchsorted(np.cumsum(w[order]),w.sum()/2)])
