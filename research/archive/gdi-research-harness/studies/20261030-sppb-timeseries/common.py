import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if (ROOT/'outputs/gdi-ridge-experiment-20261016/deps').exists():
    sys.path.insert(0, str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np

def save(name, obj):
    p = HERE/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load():
    audit = json.loads((HERE/'input-audit.json').read_text(encoding='utf8'))
    assert sha(HERE/'input.npz') == audit['input_sha256']
    assert sha(HERE/'protocol.json') == audit['protocol_sha256']
    d = dict(np.load(HERE/'input.npz', allow_pickle=False))
    assert np.isfinite(d['x']).all() and np.isfinite(d['y']).all()
    for p in np.unique(d['ids']):
        assert len(np.unique(d['folds'][d['ids']==p])) == 1
    return d

def weights(ids):
    return np.array([1/np.sum(ids==p) for p in ids], dtype=np.float64)

def mae(y, pred, ids):
    return float(np.average(np.abs(y-pred), weights=weights(ids)))

def median(y, ids):
    w = weights(ids)
    order = np.argsort(y)
    return float(y[order[np.searchsorted(np.cumsum(w[order]), w.sum()/2)]] )

def features(x, frames):
    """Native gait geometry inspired by Romeo's normalized distances/angles."""
    p = x.transpose(0,2,1).reshape(len(x),128,18,3)
    vals = []
    pairs = [(10,13),(1,8),(1,11),(3,10),(6,13),(8,11),(9,12),(2,5)]
    for a,b in pairs:
        v = np.linalg.norm(p[:,:,a,:2]-p[:,:,b,:2], axis=-1)
        ok = (p[:,:,a,2]>0)&(p[:,:,b,2]>0)
        columns = []
        for i in range(len(x)):
            z = v[i,ok[i]]
            q = [float(np.mean(z)),float(np.std(z)),*np.quantile(z,[.1,.5,.9]).tolist()] if len(z) else [0.]*5
            adjacent = ok[i,1:] & ok[i,:-1]
            q += [float(np.abs(np.diff(v[i]))[adjacent].mean()) if adjacent.any() else 0., float(ok[i].mean())]
            columns.append(q)
        vals.append(np.array(columns))
    return np.concatenate(vals+[np.log1p(frames)[:,None]], axis=1)

def metrics(y, pred, ids):
    return {'person_MAE':mae(y,pred,ids), 'person_RMSE':float(np.sqrt(np.average((y-pred)**2,weights=weights(ids))))}
