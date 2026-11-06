import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
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
    return float(np.mean([abs(np.mean(y[ids==p])-np.mean(pred[ids==p])) for p in np.unique(ids)]))

def median(y, ids):
    w = weights(ids)
    order = np.argsort(y)
    return float(y[order[np.searchsorted(np.cumsum(w[order]), w.sum()/2)]] )

def features(x, frames):
    return np.concatenate([x.mean(-1),x.std(-1),np.abs(np.diff(x,axis=-1)).mean(-1)],axis=1)

def metrics(y, pred, ids):
    errors=np.array([np.mean(y[ids==p])-np.mean(pred[ids==p]) for p in np.unique(ids)])
    return {'person_MAE':float(np.mean(np.abs(errors))),'person_RMSE':float(np.sqrt(np.mean(errors**2)))}
