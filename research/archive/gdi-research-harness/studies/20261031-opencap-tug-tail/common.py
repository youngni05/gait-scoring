"""Tail-error study: same people, three seconds and frozen nested splits."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREV=HERE.parent/'20261030-opencap415-speed-stability'
SOURCE=HERE.parent/'20261030-opencap415-tug-timeseries'
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
SEEDS=[20261030,20261031,20261032]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load():
    audit=json.loads((HERE/'preparation.json').read_text())
    for name in ['input.npz','splits.json','protocol.json']:
        assert sha(HERE/name)==audit['hashes'][name]
    d=dict(np.load(HERE/'input.npz'))
    assert d['x'].shape==(298,19,91) and len(set(d['ids']))==298
    assert np.isfinite(d['x']).all() and np.isfinite(d['y']).all()
    return d
def features(d):
    x=d['x'];v=x[:,17,:]
    full=np.concatenate([x.mean(-1),x.std(-1),np.abs(np.diff(x,axis=-1)).mean(-1)],axis=1)
    raw=full[:,[17,36,55]]
    robust=np.column_stack([np.median(v,axis=1),np.quantile(v,.75,axis=1)-np.quantile(v,.25,axis=1),np.median(np.abs(np.diff(v,axis=1)),axis=1)])
    denominator=np.maximum(raw[:,0].astype(float),.1)
    inverse=np.column_stack([1/denominator,raw[:,1]/denominator,raw[:,2]/denominator])
    return {'raw':raw,'robust':robust,'inverse':inverse}
def loss(y,p):
    error=abs(y-p);bands=np.digitize(y,[10,20])
    return {'MAE':float(error.mean()),'macro':float(np.mean([error[bands==b].mean() for b in range(3) if np.any(bands==b)]))}

METHODS={
 'BaselineRF':('raw',1,'MAE','RF'),
 'RFSelectBalanced':('raw',1,'macro','RF'),
 'WeightedRF':('raw',10,'macro','RF'),
 'RobustRF':('robust',1,'MAE','RF'),
 'RobustRFSelectBalanced':('robust',1,'macro','RF'),
 'RobustWeightedRF':('robust',10,'macro','RF'),
 'InverseRidge':('inverse',1,'MAE','Ridge'),
 'InverseRidgeBalanced':('inverse',1,'macro','Ridge')}

FOLLOWUP_METHODS={
 'WeightedRF3':('raw',3,'macro','RF'),
 'RobustWeightedRF3':('robust',3,'macro','RF')}
