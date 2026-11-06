"""Equal-person sampling and fixed-scale BMC (Ren et al., CVPR 2022)."""
import hashlib,math
import numpy as np
import torch
from torch.nn import functional as F

def sample_epoch(x,people,seed):
    ids=np.unique(people);rng=np.random.default_rng(seed)
    rounds=int(math.ceil(2*len(x.values)/len(ids)))
    tickets=rng.permutation(np.tile(np.arange(len(ids)),rounds))
    records=[np.flatnonzero(people==p) for p in ids]
    windows=[np.flatnonzero(x.owner==i) for i in range(len(people))]
    result=[]
    for ticket in tickets:
        record=int(rng.choice(records[ticket]));window=int(rng.choice(windows[record]))
        result.append((window,int(rng.integers(2))))
    result=np.asarray(result,dtype=np.int64)
    _,counts=np.unique(people[x.owner[result[:,0]]],return_counts=True)
    assert len(counts)==len(ids) and np.all(counts==rounds)
    return result

def batches(n,size):
    assert n>=2 and size>=2
    starts=list(range(0,n,size))
    if n-starts[-1]==1 and len(starts)>1:starts.pop()
    return [np.arange(a,b) for a,b in zip(starts,starts[1:]+[n])]

def steps_per_epoch(x,people,size):
    p=len(np.unique(people));n=int(math.ceil(2*len(x.values)/p))*p
    return len(batches(n,size))

def sample_digest(x):return hashlib.sha256(x.tobytes()).hexdigest()

def objective(pred,target,variant,sigma):
    assert pred.ndim==1 and target.shape==pred.shape and len(pred)>=2
    if variant=='mse':return (pred-target).square().mean()
    assert variant=='bmc' and sigma>0 and math.isfinite(sigma)
    # Official BMC, fixed positive scale in normalized target units.
    var=pred.new_tensor(sigma*sigma)
    logits=-(pred[:,None]-target[None,:]).square()/(2*var)
    return F.cross_entropy(logits,torch.arange(len(pred),device=pred.device))*(2*var).detach()
