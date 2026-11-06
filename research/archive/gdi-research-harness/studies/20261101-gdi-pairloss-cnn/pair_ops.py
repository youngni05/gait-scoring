"""Uniform-person/within-person-pair change loss on mean window predictions."""
import itertools
import numpy as np
import torch
from model import channels

def pool_for(x,people):
    if x.ages is None or len(x.ages)!=len(people):raise ValueError('Missing aligned ages')
    groups=[]
    for p in np.unique(people):
        ix=np.flatnonzero((people==p)&np.isfinite(x.ages))
        pairs=[(int(a),int(b)) for a,b in itertools.combinations(ix,2) if x.ages[a]!=x.ages[b]]
        if pairs:groups.append(pairs)
    if not groups:raise ValueError('No repeated training people for change loss')
    return groups

def change_term(model,x,target,pool,seed,device):
    rng=np.random.default_rng(seed)
    group=pool[int(rng.integers(len(pool)))];a,b=group[int(rng.integers(len(group)))]
    ia=np.flatnonzero(x.owner==a);ib=np.flatnonzero(x.owner==b)
    values=x.values[np.concatenate([ia,ib])]
    was=model.training
    # Differentiable inference-mode forward: keep training BN buffers and dropout RNG intact.
    model.eval()
    try:
        output=model(channels(torch.from_numpy(values)).reshape(-1,12,124).to(device))[:,0].reshape(-1,2)
        delta=output[len(ia):].mean(0)-output[:len(ia)].mean(0)
        expected=(target[b]-target[a]).to(device)
        return (delta-expected).square().mean()
    finally:model.train(was)

def paired_step(base_step,model,opt,x,target,c,weights,context,seed,device):
    if c['pair_weight']==0:
        loss,mse=base_step(model,opt,x,target,c,weights)
        return loss,mse,0.0
    from windows import weighted_mse
    model.train();opt.zero_grad(set_to_none=True)
    state=weighted_mse(model(x)[:,0],target,weights)
    term=change_term(model,*context,seed,device)
    loss=state+c['pair_weight']*term+c['kernel_l2']*model.penalty()
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite joint loss')
    loss.backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('Invalid joint gradients')
    opt.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('Nonfinite model parameters')
    return float(loss.detach()),float(state.detach()),float(term.detach())
