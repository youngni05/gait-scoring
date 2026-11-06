"""Same packed record batch, two loss orders. No padded windows or new parameters."""
import numpy as np
import torch
from model import channels

def packed(x, indices, device):
    indices=np.asarray(indices,dtype=np.int64)
    if len(indices)==0 or len(set(indices.tolist()))!=len(indices):
        raise ValueError('Empty or duplicate recording batch')
    selected=x[indices]
    # WindowSet preserves source window order; gather per record explicitly.
    rows=np.concatenate([np.flatnonzero(selected.owner==i) for i in range(len(indices))])
    values=channels(torch.from_numpy(selected.values[rows])).reshape(-1,12,124).to(device)
    return values,selected.counts.tolist()

def loss_from_windows(pred, target, counts, variant):
    if pred.ndim!=2 or pred.shape!=(sum(counts),2) or target.shape!=(len(counts),2):
        raise ValueError('Expected window,side predictions and recording,side targets')
    if any(type(n) is not int or n<=0 for n in counts):raise ValueError('Invalid window counts')
    parts=torch.split(pred,counts,dim=0)
    if variant=='record_mean':
        pooled=torch.stack([p.mean(0) for p in parts])
        return (pooled-target).square().mean()
    if variant=='window_control':
        return torch.stack([(p-target[i]).square().mean() for i,p in enumerate(parts)]).mean()
    raise ValueError('Unknown loss variant')

def recording_step(model,opt,x,target,counts,c):
    model.train();opt.zero_grad(set_to_none=True)
    mse=loss_from_windows(model(x)[:,0].reshape(-1,2),target,counts,c['loss_variant'])
    loss=mse+c['kernel_l2']*model.penalty()
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
    loss.backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('Invalid gradient')
    opt.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('Invalid weights')
    return float(loss.detach()),float(mse.detach())
