"""Ragged windows indexed by recording; labels and folds always stay recording-level."""
import numpy as np
import torch

class WindowSet:
    def __init__(self,values,owner,nrecords,starts=None,fallback=None,record_weight=None):
        self.values=values
        self.owner=np.asarray(owner,dtype=np.int64)
        self.nrecords=int(nrecords)
        self.record_weight=np.ones(nrecords) if record_weight is None else np.asarray(record_weight,dtype=float)
        assert self.record_weight.shape==(nrecords,) and np.isfinite(self.record_weight).all() and (self.record_weight>0).all()
        self.starts=np.zeros(len(values),dtype=np.int64) if starts is None else starts
        self.fallback=np.zeros(len(values),dtype=bool) if fallback is None else fallback
        self.counts=np.bincount(self.owner,minlength=self.nrecords)

    def __len__(self):return self.nrecords

    def validate(self):
        if (self.values.dtype!=np.float32 or self.values.shape!=(len(self.owner),2,124,12)
            or not np.isfinite(self.values).all() or self.owner.shape!=(len(self.values),)
            or self.counts.shape!=(self.nrecords,) or not (self.counts>0).all()
            or self.starts.shape!=self.owner.shape or self.fallback.shape!=self.owner.shape
            or not (self.starts>=0).all()):
            raise ValueError('Invalid multiwindow data or empty record')

    def __getitem__(self,selection):
        indices=np.atleast_1d(np.arange(self.nrecords)[selection])
        if not len(indices) or len(np.unique(indices))!=len(indices):
            raise ValueError('Record selection must be nonempty and unique')
        mapping=np.full(self.nrecords,-1,dtype=np.int64);mapping[indices]=np.arange(len(indices))
        keep=mapping[self.owner]>=0
        return WindowSet(self.values[keep],mapping[self.owner[keep]],len(indices),self.starts[keep],self.fallback[keep],self.record_weight[indices])

    def side_weights(self):
        # Full epoch mean = mean across records of mean across their window-side errors.
        # Normalize ONCE on the training subset, never separately within each minibatch.
        weight=len(self.owner)*self.record_weight[self.owner]/(self.record_weight.sum()*self.counts[self.owner])
        return np.repeat(weight,2).astype(np.float32)

    def aggregate(self,window_predictions):
        if window_predictions.shape!=(len(self.owner),2) or not np.isfinite(window_predictions).all():
            raise ValueError('Invalid per-window predictions')
        total=np.zeros((self.nrecords,2),dtype=np.float64)
        np.add.at(total,self.owner,window_predictions)
        return total/self.counts[:,None]

def weighted_mse(pred,target,weights=None):
    if pred.ndim!=1 or target.shape!=pred.shape:
        raise ValueError('Prediction/target must be matching vectors; no broadcasting')
    if weights is None:return (pred-target).square().mean()
    if weights.shape!=pred.shape or not bool(torch.isfinite(weights).all()) or not bool((weights>0).all()):
        raise ValueError('Weights must be positive finite matching vectors')
    return ((pred-target).square()*weights).mean()
