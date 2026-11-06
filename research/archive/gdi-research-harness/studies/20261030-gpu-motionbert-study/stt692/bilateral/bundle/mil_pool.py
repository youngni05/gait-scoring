"""CPU prototype only: masked attention pooling within one recording, no GDI fitting."""
import torch
from torch import nn
class WindowAttentionPool(nn.Module):
    def __init__(self,features):
        super().__init__();self.score=nn.Sequential(nn.Linear(features,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x,valid):
        if x.ndim!=3 or valid.shape!=x.shape[:2] or valid.dtype!=torch.bool:raise ValueError('Expected [record,window,feature] and boolean mask')
        if not bool(valid.any(1).all()):raise ValueError('Every recording requires an observed window')
        safe=x.masked_fill(~valid[...,None],0)
        logits=self.score(safe).squeeze(-1).masked_fill(~valid,float('-inf'))
        weights=torch.softmax(logits,dim=1)
        return (safe*weights[...,None]).sum(1),weights
