"""Matched mean vs learned weighting of window scores; no bilateral interaction."""
import torch
from torch import nn
from model.Transformer import SpatioTemporalTransformer

class RecordingSTT(nn.Module):
    def __init__(self,mode='mean'):
        super().__init__()
        if mode not in ('mean','attention'):raise ValueError('Unknown pooling')
        self.mode=mode
        self.encoder=SpatioTemporalTransformer(num_frame=124,num_joints=4,in_chans=2,joint_embed_dim=12,num_heads=2,drop_rate=0.)
        if mode=='attention':
            self.gate=nn.Linear(48,1,bias=False)
            nn.init.zeros_(self.gate.weight)  # Start with exactly uniform weights.

    def forward(self,x,valid,details=False):
        if x.ndim!=6 or tuple(x.shape[2:])!=(2,124,4,2) or valid.shape!=x.shape[:2] or valid.dtype!=torch.bool:
            raise ValueError('Expected [record,window,2,124,4,xy] and boolean window mask')
        if not bool(valid.any(1).all()):raise ValueError('Empty recording')
        observed=x[valid]
        if not bool(torch.isfinite(observed).all()):raise ValueError('Nonfinite observed window')
        # Padding never reaches the encoder or its gradient graph.
        legs=observed.reshape(-1,124,4,2).permute(0,3,1,2)
        z=self.encoder.temporal_forward(self.encoder.spatial_forward(legs))
        per_window=self.encoder.regression_head(z).reshape(-1,2)
        scores=x.new_zeros((*valid.shape,2));scores[valid]=per_window
        logits=x.new_full((*valid.shape,2),float('-inf'))
        logits[valid]=self.gate(z).reshape(-1,2) if self.mode=='attention' else 0.
        weights=torch.softmax(logits,dim=1)
        prediction=(weights*scores).sum(1)
        return (prediction,weights,scores) if details else prediction
