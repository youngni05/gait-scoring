"""Paired legs, shared official STT encoder, swap-equivariant mean/difference head."""
import torch
from torch import nn
from model.Transformer import SpatioTemporalTransformer

class BilateralSTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=SpatioTemporalTransformer(num_frame=124,num_joints=4,in_chans=2,joint_embed_dim=12,num_heads=2,drop_rate=0.)
        self.encoder.regression_head=nn.Identity()
        self.common=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))
        self.difference=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):
        if x.ndim!=5 or tuple(x.shape[1:])!=(2,124,4,2):raise ValueError('Expected paired windows [N,2,124,4,2]')
        z=self.encoder(x.reshape(-1,124,4,2)).reshape(-1,2,48)
        avg=(z[:,0]+z[:,1])*0.5;diff=(z[:,0]-z[:,1])*0.5
        # Identical leg inputs imply zero anatomical difference. Batched kernels
        # can round identical rows differently; enforce this exact special case.
        identical=(x[:,0]==x[:,1]).flatten(1).all(1)
        diff=torch.where(identical[:,None],torch.zeros_like(diff),diff)
        common=self.common(torch.cat([avg,diff.abs()],dim=-1))
        delta=(self.difference(torch.cat([avg,diff],dim=-1))-self.difference(torch.cat([avg,-diff],dim=-1)))*0.5
        return torch.cat([common+delta,common-delta],dim=-1).reshape(-1)
