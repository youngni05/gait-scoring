"""Shared CNN features, record pooling, ordered thresholds and optional point head."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from base_model import GdiCNN

def boundaries(width):
    assert width in (10,20)
    return np.arange(0,161,width,dtype=np.float64)

def targets(y,width):
    # [a,b): a point equal to a boundary belongs to its upper interval.
    return np.searchsorted(boundaries(width),np.asarray(y),side='right')

def probabilities(logits):
    q=torch.sigmoid(logits)
    return torch.cat((1-q[...,:1],q[...,:-1]-q[...,1:],q[...,-1:]),dim=-1)

class OrdinalCNN(nn.Module):
    def __init__(self,width,mode):
        super().__init__();assert mode in ('regression','ordinal','joint')
        self.width=width;self.mode=mode;self.base=GdiCNN(.5)
        # Reuse the exact six-convolution topology, without its old output layer.
        self.base.output=nn.Identity()
        if mode!='ordinal':self.reg=nn.Linear(10,1)
        if mode!='regression':
            self.rank=nn.Linear(10,1);nn.init.zeros_(self.rank.weight);nn.init.constant_(self.rank.bias,3.75)
            self.first=nn.Parameter(torch.tensor(0.))
            self.gaps=nn.Parameter(torch.full((len(boundaries(width))-1,),float(np.log(np.expm1(width/20)))))
        if mode!='ordinal':
            nn.init.xavier_uniform_(self.reg.weight,generator=torch.Generator().manual_seed(torch.initial_seed()+17))
            nn.init.zeros_(self.reg.bias)

    def forward(self,values,counts):
        h=self.base(values).reshape(-1,2,10)
        pooled=torch.stack([a.mean(0) for a in torch.split(h,counts,dim=0)])
        score=self.reg(pooled)[...,0] if self.mode!='ordinal' else None
        logits=None
        if self.mode!='regression':
            # CUDA2.4 cumsum rejects deterministic mode; fixed-order additions
            # compute the same ordered thresholds without relaxing determinism.
            cuts=[self.first]
            for gap in (F.softplus(self.gaps)+1e-6).unbind():cuts.append(cuts[-1]+gap)
            cuts=torch.stack(cuts)
            logits=self.rank(pooled)-cuts
        return score,logits

    def penalty(self):return self.base.penalty()

def objective(model,score,logits,y,weights,lo,span,lam):
    parts={};loss=0
    if score is not None:
        per=((score-(y-lo)/span)**2).mean(1)
        parts['regression']=torch.mean(per*weights);loss=loss+parts['regression']
    if logits is not None:
        cuts=torch.as_tensor(boundaries(model.width),device=y.device,dtype=y.dtype)
        binary=(y[...,None]>=cuts).to(y.dtype)
        per=F.binary_cross_entropy_with_logits(logits,binary,reduction='none').mean((1,2))
        parts['ordinal']=torch.mean(per*weights)
        loss=loss+(lam if score is not None else 1.)*parts['ordinal']
    loss=loss+0.00031622776601683794*model.penalty()
    return loss,parts
