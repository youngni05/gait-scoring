"""PyTorch adaptation of mobile-gaitlab's doublesided CNN topology.

Source and Apache-2.0 license: sources/. Input and target handling are changed:
524 channels per side: unchanged 12 geometry channels plus 512 frozen temporal features; output 0 predicts that side; output 1 has zero loss weight, as in source.
This is not a numerical reproduction of the original Keras model.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SameConv(nn.Conv1d):
    def forward(self, x):
        # Keras SAME padding for even kernel 8: three left, four right.
        k = self.kernel_size[0] - 1
        return super().forward(F.pad(x, (k // 2, k - k // 2)))


class GdiCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([SameConv(524 if i == 0 else 32, 32, 8)
                                    for i in range(6)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(32, eps=0.001, momentum=0.01)
                                   for _ in range(6)])
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(10 * 32, 10)
        self.output = nn.Linear(10, 2)
        self.register_buffer("feature_mean",torch.zeros(512))
        self.register_buffer("feature_scale",torch.ones(512))
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (524, 124):
            raise ValueError('CNN expects B,524,124; 524 is channels, 124 is time')
        motion=(x[:,12:,:]-self.feature_mean[None,:,None])/self.feature_scale[None,:,None]
        x=torch.cat([x[:,:12,:],motion],dim=1)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = norm(F.relu(conv(x)))
            if i % 2:
                x = self.dropout(F.max_pool1d(x, 3 if i == 5 else 2))
        # Preserve Keras time-major Flatten ordering, although weights start fresh.
        x = x.transpose(1, 2).reshape(x.shape[0], -1)
        return self.output(F.relu(self.hidden(x)))

    def penalty(self):
        # Original notebook regularizes kernels of convolution layers 3..6 only.
        return sum(c.weight.square().sum() for c in self.convs[2:])


def channels(x):
    if x.ndim!=4 or tuple(x.shape[1:])!=(2,124,524):
        raise ValueError('Expected record, left/right, time124, features524')
    return x.transpose(2,3).contiguous()


def training_feature_stats(x):
    # Both sides receive the same full-body feature; use one copy, training records only.
    if x.ndim!=4 or x.shape[1:]!=(2,124,524):raise ValueError('Wrong fusion axes')
    v=x[:,0,:,12:].astype('float64')
    mean=v.mean(axis=(0,1));scale=v.std(axis=(0,1))
    scale[scale<1e-6]=1.0
    return torch.from_numpy(mean.astype('float32')),torch.from_numpy(scale.astype('float32'))
