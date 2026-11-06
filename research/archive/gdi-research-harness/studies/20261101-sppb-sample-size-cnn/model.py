"""Coordinate-only adaptation of the existing GDI six-block CNN (Apache-2.0)."""
import torch
from torch import nn
from torch.nn import functional as F


class SameConv(nn.Conv1d):
    def forward(self, x):
        k = self.kernel_size[0] - 1
        return super().forward(F.pad(x, (k//2, k-k//2)))


class TransferCNN(nn.Module):
    def __init__(self, outputs=1, dropout=0.5, channels=32):
        super().__init__()
        self.convs = nn.ModuleList([SameConv(6 if i == 0 else channels, channels, 8) for i in range(6)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(channels, eps=0.001, momentum=0.01) for _ in range(6)])
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(10*channels, 10)
        self.output = nn.Linear(10, outputs)
        self.frozen_blocks = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (2,6,124):
            raise ValueError('Expected windows,side2,channel6,time124')
        n = x.shape[0]
        x = x.reshape(-1,6,124)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = norm(F.relu(conv(x)))
            if i % 2:
                x = self.dropout(F.max_pool1d(x, 3 if i == 5 else 2))
        x = x.transpose(1,2).reshape(2*n,-1)
        h = F.relu(self.hidden(x)).reshape(n,2,10).mean(1)
        return self.output(h)

    def initialize(self, checkpoint, source):
        if source == 'scratch':
            return []
        state = checkpoint['model']
        current = self.state_dict()
        copied = []
        for k, v in state.items():
            if k.startswith('output.'):
                continue
            if k == 'convs.0.weight' and source == 'gdi':
                v = v[:,:6,:]
            if k not in current or current[k].shape != v.shape:
                raise ValueError('Incompatible pretrained tensor: '+k)
            current[k] = v.clone()
            copied.append(k)
        expected = [k for k in current if not k.startswith('output.')]
        if set(copied) != set(expected):
            raise ValueError('Missing pretrained backbone tensors')
        self.load_state_dict(current)
        return copied

    def freeze(self, blocks):
        self.frozen_blocks = blocks
        for i in range(blocks):
            for module in (self.convs[i], self.norms[i]):
                for p in module.parameters():
                    p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for i in range(self.frozen_blocks):
                self.norms[i].eval()
        return self
