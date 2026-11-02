"""Small temporal correction; the baseline CNN and MotionBERT are never optimized."""
import numpy as np
import torch
from torch import nn


class Correction(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(512))
        self.register_buffer('scale', torch.ones(512))
        self.project = nn.Conv1d(512, 16, 1)
        self.temporal = nn.Conv1d(16, 16, 3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(16, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (124, 512):
            raise ValueError('Expected record,time124,feature512')
        x = ((x-self.mean)/self.scale).transpose(1, 2)
        x = torch.relu(self.project(x))
        x = torch.relu(self.temporal(x)).mean(dim=2)
        return self.output(self.dropout(x)) * 10.0


def feature_stats(train):
    value = np.asarray(train, dtype=np.float64)
    mean = value.mean(axis=(0, 1)).astype(np.float32)
    scale = value.std(axis=(0, 1)).astype(np.float32)
    scale[scale < 1e-6] = 1
    return torch.from_numpy(mean), torch.from_numpy(scale)
