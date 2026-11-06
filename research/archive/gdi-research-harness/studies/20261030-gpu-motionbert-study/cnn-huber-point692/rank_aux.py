"""Adapt official RankSim to a mean auxiliary loss with deterministic tie sampling.
Official source: BorealisAI/ranksim-imbalanced-regression (CC BY-NC-SA 4.0).
Original files and license are preserved alongside this research-only adapter.
"""
import random,hashlib
import torch
from ranksim import batchwise_ranking_regularizer

def feature_loss(features,targets,interpolation):
    assert features.ndim==2 and targets.ndim==1 and len(features)==len(targets)
    unique=len(torch.unique(targets))
    if unique<2:return features.sum()*0
    # Tie choices depend only on this training batch, so restart does not need
    # a separate mutable Python RNG stream. Restore caller state afterwards.
    state=random.getstate()
    random.seed(int(hashlib.sha256(targets.detach().cpu().numpy().tobytes()).hexdigest()[:8],16))
    try:return batchwise_ranking_regularizer(features,targets[:,None],interpolation)/unique
    finally:random.setstate(state)
