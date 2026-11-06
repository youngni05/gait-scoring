"""One training-only population pass; dropout is disabled for inference calibration."""
import hashlib
import numpy as np
import torch

POLICY = 'train_population_single_pass_v1'


def reset_running_statistics(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.reset_running_stats()


def calibrate(model, d, train, device, epoch, input_sha256):
    windows = np.flatnonzero(train[d['owner']]).astype('<i8')
    assert 0 < len(windows) <= 3000, 'Single-pass calibration capacity exceeded'
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    momenta = [m.momentum for m in layers]
    for m in layers:
        m.reset_running_stats()
        m.momentum = 1.0
        m.train()
    with torch.no_grad():
        model(torch.from_numpy(d['x'][windows]).to(device))
    for m, momentum in zip(layers, momenta):
        assert m.num_batches_tracked.item() == 1
        assert torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all()
        m.momentum = momentum
    model.eval()
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    return dict(policy=POLICY, epoch=int(epoch), input_sha256=input_sha256,
                train_people=sorted(np.unique(d['ids'][train]).tolist()),
                training_windows=len(windows), window_indices_sha256=hashlib.sha256(windows.tobytes()).hexdigest(),
                bn_layers=len(layers), batches_per_layer=1, dropout_enabled=False,
                learned_parameters_unchanged=True)
