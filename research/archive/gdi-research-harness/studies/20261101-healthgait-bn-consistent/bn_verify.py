"""Independent CPU recomputation of the saved BN buffers, using expected train IDs."""
import hashlib
import json
import numpy as np
import torch
from model import TransferCNN


def verify_bn(directory, d, expected_train, input_sha256):
    metadata = json.loads((directory/'bn_calibration.json').read_text(encoding='utf8'))
    fit = json.loads((directory/'fit_complete.json').read_text(encoding='utf8'))
    identity = json.loads((directory/'identity.json').read_text(encoding='utf8'))
    assert identity['bn_policy'] == metadata['policy'] == 'train_population_single_pass_v1'
    assert metadata['epoch'] == fit['best_epoch']
    assert metadata['input_sha256'] == input_sha256 == identity['input_sha256']
    ids = sorted(np.unique(d['ids'][expected_train]).tolist())
    assert ids == metadata['train_people'] == identity['train_people']
    assert not set(ids).intersection(identity['validation_people'] + identity['test_people'])
    indices = np.flatnonzero(expected_train[d['owner']]).astype('<i8')
    assert metadata['training_windows'] == len(indices)
    assert metadata['window_indices_sha256'] == hashlib.sha256(indices.tobytes()).hexdigest()
    assert metadata['batches_per_layer'] == 1 and not metadata['dropout_enabled']
    assert metadata['learned_parameters_unchanged']
    ck = torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    ref = TransferCNN(d['y'].shape[1],identity['config']['dropout'])
    ref.load_state_dict(ck['model'])
    ref.eval()
    count = 0
    for layer in ref.modules():
        if isinstance(layer,torch.nn.BatchNorm1d):
            layer.running_mean.zero_()
            layer.running_var.fill_(1)
            layer.num_batches_tracked.zero_()
            layer.momentum = 1.0
            layer.training = True
            count += 1
    assert count == metadata['bn_layers'] == 6
    with torch.no_grad():
        ref(torch.as_tensor(d['x'][indices]))
    max_error = 0.
    for name, value in ref.state_dict().items():
        if name.endswith(('running_mean','running_var','num_batches_tracked')):
            expected = ck['model'][name]
            if name.endswith('num_batches_tracked'):
                assert value.item() == expected.item() == 1
            else:
                np.testing.assert_allclose(value.numpy(),expected.numpy(),atol=1e-4,rtol=1e-5,err_msg=str(directory)+' '+name)
                max_error = max(max_error,float((value-expected).abs().max()))
    return max_error
