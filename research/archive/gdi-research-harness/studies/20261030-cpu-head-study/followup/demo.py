"""Read a saved linear head and precomputed features; no pose encoder or training."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import cpu_head as m
import argparse
import numpy as np
import torch


def predict(features_path, checkpoint_path, output_path):
    with np.load(features_path, allow_pickle=False) as z:
        if set(z.files) != {'features'}:
            raise ValueError('expected feature-only demonstration input')
        x = z['features'].copy()
    if x.shape != (2, 512) or not np.isfinite(x).all():
        raise ValueError('expected front/back 512-feature input')
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = ck['model']
    if set(state) != {'mu', 'scale', 'output.weight', 'output.bias'}:
        raise ValueError('not a saved head-only model')
    if state['mu'].shape != (512,) or state['scale'].shape != (512,) or not (state['scale'] > 0).all():
        raise ValueError('invalid saved scaler')
    with torch.inference_mode():
        standardized = (torch.from_numpy(x).double() - state['mu']) / state['scale']
        scores = torch.nn.functional.linear(standardized, state['output.weight'], state['output.bias']).numpy()
    independent = ((x.astype(np.float64) - state['mu'].numpy()) / state['scale'].numpy()) @ state['output.weight'].numpy().T + state['output.bias'].numpy()
    if float(np.abs(independent - scores).max()) > 1e-10:
        raise ValueError('independent linear output mismatch')
    rows = []
    for wi, window in enumerate(['front', 'back']):
        values = []
        for si, side in enumerate(['left', 'right']):
            score = float(scores[wi, si])
            low = int(np.floor(score / 10) * 10)
            values.append({'side': side, 'predicted_gdi': score, 'bin_lower_inclusive': low,
                           'bin_upper_exclusive': low + 10})
        rows.append({'window': window, 'results': values})
    result = {'input': 'one anonymous precomputed front/back feature example',
              'source_kind': 'existing public development record, evaluated by the fold model that excluded its person',
              'features_shape': [2, 512], 'windows': rows, 'bin_rule': '[10k,10(k+1)); no clipping; no clinical labels',
              'features_sha256': m.digest(features_path), 'checkpoint_sha256': m.digest(checkpoint_path),
              'encoder_used': False, 'training_performed': False, 'rgb_or_app_validation': False}
    m.save_json(output_path, result)
    print(__import__('json').dumps({'windows': rows, 'encoder_used': False, 'training_performed': False}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    predict(args.features, args.checkpoint, args.output)
