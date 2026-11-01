"""Inference for the published 124-frame CNN; input is already prepared xy/confidence."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model import GdiCNN, channels


def predict(weights, x):
    if x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (124, 17, 3) or len(x) == 0:
        raise ValueError('Expected nonempty float32 [N,124,17,3] prepared input')
    if not np.isfinite(x).all() or np.abs(x[..., :2]).max() > 1 or not (0 <= x[..., 2]).all() or not (x[..., 2] <= 1).all():
        raise ValueError('Prepared coordinates or confidence are outside the input contract')
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('architecture') != 'cnn':
        raise ValueError('Expected published CNN weights')
    torch.set_num_threads(4)
    model = GdiCNN(checkpoint['dropout']).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    values = channels(torch.from_numpy(x))
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            output.append((model(values[start:start + 128]) * checkpoint['span'] + checkpoint['lo']).numpy())
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite predictions')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.npy' or args.output.resolve() in (args.input.resolve(), args.weights.resolve()):
        parser.error('Use a separate .npy output file')
    result = predict(args.weights, np.load(args.input, allow_pickle=False))
    np.save(args.output, result, allow_pickle=False)
    print('Saved %d left/right GDI predictions to %s' % (len(result), args.output))


if __name__ == '__main__':
    main()
