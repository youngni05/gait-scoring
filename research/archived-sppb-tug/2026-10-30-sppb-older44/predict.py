"""Inference for archived 3-second functional-score experiments; prepared input only."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import torch
from common import features
from gpu import Net

INPUT_CHANNELS = 39


def predict(weights, x):
    weights = Path(weights)
    if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (INPUT_CHANNELS, 91) or len(x) == 0:
        raise ValueError('Expected nonempty float32 [N,%d,91] prepared input' % INPUT_CHANNELS)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite input')
    frames = np.full(len(x), 91)
    if weights.suffix == '.joblib':
        scaler, model = joblib.load(weights)
        values = features(x, frames)
        result = model.predict(scaler.transform(values))
    elif weights.suffix == '.pt':
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1 or checkpoint.get('input_channels') != INPUT_CHANNELS:
            raise ValueError('Weights do not match this experiment')
        model = Net(checkpoint['name']).eval()
        model.load_state_dict(checkpoint['state'], strict=True)
        norm = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in checkpoint['normalization'].items()}
        values = torch.as_tensor((x - norm['xmean']) / norm['xstd'], dtype=torch.float32)
        frame_feature = torch.as_tensor((np.log1p(frames)[:, None] - norm['fmean']) / norm['fstd'], dtype=torch.float32)
        with torch.inference_mode():
            result = model(values, frame_feature).numpy() * norm['ystd'] + norm['ymean']
    else:
        raise ValueError('Expected published .joblib or .pt model')
    if result.shape != (len(x),) or not np.isfinite(result).all():
        raise ValueError('Invalid predictions')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.suffix != '.npy' or args.output.resolve() in [args.input.resolve(), args.weights.resolve()]:
        parser.error('Use a separate .npy output file')
    values = predict(args.weights, np.load(args.input, allow_pickle=False))
    np.save(args.output, values, allow_pickle=False)
    print('Saved %d predictions to %s' % (len(values), args.output))


if __name__ == '__main__':
    main()
