"""Inference for the published last-stage MotionBERT; no initialization assets needed."""
import argparse
from pathlib import Path
import numpy as np
import torch
from lib.model.DSTformer import DSTformer


class Scorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                                 depth=5, num_heads=8, mlp_ratio=4, num_joints=17,
                                 maxlen=243, qkv_bias=True, att_fuse=True)
        self.register_buffer('mu', torch.zeros(512, dtype=torch.float64))
        self.register_buffer('scale', torch.ones(512, dtype=torch.float64))
        self.output = torch.nn.Linear(512, 2).double()

    def forward(self, x):
        features = self.encoder.get_representation(x).mean((1, 2))
        return self.output((features.double() - self.mu) / self.scale)


def predict(weights, x):
    if x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (124, 17, 3) or len(x) == 0:
        raise ValueError('Expected nonempty float32 [N,124,17,3] prepared input')
    if not np.isfinite(x).all() or np.abs(x[..., :2]).max() > 1 or not (0 <= x[..., 2]).all() or not (x[..., 2] <= 1).all():
        raise ValueError('Prepared coordinates or confidence are outside the input contract')
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('architecture') != 'motionbert' or checkpoint.get('variant') != 'last_stage':
        raise ValueError('Expected published last-stage MotionBERT weights')
    torch.set_num_threads(4)
    model = Scorer().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    output = []
    with torch.inference_mode():
        for row in x:
            output.append(model(torch.from_numpy(row[None])).numpy()[0])
    result = np.stack(output)
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
