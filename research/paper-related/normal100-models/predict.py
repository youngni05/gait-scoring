"""Inference with frozen, published parameters; no fitting or score clipping."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def new_model(family, stt_source=None):
    if family == 'CNN':
        from cnn import GdiCNN
        return GdiCNN()
    if stt_source is None:
        raise ValueError('STT and fusion require --stt-source pointing to the upstream source checkout')
    sys.path.insert(0, str(stt_source))
    if family == 'STT':
        from model.Transformer import SpatioTemporalTransformer
        return SpatioTemporalTransformer(num_frame=124, num_joints=4, in_chans=2,
                                        joint_embed_dim=12, num_heads=2, drop_rate=0.)
    from bilateral import BilateralSTT
    return BilateralSTT()


def raw_forward(model, family, x, lo, span):
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), 16):
            v = torch.from_numpy(np.ascontiguousarray(x[start:start + 16], dtype=np.float32))
            if family == 'CNN':
                from cnn import channels
                y = model(channels(v).reshape(-1, 12, 124))[:, 0]
            elif family == 'STT':
                y = model(v[..., :8].reshape(-1, 124, 4, 2))
            else:
                y = model(v[..., :8].reshape(-1, 2, 124, 4, 2))
            values.append((y * float(span) + float(lo)).reshape(-1, 2).numpy())
    return np.concatenate(values).astype(np.float64)


def combine(cnn, stt, bilateral, weight, fraction):
    base = weight * cnn + (1 - weight) * stt
    mean = base.mean(axis=1)
    difference = ((1 - fraction) * (base[:, 0] - base[:, 1])
                  + fraction * (bilateral[:, 0] - bilateral[:, 1]))
    return np.column_stack([mean + difference / 2, mean - difference / 2])


def predict(x, owners, manifest, weights_root, stt_source=None, calibration='original', mode='fusion'):
    if x.ndim != 4 or x.shape[1:] != (2, 124, 12) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Expected finite, nonempty x with shape [windows,2,124,12]')
    if (owners.shape != (len(x),) or owners.dtype.kind not in 'iu' or owners.min() < 0
            or not np.array_equal(np.unique(owners), np.arange(owners.max() + 1))):
        raise ValueError('window_record must contain consecutive nonnegative record indexes')
    n = int(owners.max()) + 1
    outputs = {}
    for member in manifest['members']:
        family = member['family']
        if mode.startswith('cnn') and family != 'CNN':
            continue
        if mode == 'cnn_refit' and member['tag'] != 'refit':
            continue
        path = weights_root / member['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != member['public_sha256']:
            raise ValueError('Weight fingerprint mismatch: ' + member['file'])
        ck = torch.load(path, map_location='cpu', weights_only=True)
        model = new_model(family, stt_source)
        model.load_state_dict(ck['model'], strict=True)
        raw = raw_forward(model, family, x, ck['lo'], ck['span'])
        sums = np.zeros((n, 2), dtype=np.float64)
        np.add.at(sums, owners, raw)
        record = sums / np.bincount(owners, minlength=n)[:, None]
        slope, intercept = member['stanford_calibration'] if calibration == 'stanford' else member['calibration']
        outputs[(family, member['fold'], member['tag'])] = record * slope + intercept
    folds = []
    for fold in range(5):
        def bag(family):
            selected = [outputs[(family, fold, m['tag'])] for m in manifest['members']
                        if m['family'] == family and m['fold'] == fold and not m['excluded_from_bag']]
            return np.mean(selected, axis=0)
        if mode == 'cnn_refit':
            value = outputs[('CNN', fold, 'refit')]
        elif mode == 'cnn_bag':
            value = bag('CNN')
        else:
            selection = manifest['selections'][fold]
            value = combine(bag('CNN'), bag('STT'), outputs[('bilateral', fold, 'refit')],
                            selection['cnn_weight'], selection['fraction'])
        folds.append(value)
    return np.mean(folds, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--weights', type=Path, default=HERE)
    parser.add_argument('--manifest', type=Path, default=HERE / 'models.json')
    parser.add_argument('--stt-source', type=Path)
    parser.add_argument('--mode', choices=['cnn_refit', 'cnn_bag', 'fusion'], default='fusion')
    parser.add_argument('--calibration', choices=['original', 'stanford'], default='original')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.out.exists():
        raise FileExistsError('Output already exists')
    with np.load(args.input, allow_pickle=False) as data:
        x = data['x'].astype(np.float32)
        owners = data['window_record'] if 'window_record' in data else np.arange(len(x))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    result = predict(x, owners, manifest, args.weights, args.stt_source, args.calibration, args.mode)
    np.savez_compressed(args.out, prediction=result)
    print(json.dumps({'records': len(result), 'mode': args.mode, 'calibration': args.calibration,
                      'shape': list(result.shape), 'fitting_steps': 0}))


if __name__ == '__main__':
    main()
