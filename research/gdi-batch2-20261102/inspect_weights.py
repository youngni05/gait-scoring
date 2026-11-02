"""Inspect a model file from this release without loading individual research data."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    path = args.path
    if path.suffix == '.pt':
        import torch
        value = torch.load(path, map_location='cpu', weights_only=True)
        report = {'fields': list(value), 'tensors': {k: list(v.shape) for k, v in value['model'].items()},
                  'fold': value.get('fold'), 'calibration': value.get('calibration')}
    elif path.suffix == '.npz':
        import numpy as np
        with np.load(path, allow_pickle=False) as value:
            report = {'arrays': {k: {'shape': list(value[k].shape), 'dtype': str(value[k].dtype)} for k in value.files}}
    elif path.suffix == '.json':
        value = json.loads(path.read_text(encoding='utf-8'))
        report = {'fields': list(value), 'selected_family': value.get('selected_family')}
    elif path.suffix == '.joblib':
        import joblib
        value = joblib.load(path)
        report = {'class': type(value).__name__, 'input_features': value.n_features_in_, 'trees': len(value.estimators_)}
    else:
        parser.error('Expected a .pt, .npz, .json or .joblib model file from this release')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
