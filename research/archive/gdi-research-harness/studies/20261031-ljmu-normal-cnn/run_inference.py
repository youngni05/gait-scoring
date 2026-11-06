"""Replay the existing sample and run unchanged frozen CNNs on the supplied input."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = ROOT / 'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
sys.path.insert(0, str(BUNDLE))
import check_sample
import infer


def main():
    sample = check_sample.check(HERE / 'sample_replay')
    print(json.dumps(sample), flush=True)
    source = HERE / 'input/normal_cnn_v2/prepared_normal/prepared.npz'
    with np.load(source, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    infer.validate_input(data)
    assert data['x'].shape == (3307, 2, 124, 12)
    assert len(data['record_id']) == 882
    assert len(set(data['person_id'])) == 64
    assert np.isnan(data['fps']).all()
    result = infer.infer(source, HERE / 'predictions', list(range(5)))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
