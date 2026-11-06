"""Frozen 3-second cohort and paired person splits for the follow-up."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE.parent / '20261030-opencap415-tug-timeseries'
sys.path.insert(0, str(ROOT / 'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np

SEEDS = [20261030, 20261031, 20261032]
VIEWS = ['mean_speed', 'speed_summary', 'pelvis_motion', 'full']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, obj):
    path = HERE / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def load():
    audit = json.loads((HERE / 'preparation.json').read_text())
    assert sha(HERE / 'input.npz') == audit['source_input_sha256']
    assert sha(HERE / 'protocol.json') == audit['protocol_sha256']
    d = dict(np.load(HERE / 'input.npz', allow_pickle=False))
    assert len(d['ids']) == len(set(d['ids'])) == 298
    assert d['x'].shape == (298, 19, 91)
    assert np.isfinite(d['x']).all() and np.isfinite(d['y']).all()
    return d

def features(x):
    summary = np.concatenate([x.mean(-1), x.std(-1), np.abs(np.diff(x, axis=-1)).mean(-1)], axis=1)
    return {'mean_speed': summary[:, [17]],
            'speed_summary': summary[:, [17, 36, 55]],
            'pelvis_motion': summary[:, [17, 18, 36, 37, 55, 56]],
            'full': summary}

def lower_median(y):
    return float(np.sort(y)[(len(y)-1)//2])
