"""Paths and the explicit image-coordinate adapter for the paired-video study."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / '20261101-self-video-gdi'
BIG = Path('D:/gait-research-data/D_P/self-video-mediapipe-20261101')
OLD_BIG = Path('D:/gait-research-data/D_P/self-video-gdi-20261101')
MP_HOME = ROOT / 'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot'
TASK = MP_HOME / 'pose_landmarker_full.task'
sys.path.insert(0, str(MP_HOME / 'deps'))
import numpy as np

# Only joints actually consumed by the frozen GDI pipeline are mapped.
# MP foot_index is a proxy for BODY25 big_toe, not an identical landmark.
MAP = {9: 24, 10: 26, 11: 28, 12: 23, 13: 25, 14: 27, 19: 31, 22: 32}
JOINTS = {'left_hip': 12, 'left_knee': 13, 'left_ankle': 14, 'left_foot_proxy': 19,
          'right_hip': 9, 'right_knee': 10, 'right_ankle': 11, 'right_foot_proxy': 22}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def adapt(landmarks, width, height):
    raw = np.full((len(landmarks), 25, 3), np.nan, dtype=np.float64)
    for op_index, mp_index in MAP.items():
        raw[:, op_index, 0] = landmarks[:, mp_index, 0] * width
        raw[:, op_index, 1] = landmarks[:, mp_index, 1] * height
        raw[:, op_index, 2] = landmarks[:, mp_index, 3]
    return raw.reshape(-1, 75)
