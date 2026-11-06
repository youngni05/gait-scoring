"""Toronto XY/confidence -> current 124-frame input -> frozen 512-D features."""
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MB = ROOT / 'outputs/motionbert-smoke-20261015'
DATA = ROOT / 'outputs/gait-public-access-20261015/Toronto'
sys.dont_write_bytecode = True
sys.path[:0] = [str(MB/'vendor'), str(MB/'deps')]
import numpy as np
import openpyxl


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def functions():
    namespace = {'np': np}
    for path, name in [(MB/'assets/mmpose_inference_3d.py', 'convert_keypoint_definition'),
                       (ROOT/'tools/stanford-sample-check/prepare_ridge_data.py', 'normalize')]:
        node = next(x for x in ast.walk(ast.parse(path.read_text(encoding='utf8')))
                    if isinstance(x, ast.FunctionDef) and x.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['convert_keypoint_definition'], namespace['normalize']


def adapt(values, convert, normalize):
    assert values.ndim == 2 and values.shape[1] == 52
    t = values[:, 0]
    steps = np.diff(t)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    if not len(positive):
        return None, {'status': 'excluded_invalid_time'}
    nominal_step = float(np.median(positive))
    motion = values[:, 1:].reshape(-1, 17, 3)
    valid = (np.isfinite(values).all(1) & ((motion[..., 2] >= 0) & (motion[..., 2] <= 1)).all(1)
             & (motion[:, 11:17, 2] > 0).all(1))
    runs, start = [], None
    for i, good in enumerate(valid):
        connected = i > 0 and valid[i-1] and 0 < t[i]-t[i-1] <= 1.5*nominal_step
        if start is not None and (not good or not connected):
            runs.append((start, i-1))
            start = None
        if good and start is None:
            start = i
    if start is not None:
        runs.append((start, len(t)-1))
    if not runs:
        return None, {'status': 'excluded_no_valid_run'}
    first, last = max(runs, key=lambda pair: t[pair[1]]-t[pair[0]])
    span = float(t[last]-t[first])
    info = {'source_rows': len(t), 'valid_source_rows': int(valid.sum()), 'median_source_step_seconds': nominal_step,
            'longest_valid_span_seconds': span, 'source_start_row': first, 'source_end_row': last}
    target_time = t[first]+np.arange(124)/30
    if span+1e-10 < 123/30:
        return None, {**info, 'status': 'excluded_shorter_than_4_1_seconds'}
    assert target_time[-1] <= t[last]+1e-10 and valid[first:last+1].all()
    assert np.all((np.diff(t[first:last+1]) > 0) & (np.diff(t[first:last+1]) <= 1.5*nominal_step))
    flat = values[first:last+1, 1:]
    sampled = np.stack([np.interp(target_time, t[first:last+1], flat[:, j]) for j in range(51)], axis=1)
    coco = sampled.reshape(124, 17, 3).astype(np.float32)
    h36m = convert(coco, 'coco', 'h36m')
    result = normalize(h36m)
    assert result.shape == (124, 17, 3) and result.dtype == np.float32 and np.isfinite(result).all()
    assert np.array_equal(result[..., 2], h36m[..., 2])
    assert np.max(np.abs(result)) <= 1
    return result, {**info, 'status': 'included', 'output_start_seconds': float(target_time[0]),
                    'output_end_seconds': float(target_time[-1]), 'zero_confidence_joints': int((result[..., 2] == 0).sum())}


def labels():
    wb = openpyxl.load_workbook(DATA/'Table 1.xlsx', read_only=True, data_only=True)
    rows = list(wb['Baseline Information'].values)
    columns = {name.strip(): i for i, name in enumerate(rows[0]) if isinstance(name, str)}
    result = {}
    for row in rows[1:]:
        if type(row[0]) in (float, int):
            person = int(row[0])
            assert person not in result
            y = [row[columns['POMA-Gait']], row[columns['TUG (s)']]]
            assert all(isinstance(x, (float, int)) for x in y) and np.isfinite(y).all()
            result[person] = y
    wb.close()
    assert len(result) == 14
    return result


def encoder():
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5, num_heads=8,
                     mlp_ratio=4, num_joints=17, maxlen=243, qkv_bias=True, att_fuse=True)
    cp = torch.load(MB/'assets/MB_lite.bin', map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in cp['model_pos'].items()}, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, torch


def main():
    tick = time.monotonic()
    protocol = json.loads((HERE/'protocol.json').read_text())
    assert not (HERE/'features.npz').exists(), 'Existing experiment must not be overwritten'
    sources = [DATA/'PoseTracking.zip', DATA/'Table 1.xlsx', MB/'assets/MB_lite.bin',
               MB/'vendor/lib/model/DSTformer.py', MB/'vendor/lib/model/drop.py',
               MB/'assets/mmpose_inference_3d.py', ROOT/'tools/stanford-sample-check/prepare_ridge_data.py',
               HERE/'protocol.json', Path(__file__)]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    assert hashes[(MB/'assets/MB_lite.bin').relative_to(ROOT).as_posix()] == protocol['encoder']['weights_sha256']
    save('source_receipt.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'sha256': hashes})
    convert, normalize = functions()
    names = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
    header = ['time']+[f'{name}_{axis}' for name in names for axis in ['x','y','conf']]
    inputs, people, records = [], [], []
    with zipfile.ZipFile(DATA/'PoseTracking.zip') as archive:
        for member in sorted(archive.namelist()):
            if '/Alphapose/' not in member or not member.endswith('.csv'):
                continue
            raw = archive.read(member)
            rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
            base = {'member': member, 'sha256': hashlib.sha256(raw).hexdigest()}
            if rows[0] != header:
                records.append({**base, 'status': 'excluded_incomplete_header'})
                continue
            x, info = adapt(np.asarray(rows[1:], dtype=np.float64), convert, normalize)
            if x is not None:
                person = int(re.search(r'OAW(\d+)-', member).group(1))
                assert person == int(Path(member).parent.name)
                records.append({**base, **info, 'person': person, 'input_row': len(inputs)})
                inputs.append(x)
                people.append(person)
            else:
                records.append({**base, **info})
    save('preparation.json', {'records': records, 'counts': dict(Counter(r['status'] for r in records)),
                              'people': len(set(people)), 'windows': len(inputs)})
    x = np.stack(inputs)
    person = np.asarray(people, dtype=np.int64)
    np.savez_compressed(HERE/'inputs.npz', input=x, person=person)
    clinical = labels()
    assert set(people) <= clinical.keys()
    model, torch = encoder()
    features = []
    with torch.inference_mode():
        for i, value in enumerate(x):
            feature = model.get_representation(torch.from_numpy(value[None])).mean((1, 2)).numpy()[0]
            assert feature.shape == (512,) and np.isfinite(feature).all()
            features.append(feature.copy())
            if (i+1) % 10 == 0:
                print(json.dumps({'frozen_features': i+1, 'total': len(x), 'seconds': round(time.monotonic()-tick, 1)}), flush=True)
        repeat = model.get_representation(torch.from_numpy(x[:1])).mean((1, 2)).numpy()[0]
    np.testing.assert_array_equal(features[0], repeat)
    person_ids = np.unique(person)
    X = np.stack([np.mean(np.stack(features)[person == p].astype(np.float64), axis=0) for p in person_ids])
    Y = np.asarray([clinical[int(p)] for p in person_ids], dtype=np.float64)
    np.savez_compressed(HERE/'features.npz', record_features=np.stack(features), record_person=person,
                        X=X, Y=Y, person=person_ids, windows_per_person=np.asarray([(person == p).sum() for p in person_ids]))
    assert all(sha(ROOT/path) == value for path, value in hashes.items())
    save('feature_receipt.json', {'people': len(person_ids), 'windows': len(x), 'input_shape': list(x.shape),
                                'person_features_shape': list(X.shape), 'repeat_exactly_equal': True,
                                'source_files_unchanged': True, 'trainable_encoder_parameters': 0,
                                'runtime': {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'numpy': np.__version__, 'device': 'cpu'},
                                'elapsed_seconds': time.monotonic()-tick,
                                'sha256': {name: sha(HERE/name) for name in ['inputs.npz','features.npz','preparation.json','protocol.json']}})
    print(json.dumps({'people': len(person_ids), 'windows': len(x), 'status': 'features_prepared'}), flush=True)


if __name__ == '__main__':
    main()
