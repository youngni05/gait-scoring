"""Frozen-model diagnostics. Transformations are sensitivity probes, not corrections."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = ROOT / 'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
DEV = HERE.parent / '20261030-gpu-motionbert-study/cnn-multiwindow695'
OUT = HERE / 'diagnostics'
sys.path.insert(0, str(BUNDLE))
import infer
import preprocess

XCOLS = [0, 2, 4, 6, 11]


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def reflect(x):
    result = x.copy()
    result[..., XCOLS] *= -1
    return result


def full_predict(d, name):
    path = OUT / (name + '.npz')
    if path.exists():
        return load(path)['predictions']
    scores = []
    for f in range(5):
        _, pred, _ = infer.predict_records(d['x'], d['window_record'], len(d['person_id']), f)
        scores.append(pred)
        print(json.dumps({'stage': name, 'fold': f, 'records': len(pred)}), flush=True)
    result = np.stack(scores)
    np.savez_compressed(path, predictions=result, person_id=d['person_id'])
    return result


def speed_inputs(d):
    """Stitch only observed feature frames; same centers/windows for all rates."""
    x, owner, starts = d['x'], d['window_record'], d['window_start']
    values = {r: [] for r in (.8, 1., 1.2)}
    owners, selected = [], []
    max_overlap_error = 0.
    for rid in range(len(d['person_id'])):
        ww = np.flatnonzero(owner == rid)
        last = int(starts[ww].max()) + 124
        timeline = np.full((2, last, 12), np.nan, dtype=np.float32)
        for w in ww:
            st = int(starts[w])
            old = timeline[:, st:st + 124]
            finite = np.isfinite(old)
            if finite.any():
                max_overlap_error = max(max_overlap_error, float(np.abs(old[finite] - x[w][finite]).max()))
            timeline[:, st:st + 124] = x[w]
        for w in ww:
            center = starts[w] + 61.5
            tmax = center + (np.arange(124) - 61.5) * 1.2
            a, b = int(np.floor(tmax[0])), int(np.ceil(tmax[-1]))
            if a < 0 or b >= last or not np.isfinite(timeline[:, a:b + 1]).all():
                continue
            for rate in values:
                t = center + (np.arange(124) - 61.5) * rate
                lo, hi = np.floor(t).astype(int), np.ceil(t).astype(int)
                frac = (t - lo)[None, :, None]
                value = timeline[:, lo] * (1 - frac) + timeline[:, hi] * frac
                values[rate].append(value.astype(np.float32))
            owners.append(rid)
            selected.append(int(w))
    assert max_overlap_error == 0
    assert selected
    np.testing.assert_array_equal(np.stack(values[1.]), x[selected])
    records = np.unique(owners)
    remap = np.full(len(d['person_id']), -1, dtype=int)
    remap[records] = np.arange(len(records))
    prepared = {r: dict(x=np.stack(v), window_record=remap[owners], person_id=d['person_id'][records])
                for r, v in values.items()}
    return prepared, dict(records=records.tolist(), original_windows=selected,
                         overlap_max_error=max_overlap_error, identity_exact=True,
                         input_feature_interpolation=True, actual_fps_not_inferred=True)


def main():
    OUT.mkdir(exist_ok=True)
    infer.verify_bundle()
    infer.configure()
    dev_path = DEV / 'bundle/data.npz'
    normal_path = HERE / 'input/normal_cnn_v2/prepared_normal/prepared.npz'
    source = {'dev_sha256': infer.sha(dev_path), 'normal_sha256': infer.sha(normal_path),
              'models_sha256': infer.sha(BUNDLE / 'models.json'),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    old = OUT / 'sources.json'
    if old.exists():
        previous = json.loads(old.read_text())
        for key in ('dev_sha256', 'normal_sha256', 'models_sha256'):
            assert previous[key] == source[key], 'Source changed; use a separate diagnostics directory'
        source['previous_script_sha256'] = previous.get('previous_script_sha256', [])
        if previous['script_sha256'] != source['script_sha256']:
            source['previous_script_sha256'].append(previous['script_sha256'])
    old.write_text(json.dumps(source, indent=2), encoding='utf-8')
    dev, normal = load(dev_path), load(normal_path)
    saved = load(DEV / 'analysis/cnn_oof.npz')
    np.testing.assert_array_equal(saved['person_id'], dev['person_id'])
    np.testing.assert_array_equal(saved['y'], dev['y'])
    original = full_predict(dev, 'development_all_models')
    actual_oof = original[dev['fold'], np.arange(len(dev['y']))]
    replay_error = float(np.abs(actual_oof - saved['predictions'][:, 0]).max())
    assert replay_error < 1e-4, replay_error
    # Check reflection at raw BODY25 level using the public reproduction sample.
    sample = load(BUNDLE / 'sample/expected.npz')
    raw = sample['selected_body25'].copy()
    valid = np.isfinite(raw[:, :, 0]) & (raw[:, :, 0] != 0)
    raw[:, :, 0][valid] = 4096 - raw[:, :, 0][valid]
    recomputed = preprocess.extract_windows(raw.reshape(-1, 75))
    np.testing.assert_array_equal(recomputed['window_start'], sample['window_start'])
    reflection_error = float(np.abs(recomputed['x'] - reflect(sample['x'])).max())
    assert reflection_error < 1e-5, reflection_error
    np.testing.assert_array_equal(reflect(reflect(normal['x'])), normal['x'])
    full_predict({**normal, 'x': reflect(normal['x'])}, 'normal_reflected')
    reflected_oof = np.empty_like(saved['predictions'][:, 0])
    for f in range(5):
        idx = np.flatnonzero(dev['fold'] == f)
        keep = np.isin(dev['window_record'], idx)
        remap = np.full(len(dev['y']), -1, dtype=int)
        remap[idx] = np.arange(len(idx))
        _, pred, _ = infer.predict_records(reflect(dev['x'][keep]), remap[dev['window_record'][keep]], len(idx), f)
        reflected_oof[idx] = pred
        print(json.dumps({'stage': 'development_reflected_oof', 'fold': f}), flush=True)
    np.savez_compressed(OUT / 'development_reflected_oof.npz', predictions=reflected_oof)
    speed, info = speed_inputs(normal)
    (OUT / 'time_probe_metadata.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    for rate, data in speed.items():
        full_predict(data, 'normal_time_' + str(rate).replace('.', '_'))
    verification = dict(passed=True, development_replay_max_error=replay_error,
        reflection_raw_preprocessing_max_error=reflection_error, reflection_involution_exact=True,
        time_identity_exact=True, time_overlap_max_error=info['overlap_max_error'],
        normal_records=len(normal['person_id']), development_records=len(dev['person_id']),
        no_training=True, no_calibration_fit=True, protected_data_used=False,
        transformation_is_sensitivity_probe_not_validated_correction=True)
    (OUT / 'model_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
