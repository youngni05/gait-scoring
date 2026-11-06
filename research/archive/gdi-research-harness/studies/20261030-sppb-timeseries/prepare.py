"""Link scores by exact full ID/date; central-camera WALKING ONLY input."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import scipy.io
import pandas as pd
import hashlib
import json
from sklearn.model_selection import KFold

def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def make_sequence(frames):
    points = np.zeros((len(frames), 18, 3), dtype=np.float32)
    ambiguous = 0
    for t, frame in enumerate(frames):
        frame = np.asarray(frame).reshape(-1, 4)
        positive_ids = frame[frame[:, 3] > 0, 3]
        if len(positive_ids) != len(set(positive_ids)):
            ambiguous += 1
            continue  # Source has duplicate joint IDs: mask whole ambiguous frame.
        seen = set()
        for a, b, confidence, jid in frame:
            if jid <= 0:
                continue
            assert jid == int(jid) and 1 <= jid <= 18 and jid not in seen
            seen.add(jid)
            if np.isfinite([a, b, confidence]).all() and 0 < confidence <= 1 and a > 0 and b > 0:
                points[t, int(jid)-1] = [a, b, confidence]
    good = (points[:, [8, 11, 1], 2] > 0).all(1)
    if len(frames) < 32 or good.mean() < .5:
        return None, {'frames': len(frames), 'root_valid_fraction': float(good.mean())}
    root = points[:, [8, 11], :2].mean(1)
    scale = np.median(np.linalg.norm(points[good, 1, :2] - root[good], axis=1))
    assert scale > 0 and np.isfinite(scale)
    points[:, :, :2] = (points[:, :, :2] - root[:, None, :]) / scale
    valid = (points[:, :, 2] > 0) & good[:, None]
    points[~valid] = 0
    # Nearest observed frame: no coordinate interpolation across missing joints.
    idx = np.rint(np.linspace(0, len(points)-1, 128)).astype(int)
    x = points[idx].reshape(128, 54).T
    return x, {'frames': len(frames), 'ambiguous_frames': ambiguous, 'root_valid_fraction': float(good.mean()),
               'joint_valid_fraction': float(valid.mean()), 'scale_pixels': float(scale)}

def run():
    protocol = {
        'declared_before_fitting': True,
        'input': 'walkingSet.mat skeletonJointsCx ONLY; 18 native OpenPose joints x (root-relative x,y,confidence); 128 nearest observed frames across entire walk, no synthetic time axis',
        'other_inputs': 'log1p source walking frame count only; no age, sex, site, ID, date or component scores',
        'target': 'same full participant ID AND exact date; sum walking/balance/ststand evaluations, each integral 0..4; reconstructed SPPB component total, not separately supplied total',
        'quality': '>=32 source frames and >=50% frames with neck and both hips; frames with duplicate joint IDs fully masked; missing joints zero with confidence zero; no between-visit/camera imputation',
        'normalization': 'each frame hip center; per-sequence median neck-to-hip length; no population-fitted preprocessing',
        'split': 'fixed label-independent shuffled unique full IDs, KFold 5 seed20261030; nested 3-fold unique training IDs seed20261031',
        'weight': 'equal total weight per person during training and evaluation; all visits stay grouped',
        'baseline': 'training person-weighted mean and median; frame-count-only Ridge as acquisition-duration control',
        'cpu': {'Ridge_alpha': [1., 100.], 'RF_leaf': [2, 5], 'RF_trees': 128},
        'gpu': {'models': ['CNN', 'TCN'], 'width': 24, 'kernel': 5, 'layers': 3, 'dropout': .2, 'lr': .001, 'weight_decay': .001, 'max_epochs': 80, 'patience': 12, 'batch': 'full training set with person weights', 'selection': 'inner 3-fold best epoch; median epoch refit on all outer training people', 'seeds': [20261030]},
        'metrics': 'primary mean of each person mean absolute visit error; also person-weighted RMSE; no clipping, rounding, clinical success threshold or outer-fold selection',
        'pilot': 'CUDA allocate/calculate/synchronize; real fold0 training subset 3 epochs; finite loss/prediction; checkpoint save/reload equal predictions; timing/peak allocated/reserved memory',
        'scope': 'small exploratory single-source video-derived skeleton study; not raw RGB validation or clinical performance claim; no GDI input, split or job changes',
        'method_basis': 'Romeo2020 dataset: real 18-joint OpenPose sequences and SPPB component evaluations; Romeo2023: height-normalized interjoint time series and convolution/recurrent comparison; TCAI-FallNet2026: lightweight temporal convolution candidate, different all-task classification objective',
        'permissions': 'public author repository README invites use with paper citation; explicit repository data LICENSE not found. Research-only local derived inputs; no raw/derived dataset publication; no assertion of CC license transfer from author website or paper',
        'fps': 'paper describes nominal20fps but resynchronization to lowest recorded rate; no per-record timestamps in MAT verified. Native frame count used; no exact seconds claim'
    }
    save('protocol.json', protocol)
    sets = {k: scipy.io.loadmat(HERE/'raw'/f'{k}Set.mat', simplify_cells=True)[k+'Set'] for k in ['walking', 'balance', 'ststand']}
    lookups = {}
    for name, rows in sets.items():
        lookup = {}
        for i, r in enumerate(rows):
            key = (str(r[0]), str(r[1]))
            assert key not in lookup, (name, key)
            assert float(r[4]) == int(r[4]) and 0 <= int(r[4]) <= 4
            lookup[key] = (i, r)
        lookups[name] = lookup
    common = set.intersection(*(set(v) for v in lookups.values()))
    manifest, rejected, xs = [], [], []
    for key in sorted(lookups['walking']):
        if key not in common:
            rejected.append({'id': key[0], 'date': key[1], 'reason': 'missing same-date component label'})
            continue
        rows = {k: lookups[k][key] for k in sets}
        assert len({str(r[1][2]) for r in rows.values()}) == 1
        assert len({int(r[1][3]) for r in rows.values()}) == 1
        walk = rows['walking'][1]
        x, qc = make_sequence(walk[5].skeletonJointsCx)
        if x is None:
            rejected.append({'id': key[0], 'date': key[1], 'reason': 'pose quality', **qc})
            continue
        record = {'sample': len(xs), 'id': key[0], 'date': key[1], 'target': sum(int(r[1][4]) for r in rows.values()), **qc}
        record.update({k+'_score': int(v[1][4]) for k,v in rows.items()})
        record.update({k+'_source_row0': int(v[0]) for k,v in rows.items()})
        manifest.append(record)
        xs.append(x)
    df = pd.DataFrame(manifest)
    people = np.array(sorted(df.id.unique()))
    folds = {}
    for fold, (_, test) in enumerate(KFold(5, shuffle=True, random_state=20261030).split(people)):
        folds.update({p: fold for p in people[test]})
    df['fold'] = df.id.map(folds)
    x = np.asarray(xs)
    assert x.shape == (len(df), 54, 128) and np.isfinite(x).all()
    np.savez_compressed(HERE/'input.npz', x=x, y=df.target.to_numpy(dtype=np.float32), ids=df.id.to_numpy(dtype=str), folds=df.fold.to_numpy(), frames=df.frames.to_numpy(dtype=np.float32))
    df.to_csv(HERE/'manifest.csv', index=False)
    save('rejected.json', rejected)
    save('input-audit.json', {'source_sizes': {k: {'records': len(v), 'people':len(set(v[:,0]))} for k,v in sets.items()},
        'same_date_common_records':len(common), 'same_date_common_people':len(set(k[0] for k in common)),
        'eligible_records':len(df), 'eligible_people':len(people), 'score_counts':{str(k):int(v) for k,v in df.target.value_counts().sort_index().items()},
        'frames_range': [int(df.frames.min()),int(df.frames.max())], 'root_valid_fraction_range':[float(df.root_valid_fraction.min()),float(df.root_valid_fraction.max())],
        'shape':list(x.shape), 'rejected':len(rejected), 'raw_files_sha256':{p.name:sha(p) for p in (HERE/'raw').glob('*.mat')},
        'input_sha256':sha(HERE/'input.npz'), 'manifest_sha256':sha(HERE/'manifest.csv'), 'protocol_sha256':sha(HERE/'protocol.json')})
    print((HERE/'input-audit.json').read_text(encoding='utf8'))

if __name__ == '__main__':
    run()
