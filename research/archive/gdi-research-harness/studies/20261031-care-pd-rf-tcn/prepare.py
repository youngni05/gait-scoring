"""Freeze full-walk inputs and official participant folds; no fitted preprocessing."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = ROOT / 'outputs/care-pd-audit-20261031'
sys.path.insert(0, str(AUDIT))
from audit import read


def rotations(v):
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    a = v / np.maximum(theta, 1e-15)
    k = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
    k[..., 0, 1], k[..., 0, 2] = -a[..., 2], a[..., 1]
    k[..., 1, 0], k[..., 1, 2] = a[..., 2], -a[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -a[..., 1], a[..., 0]
    t = theta[..., None]
    return np.eye(3) + np.sin(t)*k + (1-np.cos(t))*(k@k)


def sequence(pose, fps):
    # Exclude global orientation; keep local rotations of all 23 articulated joints.
    m = rotations(pose.reshape(-1, 24, 3)[:, 1:].astype(np.float64))
    six = m[..., :2].reshape(len(pose), 138)
    # Non-overlapping 40 ms bins average every original frame, including the final bin.
    bins = np.floor(np.arange(len(pose)) * 25 / fps + 1e-8).astype(int)
    sums = np.zeros((bins.max()+1, 138))
    np.add.at(sums, bins, six)
    counts = np.bincount(bins)
    assert (counts > 0).all()
    return (sums/counts[:, None]).astype(np.float32)


def features(x):
    parts = [x.mean(0), x.std(0), np.quantile(x, .1, axis=0),
             np.quantile(x, .9, axis=0), np.abs(np.diff(x, axis=0)).mean(0)*25]
    centered = x-x.mean(0)
    ac = []
    for lag in (5, 10, 15, 20, 25, 30, 35, 40):
        if len(x) <= lag:
            ac.append(0.)
        else:
            u, v = centered[:-lag], centered[lag:]
            ac.append(float((u*v).sum() / max(np.sqrt((u*u).sum()*(v*v).sum()), 1e-12)))
    return np.concatenate(parts + [np.array(ac)]).astype(np.float32)


def main():
    manifests = json.loads((AUDIT/'original-file-verification.json').read_text())
    arrays, rows, rf = [], [], []
    for entry in manifests:
        path = Path(entry['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        cohort = entry['cohort']
        split_file = ROOT / 'outputs/dataset-compatibility-20261030/CARE-PD' / (cohort+'_6fold_participants.pkl')
        folds = read(split_file)
        assignment = {p: f-1 for f, split in folds.items() for p in split['eval']}
        for person, walks in sorted(read(path).items()):
            for walk, record in sorted(walks.items()):
                x = sequence(record['pose'], float(record['fps']))
                assert np.isfinite(x).all() and len(x) > 1
                arrays.append(x)
                rf.append(features(x))
                rows.append(dict(index=len(rows), cohort=cohort, person=cohort+':'+str(person),
                    walk=str(walk), y=int(record['UPDRS_GAIT']), fold=assignment[str(person)],
                    frames=len(x), original_frames=len(record['pose']), fps=float(record['fps'])))
    lengths = np.array([len(x) for x in arrays])
    offsets = np.r_[0, np.cumsum(lengths)]
    np.savez_compressed(HERE/'data.npz', x=np.concatenate(arrays), offsets=offsets,
        rf=np.stack(rf), y=np.array([r['y'] for r in rows], dtype=np.float32),
        person=np.array([r['person'] for r in rows]),
        cohort=np.array([r['cohort'] for r in rows]), fold=np.array([r['fold'] for r in rows]))
    with (HERE/'manifest.csv').open('w', newline='', encoding='utf8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    protocol = dict(people=len(set(r['person'] for r in rows)), walks=len(rows),
        seeds=[20261030,20262030,20263030,20264030,20265030], outer_folds=6,
        inner='For outer f, validation=(f+1)%6; remaining four folds train; refit on five outer-train folds',
        input='All complete original walks, 23 local SMPL rotations represented by first two matrix columns, 25 Hz bin means; no global orientation, translation, body shape, cohort or medication input',
        rf_features='698: mean/std/p10/p90/mean absolute temporal difference per 138 channels plus 8 global normalized autocorrelations (missing lag=0)',
        model='Three masked Conv1d layers, width32, kernel5, dilation1/2/4, global masked mean, sigmoid score 0..3',
        rf='RandomForestRegressor 200 trees, max_features=.5, min_samples_leaf in [2,8], selected by inner person-equal MAE',
        optimizer='AdamW lr .001 weight_decay .001, SmoothL1 beta .5, batch32, max60 epochs, patience10',
        weights='Each participant equal total training weight; evaluation averages walk errors within each person before averaging people',
        baselines='Outer-training person-weighted mean and median, plus person-weighted mode for classification',
        high_score='Observed y>=2 and y==3 separately; continuous absolute error >=2 is large error',
        evaluation='MAE, RMSE, signed error, large-error rate, rounded accuracy, macro recall; seed-level results, no seed ensemble',
        source_files=manifests, prepared_frames=int(offsets[-1]), max_frames=int(lengths.max()),
        data_sha256=hashlib.sha256((HERE/'data.npz').read_bytes()).hexdigest())
    (HERE/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in protocol.items() if k != 'source_files'}, indent=2))


if __name__ == '__main__':
    main()
