"""Shared immutable inputs, exact existing splits and person-weighted metrics."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CPU = ROOT/'gdi-research-harness/studies/20261030-cpu-head-study/followup'
GPU = ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-followup695/bundle'
MB = ROOT/'outputs/motionbert-smoke-20261015'
sys.dont_write_bytecode = True
sys.path[:0] = [str(MB/'deps'), str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps')]
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    path = HERE/path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def data():
    receipt = load(HERE/'sources.json')
    assert all(sha(ROOT/path) == value for path, value in receipt['sha256'].items())
    assert sha(HERE/'data.npz') == receipt['local_dataset_sha256']
    return arrays(HERE/'data.npz')


def split(d, outer, inner=None):
    te = np.flatnonzero(d['fold'] == outer)
    tr = np.flatnonzero(d['fold'] != outer)
    if inner is None:
        assert len(set(d['person_id'][tr])) == 556 and len(set(d['person_id'][te])) == 139
        assert not set(d['person_id'][tr]) & set(d['person_id'][te])
        return tr, te
    assignment = arrays(HERE/f'splits/outer{outer}.npz')['inner_assignment']
    it, iv = tr[assignment != inner], tr[assignment == inner]
    assert len(set(d['person_id'][it])) == 417 and len(set(d['person_id'][iv])) == 139
    assert not (set(d['person_id'][it]) & set(d['person_id'][iv]) or
                (set(d['person_id'][it]) | set(d['person_id'][iv])) & set(d['person_id'][te]))
    return it, iv


def errors(y, pred, people):
    assert y.shape == pred.shape == (len(people),2) and np.isfinite(pred).all()
    return np.asarray([np.abs(pred[people == p]-y[people == p]).mean() for p in np.unique(people)])


def metrics(y, pred, people):
    def macro(value):
        return float(np.mean([value[people == p].mean() for p in np.unique(people)]))
    distance = abs(np.floor(pred/10)-np.floor(y/10))
    return {'people': len(np.unique(people)), 'records': len(y), 'MAE': float(errors(y,pred,people).mean()),
            'bias': macro(pred-y), 'exact_10point_bin_rate': macro(distance == 0),
            'two_or_more_bins_rate': macro(distance >= 2), 'prediction_range': [float(pred.min()),float(pred.max())]}


def summary(d, methods):
    result = {'metrics': {}, 'fold_metrics': {}, 'groups': {}, 'comparisons_with_PLS': {}}
    rng = np.random.default_rng(20261030)
    samples = rng.integers(0,695,size=(10000,695))
    baseline = errors(d['y'],d['pls'],d['person_id'])
    for name, pred in methods.items():
        result['metrics'][name] = metrics(d['y'],pred,d['person_id'])
        result['fold_metrics'][name] = [metrics(d['y'][d['fold']==f],pred[d['fold']==f],d['person_id'][d['fold']==f]) for f in range(5)]
        result['groups'][name] = {str(c): metrics(d['y'][d['cohort']==c],pred[d['cohort']==c],d['person_id'][d['cohort']==c]) for c in [0,1]}
        diff = baseline-errors(d['y'],pred,d['person_id'])
        result['comparisons_with_PLS'][name] = {'MAE_reduction': float(diff.mean()),
            'conditional_bootstrap95': np.quantile(diff[samples].mean(1),[.025,.975]).tolist(),
            'people_improved_equal_worse': [int((diff > 1e-12).sum()),int((abs(diff)<=1e-12).sum()),int((diff < -1e-12).sum())]}
    return result
