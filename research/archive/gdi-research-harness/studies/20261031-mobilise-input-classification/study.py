"""Fixed-band input ablation; person-disjoint nested validation, CPU execution."""
from pathlib import Path
import sys, os, json, time, itertools, argparse, importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / '20261031-mobilise-ordinal-bands'
FACTORIAL = HERE.parent / '20261030-mobilise-factorial'
STORE = Path('D:/gait-research-data/D_P/mobilise-input-classification-20261031')
sys.path[:0] = [str(PREVIOUS), str(ROOT / 'outputs/motionbert-smoke-20261015/deps')]
spec = importlib.util.spec_from_file_location('ordinal_reference', PREVIOUS / 'study.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
import torch
import joblib

sha, save, weights = old.sha, old.save, old.weights
SEEDS, KEY, FEATURES = old.SEEDS, old.KEY, old.FEATURES
METHODS = ['SpeedRF', 'MedianRF', 'BoutRF', 'DeepSets']
EPOCHS = list(range(10, 601, 10))

def initialize():
    global BAGS, LIMITS
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LIMITS = threadpool_limits(limits=1)
    old.init()
    selected = pd.read_csv(old.BASE / 'selected-bouts.csv')
    selected = selected[selected.draw.eq(0)].copy()
    audit = pd.read_csv(FACTORIAL / 'audited-bouts.csv')
    assert audit.turn_valid.eq(True).all()
    selected = selected.merge(audit[KEY + ['wb_id', 'turn_rate_per_second']],
                              on=KEY + ['wb_id'], validate='many_to_one')
    bkey = KEY + ['band']
    selected = selected.sort_values(bkey + ['rank'])
    assert selected.groupby(bkey).size().eq(5).all()
    assert not selected.duplicated(bkey + ['rank']).any()
    BAGS = {tuple(key): g[FEATURES + ['duration', 'turn_rate_per_second']].to_numpy()
            for key, g in selected.groupby(bkey, sort=False)}
    assert all(np.isfinite(v).all() for v in BAGS.values())

def bag_array(frame, k):
    x = np.stack([BAGS[(p, v, b)][:k] for p, v, b in frame[KEY + ['band']].itertuples(index=False, name=None)])
    np.testing.assert_allclose(np.median(x[:, :, :4], axis=1), frame[FEATURES], rtol=1e-12, atol=1e-12)
    return x

def inputs(frame, k):
    x = bag_array(frame, k)
    # Lossless set representation for RF: speed first, all remaining values break ties.
    ordered = np.array([sorted(row.tolist(), key=lambda t: (t[2], *t)) for row in x])
    return {'SpeedRF': frame[['averagestridespeed']].to_numpy(),
            'MedianRF': frame[FEATURES].to_numpy(),
            'BoutRF': ordered.reshape(len(x), -1), 'DeepSets': x}

class DeepSet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(6, 32)
        self.hidden = torch.nn.Linear(32, 16)
        self.head = torch.nn.Linear(16, 4)

    def forward(self, x):
        h = torch.relu(self.encoder(x)).mean(1)
        return self.head(torch.relu(self.hidden(h)))

def predict_ds(bundle, x):
    """Independent float64 NumPy forward pass of saved PyTorch parameters."""
    z = (x - bundle['mean']) / bundle['std']
    h = np.maximum(z @ bundle['encoder.weight'].T + bundle['encoder.bias'], 0).mean(1)
    h = np.maximum(h @ bundle['hidden.weight'].T + bundle['hidden.bias'], 0)
    return old.softmax(h @ bundle['head.weight'].T + bundle['head.bias'], axis=1)

def ds_fit(x, y, ids, seed, epochs, valid=None):
    w = weights(ids); w = w / w.sum()
    flat = x.reshape(-1, 6); fw = np.repeat(w / x.shape[1], x.shape[1])
    mean = np.average(flat, axis=0, weights=fw)
    std = np.maximum(np.sqrt(np.average((flat - mean)**2, axis=0, weights=fw)), 1e-8)
    xt = torch.tensor((x - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long); wt = torch.tensor(w, dtype=torch.float32)
    torch.manual_seed(seed); model = DeepSet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
    curves, snapshots, probabilities = [], [], []
    if valid is not None:
        vx, vy, vi = valid
        vt = torch.tensor((vx - mean) / std, dtype=torch.float32)
        vw = weights(vi)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = (torch.nn.functional.cross_entropy(model(xt), yt, reduction='none') * wt).sum()
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
        if valid is not None and epoch % 10 == 0:
            with torch.no_grad():
                prob = model(vt).softmax(1).numpy().astype(float)
                prob = prob / prob.sum(1, keepdims=True)
            curves.append(float(np.average(-np.log(np.clip(prob[np.arange(len(vy)), vy], 1e-12, 1)), weights=vw)))
            snapshots.append({'mean': mean.copy(), 'std': std.copy(),
                              **{n: p.detach().numpy().copy() for n, p in model.state_dict().items()}})
            probabilities.append(prob)
    bundle = {'mean': mean, 'std': std, **{n: p.detach().numpy().copy() for n, p in model.state_dict().items()}}
    return bundle, curves, snapshots, probabilities

def save_model(path, model):
    if path.suffix == '.npz': np.savez_compressed(path, **model)
    else: joblib.dump(model, path, compress=3)
    return {'path': str(path), 'sha256': sha(path)}

def work(task):
    target, band, k, fold, seed = task
    tag = f'{target}-{band}-{k}-f{fold}-s{seed}'
    done = STORE / 'folds' / f'{tag}.json'
    if done.exists(): return tag, 'cached'
    started = time.time()
    tr, te, inner, ref = old.cell_data(target, band, k, fold)
    x, xt = inputs(tr, k), inputs(te, k)
    col = 'total_sppb_score' if target == 'sppb' else 'tugtime'
    y = old.classes(tr[col].to_numpy(), target); ids = tr.participantid.to_numpy()
    out = old.outframe(te, target, band, k, fold, seed)
    out['speed'] = te.averagestridespeed
    arrays = {'ids': ids, 'visits': tr.visit_number.to_numpy(dtype='U'), 'y': y, 'inner': inner}
    prior = np.bincount(y, weights=weights(ids), minlength=4); prior /= prior.sum()
    for j in range(4): out[f'Prior_p{j}'] = prior[j]
    ledger = {'task': list(task), 'split_sha256': sha(old.SETS / 'folds' / f'{target}-{band}-{k}-f{fold}.json'), 'models': {}}
    for name in METHODS:
        oof = np.zeros((len(tr), 4)); model_info = {}
        if name == 'MedianRF':
            source = old.STORE / 'folds' / f'{tag}.json'
            previous = json.loads(source.read_text())
            ins = np.load(old.STORE / 'folds' / f'{tag}-inner.npz')
            np.testing.assert_array_equal(ins['ids'], ids); np.testing.assert_array_equal(ins['inner'], inner)
            np.testing.assert_array_equal(ins['y'], y)
            oof = ins['RF_oof'].copy()
            model_path = Path(previous['models']['RF']['path'])
            assert sha(model_path) == previous['models']['RF']['sha256']
            model = joblib.load(model_path); raw = old.probabilities(model, xt[name], 'RF')
            model_info = {'path': str(model_path), 'sha256': sha(model_path), 'reused_reference': str(source), 'reference_sha256': sha(source)}
        elif name in ['SpeedRF', 'BoutRF']:
            saved_inner = []
            for i in range(3):
                a = inner != i; b = ~a
                m = old.fit(old.make('RF', seed + fold*10 + i), x[name][a], y[a], ids[a], 'RF')
                oof[b] = old.probabilities(m, x[name][b], 'RF')
                saved_inner.append(save_model(STORE / 'models' / f'{tag}-{name}-inner{i}.joblib', m))
            m = old.fit(old.make('RF', seed + fold*10 + 9), x[name], y, ids, 'RF')
            raw = old.probabilities(m, xt[name], 'RF')
            model_info = save_model(STORE / 'models' / f'{tag}-{name}.joblib', m)
            model_info['inner_models'] = saved_inner
        else:
            curves, snapshots, ps, sizes = [], [], [], []
            for i in range(3):
                a = inner != i; b = ~a
                _, c, bundles, prob = ds_fit(x[name][a], y[a], ids[a], seed + fold*10 + i, 600,
                                              (x[name][b], y[b], ids[b]))
                curves.append(c); snapshots.append(bundles); ps.append(prob); sizes.append(len(np.unique(ids[b])))
            scores = np.average(curves, axis=0, weights=sizes)
            best = int(np.argmin(scores)); epochs = EPOCHS[best]; saved_inner = []
            for i in range(3):
                oof[inner == i] = ps[i][best]
                saved_inner.append(save_model(STORE / 'models' / f'{tag}-{name}-inner{i}.npz', snapshots[i][best]))
            bundle, _, _, _ = ds_fit(x[name], y, ids, seed + fold*10 + 9, epochs)
            raw = predict_ds(bundle, xt[name])
            model_info = save_model(STORE / 'models' / f'{tag}-{name}.npz', bundle)
            model_info.update(inner_models=saved_inner, epochs=epochs, inner_curves=curves,
                              inner_people=sizes, combined_curve=scores.tolist(), width=32)
        info = old.finish_prob(name, oof, raw, y, ids, inner, out, target, arrays)
        info.update(model_info); ledger['models'][name] = info
    np.savez_compressed(STORE / 'folds' / f'{tag}-inner.npz', **arrays)
    out.to_csv(STORE / 'folds' / f'{tag}.csv.gz', index=False)
    ledger.update(seconds=time.time()-started, predictions_sha256=sha(STORE / 'folds' / f'{tag}.csv.gz'),
                  inner_sha256=sha(STORE / 'folds' / f'{tag}-inner.npz'), code_sha256=sha(__file__))
    save(done, ledger)
    return tag, round(ledger['seconds'], 1)

def prepare():
    files = [old.BASE / 'features.csv', old.BASE / 'selected-bouts.csv', old.BASE / 'person-folds.csv',
             FACTORIAL / 'audited-bouts.csv', FACTORIAL / 'quality.json', PREVIOUS / 'study.py', PREVIOUS / 'verification.json']
    p = {'code_sha256': sha(__file__), 'source_hashes': {str(f): sha(f) for f in files},
         'seeds': SEEDS, 'cuts': old.CUTS, 'primary': 'long5 draw0; all six duration/count conditions reported',
         'people': {'sppb': 2174, 'tug': 1939}, 'folds': 'exact preceding outer5 / inner3 person splits; repeated visits stay together',
         'methods': {'SpeedRF': 'median stride speed only, 128 trees, leaf20, max_features1',
                     'MedianRF': 'reuse exact preceding four-class RF, median four features',
                     'BoutRF': 'same RF; all k x6 values sorted lexicographically by speed then all features and flattened; no bout removed',
                     'DeepSets': 'six features per bout -> Linear6,32 ReLU mean pooling -> Linear32,16 ReLU -> Linear16,4'},
         'extended_features': FEATURES + ['duration', 'turn_rate_per_second'],
         'training': 'person-equal weights; no metadata/class weighting; CPU fullbatch AdamW lr.01 wd.001 CE; DS epoch10..600 step10 selected by inner person log loss, earlier tie; fixed width32',
         'calibration': 'same reference temperature [.25,4] on inner OOF; cross-fitted temperature for internal fpr10/sens80 thresholds; raw also retained',
         'selection_limit': 'DS epoch chosen over inner validation predictions reused for calibration, all inside outer training; inner metrics are selection metrics, not generalization estimates',
         'speed_analysis': 'outer-test only, each low-function visit matched to nearest-speed nonlow visit from different person in SAME outer fold, max gap .05m/s; ties participantid/visit; with-replacement controls; repeated visits person weighted; repeat with same cohort; descriptive paired risk-order accuracy and coverage; no model/threshold selection on these pairs',
         'metrics': 'Brier, RPS, logloss, accuracy, perclass recall, extreme error, low miss/fpr, calibrated fpr10 miss/fpr; paired person bootstrap long5 differences; five seeds separate, not ensemble',
         'storage': {'small': str(HERE), 'large': str(STORE)},
         'limits': 'development dataset reused; no video/external clinical validity; matched comparisons descriptive, controls reusable; six summaries not raw step dynamics'}
    if (HERE / 'protocol.json').exists(): assert json.loads((HERE / 'protocol.json').read_text()) == p
    save(HERE / 'protocol.json', p); print('PROTOCOL_READY', flush=True)

def run(mode, workers):
    p = json.loads((HERE / 'protocol.json').read_text()); assert p['code_sha256'] == sha(__file__)
    for path, h in p['source_hashes'].items(): assert sha(path) == h
    assert HERE.resolve().drive.upper() == 'C:' and STORE.resolve().drive.upper() == 'D:'
    for n in ['models', 'folds']: (STORE / n).mkdir(parents=True, exist_ok=True)
    for n in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[n] = '1'
    tasks = list(itertools.product(['sppb', 'tug'], ['short', 'long'], [1,3,5], range(5), SEEDS))
    if mode == 'pilot': tasks = [(t, 'long', 5, 0, SEEDS[0]) for t in ['sppb', 'tug']]
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        fs = [pool.submit(work, t) for t in tasks]
        for n, f in enumerate(as_completed(fs), 1):
            tag, secs = f.result()
            print(n, '/', len(tasks), tag, secs, 'elapsed', round(time.time()-start), flush=True)
    save(HERE / f'{mode}-completion.json', {'tasks': len(tasks), 'seconds': time.time()-start, 'code_sha256': sha(__file__)})

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prepare','pilot','main'])
    ap.add_argument('--workers', type=int, default=4); args = ap.parse_args()
    if args.mode == 'prepare': prepare()
    else: run(args.mode, args.workers)
