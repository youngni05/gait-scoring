"""Cached-feature CPU control. Source bundle is read-only; no encoder is loaded."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE = ROOT / 'control/manual-sessions/20261029-gpu-finetune-preparation/bundle'
WORKSPACE = ROOT.parent
for folder in ('outputs/motionbert-smoke-20261015/deps', 'outputs/gdi-ridge-experiment-20261016/deps'):
    sys.path.insert(0, str(WORKSPACE / folder))
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, obj):
    path = Path(path).resolve()
    path.relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def source_data():
    manifest = load_json(BUNDLE / 'manifest.json')
    fingerprints = {}
    for name in ('train_only.npz', 'baseline_heads.npz', 'baseline_predictions.npz', 'run.py', 'protocol.draft.json'):
        actual = digest(BUNDLE / name)
        if actual != manifest[name]:
            raise ValueError('source fingerprint changed: ' + name)
        fingerprints[name] = actual
    with np.load(BUNDLE / 'train_only.npz', allow_pickle=False) as z:
        d = {name: z[name].copy() for name in ('features', 'y', 'person_id', 'video_id', 'fold')}
    with np.load(BUNDLE / 'baseline_heads.npz', allow_pickle=False) as z:
        heads = {name: z[name].copy() for name in z.files}
    with np.load(BUNDLE / 'baseline_predictions.npz', allow_pickle=False) as z:
        baseline = z['predictions'].copy()
    if d['features'].shape != (203, 2, 512) or d['y'].shape != (203, 2) or baseline.shape != (203, 2, 2):
        raise ValueError('shape mismatch')
    if len(set(d['person_id'])) != 175 or len(set(d['video_id'])) != 203 or set(d['fold']) != set(range(5)):
        raise ValueError('identity/fold mismatch')
    if not all(np.isfinite(a).all() for a in (d['features'], d['y'], baseline)):
        raise ValueError('nonfinite source')
    for fold in range(5):
        tr = set(d['person_id'][d['fold'] != fold])
        te = set(d['person_id'][d['fold'] == fold])
        if tr & te or len(tr) != 140 or len(te) != 35:
            raise ValueError('person split mismatch')
    return d, heads, baseline, fingerprints


class Head(torch.nn.Module):
    def __init__(self, heads, fold):
        super().__init__()
        prefix = f'fold{fold}__'
        self.register_buffer('mu', torch.as_tensor(heads[prefix + 'scaler_mean']).double().clone())
        self.register_buffer('scale', torch.as_tensor(heads[prefix + 'scaler_scale']).double().clone())
        self.output = torch.nn.Linear(512, 2).double()
        with torch.no_grad():
            self.output.weight.copy_(torch.as_tensor(heads[prefix + 'coef']))
            self.output.bias.copy_(torch.as_tensor(heads[prefix + 'intercept']))

    def forward(self, x):
        return self.output((x.double() - self.mu) / self.scale)


def optimizer(model, config):
    return torch.optim.AdamW(model.output.parameters(), lr=config['head_lr'], weight_decay=config['weight_decay'])


def update(model, opt, x, y):
    opt.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss')
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.output.parameters()):
        raise ValueError('invalid gradient')
    opt.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite weights')
    return float(loss.detach())


def checkpoint(path, model, opt, done, identity):
    path = Path(path).resolve()
    path.relative_to(HERE)
    temp = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'done': done,
                'identity': identity, 'rng': torch.get_rng_state()}, temp)
    os.replace(temp, path)


def restore(path, model, opt, identity):
    c = torch.load(path, map_location='cpu', weights_only=True)
    if c['identity'] != identity:
        raise ValueError('checkpoint identity mismatch')
    model.load_state_dict(c['model'], strict=True)
    opt.load_state_dict(c['optimizer'])
    torch.set_rng_state(c['rng'])
    return int(c['done'])


def config_ready(path, synthetic_only=False):
    c = load_json(path)
    if c.get('status') != 'ready':
        raise ValueError('actual training requires an approved ready config')
    expected_kind = 'synthetic_only' if synthetic_only else 'cached_development_head_only'
    if c.get('data_kind') != expected_kind:
        raise ValueError('protocol data kind mismatch')
    if c.get('variants') != ['head_only'] or c.get('folds') != list(range(5)) or c.get('seeds') != [20261028]:
        raise ValueError('unexpected job scope')
    for key in ('steps', 'batch_size', 'checkpoint_every'):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError('invalid ' + key)
    if not isinstance(c.get('head_lr'), (int, float)) or not math.isfinite(c['head_lr']) or c['head_lr'] <= 0:
        raise ValueError('invalid learning rate')
    if not isinstance(c.get('weight_decay'), (int, float)) or not math.isfinite(c['weight_decay']) or c['weight_decay'] < 0:
        raise ValueError('invalid weight decay')
    expected = {'loss': 'record_equal_MSE_left_right', 'train_window': 'front',
                'evaluate_windows': ['front', 'back'], 'optimizer': 'AdamW', 'dropout': 'disabled_all_arms'}
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('comparison contract changed')
    return c


def metrics(d, pred):
    people = np.unique(d['person_id'])
    errors = np.abs(pred - d['y'][:, None, :]).mean(axis=2)
    gap = np.abs(pred[:, 0] - pred[:, 1]).mean(axis=1)
    per = np.array([[errors[d['person_id'] == person, 0].mean(), errors[d['person_id'] == person, 1].mean(),
                     gap[d['person_id'] == person].mean()] for person in people])
    distribution = [float(np.array([pred[d['person_id'] == person, w].mean() for person in people]).std(ddof=1)) for w in range(2)]
    return {'front_mae': float(per[:, 0].mean()), 'back_mae': float(per[:, 1].mean()),
            'front_back_gap': float(per[:, 2].mean()), 'person_prediction_sd_front_back': distribution}, per


def train_all(config_path, out, synthetic_only=False):
    config = config_ready(config_path, synthetic_only=synthetic_only)
    d, heads, baseline, fingerprints = source_data()
    out = Path(out).resolve()
    out.relative_to(HERE)
    out.mkdir(parents=True, exist_ok=True)
    predictions = np.full((203, 2, 2), np.nan)
    times = []
    for fold in config['folds']:
        job = out / f'head_only_seed20261028_fold{fold}'
        job.mkdir(exist_ok=True)
        identity = {'config': digest(config_path), 'sources': fingerprints, 'fold': fold, 'seed': 20261028,
                    'torch': str(torch.__version__), 'device': 'cpu', 'implementation': digest(__file__)}
        torch.manual_seed(20261028)
        torch.use_deterministic_algorithms(True)
        model = Head(heads, fold)
        opt = optimizer(model, config)
        tr = np.flatnonzero(d['fold'] != fold)
        te = np.flatnonzero(d['fold'] == fold)
        if config['batch_size'] > len(tr):
            raise ValueError('batch exceeds training records')
        if (job / 'complete.json').exists():
            done = load_json(job / 'complete.json')
            if done['identity'] != identity or digest(job / 'predictions.npz') != done['predictions_sha256']:
                raise ValueError('completed evidence mismatch')
            with np.load(job / 'predictions.npz', allow_pickle=False) as z:
                if not np.array_equal(z['indices'], te):
                    raise ValueError('saved evaluation coverage changed')
                predictions[te] = z['predictions']
            times.append(done['elapsed_seconds'])
            continue
        save_json(job / 'job.json', identity)
        ck = job / 'checkpoint.pt'
        start = restore(ck, model, opt, identity) if ck.exists() else 0
        if start > config['steps']:
            raise ValueError('checkpoint exceeds declared budget')
        batches = math.ceil(len(tr) / config['batch_size'])
        before_mu, before_scale = model.mu.clone(), model.scale.clone()
        t0 = time.perf_counter()
        loss = None
        for j in range(start, config['steps']):
            epoch, part = divmod(j, batches)
            order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(20261028 + epoch)).numpy()
            ix = tr[order[part * config['batch_size']:(part + 1) * config['batch_size']]]
            loss = update(model, opt, torch.from_numpy(d['features'][ix, 0]), torch.from_numpy(d['y'][ix]).double())
            if (j + 1) % config['checkpoint_every'] == 0 or j + 1 == config['steps']:
                checkpoint(ck, model, opt, j + 1, identity)
        if not torch.equal(before_mu, model.mu) or not torch.equal(before_scale, model.scale):
            raise ValueError('fixed scaler changed')
        with torch.inference_mode():
            p = np.stack([model(torch.from_numpy(d['features'][te, w])).numpy() for w in range(2)], axis=1)
        if not np.isfinite(p).all():
            raise ValueError('nonfinite predictions')
        predictions[te] = p
        np.savez_compressed(job / 'predictions.npz', indices=te, predictions=p)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        save_json(job / 'complete.json', {'identity': identity, 'steps': config['steps'], 'elapsed_seconds': elapsed,
                                         'last_training_loss': loss, 'predictions_sha256': digest(job / 'predictions.npz')})
    if not np.isfinite(predictions).all():
        raise ValueError('incomplete five-fold coverage')
    base_metrics, base_people = metrics(d, baseline)
    trained_metrics, trained_people = metrics(d, predictions)
    np.savez_compressed(out / 'all_predictions.npz', predictions=predictions, baseline=baseline,
                        person_id=d['person_id'], video_id=d['video_id'], fold=d['fold'])
    delta = base_people - trained_people
    rng = np.random.default_rng(20261029)
    draws = rng.integers(0, 175, size=(10000, 175))
    intervals = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5], axis=0).T.tolist()
    save_json(out / 'summary.json', {'people': 175, 'records': 203, 'optimizer_updates': config['steps'] * 5,
                                    'synthetic_only': synthetic_only,
                                    'actual_gdi_training_updates': 0 if synthetic_only else config['steps'] * 5,
                                    'baseline': base_metrics, 'head_only': trained_metrics,
                                    'person_paired_reduction_front_back_gap': delta.mean(axis=0).tolist(),
                                    'exploratory_person_bootstrap95_front_back_gap': intervals,
                                    'fold_seconds': times, 'gpu_used': False, 'encoder_loaded': False,
                                    'independent_final_evaluation': False, 'source_fingerprints': fingerprints})
    print(json.dumps({'finished': str(out), 'baseline': base_metrics, 'head_only': trained_metrics}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train_all(args.config, args.output)
