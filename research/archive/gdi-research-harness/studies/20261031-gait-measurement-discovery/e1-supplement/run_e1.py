"""Frozen-condition E1 follow-up; parent experiment is read-only."""
import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as parent
import numpy as np
import pandas as pd

OUT = parent.OUT / 'e1-supplement'
PROTOCOL = HERE / 'protocol.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def protocol():
    return read(PROTOCOL)


def names():
    p = protocol()
    return p['primary_datasets'] + p['supporting_datasets']


def write(path, value):
    parent.save(path, value)


def freeze():
    assert not (HERE / 'source-freeze.json').exists(), 'Already frozen; never overwrite.'
    paths = [PROTOCOL, HERE / 'run_e1.py', HERE.parent / 'common.py',
             HERE.parent / 'comparison-protocol.json', HERE.parent / 'comparison-preparation.json']
    prep = read(HERE.parent / 'comparison-preparation.json')
    assert parent.sha(HERE.parent / 'comparison-protocol.json') == prep['protocol_sha256']
    for path, digest in prep['source_sha256'].items():
        assert parent.sha(path) == digest, path
    paths.extend(Path(p) for p in prep['source_sha256'])
    for name in names():
        d, meta = parent.load(name)
        assert len(d['y']) == protocol()['fixed_people'][name] == len(set(d['ids']))
        assert parent.sha(parent.OUT / name / 'input.npz') == meta['input_sha256']
        paths.extend([parent.OUT / name / 'input.npz', parent.OUT / name / 'metadata.json'])
        for repeat in range(3):
            for fold in range(5):
                prefix = parent.OUT / name / f'r{repeat}f{fold}'
                marker = prefix.with_suffix('.complete.json')
                for path, digest in read(marker)['outputs'].items():
                    assert parent.sha(path) == digest, path
                    paths.append(Path(path))
                paths.append(marker)
    fingerprints = {str(p): parent.sha(p) for p in dict.fromkeys(paths)}
    write(HERE / 'source-freeze.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
          'sha256': fingerprints, 'protocol_sha256': parent.sha(PROTOCOL),
          'datasets': names(), 'all_parent_sources_checked': True})
    print('FROZEN', len(fingerprints), 'files', parent.sha(PROTOCOL), flush=True)


def check_frozen(all_sources=False):
    frozen = read(HERE / 'source-freeze.json')
    assert parent.sha(PROTOCOL) == frozen['protocol_sha256']
    for path, digest in frozen['sha256'].items():
        if all_sources or Path(path) in [PROTOCOL, HERE / 'run_e1.py', HERE.parent / 'common.py']:
            assert parent.sha(path) == digest, path
    return frozen


def cell_predictions(name, repeat, fold):
    path = parent.OUT / name / f'r{repeat}f{fold}.predictions.csv.gz'
    frame = pd.read_csv(path, dtype={'id': str}, float_precision='round_trip')
    return frame


def variant(meta, families):
    return 'RF|family:' + '+'.join(k for k in meta['groups'] if k in families)


def columns(meta, families):
    return sorted({i for k in families for i in meta['groups'][k]})


def shuffled_input(x, meta, family, train, test, seed, dataset_index, repeat, fold):
    cols = columns(meta, ['speed', family])
    z = x[:, cols].copy()
    block = [cols.index(i) for i in meta['groups'][family]]
    mappings = []
    for partition, people in enumerate([train, test]):
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, dataset_index, repeat, fold, list(meta['groups']).index(family), partition]))
        donor = rng.permutation(people)
        z[np.ix_(people, block)] = x[np.ix_(donor, meta['groups'][family])]
        mappings.append(donor.tolist())
    return z, mappings


def run(name):
    started = time.time()
    check_frozen()
    d, meta = parent.load(name)
    assert parent.sha(parent.OUT / name / 'input.npz') == meta['input_sha256']
    index = names().index(name)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    p = protocol()
    for repeat in range(3):
        for fold in range(5):
            prefix = folder / f'r{repeat}f{fold}'
            marker = prefix.with_suffix('.complete.json')
            if marker.exists():
                result = read(marker)
                assert result['protocol_sha256'] == parent.sha(PROTOCOL)
                for path, digest in result['sha256'].items():
                    assert parent.sha(path) == digest, path
                continue
            train = np.flatnonzero(d['folds'][repeat] != fold)
            test = np.flatnonzero(d['folds'][repeat] == fold)
            ledger = read(parent.OUT / name / f'r{repeat}f{fold}.ledger.json')
            assert train.tolist() == ledger['train'] and test.tolist() == ledger['test']
            rf_seed = 9000 + 100 * repeat + fold
            assert rf_seed == ledger['seed']
            results, maps = [], []
            for family in meta['groups']:
                if family == 'speed':
                    continue
                for seed in p['shuffle_control']['seeds']:
                    z, donors = shuffled_input(d['x'], meta, family, train, test,
                                              seed, index, repeat, fold)
                    model = parent.estimator('RF', len(d['y']), rf_seed)
                    model.fit(z[train], d['y'][train], randomforestregressor__sample_weight=np.ones(len(train)))
                    pred = model.predict(z[test])
                    assert np.isfinite(pred).all()
                    results.append(pd.DataFrame({'repeat': repeat, 'fold': fold, 'family': family,
                        'shuffle_seed': seed, 'sample': test, 'id': d['ids'][test],
                        'target': d['y'][test], 'prediction': pred}))
                    maps.append({'family': family, 'shuffle_seed': seed,
                                 'train_donors': donors[0], 'test_donors': donors[1]})
                    # One fit per dataset is retained for checkpoint reload verification.
                    if repeat == 0 and fold == 0 and family == next(k for k in meta['groups'] if k != 'speed') and seed == p['shuffle_control']['seeds'][0]:
                        with (folder / 'representative-model.pkl').open('wb') as stream:
                            pickle.dump(model, stream)
            prediction_path = prefix.with_suffix('.controls.csv.gz')
            mapping_path = prefix.with_suffix('.permutations.json')
            pd.concat(results, ignore_index=True).to_csv(prediction_path, index=False)
            write(mapping_path, {'train': train.tolist(), 'test': test.tolist(), 'maps': maps})
            files = [prediction_path, mapping_path]
            if repeat == 0 and fold == 0:
                files.append(folder / 'representative-model.pkl')
            write(marker, {'protocol_sha256': parent.sha(PROTOCOL),
                  'sha256': {str(path): parent.sha(path) for path in files}})
            print(name, f'r{repeat}f{fold}', f'fits={len(maps)}',
                  f'seconds={time.time()-started:.1f}', flush=True)
    write(folder / 'execution.json', {'status': 'complete', 'people': len(d['y']),
          'outer_cells': 15, 'control_fits': 15 * (len(meta['groups']) - 1) * len(p['shuffle_control']['seeds']),
          'protocol_sha256': parent.sha(PROTOCOL), 'seconds': time.time()-started})


def paired(delta):
    delta = np.asarray(delta, float)
    rng = np.random.default_rng(20261031)
    estimates = np.empty(10000)
    for start in range(0, 10000, 100):
        estimates[start:start+100] = delta[rng.integers(0, len(delta), (100, len(delta)))].mean(axis=1)
    return float(delta.mean()), *np.quantile(estimates, [.025, .975]).tolist()


def analyze(name):
    check_frozen()
    assert read(OUT / name / 'execution.json')['status'] == 'complete'
    d, meta = parent.load(name)
    n = len(d['y'])
    families = list(meta['groups'])
    needed = {variant(meta, ['speed']), variant(meta, families)}
    needed.update(variant(meta, ['speed', k]) for k in families if k != 'speed')
    needed.update(variant(meta, [g for g in families if g != k]) for k in families)
    parent_frames, control_frames = [], []
    for repeat in range(3):
        for fold in range(5):
            frame = cell_predictions(name, repeat, fold)
            frame = frame[frame.variant.isin(needed)].copy()
            assert set(frame.variant) == needed
            parent_frames.append(frame)
            control_frames.append(pd.read_csv(OUT / name / f'r{repeat}f{fold}.controls.csv.gz',
                                               dtype={'id': str}, float_precision='round_trip'))
    original = pd.concat(parent_frames, ignore_index=True)
    control = pd.concat(control_frames, ignore_index=True)
    for frame in [original, control]:
        idx = frame['sample'].to_numpy(int)
        np.testing.assert_array_equal(frame.id.astype(str), d['ids'][idx])
        np.testing.assert_array_equal(frame.target, d['y'][idx])
        assert np.isfinite(frame.prediction).all()
        frame['error'] = abs(frame.prediction - frame.target)
    errors = {}
    for key, frame in original.groupby('variant'):
        assert len(frame) == n * 3 and not frame.duplicated(['repeat', 'sample']).any()
        errors[key] = frame.pivot(index='repeat', columns='sample', values='error').reindex(index=range(3), columns=range(n)).to_numpy()
        assert np.isfinite(errors[key]).all()
    controls = {}
    for (family, seed), frame in control.groupby(['family', 'shuffle_seed']):
        assert len(frame) == n * 3 and not frame.duplicated(['repeat', 'sample']).any()
        controls[family, seed] = frame.pivot(index='repeat', columns='sample', values='error').reindex(index=range(3), columns=range(n)).to_numpy()
        assert np.isfinite(controls[family, seed]).all()
    rows = []
    speed = errors[variant(meta, ['speed'])]
    full = errors[variant(meta, families)]
    for family in families:
        row = {'dataset': name, 'people': n, 'family': family, 'speed_MAE': speed.mean(), 'full_MAE': full.mean()}
        without = errors[variant(meta, [g for g in families if g != family])]
        delta = without - full
        row.update(zip(['removal_delta', 'removal_ci_low', 'removal_ci_high'], paired(delta.mean(axis=0))))
        row['without_MAE'] = without.mean()
        row['removal_positive_repeats'] = int((delta.mean(axis=1) > 0).sum())
        row['removal_support'] = bool(row['removal_ci_low'] > 0 and row['removal_positive_repeats'] >= 2)
        if family != 'speed':
            real = errors[variant(meta, ['speed', family])]
            delta = real - speed
            row['addition_MAE'] = real.mean()
            row.update(zip(['addition_delta', 'addition_ci_low', 'addition_ci_high'], paired(delta.mean(axis=0))))
            row['addition_negative_repeats'] = int((delta.mean(axis=1) < 0).sum())
            seeds = protocol()['shuffle_control']['seeds']
            shuffled = np.stack([controls[family, seed] for seed in seeds])
            row['shuffle_MAE'] = shuffled.mean()
            row['shuffle_seed_MAEs'] = json.dumps(shuffled.mean(axis=(1, 2)).tolist())
            row['real_better_shuffle_seeds'] = int((real.mean() < shuffled.mean(axis=(1, 2))).sum())
            row.update(zip(['real_minus_shuffle', 'shuffle_ci_low', 'shuffle_ci_high'], paired((real - shuffled.mean(axis=0)).mean(axis=0))))
            row['additional_information'] = bool(row['addition_delta'] < 0 and row['addition_ci_high'] < 0
                and row['addition_negative_repeats'] >= 2 and row['shuffle_ci_high'] < 0
                and row['real_better_shuffle_seeds'] >= 4)
            for repeat in range(3):
                row[f'addition_repeat{repeat}_delta'] = float(delta[repeat].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / f'{name}-groups.csv', index=False, encoding='utf-8-sig')
    combined = paired((full-speed).mean(axis=0))
    summary = {'dataset': name, 'people': n, 'speed_MAE': float(speed.mean()), 'full_MAE': float(full.mean()),
        'full_minus_speed': dict(zip(['delta', 'ci_low', 'ci_high'], combined)),
        'full_improving_repeats': int(((full-speed).mean(axis=1) < 0).sum()),
        'additional_families': [r['family'] for r in rows if r.get('additional_information', False)],
        'group_table_sha256': parent.sha(HERE / f'{name}-groups.csv'), 'protocol_sha256': parent.sha(PROTOCOL)}
    write(HERE / f'{name}-summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run', 'analyze', 'check'])
    parser.add_argument('dataset', nargs='?', choices=names())
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'run':
        run(args.dataset)
    elif args.action == 'analyze':
        analyze(args.dataset)
    else:
        check_frozen(all_sources=True)
        print('ALL_FROZEN_HASHES_UNCHANGED', flush=True)
