"""Independent arithmetic, permutation, fit-replay and reload checks for E1."""
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as base
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def independent_interval(values):
    rng = np.random.default_rng(20261031)
    draws = np.concatenate([np.mean(values[rng.choice(len(values), size=(250, len(values)), replace=True)], axis=1)
                            for _ in range(40)])
    return [float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def build(n, seed):
    return Pipeline([('missing', base.SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                     ('scale', base.StandardScaler()),
                     ('forest', base.RandomForestRegressor(n_estimators=128, min_samples_leaf=10 if n >= 500 else 2,
                         max_features=1.0, random_state=seed, n_jobs=2))])


def check(name):
    p = read(HERE / 'protocol.json')
    order = p['primary_datasets'] + p['supporting_datasets']
    source = read(HERE / 'source-freeze.json')
    assert base.sha(HERE / 'protocol.json') == source['protocol_sha256']
    d, meta = base.load(name)
    n = len(d['y'])
    families = list(meta['groups'])
    folder = base.OUT / 'e1-supplement' / name
    originals, controls = [], []
    total_permutations = 0
    max_replay_error = 0.0
    for repeat in range(3):
        for fold in range(5):
            train = np.flatnonzero(d['folds'][repeat] != fold)
            test = np.flatnonzero(d['folds'][repeat] == fold)
            assert set(d['ids'][train]).isdisjoint(d['ids'][test])
            marker = read(folder / f'r{repeat}f{fold}.complete.json')
            assert marker['protocol_sha256'] == source['protocol_sha256']
            for path, digest in marker['sha256'].items():
                assert base.sha(path) == digest, path
            parent_path = base.OUT / name / f'r{repeat}f{fold}.predictions.csv.gz'
            assert base.sha(parent_path) == source['sha256'][str(parent_path)]
            raw = pd.read_csv(parent_path, dtype={'id': str}, float_precision='round_trip')
            raw = raw[raw.variant.str.startswith('RF|family:')].copy()
            originals.append(raw)
            c = pd.read_csv(folder / f'r{repeat}f{fold}.controls.csv.gz', dtype={'id': str}, float_precision='round_trip')
            assert len(c) == len(test) * (len(families) - 1) * 5
            assert not c.duplicated(['family', 'shuffle_seed', 'sample']).any()
            assert set(c['repeat']) == {repeat} and set(c.fold) == {fold}
            mappings = read(folder / f'r{repeat}f{fold}.permutations.json')
            assert mappings['train'] == train.tolist() and mappings['test'] == test.tolist()
            assert len(mappings['maps']) == (len(families) - 1) * 5
            assert len({(m['family'], m['shuffle_seed']) for m in mappings['maps']}) == len(mappings['maps'])
            for item in mappings['maps']:
                family, seed = item['family'], item['shuffle_seed']
                assert family in families and family != 'speed' and seed in p['shuffle_control']['seeds']
                selected = c[(c.family == family) & (c.shuffle_seed == seed)].sort_values('sample')
                np.testing.assert_array_equal(selected['sample'], test)
                for partition, (people, donors) in enumerate([(train, item['train_donors']), (test, item['test_donors'])]):
                    assert sorted(donors) == people.tolist()
                    expected = np.random.default_rng(np.random.SeedSequence(
                        [seed, order.index(name), repeat, fold, families.index(family), partition])).permutation(people)
                    np.testing.assert_array_equal(donors, expected)
                total_permutations += 2
            controls.append(c)
            if repeat == 0 and fold == 0:
                first = next(f for f in families if f != 'speed')
                candidates = [['speed'], families, ['speed', first]]
                for selected_families in candidates:
                    label = 'RF|family:' + '+'.join(f for f in families if f in selected_families)
                    cols = sorted({j for f in selected_families for j in meta['groups'][f]})
                    model = build(n, 9000)
                    model.fit(d['x'][np.ix_(train, cols)], d['y'][train], forest__sample_weight=np.ones(len(train)))
                    prediction = model.predict(d['x'][np.ix_(test, cols)])
                    expected = raw[raw.variant == label].sort_values('sample').prediction.to_numpy()
                    np.testing.assert_allclose(prediction, expected, atol=1e-9, rtol=0)
                    max_replay_error = max(max_replay_error, float(abs(prediction-expected).max()))
                cols = sorted(meta['groups']['speed'] + meta['groups'][first])
                z = d['x'][:, cols].copy()
                block = [cols.index(j) for j in meta['groups'][first]]
                item = next(m for m in mappings['maps'] if m['family'] == first and m['shuffle_seed'] == 51001)
                for people, donors in [(train, item['train_donors']), (test, item['test_donors'])]:
                    z[np.ix_(people, block)] = d['x'][np.ix_(donors, meta['groups'][first])]
                speed_local = [cols.index(j) for j in meta['groups']['speed']]
                np.testing.assert_allclose(z[:, speed_local], d['x'][:, meta['groups']['speed']], equal_nan=True, rtol=0, atol=0)
                model = build(n, 9000)
                model.fit(z[train], d['y'][train], forest__sample_weight=np.ones(len(train)))
                expected = c[(c.family == first) & (c.shuffle_seed == 51001)].sort_values('sample').prediction.to_numpy()
                prediction = model.predict(z[test])
                np.testing.assert_allclose(prediction, expected, atol=1e-9, rtol=0)
                with (folder / 'representative-model.pkl').open('rb') as stream:
                    restored = pickle.load(stream)
                np.testing.assert_allclose(restored.predict(z[test]), expected, atol=1e-9, rtol=0)
                max_replay_error = max(max_replay_error, float(abs(prediction-expected).max()))
    original = pd.concat(originals, ignore_index=True)
    control = pd.concat(controls, ignore_index=True)
    for frame in [original, control]:
        idx = frame['sample'].to_numpy(int)
        np.testing.assert_array_equal(frame.id.astype(str), d['ids'][idx])
        np.testing.assert_array_equal(frame.target, d['y'][idx])
        assert np.isfinite(frame.prediction).all()
        np.testing.assert_array_equal(frame.fold, d['folds'][frame['repeat'].to_numpy(int), idx])
        frame['ae'] = np.abs(frame.target.to_numpy() - frame.prediction.to_numpy())
    err = {}
    for label, group in original.groupby('variant'):
        assert len(group) == 3*n and not group.duplicated(['repeat', 'sample']).any()
        err[label] = group.pivot(index='repeat', columns='sample', values='ae').sort_index().sort_index(axis=1).to_numpy()
    ctrl = {}
    for key, group in control.groupby(['family', 'shuffle_seed']):
        assert len(group) == 3*n and not group.duplicated(['repeat', 'sample']).any()
        ctrl[key] = group.pivot(index='repeat', columns='sample', values='ae').sort_index().sort_index(axis=1).to_numpy()
    table = pd.read_csv(HERE / f'{name}-groups.csv')
    assert set(table.family) == set(families) and len(table) == len(families)
    speed = err['RF|family:speed']
    full = err['RF|family:' + '+'.join(families)]
    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    checked_intervals = 0
    for family in families:
        row = table[table.family == family].iloc[0]
        close([row.speed_MAE, row.full_MAE], [speed.mean(), full.mean()])
        without = err['RF|family:' + '+'.join(f for f in families if f != family)]
        remove = without - full
        interval = independent_interval(remove.mean(0))
        close([row.removal_delta, row.removal_ci_low, row.removal_ci_high], interval)
        close(row.without_MAE, without.mean())
        assert row.removal_positive_repeats == sum(remove.mean(1) > 0)
        assert bool(row.removal_support) == (interval[1] > 0 and sum(remove.mean(1) > 0) >= 2)
        checked_intervals += 1
        if family == 'speed':
            continue
        real = err['RF|family:' + '+'.join(f for f in families if f in ['speed', family])]
        addition = real - speed
        interval = independent_interval(addition.mean(0))
        close([row.addition_delta, row.addition_ci_low, row.addition_ci_high], interval)
        close(row.addition_MAE, real.mean())
        shuffled = np.stack([ctrl[family, seed] for seed in p['shuffle_control']['seeds']])
        control_interval = independent_interval((real-shuffled.mean(0)).mean(0))
        close([row.real_minus_shuffle, row.shuffle_ci_low, row.shuffle_ci_high], control_interval)
        close(row.shuffle_MAE, shuffled.mean())
        close(json.loads(row.shuffle_seed_MAEs), shuffled.mean(axis=(1, 2)))
        better_repeats = int(sum(addition.mean(1) < 0))
        better_seeds = int(sum(real.mean() < shuffled.mean(axis=(1, 2))))
        assert row.addition_negative_repeats == better_repeats and row.real_better_shuffle_seeds == better_seeds
        for repeat in range(3):
            close(row[f'addition_repeat{repeat}_delta'], addition[repeat].mean())
        expected = interval[0] < 0 and interval[2] < 0 and better_repeats >= 2 and control_interval[2] < 0 and better_seeds >= 4
        assert bool(row.additional_information) == expected
        checked_intervals += 2
    summary = read(HERE / f'{name}-summary.json')
    close([summary['full_minus_speed'][k] for k in ['delta', 'ci_low', 'ci_high']], independent_interval((full-speed).mean(0)))
    assert summary['additional_families'] == table.loc[table.additional_information == True, 'family'].tolist()
    assert summary['group_table_sha256'] == base.sha(HERE / f'{name}-groups.csv')
    result = {'passed': True, 'dataset': name, 'people': n, 'outer_cells': 15,
        'all_control_rows_checked': len(control), 'partition_permutations_checked': total_permutations,
        'independent_10000_draw_intervals_checked': checked_intervals + 1,
        'original_RF_fit_replays': 3, 'shuffled_RF_fit_replays': 1, 'checkpoint_reload': True,
        'max_replay_prediction_difference': max_replay_error,
        'all_group_MAEs_repeat_directions_and_decisions_recomputed': True,
        'protocol_sha256': base.sha(HERE / 'protocol.json'), 'verifier_sha256': base.sha(Path(__file__)),
        'limits': 'Representative models replayed, not all1800 controls; conditional exploratory inference, no independent clinical or video measurement validation.'}
    base.save(HERE / f'{name}-verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    check(sys.argv[1])
