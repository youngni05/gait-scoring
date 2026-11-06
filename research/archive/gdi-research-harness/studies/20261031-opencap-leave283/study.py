"""Frozen-person deletion sensitivity and feature-block ablation."""
from pathlib import Path
import sys, json, hashlib, importlib.util
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / '20261031-opencap-extrapolation'
spec = importlib.util.spec_from_file_location('extrapolation_study', PRIOR/'study.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
np, pd, ridge_fit = previous.np, previous.pd, previous.fit
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
SEEDS = [20261030, 20261031, 20261032]
REMOVED = 'sub-283'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, value):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')

def fit(x, y, tr, te, family, setting, seed):
    if family == 'Ridge':
        pred, raw, pair = ridge_fit(x, y, tr, te, 'Ridge', setting)
        return pred, pair
    scale = StandardScaler().fit(x[tr], sample_weight=np.ones(len(tr)))
    model = RandomForestRegressor(n_estimators=128, min_samples_leaf=setting,
        random_state=seed, n_jobs=1, max_features=1.)
    model.fit(scale.transform(x[tr]), y[tr], sample_weight=np.ones(len(tr)))
    return model.predict(scale.transform(x[te])), (scale, model)

def prepare():
    assert not (HERE / 'protocol.json').exists()
    sources = {'prep': HERE.parent/'20261031-opencap-chair/prep_event',
               'geometry': HERE.parent/'20261031-opencap-turn/geometry'}
    configs = {}
    hashes = {}
    for cohort, src in sources.items():
        d = dict(np.load(src/'input.npz'))
        full = d['PrepAll' if cohort == 'prep' else 'Walk3Turn']
        views = {'Full': full.copy()}
        mapping = {'Full': {'source_view': 'PrepAll' if cohort == 'prep' else 'Walk3Turn', 'columns': list(range(full.shape[1]))}}
        def drop(name, columns):
            kept = [i for i in range(full.shape[1]) if i not in columns]
            views[name] = full[:, kept]
            mapping[name] = {'source_view': mapping['Full']['source_view'], 'columns': kept}
        drop('NoWalk', range(3))
        drop('NoTurnTime', [3])
        if cohort == 'prep':
            views['Context'] = d['TurnContext']
            mapping['Context'] = {'source_view': 'TurnContext', 'columns': list(range(36))}
            drop('NoTurn', range(3, 13))
            drop('NoRise', range(13, 21))
            drop('NoSit', range(21, 29))
            drop('NoPhaseTime', [13, 21])
            drop('NoRiseMotion', range(14, 21))
            drop('NoSitMotion', range(22, 29))
            drop('ChairTimeOnly', list(range(14, 21)) + list(range(22, 29)))
            core_src = HERE.parent/'20261031-opencap-chair/event_strict'
            core = dict(np.load(core_src/'input.npz'))
            ix = [list(core['ids']).index(pid) for pid in d['ids']]
            assert np.array_equal(core['y'][ix], d['y'])
            assert np.array_equal(core['ChairAll'][ix, :13], full[:, :13])
            assert np.array_equal(core['ChairAll'][ix, 21:], full[:, 21:])
            views['CoreRise'] = core['ChairAll'][ix].copy()
            mapping['CoreRise'] = {'source_view': 'event_strict/ChairAll mapped by participant', 'replaced_columns': list(range(13, 21))}
            for name in ['input.npz', 'manifest.json']: hashes[str((core_src/name).relative_to(HERE.parent))] = sha(core_src/name)
            rf_views = ['Full', 'Context']
        else:
            views['MatchedWalk'] = d['WalkMatchedSpeed']
            mapping['MatchedWalk'] = {'source_view': 'WalkMatchedSpeed', 'columns': list(range(3))}
            drop('TurnTimeOnly', range(4, 13))
            rf_views = ['Full', 'MatchedWalk']
        keep = np.flatnonzero(d['ids'] != REMOVED)
        assert sum(d['ids'] == REMOVED) == 1
        old_new = {int(k): i for i, k in enumerate(keep)}
        remap = lambda a: [old_new[i] for i in a if i in old_new]
        plans = []
        for rep in json.loads((src/'splits.json').read_text()):
            folds = []
            for f in rep['splits']:
                folds.append({'fold': f['fold'], 'train': remap(f['train']), 'test': remap(f['test']),
                    'inner': [{'train': remap(v['train']), 'validation': remap(v['validation'])} for v in f['inner']]})
            plans.append({**rep, 'splits': folds})
        out = HERE/cohort
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out/'input.npz', ids=d['ids'][keep], y=d['y'][keep], **{k:v[keep] for k,v in views.items()})
        manifests = json.loads((src/'manifest.json').read_text())
        save(cohort+'/manifest.json', [manifests[i] for i in keep])
        save(cohort+'/splits.json', plans)
        for name in ['input.npz', 'manifest.json', 'splits.json', 'predictions.csv']:
            hashes[str((src/name).relative_to(HERE.parent))] = sha(src/name)
        for name in ['predictions.csv', 'person-results.csv']:
            path=PRIOR/('prep_event' if cohort=='prep' else 'geometry')/name
            hashes[str(path.relative_to(HERE.parent))] = sha(path)
        configs[cohort] = {'source': str(src.relative_to(HERE.parent)), 'people': len(keep), 'long_people': int(sum(d['y'][keep]>=20)),
            'ridge_views': list(views), 'rf_views': rf_views, 'mapping': mapping,
            'hashes': {n:sha(out/n) for n in ['input.npz', 'manifest.json', 'splits.json']}}
    protocol = {'removed_id': REMOVED, 'purpose': 'evaluation-only deletion then complete train/inner/test deletion; remaining person memberships frozen',
        'cohorts': configs, 'source_hashes': hashes, 'Ridge_alpha': [.1, 1., 10., 100.], 'RF_leaf': [2,5], 'RF_seeds': SEEDS,
        'selection': 'same inner MAE, train-only standardization; each ablation refits and retunes using only training people',
        'observation': 'original windows unchanged; context controls same duration; column ablations acquire same clips but withhold feature groups, not claims of equal informative frames',
        'CoreRise': 'same acquired preparation clip; withhold pre-core frames by replacing rise descriptors with already extracted core descriptors; no new video or extra seconds',
        'interpretation': 'leave-one-block-out measures conditional model utility, not biological causality; correlated features and retuning affect differences',
        'metrics': 'person-equal MAE, under/overprediction, five frozen partitions; >=20 descriptive band; paired-person conditional bootstrap 10000 draws seed20261031; report each long person',
        'no_external_validation': True, 'fit_code_sha256':sha(HERE/'study.py')}
    save('protocol.json', protocol)
    print(json.dumps({k:{a:v[a] for a in ['people','long_people','ridge_views']} for k,v in configs.items()}), flush=True)

if __name__ == '__main__': prepare()
