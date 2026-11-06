"""Freeze real source linkage, candidate definitions and splits before fitting."""
from common import *
import ast
import re
import openpyxl

SOURCES = {}


def tracked(path):
    SOURCES[str(path)] = sha(path)
    return path


def assigned(path, name):
    tree = ast.parse(tracked(path).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


def freeze(name, x, y, ids, cohort, names, groups, scope, extra=None):
    assert x.shape == (len(y), len(names)) and len(ids) == len(set(ids)) == len(y)
    assert np.isfinite(y).all() and not np.isinf(x).any()
    assert set(i for g in groups.values() for i in g) == set(range(len(names)))
    folder = OUT/name
    folder.mkdir(parents=True, exist_ok=True)
    folds = np.stack([outer_folds(cohort, r) for r in range(3)])
    np.savez_compressed(folder/'input.npz', x=x, y=y, ids=np.asarray(ids).astype(str), cohort=np.asarray(cohort).astype(str), folds=folds)
    meta = {'name': name, 'people': len(y), 'features': names, 'groups': groups, 'scope': scope,
            'target_range': [float(min(y)), float(max(y))], 'missing_per_feature': np.isnan(x).sum(0).tolist(),
            'input_sha256': sha(folder/'input.npz'), 'extra': extra or {}}
    save(folder/'metadata.json', meta)
    return meta


def mobilise():
    source = STUDIES/'20261031-mobilised-uncertainty-review'
    master = pd.read_csv(tracked(source/'extracted/Main datasets for analysis/CSV files/All_cohorts_dataset.csv'), low_memory=False)
    first = pd.read_csv(tracked(source/'first-eligible-per-target.csv'))
    keys = ['participantid', 'visit.number']
    assert not master.duplicated(keys).any()
    groups = {'speed': list(range(5)), 'stride_length': list(range(5,8)), 'cadence': list(range(8,12)), 'cycle_duration': list(range(12,15))}
    names = SPEED+STRIDE+CADENCE+CYCLE
    metadata = []
    for outcome in ['sppb', 'tug', 'nonwalk']:
        if outcome == 'tug':
            prior = pd.read_csv(tracked(Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-mobilised-function-models/TUG-cohort.csv')))
            selected = prior[keys]
        else:
            selected = first[first.audit_target == 'SPPB'][keys]
        g = selected.merge(master, on=keys, validate='one_to_one').sort_values('participantid').reset_index(drop=True)
        if outcome == 'tug':
            y = g.tugtime.to_numpy(float)
            linked = g[keys+['tugtime']].merge(prior[keys+['target']], on=keys, validate='one_to_one')
            np.testing.assert_array_equal(linked.tugtime, linked.target)
            assert len(g) == 1980 and (y >= 3).all()
        else:
            cols = ['balance_score', '4mwt_speed_score', '5crt_score']
            complete = g[cols].notna().all(axis=1)
            np.testing.assert_array_equal(g.loc[complete, cols].sum(1), g.loc[complete, 'total_sppb_score'])
            if outcome == 'nonwalk':
                g = g[g[['balance_score', '5crt_score']].notna().all(axis=1)].reset_index(drop=True)
                y = g[['balance_score', '5crt_score']].sum(1).to_numpy(float)
            else:
                y = g.total_sppb_score.to_numpy(float)
                assert len(g) == 2232
            assert (y == y.astype(int)).all() and (y >= 0).all() and (y <= (8 if outcome == 'nonwalk' else 12)).all()
        assert (g.n_days_w >= 3).all() and (g.ws_1030_avg_w > 0).all() and g.proto_dev.eq('N').all()
        original_form = pd.read_csv(tracked(source/f'extracted/Clinical Data/T1-T5-study-instances-{"tug" if outcome == "tug" else "sppb"}.csv')).rename(columns={'visit-number':'visit.number'})
        labels = ['tugtime'] if outcome == 'tug' else ['total_sppb_score', 'balance_score', '4mwt_speed_score', '5crt_score']
        # Some original forms use component-level fields rather than computed scores.
        available = [c for c in labels if c in original_form]
        if available:
            j = g[keys+available].merge(original_form[keys+available], on=keys, suffixes=('_main','_form'), validate='one_to_one')
            assert len(j) == len(g)
            for c in available:
                np.testing.assert_allclose(j[c+'_main'], j[c+'_form'], equal_nan=True)
        g[keys+['cohort']+names].assign(target=y).to_csv(OUT/f'mobilise_{outcome}_linked.csv', index=False)
        metadata.append(freeze('mobilise_'+outcome, g[names].to_numpy(float), y, g.participantid.to_numpy(), g.cohort.to_numpy(), names, groups,
            'Provider-filtered weekly wearable summaries; first eligible target-specific visit, >=3 valid days and +/-14day clinical/sensor timing. Not short RGB video.',
            {'target': 'balance+chair components, excluding walking score' if outcome == 'nonwalk' else ('measured TUG seconds' if outcome == 'tug' else 'supplied SPPB total'), 'clinical_form_fields_rechecked': available, 'over60_TUG_retained': int((y>60).sum()) if outcome=='tug' else 0}))
    return metadata


def opencap():
    base = STUDIES/'20261030-opencap415-speed-stability'
    d = dict(np.load(tracked(base/'input.npz'), allow_pickle=False))
    audit = json.loads(tracked(base/'preparation.json').read_text())
    assert sha(base/'input.npz') == audit['source_input_sha256']
    protocol = json.loads(tracked(STUDIES/'20261030-opencap415-tug-timeseries/all3s/protocol.json').read_text())
    channels = protocol['channels']
    x = np.concatenate([d['x'].mean(-1), d['x'].std(-1), abs(np.diff(d['x'],axis=-1)).mean(-1)], axis=1)
    names = [channel+'__'+stat for stat in ['mean','sd','mean_abs_adjacent_change'] for channel in channels]
    groups = {}
    sets = {'speed':[17,18], 'pelvis_angles':[0,1], 'hip_angles':[2,3,4,8,9,10], 'knee_angles':[5,11], 'ankle_angles':[6,7,12,13], 'lumbar_angles':[14,15,16]}
    for name, indices in sets.items():
        groups[name] = [i+offset for offset in [0,19,38] for i in indices]
    source_labels = pd.read_csv(tracked(STUDIES/'20261030-opencap415-tug-timeseries/raw/participant_info.csv'))
    manifest = pd.read_csv(tracked(STUDIES/'20261030-opencap415-tug-timeseries/all3s/manifest.csv'))
    lookup = source_labels.set_index(['subid','visit'])
    cohort = []
    for pid, y in zip(d['ids'], d['y']):
        visit = int(manifest.loc[manifest.id == pid, 'visit'].iloc[0])
        row = lookup.loc[(pid,visit)]
        assert float(row.tft_tug) == y
        cohort.append(row.diag)
    metadata = [freeze('opencap_tug', x, d['y'], d['ids'], cohort, names, groups,
        '298 people; one 3s 19-channel OpenCap window; processed multi-camera kinematics, not single-camera validation. Uses existing mean/SD/adjacent-change definitions.')]
    prior = STUDIES/'20261031-opencap-walk-pattern'
    d = dict(np.load(tracked(prior/'TUG_common6/input.npz'), allow_pickle=False))
    pattern_names = assigned(prior/'common.py', 'PATTERN_NAMES')
    names = ['speed_mean','speed_sd','speed_mean_abs_adjacent_change']+pattern_names
    groups = {'speed':[0,1,2], 'cycle_rhythm':[3,4,5], 'leg_ROM_asymmetry':list(range(6,12)), 'hip_knee_coordination':[12], 'pelvis_lumbar_ROM':[13,14,15,16]}
    assert 'sub-283' not in d['ids']
    metadata.append(freeze('opencap_patterns_tug', d['pattern'], d['y'], d['ids'], ['all']*len(d['y']), names, groups,
        '195 people; existing exact 6s walking windows; prior study excluded sub-283 everywhere. Knee-flexion peak cycle proxies, not validated foot contacts. Do not compare MAE with 298-person cohort as a duration effect.'))
    return metadata


def older44():
    base = ROOT/'outputs/gait-public-access-20261015/OlderAdults44'
    label_path = tracked(base/'01_demography_sppb_mmse.xlsx')
    wb = openpyxl.load_workbook(label_path, data_only=True, read_only=True)
    labels = {}
    for row in wb.active.values:
        match = re.fullmatch(r'Subject\s*(\d+)', str(row[0] or '').strip())
        if match:
            labels[int(match.group(1))] = float(row[7])
    wb.close()
    path = tracked(base/'05_gait_parameters.xlsx')
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    data = list(wb.active.values)
    wb.close()
    headers = [str(c).strip() for c in data[0]]
    kinds = [headers[i].removeprefix('Left_') for i in range(2,28,2)]
    assert len(kinds) == 13 and len(labels) == 44
    records, seen, pid = {}, set(), None
    for row in data[1:]:
        if row[0] is not None:
            pid = int(re.fullmatch(r'Subject\s*(\d+)', str(row[0]).strip()).group(1))
        key = (pid, int(row[1]))
        assert key not in seen
        seen.add(key)
        pair = np.array(row[2:28], float).reshape(13,2)
        values = np.r_[pair.mean(1), abs(pair[:,0]-pair[:,1])]
        records.setdefault(pid, []).append(values)
    ids = sorted(records)
    assert len(ids) == 44 and len(seen) == 220 and all(len(records[i]) == 5 for i in ids)
    x = np.stack([np.mean(records[i],axis=0) for i in ids])
    names = [k+'__'+s for s in ['bilateral_mean','absolute_LR_difference'] for k in kinds]
    defs = {'speed':[1], 'cadence':[0], 'step_stride_time':[2,3], 'gait_events':[4,5,6], 'support_time':[7,8], 'step_geometry':[9,10,11], 'limp_index':[12]}
    groups = {k:[i for j in v for i in [j,j+13]] for k,v in defs.items()}
    meta = freeze('older44_sppb', x, np.array([labels[i] for i in ids]), ids, ['all']*44, names, groups,
        '44 independent SPPB labels; author-provided Vicon spatiotemporal parameters from five full cropped walking trials per person, not fixed3s input or RGB pose estimates. Bilateral mean and absolute left-right difference computed within each trial then averaged over five trials; crop duration excluded.',
        {'original_headers':headers, 'trials_per_person':5, 'target':'supplied SPPB total, observed5..12'})
    return [meta]


def main():
    assert not (HERE/'comparison-protocol.json').exists(), 'Comparison already frozen; do not overwrite an experiment.'
    OUT.mkdir(parents=True,exist_ok=True)
    protocol = {'purpose':'Find useful gait measurements and combinations separately for actual SPPB and TUG; no preferred winner.',
        'datasets':DATASETS, 'outer':'3 repeated person5fold (17/41/89); cohort-stratified where feasible; one row per person',
        'inner':'3 shuffled personfolds within each outer training set, seed7000+100repeat+fold; all feature/combination/model selection stays inside outer training people',
        'models':{'Ridge_alpha':10, 'RF_trees':128, 'RF_min_leaf':'10 for datasets>=500people, else2', 'RF_features':1.0, 'RF_seed':'9000+100repeat+fold; same across candidates'},
        'preprocessing':'train-only median imputation with missing indicators; StandardScaler train-only; all-missing column retained. No clinical score, cohort, age or sex in gait inputs; no clipping/rounding.',
        'candidate_library':'Every scalar alone; all combinations of <=5families, else every family/pair/leave-one-family-out/all; greedy nested Ridge forward path up to3scalars; median and cohort-median diagnostic baselines; missingness-only control.',
        'selection_policies':{'selected_single':'lowest innerMAE scalar/model', 'selected_small':'lowest innerMAE among single/model and Ridge forward2/3', 'selected_family':'lowest innerMAE family-combination/model', 'selected_any':'lowest innerMAE among all gait candidates and overall median baseline'},
        'primary_metric':'equal-person held-out mean absolute error, reported in original points/seconds; average repeat MAEs, never prediction ensemble',
        'descriptive_ranking':'all-data Spearman, BH across scalar candidates within endpoint, within-cohort Spearman, fixed-candidate OOF rankings are exploratory; no global winning candidate selected on OOF and claimed independent',
        'uncertainty':'2000 paired person bootstrap samples of repeat-averaged absolute errors, conditional on fitted models; no equivalence margin or clinical cutoffs; exploratory repeated development data',
        'resources':'local CPU, existing packages, small reportsC; derived inputs, fold ledgers and large predictionsD; no remote transfer/GPU; estimated derived data<250MB, no fitted forests retained'}
    save(HERE/'comparison-protocol.json',protocol)
    metadata = mobilise()+opencap()+older44()
    assert [m['name'] for m in metadata] == DATASETS
    sources = {**SOURCES, **{str(HERE/f):sha(HERE/f) for f in ['common.py','prepare_comparison.py','run_comparison.py']}}
    save(HERE/'comparison-preparation.json',{'datasets':metadata,'source_sha256':sources,'protocol_sha256':sha(HERE/'comparison-protocol.json')})
    print(json.dumps([{k:m[k] for k in ['name','people','target_range']} for m in metadata],indent=2),flush=True)


if __name__ == '__main__':
    main()
