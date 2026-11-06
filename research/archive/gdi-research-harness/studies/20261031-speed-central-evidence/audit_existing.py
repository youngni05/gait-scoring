"""Recalculate existing comparisons; no training, feature selection or new hypotheses."""
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
MOBILISE = Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-mobilised-function-models')
SOURCES = {}


def track(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(track(path).read_text(encoding='utf-8-sig'))


def rows(path):
    track(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def equal(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=0), (actual, expected)


def main():
    source = STUDIES / '20261030-opencap415-speed-stability'
    summary = read_json(source / 'summary.json')
    groups = defaultdict(dict)
    targets = {}
    for row in rows(source / 'cpu-predictions.csv'):
        if row['model'] not in ('RF', 'Median'):
            continue
        key = (row['view'], row['model'], int(row['repeat']), int(row['seed']))
        person, target, prediction = row['id'], float(row['target']), float(row['prediction'])
        assert person not in groups[key]
        if person in targets:
            equal(target, targets[person])
        targets[person] = target
        groups[key][person] = abs(target - prediction)
    assert len(targets) == 298
    metrics, run_errors = [], {}
    for view, model in [('none', 'Median')] + [(v, 'RF') for v in ('mean_speed', 'speed_summary', 'pelvis_motion', 'full')]:
        runs = {key[2:]: values for key, values in groups.items() if key[:2] == (view, model)}
        assert len(runs) == (5 if model == 'Median' else 15)
        assert all(set(values) == set(targets) for values in runs.values())
        mae = fmean(fmean(values.values()) for values in runs.values())
        saved = next(item for item in summary['models'] if (item['view'], item['model']) == (view, model))
        equal(mae, saved['mean_MAE'])
        run_errors[view] = runs
        metrics.append({'view': view, 'model': model, 'people': 298, 'runs': len(runs), 'MAE_seconds': mae})
    comparisons = {}
    for other in ('mean_speed', 'speed_summary', 'pelvis_motion'):
        full, reference = run_errors['full'], run_errors[other]
        assert full.keys() == reference.keys()
        delta = fmean(fmean(full[key].values()) - fmean(reference[key].values()) for key in full)
        saved = summary['paired_comparisons']['full_RF_minus_' + other]
        equal(delta, saved['mean_difference_seconds'])
        assert sum(fmean(full[key].values()) < fmean(reference[key].values()) for key in full) == saved['full_lower_MAE_runs']
        comparisons[other] = {'full_minus_reference_MAE_seconds': delta, 'previous_conditional_bootstrap95': saved['conditional_person_bootstrap95'], 'interval_recomputed_this_audit': False}

    older = STUDIES / '20261030-older44-sppb-timeseries'
    older_summary = read_json(older / 'summary.json')
    people = rows(older / 'person-predictions.csv')
    assert len(people) == len({row['id'] for row in people}) == 44
    assert sum(int(row['clips']) for row in people) == 220
    older_metrics = {}
    for model in ('Mean', 'Median', 'Ridge', 'RF', 'CNN', 'TCN'):
        mae = fmean(abs(float(row['target']) - float(row[model])) for row in people)
        equal(mae, older_summary['metrics'][model]['person_MAE'])
        older_metrics[model] = mae

    mobilise_saved = rows(MOBILISE / 'summary.csv')
    mobilise_groups = defaultdict(dict)
    for row in rows(MOBILISE / 'predictions.csv.gz'):
        if row['outcome'] != 'TUG' or row['model'] not in ('Ridge', 'RFreg') or row['view'] not in ('speed', 'pattern'):
            continue
        assert row['role'] == 'test'
        key = (row['model'], row['view'], int(row['repeat']))
        person = row['participantid']
        assert person not in mobilise_groups[key]
        mobilise_groups[key][person] = (float(row['target']), abs(float(row['target']) - float(row['prediction_seconds'])))
    mobilise_metrics = []
    for model in ('Ridge', 'RFreg'):
        for view in ('speed', 'pattern'):
            repeats = [mobilise_groups[model, view, rep] for rep in range(3)]
            assert all(len(group) == 1980 for group in repeats)
            reference = mobilise_groups[model, 'speed', 0]
            assert all(group.keys() == reference.keys() for group in repeats)
            assert all(all(group[person][0] == reference[person][0] for person in reference) for group in repeats)
            mae = fmean(fmean(value[1] for value in group.values()) for group in repeats)
            saved = next(row for row in mobilise_saved if (row['outcome'], row['model'], row['view'], row['cohort']) == ('TUG', model, view, 'all'))
            equal(mae, float(saved['mae']))
            mobilise_metrics.append({'model': model, 'view': view, 'people': 1980, 'repeats': 3, 'MAE_seconds': mae})

    result = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'passed': True,
        'scope': 'Recalculation of existing held-out MAE only. No new models, split selection, clinical validation or correlation ranking.',
        'opencap_TUG': {'metrics': metrics, 'paired_comparisons': comparisons, 'limitation': 'Multi-camera-derived OpenCap data; full input includes speed. No joint-only comparison in these saved views. Intervals conditional on existing fitted models.'},
        'older44_SPPB': {'people': 44, 'clips': 220, 'MAE_points': older_metrics, 'limitation': 'Full marker input already includes pelvis velocity. These predictions cannot isolate speed from marker information. Only supplied total labels; no non-walking SPPB components evaluated here.'},
        'mobilise_TUG': {'metrics': mobilise_metrics, 'limitation': 'Weekly wearable summaries, not joint angles or short RGB videos. Pattern view includes the same speed features plus stride/cadence information.'},
        'source_sha256': SOURCES,
    }
    # Fail before saving if any evidence changed during calculation.
    for path, digest in SOURCES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    (HERE / 'existing-evidence-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in ('source_sha256',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
