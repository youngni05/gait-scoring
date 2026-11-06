"""Fixed 50:50 blending of existing, aligned held-person predictions; no fitting."""
import os
import sys
import csv
import json
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parents[1]))
import cpu_head as m
import numpy as np

PLS = HERE.parent / 'expanded695-pls64-study'
DATA = HERE.parent / 'expanded695-pls-study'
GPU = m.ROOT / 'studies/20261030-gpu-motionbert-study/expansion695'
spec = importlib.util.spec_from_file_location('parent_metrics', DATA / 'run.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
METHODS = ['ridge', 'pls32', 'gpu_last_stage', 'equal_blend']


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sources():
    paths = [HERE / 'protocol.json', DATA / 'development695.npz', DATA / 'run.py',
             PLS / 'oof.metadata.json', PLS / 'artifact_manifest.json',
             GPU / 'analysis/gpu/verification.json', GPU / 'analysis/gpu/last_stage_oof.npz',
             GPU / 'analysis/gpu/last_stage_provenance.json', GPU / 'gpu-bundle/protocol.ready.json',
             GPU / 'train_only.npz']
    # Preserve every recorded PLS result, including parent datasets and models.
    for folder in [DATA, PLS]:
        for name, digest in m.load_json(folder / 'artifact_manifest.json').items():
            path = folder / name
            assert m.digest(path) == digest, path
            paths.append(path)
    return {p.relative_to(m.WORKSPACE).as_posix(): m.digest(p) for p in set(paths)}


def main():
    assert not (HERE / 'summary.json').exists(), 'Do not overwrite completed results'
    protocol = m.load_json(HERE / 'protocol.json')
    assert protocol['blend'] == {'pls32': 0.5, 'gpu_last_stage': 0.5}
    fingerprints = sources()
    save('source_hashes.json', fingerprints)
    d = npz(DATA / 'development695.npz')
    p = npz(PLS / 'oof_predictions.npz')
    g = npz(GPU / 'analysis/gpu/last_stage_oof.npz')
    with np.load(GPU / 'train_only.npz', allow_pickle=False) as z:
        gd = {k: z[k].copy() for k in ['person_id', 'video_id', 'fold', 'y', 'cohort']}
    meta = m.load_json(PLS / 'oof.metadata.json')
    provenance = m.load_json(GPU / 'analysis/gpu/last_stage_provenance.json')
    assert m.digest(PLS / 'oof_predictions.npz') == meta['predictions_sha256']
    assert m.digest(DATA / 'development695.npz') == meta['dataset_sha256']
    assert m.digest(GPU / 'analysis/gpu/last_stage_oof.npz') == provenance['prediction_sha256']
    assert m.digest(GPU / 'train_only.npz') == provenance['dataset_sha256']
    assert m.load_json(PLS / 'verification.json')['verified']
    assert m.load_json(GPU / 'analysis/gpu/verification.json')['passed']
    assert provenance['sides'] == ['left_GDI', 'right_GDI']
    for key in ['person_id', 'video_id', 'fold']:
        for other in [p, g, gd]:
            np.testing.assert_array_equal(d[key], other[key])
    np.testing.assert_array_equal(d['y'], gd['y'])
    np.testing.assert_array_equal(d['source_group'] == 'original175', gd['cohort'] == 0)
    audit = []
    for f in range(5):
        model = npz(PLS / f'models/fold{f}_selected.npz')
        tr, te = np.flatnonzero(d['fold'] != f), np.flatnonzero(d['fold'] == f)
        np.testing.assert_array_equal(tr, model['train_indices'])
        np.testing.assert_array_equal(te, model['evaluation_indices'])
        assert int(model['selected_k']) == 32
        gp = next(v for v in provenance['fold_models'] if v['fold'] == f)
        train, test = set(d['person_id'][tr]), set(d['person_id'][te])
        assert train == set(gp['training_people']) and test == set(gp['evaluation_people'])
        assert not train & test and len(train) == 556 and len(test) == 139
        audit.append(dict(fold=f, train_people=556, held_people=139, held_records=len(te), overlap=0))
    preds = [p['predictions'][p['methods'].tolist().index(k)] for k in ['ridge', 'pls64']]
    preds.append(g['predictions'][:, 0])
    np.testing.assert_allclose(preds[0], g['baseline'][:, 0], rtol=0, atol=1e-10)
    preds.append((preds[1] + preds[2]) / 2)
    preds = np.array(preds)
    assert preds.shape == (4, 1064, 2) and np.isfinite(preds).all()
    np.savez_compressed(HERE / 'oof_predictions.npz', predictions=preds, methods=np.array(METHODS),
                        **{k: d[k] for k in ['person_id', 'video_id', 'fold', 'y', 'source_group']})
    metrics, bands, folds, persons, records = [], [], [], [], []
    people = np.unique(d['person_id'])
    assert len(people) == 695 and len(set(d['video_id'])) == 1064
    errors = np.array([parent.person_errors(d['y'], pred, d['person_id']) for pred in preds])
    means = np.array([[pred[d['person_id'] == pid].mean() for pid in people] for pred in preds])
    true_means = np.array([d['y'][d['person_id'] == pid].mean() for pid in people])
    for mi, (method, pred) in enumerate(zip(METHODS, preds)):
        for group in ['all695', 'original175', 'additional520']:
            use = np.ones(1064, dtype=bool) if group == 'all695' else d['source_group'] == group
            metrics.append(dict(method=method, group=group, records=int(use.sum()),
                raw_prediction_min=float(pred[use].min()), raw_prediction_max=float(pred[use].max()),
                **parent.evaluate(d, pred, use)))
        for band in np.unique(np.floor(d['y'] / 10).astype(int)):
            bands.append(dict(method=method, actual_bin_low=int(band * 10),
                record_sides=int((np.floor(d['y'] / 10) == band).sum()), **parent.evaluate(d, pred, band=band)))
        for f in range(5):
            folds.append(dict(method=method, fold=f, **parent.evaluate(d, pred, d['fold'] == f)))
    for j, pid in enumerate(people):
        mask = d['person_id'] == pid
        a, b = preds[1, mask] - d['y'][mask], preds[2, mask] - d['y'][mask]
        row = dict(person_id=str(pid), fold=int(d['fold'][mask][0]), group=str(d['source_group'][mask][0]),
                   records=int(mask.sum()), actual_mean=float(true_means[j]),
                   opposite_error_fraction=float((a * b < 0).mean()))
        for mi, method in enumerate(METHODS):
            row[method + '_MAE'] = float(errors[mi, j])
            row[method + '_mean'] = float(means[mi, j])
            row[method + '_bias'] = float(means[mi, j] - true_means[j])
        row['blend_MAE_reduction_vs_pls32'] = float(errors[1, j] - errors[3, j])
        persons.append(row)
    for i in range(1064):
        for side, label in enumerate(['left_GDI', 'right_GDI']):
            row = dict(person_id=str(d['person_id'][i]), video_id=str(d['video_id'][i]),
                       fold=int(d['fold'][i]), side=label, actual=float(d['y'][i, side]))
            row.update({k: float(preds[j, i, side]) for j, k in enumerate(METHODS)})
            records.append(row)
    draws = np.random.default_rng(20261029).integers(0, 695, (10000, 695))
    comparisons = []
    for base, candidate in [(1, 2), (1, 3), (2, 3), (0, 3)]:
        delta = errors[base] - errors[candidate]
        comparisons.append(dict(baseline=METHODS[base], candidate=METHODS[candidate],
            MAE_reduction=float(delta.mean()), exploratory_bootstrap95=np.percentile(delta[draws].mean(1), [2.5, 97.5]).tolist(),
            people_improved=int((delta > 1e-12).sum()), people_worsened=int((delta < -1e-12).sum()),
            people_tied=int((abs(delta) <= 1e-12).sum())))
    complement = dict(person_MAE_correlation=float(np.corrcoef(errors[1], errors[2])[0, 1]),
        person_mean_signed_error_correlation=float(np.corrcoef(means[1] - true_means, means[2] - true_means)[0, 1]),
        person_weighted_opposite_error_fraction=float(np.mean([r['opposite_error_fraction'] for r in persons])),
        blend_better_than_both_people=int((errors[3] < np.minimum(errors[1], errors[2]) - 1e-12).sum()))
    for name, rows in [('metrics.csv', metrics), ('score_bands.csv', bands), ('fold_metrics.csv', folds),
                       ('person_comparison.csv', persons), ('record_side_predictions.csv', records), ('split_audit.csv', audit)]:
        table(name, rows)
    for path, digest in fingerprints.items():
        assert m.digest(m.WORKSPACE / path) == digest, path
    save('summary.json', dict(people=695, records=1064, methods=METHODS, metrics=metrics, score_bands=bands,
        fold_metrics=folds, comparisons=comparisons, complementarity=complement,
        preserved_source_files=len(fingerprints), training=False, GPU_used=False, protected_numeric_data_used=False,
        independent_final_evaluation=False, protocol_sha256=m.digest(HERE / 'protocol.json')))
    print(json.dumps({'metrics': [r for r in metrics if r['group'] == 'all695'],
                      'comparisons': comparisons, 'complementarity': complement}, indent=2))


if __name__ == '__main__':
    main()
