"""Independent checks of fold membership, replay, and unchanged time-probe baseline."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'diagnostics'
DEV = HERE.parent / '20261030-gpu-motionbert-study/cnn-multiwindow695'
ROOT = HERE.parents[2]


def main():
    source = json.loads((OUT / 'sources.json').read_text())
    for key, p in [('dev_sha256', DEV / 'bundle/data.npz'),
                   ('normal_sha256', HERE / 'input/normal_cnn_v2/prepared_normal/prepared.npz'),
                   ('models_sha256', ROOT / 'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle/models.json')]:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == source[key]
    with np.load(DEV / 'bundle/data.npz', allow_pickle=False) as z:
        ids, folds, y = z['person_id'], z['fold'], z['y']
    for f in range(5):
        complete = json.loads((DEV / f'received/results/fold{f}/complete.json').read_text())
        assert set(complete['identity']['train_people']) == set(ids[folds != f])
        assert set(complete['identity']['outer_evaluation_people']) == set(ids[folds == f])
    with np.load(OUT / 'development_all_models.npz', allow_pickle=False) as z:
        cube = z['predictions']
    with np.load(DEV / 'analysis/cnn_oof.npz', allow_pickle=False) as z:
        oof_error = float(np.abs(cube[folds, np.arange(len(folds))] - z['predictions'][:, 0]).max())
    assert oof_error < 1e-4
    with np.load(HERE / 'input/normal_cnn_v2/prepared_normal/prepared.npz', allow_pickle=False) as z:
        owner, records = z['window_record'], z['record_id']
    info = json.loads((OUT / 'time_probe_metadata.json').read_text())
    selected = set(info['original_windows'])
    rmap = {original: i for i, original in enumerate(info['records'])}
    total = np.zeros((5, len(rmap), 2))
    count = np.zeros((5, len(rmap)))
    with (HERE / 'predictions/window_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            w, model = int(row['window_index']), int(row['fold'])
            if w not in selected:
                continue
            i = rmap[int(owner[w])]
            assert row['record_id'] == records[owner[w]]
            total[model, i] += [float(row['left_GDI']), float(row['right_GDI'])]
            count[model, i] += 1
    assert (count > 0).all()
    with np.load(OUT / 'normal_time_1_0.npz', allow_pickle=False) as z:
        baseline_error = float(np.abs(total / count[..., None] - z['predictions']).max())
    assert baseline_error < 1e-4
    with np.load(OUT / 'development_age_metadata.npz', allow_pickle=False) as z:
        ages = z['age']
        np.testing.assert_array_equal(z['person_id'], ids)
        assert np.isfinite(ages).all()
        adult_people = len(set(ids[ages >= 18]))
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    assert sum(r['side_scores'] for r in summary['score_bins']) == y.size
    for name in ['development_all_models', 'normal_reflected', 'development_reflected_oof',
                 'normal_time_0_8', 'normal_time_1_0', 'normal_time_1_2']:
        with np.load(OUT / (name + '.npz'), allow_pickle=False) as z:
            assert np.isfinite(z['predictions']).all()
    result = dict(passed=True, five_fold_membership_checked_against_training_identity=True,
        oof_replay_max_error=oof_error, time_baseline_vs_original_window_csv_max_error=baseline_error,
        adult_development_people=adult_people, age_records=len(ages),
        age_groups_can_share_people_across_visits=True, all_six_prediction_files_finite=True,
        original_inputs_and_model_manifest_unchanged=True)
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
