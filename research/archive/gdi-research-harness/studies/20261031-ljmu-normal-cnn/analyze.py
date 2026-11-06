"""Descriptive analysis; participants are equally weighted, no assumed GDI labels."""
import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stats(x):
    x = np.asarray(x)
    return dict(n=int(x.size), mean=float(x.mean()), sd=float(x.std()),
                minimum=float(x.min()), q05=float(np.quantile(x, .05)),
                median=float(np.median(x)), q95=float(np.quantile(x, .95)),
                maximum=float(x.max()))


def main():
    with np.load(HERE / 'predictions/predictions.npz', allow_pickle=False) as z:
        cube, ids, people = z['predictions'], z['record_id'], z['person_id']
    with np.load(HERE / 'input/normal_cnn_v2/prepared_normal/prepared.npz', allow_pickle=False) as z:
        owner = z['window_record']
        missing = z['window_missing_fraction']
    assert cube.shape == (5, 882, 2) and np.isfinite(cube).all()
    unique = np.unique(people)
    # Average records within each participant, separately for each frozen model.
    pcube = np.stack([cube[:, people == p].mean(axis=1) for p in unique], axis=1)
    record_pair = cube.mean(axis=0)
    person_pair = pcube.mean(axis=0)
    person_mean = person_pair.mean(axis=1)
    record_mean = record_pair.mean(axis=1)
    record_sd = np.array([record_mean[people == p].std() for p in unique])
    model_sd = pcube.mean(axis=2).std(axis=0)
    record_ranges = np.array([np.ptp(record_mean[people == p]) for p in unique])
    participants = []
    for i, p in enumerate(unique):
        keep = people == p
        participants.append(dict(person_id=p, records=int(keep.sum()),
            left_GDI=person_pair[i, 0], right_GDI=person_pair[i, 1],
            mean_GDI=person_mean[i], abs_left_right=abs(np.diff(person_pair[i])[0]),
            record_mean_sd=record_sd[i], record_mean_min=record_mean[keep].min(),
            record_mean_max=record_mean[keep].max(), model_mean_sd=model_sd[i]))
    participants.sort(key=lambda r: r['mean_GDI'])
    write_csv(HERE / 'participant_summary.csv', participants)
    quality = {r['record_id']: r for r in read_csv(HERE / 'input/normal_cnn_v2/prepared_normal/quality.csv')}
    records = []
    for i, rid in enumerate(ids):
        q = quality[rid]
        records.append(dict(person_id=people[i], record_id=rid,
            experiment=re.search(r'_E(\d+)', rid).group(1), alt='_ALT_' in rid,
            windows=int((owner == i).sum()), left_GDI=record_pair[i, 0],
            right_GDI=record_pair[i, 1], mean_GDI=record_mean[i],
            abs_left_right=abs(np.diff(record_pair[i])[0]),
            model_mean_sd=cube[:, i].mean(axis=1).std(),
            max_window_missing_fraction=missing[owner == i].max(),
            mean_confidence=float(q['mean_confidence']),
            raw_zero_or_nan_fraction=float(q['raw_zero_or_nan_fraction'])))
    write_csv(HERE / 'record_summary.csv', records)
    models = []
    for f in range(5):
        values = pcube[f].mean(axis=1)
        models.append(dict(fold=f, left_mean=float(pcube[f, :, 0].mean()),
            right_mean=float(pcube[f, :, 1].mean()), **stats(values)))
    write_csv(HERE / 'model_summary.csv', models)
    # Verify supplied CSV outputs against NPZ, and independently aggregate windows.
    record_rows = read_csv(HERE / 'predictions/record_predictions.csv')
    window_rows = read_csv(HERE / 'predictions/window_predictions.csv')
    person_rows = read_csv(HERE / 'predictions/participant_predictions.csv')
    index = {rid: i for i, rid in enumerate(ids)}
    pindex = {p: i for i, p in enumerate(unique)}
    sums = np.zeros_like(cube)
    counts = np.zeros(cube.shape[:2], dtype=int)
    for r in window_rows:
        f, i = int(r['fold']), index[r['record_id']]
        sums[f, i] += [float(r['left_GDI']), float(r['right_GDI'])]
        counts[f, i] += 1
    np.testing.assert_allclose(sums / counts[:, :, None], cube, rtol=0, atol=1e-10)
    raw_cube = np.empty_like(cube)
    for r in record_rows:
        f, i = int(r['fold']), index[r['record_id']]
        np.testing.assert_allclose([float(r['left_GDI']), float(r['right_GDI'])], cube[f, i], rtol=0, atol=1e-10)
        raw_cube[f, i] = [float(r['left_raw']), float(r['right_raw'])]
    for r in person_rows:
        f, i = int(r['fold']), pindex[r['person_id']]
        np.testing.assert_allclose([float(r['left_GDI']), float(r['right_GDI'])], pcube[f, i], rtol=0, atol=1e-10)
    raw_person = np.array([raw_cube[:, people == p].mean() for p in unique])
    groups = []
    for field in ('experiment', 'alt'):
        for value in sorted(set(r[field] for r in records)):
            idx = np.array([r[field] == value for r in records])
            vals = [record_mean[idx & (people == p)].mean() for p in unique if (idx & (people == p)).any()]
            groups.append(dict(field=field, value=value, records=int(idx.sum()), **stats(vals)))
    write_csv(HERE / 'condition_summary.csv', groups)
    correlations = {}
    for field in ('max_window_missing_fraction', 'mean_confidence', 'raw_zero_or_nan_fraction', 'windows'):
        correlations[field] = float(np.corrcoef([r[field] for r in records], record_mean)[0, 1])
    quality_groups = []
    for label, rule in [('zero_missing', lambda r: r['max_window_missing_fraction'] == 0),
                        ('missing_at_most_5pct', lambda r: r['max_window_missing_fraction'] <= .05),
                        ('missing_over_5pct', lambda r: r['max_window_missing_fraction'] > .05)]:
        idx = np.array([rule(r) for r in records])
        values = [record_mean[idx & (people == p)].mean() for p in unique if (idx & (people == p)).any()]
        quality_groups.append(dict(group=label, records=int(idx.sum()), **stats(values)))
    development_path = HERE.parent / '20261030-gpu-motionbert-study/cnn-multiwindow695/analysis/cnn_oof.npz'
    with np.load(development_path, allow_pickle=False) as dev:
        y, pred = dev['y'], dev['predictions'][:, 0]
        development = dict(source=str(development_path), unit='record-side scores; repeated people are not independent',
                           actual_mean=float(y.mean()), prediction_mean=float(pred.mean()), bins=[])
        for low, high in [(90, 100), (100, 120)]:
            mask = (y >= low) & (y < high)
            development['bins'].append(dict(actual_gdi_lower=low, actual_gdi_upper_exclusive=high,
                side_scores=int(mask.sum()), records=int(mask.any(axis=1).sum()),
                people=len(set(dev['person_id'][mask.any(axis=1)])),
                actual_mean=float(y[mask].mean()), predicted_mean=float(pred[mask].mean())))
    summary = dict(people=len(unique), records=len(ids), windows=len(owner), models=models,
        weighting='equal records within participant, then equal participants; five-model mean is descriptive only',
        participant_mean=stats(person_mean), participant_left=stats(person_pair[:, 0]),
        participant_right=stats(person_pair[:, 1]), record_mean=stats(record_mean),
        participant_mean_raw=stats(raw_person),
        participant_abs_left_right=stats(np.abs(person_pair[:, 0] - person_pair[:, 1])),
        record_abs_left_right=stats(np.abs(record_pair[:, 0] - record_pair[:, 1])),
        within_person_record_sd=stats(record_sd), within_person_record_range=stats(record_ranges),
        between_models_participant_sd=stats(model_sd),
        person_counts={'below80':int((person_mean < 80).sum()),
                       '80to90':int(((person_mean >= 80) & (person_mean < 90)).sum()),
                       '90to100':int(((person_mean >= 90) & (person_mean < 100)).sum()),
                       '100plus':int((person_mean >= 100).sum())},
        condition_groups=groups, exploratory_record_pearson_correlations=correlations,
        descriptive_quality_subsets=quality_groups, existing_development_oof_reference=development,
        filename_experiment3_records=[r for r in records if r['experiment'] == '3'],
        lowest_people=participants[:5], highest_people=participants[-5:],
        no_actual_GDI_labels=True, no_accuracy_metrics=True,
        note='100 is a reference marker, not an assigned individual ground truth; bins are descriptive, not diagnoses')
    (HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    verification = dict(passed=True, record_csv_rows=len(record_rows), window_csv_rows=len(window_rows),
        participant_csv_rows=len(person_rows), prediction_shape=list(cube.shape),
        window_to_record_max_difference=float(np.max(np.abs(sums / counts[:, :, None] - cube))),
        participant_csv_matches=True, record_csv_matches=True, finite_predictions=True)
    (HERE / 'analysis_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic', 'axes.unicode_minus':False, 'font.size':11})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    axs[0].hist(person_mean, bins=np.arange(40, 111, 5), color='#367d9d', edgecolor='white')
    axs[0].axvline(100, color='#d4663a', linestyle='--', label='참조선 100 (개인 정답 아님)')
    axs[0].axvline(person_mean.mean(), color='#243e50', label=f'평균 {person_mean.mean():.2f}')
    axs[0].set(xlabel='사람별 예측 GDI (좌우·5모델 평균)', ylabel='사람 수', title='성인 64명의 예측 분포')
    axs[0].legend(fontsize=9)
    axs[1].boxplot([pcube[f].mean(axis=1) for f in range(5)], tick_labels=[f'모델 {f+1}' for f in range(5)])
    axs[1].axhline(100, color='#d4663a', linestyle='--')
    axs[1].set(ylabel='사람별 예측 GDI (좌우 평균)', title='같은 64명을 5개 학습 모델로 각각 예측')
    fig.suptitle('LJMU 정상 보행 자료 → 기존 다구간 1D-CNN\n64명 · 882기록 · 3,307구간 | 실제 GDI 정답 없음', fontsize=14)
    fig.savefig(HERE / 'prediction_distribution.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
