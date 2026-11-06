"""Describe score compression and input shifts without assigning LJMU true GDI."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'diagnostics'
ROOT = HERE.parents[2]
DEV = HERE.parent / '20261030-gpu-motionbert-study/cnn-multiwindow695'
sys.path.insert(0, str(HERE))
from analyze import read_csv, write_csv, stats
from diagnose_models import load


def person_mean(values, people):
    return np.array([values[people == p].mean(axis=0) for p in np.unique(people)])


def paired(delta, people):
    values = person_mean(delta, people).reshape(-1)
    rng = np.random.default_rng(20261031)
    boot = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return dict(people=len(values), pairs=len(delta), mean_delta=float(values.mean()),
                median_delta=float(np.median(values)), mean_abs_person_delta=float(np.abs(values).mean()),
                descriptive_person_bootstrap95=np.quantile(boot, [.025, .975]).tolist())


def age_metadata(dev):
    source = ROOT / 'outputs/stanford-sample-check-20261016'
    people, videos = set(dev['person_id']), set(dev['video_id'])
    vmap = {}
    with (source / 'video_list.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            video = Path(row['Video_File']).stem.replace('-processed', '')
            if video in videos:
                vmap[video] = row['Exam_ID']
    allowed = set(vmap.values())
    ages = {}
    # Retain only age metadata for approved development participant+visit IDs.
    with (source / 'alldata.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        ia, ip, ie = [header.index(k) for k in ('age', 'Patient_ID', 'examid')]
        for row in reader:
            if row[ip] not in people or row[ie] not in allowed:
                continue
            if row[ia]:
                ages.setdefault((row[ip], row[ie]), []).append(float(row[ia]))
    # Duplicated side rows can differ only at decimal serialization precision.
    assert all(np.ptp(v) < 1e-8 for v in ages.values())
    assert all(not (min(v) < 18 <= max(v)) for v in ages.values())
    age = np.array([ages[p, vmap[v]][0] for p, v in zip(dev['person_id'], dev['video_id'])])
    np.savez_compressed(OUT / 'development_age_metadata.npz', age=age,
                        person_id=dev['person_id'], video_id=dev['video_id'])
    return age


def aggregate_windows(v, owner, n):
    total = np.zeros((n, *v.shape[1:]))
    np.add.at(total, owner, v)
    return total / np.bincount(owner).reshape(n, *([1] * (v.ndim - 1)))


def features(d):
    x = d['x']
    # All outputs are record-level, side-averaged physical/temporal summaries.
    n, owner = len(d['person_id']), d['window_record']
    feat = {
        'ankle_height_over_thigh': x[..., 1].mean(axis=(1, 2)),
        'knee_angle_deg': np.rad2deg(x[..., 8]).mean(axis=(1, 2)),
        'ankle_angle_deg': np.rad2deg(x[..., 9]).mean(axis=(1, 2)),
        'toe_ankle_distance_over_thigh': x[..., 10].mean(axis=(1, 2)),
        'ankle_x_temporal_sd': x[..., 0].std(axis=2).mean(axis=1),
        'ankle_separation_temporal_sd': x[..., 11].std(axis=2).mean(axis=1),
        'ankle_x_change_per_frame': np.abs(np.diff(x[..., 0], axis=2)).mean(axis=(1, 2)),
        'knee_angle_change_deg_per_frame': np.rad2deg(np.abs(np.diff(x[..., 8], axis=2))).mean(axis=(1, 2)),
        'absolute_toe_direction_x': np.abs(x[..., 6] - x[..., 0]).mean(axis=(1, 2)),
        'signed_toe_direction_x': (x[..., 6] - x[..., 0]).mean(axis=(1, 2)),
    }
    # Spectral peak is cycles/frame, NOT cadence in Hz (actual FPS is unknown).
    sig = x[:, 0, :, 11].astype(np.float64)
    sig = (sig - sig.mean(axis=1, keepdims=True)) * np.hanning(124)
    power = np.abs(np.fft.rfft(sig, axis=1)) ** 2
    freq = np.fft.rfftfreq(124)
    band = (freq >= 1 / 100) & (freq <= 1 / 15)
    feat['ankle_difference_spectral_peak_cycles_per_frame'] = freq[band][power[:, band].argmax(axis=1)]
    return {k: aggregate_windows(v[:, None], owner, n)[:, 0] for k, v in feat.items()}


def main():
    dev = load(DEV / 'bundle/data.npz')
    normal = load(HERE / 'input/normal_cnn_v2/prepared_normal/prepared.npz')
    cube = load(OUT / 'development_all_models.npz')['predictions']
    y, people, folds = dev['y'], dev['person_id'], dev['fold']
    oof = cube[folds, np.arange(len(y))]
    in_sample = (cube.sum(axis=0) - oof) / 4
    ages = age_metadata(dev)
    bins = []
    for lo, hi in [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 120)]:
        mask = (y >= lo) & (y < hi)
        bins.append(dict(actual_lower=lo, actual_upper_exclusive=hi, side_scores=int(mask.sum()),
            people=len(set(people[mask.any(axis=1)])), actual_mean=float(y[mask].mean()),
            training_prediction_mean=float(in_sample[mask].mean()), oof_prediction_mean=float(oof[mask].mean()),
            oof_bias=float((oof - y)[mask].mean()),
            proportion_predicted_at_least90=float((oof[mask] >= 90).mean()),
            proportion_predicted_at_least100=float((oof[mask] >= 100).mean())))
    write_csv(OUT / 'development_score_bins.csv', bins)
    age_groups = []
    for label, rr in [('under18', ages < 18), ('18plus', ages >= 18)]:
        for lo, hi in [(0, 120), (80, 90), (90, 120)]:
            mask = rr[:, None] & (y >= lo) & (y < hi)
            age_groups.append(dict(age_group=label, actual_lower=lo, actual_upper_exclusive=hi,
                records=int(mask.any(axis=1).sum()), people=len(set(people[mask.any(axis=1)])),
                side_scores=int(mask.sum()), actual_mean=float(y[mask].mean()),
                predicted_mean=float(oof[mask].mean()), bias=float((oof - y)[mask].mean())))
    write_csv(OUT / 'development_age_groups.csv', age_groups)
    base = load(HERE / 'predictions/predictions.npz')['predictions']
    reflection = load(OUT / 'normal_reflected.npz')['predictions']
    normal_people = normal['person_id']
    reflected_dev = load(OUT / 'development_reflected_oof.npz')['predictions']
    sensitivity = dict(horizontal_reflection={
        'original_mean': float(person_mean(base.mean(axis=(0, 2)), normal_people).mean()),
        'reflected_mean': float(person_mean(reflection.mean(axis=(0, 2)), normal_people).mean()),
        **paired((reflection - base).mean(axis=(0, 2)), normal_people),
        'mean_abs_record_delta': float(np.abs((reflection - base).mean(axis=(0, 2))).mean()),
        'development_oof_original_person_mae': float(person_mean(np.abs(oof - y).mean(axis=1), people).mean()),
        'development_oof_reflected_person_mae': float(person_mean(np.abs(reflected_dev - y).mean(axis=1), people).mean())})
    time_base = load(OUT / 'normal_time_1_0.npz')
    time_people = time_base['person_id']
    sensitivity['time_spacing_probes'] = []
    for rate in ['0_8', '1_0', '1_2']:
        data = load(OUT / ('normal_time_' + rate + '.npz'))
        np.testing.assert_array_equal(data['person_id'], time_people)
        sensitivity['time_spacing_probes'].append(dict(rate=float(rate.replace('_', '.')),
            records=len(time_people), mean=float(person_mean(data['predictions'].mean(axis=(0, 2)), time_people).mean()),
            **paired((data['predictions'] - time_base['predictions']).mean(axis=(0, 2)), time_people)))
    records = read_csv(HERE / 'record_summary.csv')
    index = {r['record_id']: i for i, r in enumerate(records)}
    np.testing.assert_array_equal([r['record_id'] for r in records], normal['record_id'])
    score = base.mean(axis=(0, 2))
    conditions = {}
    pairs = [(i, index[r['record_id'].replace('_ALT_', '_')]) for i, r in enumerate(records)
             if '_ALT_' in r['record_id'] and r['record_id'].replace('_ALT_', '_') in index]
    conditions['ALT_minus_original_same_person_experiment_task'] = paired(
        np.array([score[i] - score[j] for i, j in pairs]), np.array([normal_people[i] for i, j in pairs]))
    pairs = [(i, index[r['record_id'].replace('_E2T', '_E1T')]) for i, r in enumerate(records)
             if '_E2T' in r['record_id'] and r['record_id'].replace('_E2T', '_E1T') in index]
    conditions['E2_minus_E1_same_person_task_clothing_label'] = paired(
        np.array([score[i] - score[j] for i, j in pairs]), np.array([normal_people[i] for i, j in pairs]))
    fd, fn = features(dev), features(normal)
    comparison = []
    # Record-mean true GDI defines this reference group; no true GDI exists for LJMU.
    high = y.mean(axis=1) >= 90
    for key in fd:
        d = person_mean(fd[key], people)
        h = person_mean(fd[key][high], people[high])
        n = person_mean(fn[key], normal_people)
        low, upper = np.quantile(h, [.05, .95])
        comparison.append(dict(feature=key, dev_mean=float(d.mean()), high90_mean=float(h.mean()),
            ljmu_mean=float(n.mean()), high90_people=len(h), high90_sd=float(h.std()),
            standardized_mean_difference=float((n.mean() - h.mean()) / h.std()),
            high90_p05=float(low), high90_p95=float(upper),
            ljmu_people_outside_reference_central90pct=int(((n < low) | (n > upper)).sum())))
    write_csv(OUT / 'feature_comparison.csv', comparison)
    direction = fn['signed_toe_direction_x'] > 0
    deltas, pair_people = [], []
    for p in np.unique(normal_people):
        a, b = (normal_people == p) & direction, (normal_people == p) & ~direction
        if a.any() and b.any():
            deltas.append(score[a].mean() - score[b].mean())
            pair_people.append(p)
    conditions['positive_minus_negative_toe_direction_within_person'] = paired(np.array(deltas), np.array(pair_people))
    quality = {r['record_id']: r for r in read_csv(HERE / 'input/normal_cnn_v2/prepared_normal/quality.csv')}
    confidence = np.array([float(quality[r]['mean_confidence']) for r in normal['record_id']])
    # Within-person centering avoids mistaking between-person differences for record effects.
    centered = lambda a: np.array([a[i] - a[normal_people == p].mean() for i, p in enumerate(normal_people)])
    quality_r = float(np.corrcoef(centered(confidence), centered(score))[0, 1])
    summary = dict(score_bins=bins, age_groups=age_groups, age_metadata_range=[float(ages.min()), float(ages.max())],
        target_counts={'all_side_scores':int(y.size), 'at_least90':int((y >= 90).sum()),
                       'at_least100':int((y >= 100).sum())},
        no_output_cap_in_code=True, max_oof_prediction=float(oof.max()),
        max_training_prediction=float(in_sample.max()), sensitivity=sensitivity,
        matched_conditions=conditions, feature_comparison=comparison,
        within_person_confidence_score_correlation=quality_r,
        inference_limit='No causal partition into age/camera/model; no adult healthy true GDI; sensitivity probes are not corrections',
        scoring_note='Development bin tables are record-side weighted; feature comparisons and paired differences are participant weighted',
        adult_definition='age at this development visit >=18; some people may occur in both visit-age groups')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    make_plot(bins, sensitivity)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_plot(bins, sensitivity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    actual = [b['actual_mean'] for b in bins]
    ax[0].plot([50, 110], [50, 110], '--', color='#888888', label='정답과 같은 예측')
    ax[0].plot(actual, [b['training_prediction_mean'] for b in bins], 'o-', label='그 사람을 학습한 모델', color='#4c9a8b')
    ax[0].plot(actual, [b['oof_prediction_mean'] for b in bins], 'o-', label='그 사람을 제외한 평가 모델', color='#c46648')
    ax[0].set(xlabel='구간별 실제 GDI 평균', ylabel='구간별 예측 GDI 평균', title='기존 개발자료: 높은 점수를 낮게 예측')
    ax[0].legend(fontsize=9)
    vals = sensitivity['time_spacing_probes']
    ax[1].plot([v['rate'] for v in vals], [v['mean'] for v in vals], 'o-', color='#367d9d')
    ax[1].axhline(100, linestyle='--', color='#999999', label='참조선 100 (개인 정답 아님)')
    ax[1].set(xticks=[.8, 1., 1.2], xticklabels=['0.8배\n느리게', '1.0배\n원래 간격', '1.2배\n빠르게'],
              ylabel='사람별 예측 평균', title='LJMU: 시간 간격만 바꾼 민감도 검사')
    ax[1].set_ylim(min(v['mean'] for v in vals) - 5, 103)
    ax[1].legend(fontsize=9)
    fig.suptitle('77점의 원인 탐색 | 모델·보정식 고정, 재학습 없음', fontsize=15)
    fig.savefig(OUT / 'diagnostic_summary.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
