"""Existing held-person predictions only. All writes stay beside this script."""
from pathlib import Path
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BUNDLE = ROOT / 'gdi-research-harness/control/manual-sessions/20261029-gpu-finetune-preparation/bundle'
os.environ['MPLCONFIGDIR'] = str(OUT / 'mplconfig')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np

# Reuse the existing local plotting dependency, without installing or modifying it.
if __import__('importlib.util', fromlist=['find_spec']).find_spec('matplotlib') is None:
    sys.path.insert(0, str(ROOT / 'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

FONT = Path('C:/Windows/Fonts/malgun.ttf')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'figure.dpi': 130,
                     'savefig.dpi': 160, 'axes.spines.top': False, 'axes.spines.right': False})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def status(state, done, current, following):
    (OUT / 'STATUS.md').write_text(
        '# A · GDI 분포와 점수화 분석\n\n'
        f'- KST 갱신 시각: {datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")}\n'
        f'- 상태: {state}\n- 완료 내용: {done}\n- 현재 작업: {current}\n'
        '- 결정이 필요한 항목: 임상 구간·앱 점수 채택은 별도 판단. 이번에는 고정 폭10 탐색 분석\n'
        f'- 결과 경로: {OUT.as_posix()}\n- 다음 작업: {following}\n', encoding='utf-8')


def summary(a):
    return {k: float(v) for k, v in zip(
        ['min', 'q1', 'median', 'q3', 'max', 'mean', 'sd'],
        [*np.percentile(a, [0, 25, 50, 75, 100]), np.mean(a), np.std(a, ddof=1)])}


def main():
    status('실행 중', '입력 구조와 읽기 전용 범위 확인', '분석·검증 코드 실행', '그림과 보고서 검토')
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    names = ['train_only.npz', 'baseline_heads.npz', 'baseline_predictions.npz', 'README.md']
    before = {name: sha(BUNDLE / name) for name in names}
    assert all(before[n] == manifest[n] for n in names), 'input fingerprint mismatch'
    with np.load(BUNDLE / 'train_only.npz', allow_pickle=False) as z:
        # No raw RGB or protected cohort is opened; even the coordinate array is not needed.
        d = {k: z[k].copy() for k in ['y', 'features', 'person_id', 'video_id', 'fold']}
        x_shape = z['x'].shape
    with np.load(BUNDLE / 'baseline_heads.npz', allow_pickle=False) as z:
        heads = {k: z[k].copy() for k in z.files}
    with np.load(BUNDLE / 'baseline_predictions.npz', allow_pickle=False) as z:
        pred = z['predictions'].copy()
    y, features, ids, folds = d['y'], d['features'], d['person_id'], d['fold']
    people = np.unique(ids)
    assert y.shape == (203, 2) and pred.shape == (203, 2, 2)
    assert features.shape == (203, 2, 512) and x_shape == (203, 2, 124, 17, 3)
    assert len(people) == 175 and len(set(d['video_id'])) == 203
    assert all(np.isfinite(a).all() for a in [y, features, pred])
    assert set(folds) == set(range(5))
    count = np.array([(ids == p).sum() for p in ids])
    weights = np.repeat((1 / (175 * count * 2))[:, None], 2, axis=1)
    assert np.isclose(weights.sum(), 1)

    def person(a):
        return np.array([a[ids == p].mean(axis=0) for p in people])

    def macro(a):
        return float(person(a).mean())

    def correlation(a, b):
        w = weights / weights.sum()
        ac, bc = a - np.sum(w * a), b - np.sum(w * b)
        return float(np.sum(w * ac * bc) / np.sqrt(np.sum(w * ac**2) * np.sum(w * bc**2)))

    replay = np.empty_like(pred)
    mean_pred = np.empty_like(pred)
    audit = []
    for k in range(5):
        train, test = folds != k, folds == k
        assert len(set(ids[test])) == 35 and len(set(ids[train])) == 140
        assert not set(ids[test]) & set(ids[train])
        h = {n: heads[f'fold{k}__{n}'] for n in ['scaler_mean', 'scaler_scale', 'coef', 'intercept']}
        mu, sd = features[train, 0].mean(0), features[train, 0].std(0)
        sd[sd == 0] = 1
        assert np.allclose(mu, h['scaler_mean'], rtol=1e-9, atol=1e-10)
        assert np.allclose(sd, h['scaler_scale'], rtol=1e-9, atol=1e-10)
        replay[test] = ((features[test] - h['scaler_mean']) / h['scaler_scale']) @ h['coef'].T + h['intercept']
        # Mean baseline uses training records only, consistently with record-equal Ridge fitting.
        mean_pred[test] = y[train].mean(0)[None, None, :]
        xs = (features[train, 0] - h['scaler_mean']) / h['scaler_scale']
        yc = y[train] - y[train].mean(0)
        normal_residual = (xs.T @ xs + np.eye(512)) @ h['coef'].T - xs.T @ yc
        relative_residual = np.max(abs(normal_residual)) / max(1., np.max(abs(xs.T @ yc)))
        assert relative_residual < 1e-7, 'saved Ridge head not matched to training fold'
        assert np.allclose(h['intercept'], y[train].mean(0), rtol=1e-8, atol=1e-8)
        audit.append({'fold': k, 'train_people': 140, 'test_people': 35,
                      'train_records': int(train.sum()), 'test_records': int(test.sum()),
                      'normal_equation_relative_residual': float(relative_residual),
                      'replay_max_difference': float(abs(replay[test] - pred[test]).max())})
    replay_error = float(abs(replay - pred).max())
    assert replay_error < 1e-10
    assert abs(macro(abs(pred[:, 0] - y)) - 8.275834265385386) < 1e-9
    assert abs(macro(abs(pred[:, 1] - y)) - 9.033922935416292) < 1e-9
    assert abs(macro(abs(pred[:, 0] - pred[:, 1])) - 3.960482771640348) < 1e-9

    person_y = person(y)
    distributions = {'record_side': summary(y.ravel()), 'record_left': summary(y[:, 0]),
                     'record_right': summary(y[:, 1]), 'person_mean_side': summary(person_y.ravel()),
                     'person_mean_left': summary(person_y[:, 0]), 'person_mean_right': summary(person_y[:, 1])}
    distribution_rows = [{'scope': k, **v} for k, v in distributions.items()]
    table('distribution.csv', distribution_rows)
    table('fold_audit.csv', audit)
    metrics, band_rows, confusion_rows, boundary_rows = [], [], [], []
    bin_true = np.floor(y / 10).astype(int)
    all_bins = np.arange(min(bin_true.min(), np.floor(pred / 10).min(), np.floor(mean_pred / 10).min()),
                         max(bin_true.max(), np.floor(pred / 10).max(), np.floor(mean_pred / 10).max()) + 1).astype(int)
    boundary_distance = np.minimum(y % 10, 10 - y % 10)
    cms = {}
    for model, predictions in [('motionbert_ridge', pred), ('train_mean', mean_pred)]:
        for wi, window in enumerate(['front', 'back']):
            p = predictions[:, wi]
            err, signed = abs(p - y), p - y
            bin_pred = np.floor(p / 10).astype(int)
            gap = abs(bin_pred - bin_true)
            mse = macro((p - y)**2)
            variance = float(np.sum(weights * (y - np.sum(weights * y))**2))
            row = {'model': model, 'window': window, 'people': 175, 'records': 203,
                   'person_macro_mae': macro(err), 'record_side_mae': float(err.mean()),
                   'left_person_macro_mae': float(person(err)[:, 0].mean()),
                   'right_person_macro_mae': float(person(err)[:, 1].mean()),
                   'person_weighted_rmse': float(np.sqrt(mse)), 'person_macro_bias': macro(signed),
                   'prediction_min': float(p.min()), 'prediction_max': float(p.max()),
                   'person_weighted_pearson': correlation(y, p), 'pooled_weighted_r2': 1 - mse / variance,
                   'exact_bin_rate': macro(gap == 0), 'one_bin_error_rate': macro(gap == 1),
                   'two_or_more_bin_error_rate': macro(gap >= 2), 'within_one_bin_rate': macro(gap <= 1),
                   'record_exact_bin_rate': float((gap == 0).mean()),
                   'person_mean_side_prediction_sd': float(person(p).std(ddof=1)),
                   'person_mean_bilateral_prediction_sd': float(person(p).mean(1).std(ddof=1)),
                   'person_mean_side_truth_sd': float(person_y.std(ddof=1)),
                   'person_mean_bilateral_truth_sd': float(person_y.mean(1).std(ddof=1)),
                   'person_macro_front_back_gap': macro(abs(predictions[:, 0] - predictions[:, 1])),
                   'front_back_same_bin_rate': macro(np.floor(predictions[:, 0] / 10) == np.floor(predictions[:, 1] / 10))}
            metrics.append(row)
            cm = np.zeros((len(all_bins), len(all_bins)), dtype=int)
            for ai, ab in enumerate(all_bins):
                for pi, pb in enumerate(all_bins):
                    mask = (bin_true == ab) & (bin_pred == pb)
                    cm[ai, pi] = int(mask.sum())
                    confusion_rows.append({'model': model, 'window': window, 'actual_bin_low': int(ab*10),
                                           'predicted_bin_low': int(pb*10), 'record_side_count': int(mask.sum()),
                                           'person_equal_weight': float(weights[mask].sum())})
            assert cm.sum() == 406
            cms[(model, window)] = cm
            for ab in all_bins:
                mask = bin_true == ab
                if not mask.any():
                    continue
                person_band_maes, person_band_biases, person_band_hits = [], [], []
                for pid in people:
                    use = mask & (ids == pid)[:, None]
                    if use.any():
                        person_band_maes.append(float(err[use].mean()))
                        person_band_biases.append(float(signed[use].mean()))
                        person_band_hits.append(float((gap[use] == 0).mean()))
                band_rows.append({'model': model, 'window': window, 'actual_bin_low': int(ab*10),
                                  'actual_bin_high_exclusive': int((ab+1)*10),
                                  'people_in_band': len(person_band_maes), 'record_side_count': int(mask.sum()),
                                  'person_within_band_mae': float(np.mean(person_band_maes)),
                                  'person_within_band_bias': float(np.mean(person_band_biases)),
                                  'person_within_band_exact_rate': float(np.mean(person_band_hits))})
            for lo, hi in [(0,1), (1,2), (2,3), (3,4), (4,5.000001)]:
                mask = (boundary_distance >= lo) & (boundary_distance < hi)
                pp = []
                for pid in people:
                    use = mask & (ids == pid)[:, None]
                    if use.any():
                        pp.append(float((gap[use] != 0).mean()))
                boundary_rows.append({'model': model, 'window': window, 'distance_low': lo,
                                      'distance_high': min(hi, 5), 'record_side_count': int(mask.sum()),
                                      'people': len(pp), 'person_within_group_mismatch_rate': float(np.mean(pp)) if pp else None})
            assert np.isclose(row['exact_bin_rate'] + row['one_bin_error_rate'] + row['two_or_more_bin_error_rate'], 1)
    table('metrics.csv', metrics)
    table('score_bands.csv', band_rows)
    table('confusion.csv', confusion_rows)
    table('boundary_distance.csv', boundary_rows)
    unique_counts = np.array([(ids == p).sum() for p in people])
    result = {'people': 175, 'records': 203, 'record_sides': 406, 'folds': 5,
              'records_per_person': {str(k): int((unique_counts == k).sum()) for k in np.unique(unique_counts)},
              'axes': {'y': ['record', 'left_right'], 'predictions': ['record', 'front_back', 'left_right']},
              'source_hashes': before, 'bin_rule': 'floor(GDI/10); [10k,10(k+1)); no clipping; no clinical labels',
              'aggregation': 'mean over sides/records within each person, then mean over 175 people',
              'mean_baseline': 'per-side record-equal mean of training-fold labels, never evaluation-fold labels',
              'distribution': distributions, 'metrics': metrics,
              'interpretation': 'Repeatedly explored development cohort, not independent final evaluation; same-video front/back, not re-filming.'}
    dump('summary.json', result)

    # Four standalone figures; no new model or score is fitted to these results.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    hist_edges = np.arange(all_bins.min()*10, (all_bins.max()+2)*10, 10)
    for ax, label, yy, pp in [(axes[0], '기록·좌우 기준 (406개)', y.ravel(), pred[:,0].ravel()),
                              (axes[1], '사람별 좌우 평균값 기준 (350개)', person_y.ravel(), person(pred[:,0]).ravel())]:
        ax.hist(yy, bins=hist_edges, alpha=.55, color='#2474a9', label='실제 GDI')
        ax.hist(pp, bins=hist_edges, alpha=.5, color='#e68a2e', label='기존 예측: 앞 구간')
        ax.set(title=label, xlabel='GDI', ylabel='개수'); ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.suptitle('개발자료의 실제 점수와 예측 분포 — 두 집계 기준은 서로 다름', fontsize=13)
    fig.tight_layout(); fig.savefig(OUT/'01_distribution.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    limits = (float(min(y.min(), pred.min()))-3, float(max(y.max(), pred.max()))+3)
    for wi, ax in enumerate(axes):
        for si, label in enumerate(['왼쪽', '오른쪽']):
            ax.scatter(person_y[:,si], person(pred[:,wi])[:,si], s=15, alpha=.55, label=label)
        ax.plot(limits, limits, color='gray', linestyle='--', linewidth=1)
        ax.set(xlim=limits, ylim=limits, xlabel='실제 GDI (사람 안 기록 평균)', ylabel='예측 GDI (사람 안 기록 평균)',
               title=['앞 구간', '뒤 구간'][wi]); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('실제–예측 관계: 사람별 좌우 350점 / 각 축 동일 범위', fontsize=13)
    fig.tight_layout(); fig.savefig(OUT/'02_actual_predicted.png'); plt.close(fig)

    front_bands = [r for r in band_rows if r['window']=='front' and r['model']=='motionbert_ridge']
    mean_bands = [r for r in band_rows if r['window']=='front' and r['model']=='train_mean']
    bx = np.arange(len(front_bands))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(bx-.18, [r['person_within_band_mae'] for r in front_bands], width=.36, label='MotionBERT + Ridge')
    ax.bar(bx+.18, [r['person_within_band_mae'] for r in mean_bands], width=.36, label='학습 묶음 평균 예측')
    ax.set_xticks(bx, [f"[{r['actual_bin_low']}, {r['actual_bin_high_exclusive']})\nn={r['people_in_band']}명" for r in front_bands])
    ax.set(xlabel='실제 GDI 구간 (한 사람은 여러 구간에 포함될 수 있음)', ylabel='구간 내 사람별 평균 절대오차 (점)',
           title='점수대별 오차: 앞 구간 / 적은 인원 구간 해석 주의')
    ax.legend(); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'03_error_by_band.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, model, label in [(axes[0],'motionbert_ridge','MotionBERT + Ridge'),(axes[1],'train_mean','학습 묶음 평균')]:
        cm = cms[(model,'front')]
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(cms[('motionbert_ridge','front')].max(),cms[('train_mean','front')].max()))
        ax.set_xticks(np.arange(len(all_bins)), [str(b*10) for b in all_bins], rotation=45)
        ax.set_yticks(np.arange(len(all_bins)), [str(b*10) for b in all_bins])
        ax.set(xlabel='예측 구간의 하한', ylabel='실제 구간의 하한', title=label)
        for a in range(len(all_bins)):
            for p in range(len(all_bins)):
                if cm[a,p]: ax.text(p,a,str(cm[a,p]),ha='center',va='center',fontsize=8,
                                   color='white' if cm[a,p] > cm.max()*.6 else 'black')
    fig.suptitle('앞 구간 혼동표: 폭10의 고정 구간 / 칸 숫자는 기록·좌우 개수 (합계406)', fontsize=12)
    fig.tight_layout(); fig.savefig(OUT/'04_bin_confusion.png'); plt.close(fig)

    front = next(r for r in metrics if r['model']=='motionbert_ridge' and r['window']=='front')
    back = next(r for r in metrics if r['model']=='motionbert_ridge' and r['window']=='back')
    mean = next(r for r in metrics if r['model']=='train_mean' and r['window']=='front')
    dist = distributions['record_side']
    rows = '\n'.join(f"| [{r['actual_bin_low']}, {r['actual_bin_high_exclusive']}) | {r['people_in_band']} | {r['record_side_count']} | {r['person_within_band_mae']:.2f} | {r['person_within_band_exact_rate']*100:.1f}% |" for r in front_bands)
    boundary_table = '\n'.join(f"| {r['distance_low']}~{r['distance_high']}점 | {r['people']} | {r['record_side_count']} | {r['person_within_group_mismatch_rate']*100:.1f}% |"
                               for r in boundary_rows if r['model']=='motionbert_ridge' and r['window']=='front')
    report = f'''# GDI 분포와 점수화 가능성 — 기존 개발자료 분석

기준일: 2026-10-30. 이번에는 기존 예측이 실제 점수의 차이를 얼마나 구분하는지, 10점 단위 표시가 가능한지 확인했다. 새 모델을 학습하거나 임상 등급을 만들지는 않았다.

**현재 예측은 단순 평균보다 오차가 작지만, 낮고 높은 점수를 충분히 구분하지 못하고 중앙으로 모이는 경향이 있다. 10점 구간의 앞 구간 정확 일치율은 {front['exact_bin_rate']*100:.1f}%이며, 구간화만으로 세밀한 점수 구분이 해결됐다고 보기는 어렵다.**

## 1. 무엇을 분석했는가

기존 개발자료 **175명·203기록**, 기록마다 좌우 GDI 2개다. 같은 사람의 여러 기록은 같은 평가 묶음에 있다. 사람별 5묶음에서 해당 사람을 제외하고 학습한 고정 MotionBERT + Ridge의 앞·뒤124프레임 예측을 사용했다. 전체자료로 학습한 뒤 같은 자료를 예측한 값은 사용하지 않았다.

원자료의 세 번째 입력 채널은 깊이가 아니라 관측 신뢰도다. 이번 계산은 좌표를 다시 추론하지 않고 저장된 특징과 점수만 사용했다. 좌우 순서는 준비 코드의 `actual_L, actual_R`, 구간 순서는 `front, last` 적재 순서로 확인했다.

주요 성능은 **한 사람 안에서 기록과 좌우를 먼저 평균한 뒤 175명을 평균**했다. 기록별 분포는 203×2=406개 값, 사람별 분포는 기록을 평균한 175×2=350개 값이다. 두 집계의 범위·분산은 서로 다를 수 있다.

비교용 평균 예측은 각 평가 묶음을 제외한 140명의 학습 기록에서 좌우별 평균을 계산했다. 평균 예측의 학습은 기록 동일 비중이며, 평가는 사람 동일 비중이다.

## 2. 실제 점수와 예측의 범위

| 항목 | 결과 |
|---|---:|
| 실제 GDI 최소–최대 (406개 기록·좌우 값) | {dist['min']:.2f}–{dist['max']:.2f} |
| 중앙값 / 1사분위–3사분위 | {dist['median']:.2f} / {dist['q1']:.2f}–{dist['q3']:.2f} |
| 왼쪽 평균 / 오른쪽 평균 (기록 기준) | {distributions['record_left']['mean']:.2f} / {distributions['record_right']['mean']:.2f} |
| 앞 구간 예측 최소–최대 (406개 기록·좌우 값) | {front['prediction_min']:.2f}–{front['prediction_max']:.2f} |
| 뒤 구간 예측 최소–최대 (406개 기록·좌우 값) | {back['prediction_min']:.2f}–{back['prediction_max']:.2f} |
| 실제 사람별 좌우 평균값 350개의 표준편차 | {front['person_mean_side_truth_sd']:.2f} |
| 앞 구간 예측 사람별 좌우 평균값 350개의 표준편차 | {front['person_mean_side_prediction_sd']:.2f} |
| 실제 사람별 양쪽 평균 175개의 표준편차 | {front['person_mean_bilateral_truth_sd']:.2f} |
| 앞 구간 예측 사람별 양쪽 평균 175개의 표준편차 | {front['person_mean_bilateral_prediction_sd']:.2f} |

사람별 기록 수 분포는 `{result['records_per_person']}`이다. 왼쪽은 기록 수, 오른쪽은 그 기록 수를 가진 사람 수다. 예측 표준편차가 실제보다 작으면 중앙으로 모이는 경향이 있음을 뜻한다. 이것만으로 원인을 확정하지 않는다.

![실제 점수와 예측 분포](01_distribution.png)

![사람별 실제–예측 관계](02_actual_predicted.png)

## 3. 기존 예측은 단순 평균보다 얼마나 나은가

| 방법 | 앞 MAE | 뒤 MAE | 앞뒤 예측 차이 | 앞 정확 구간 일치 | 앞 한 구간 오류 | 앞 두 구간 이상 오류 |
|---|---:|---:|---:|---:|---:|---:|
| MotionBERT + Ridge | {front['person_macro_mae']:.3f} | {back['person_macro_mae']:.3f} | {front['person_macro_front_back_gap']:.3f} | {front['exact_bin_rate']*100:.1f}% | {front['one_bin_error_rate']*100:.1f}% | {front['two_or_more_bin_error_rate']*100:.1f}% |
| 학습 묶음 평균 | {mean['person_macro_mae']:.3f} | {mean['person_macro_mae']:.3f} | 0.000 | {mean['exact_bin_rate']*100:.1f}% | {mean['one_bin_error_rate']*100:.1f}% | {mean['two_or_more_bin_error_rate']*100:.1f}% |

MAE는 평균 절대오차이며 점 단위다. 정확도 백분율로 바꾸지 않는다. 앞 예측의 사람가중 Pearson 상관은 {front['person_weighted_pearson']:.3f}, 같은 가중치의 풀링 R²는 {front['pooled_weighted_r2']:.3f}다. 상관은 점수의 동반 변화를 나타내며 정밀한 개별 점수 일치를 보장하지 않는다.

평균 예측 대비 앞 MAE는 {mean['person_macro_mae']-front['person_macro_mae']:.3f}점 작고, 정확 구간 일치율은 {(front['exact_bin_rate']-mean['exact_bin_rate'])*100:.1f}%p 높다. 이는 이 개발자료의 기술적 비교이며 통계적 유의성이나 임상적 개선을 판정한 것은 아니다.

평균 예측은 같은 사람에게 앞뒤 동일 점수를 주므로 차이가 0이다. 따라서 앞뒤 차이 감소만으로 개선이라 판단하지 않으며, 정답 오차와 사람 간 구분도 함께 보려 한다. 같은 영상의 앞뒤 차이는 재촬영·시점 변화 안정성 검증이 아니다.

## 4. 어떤 점수대에서 틀리는가

앞 구간의 실제 점수대를 기준으로 나눴다. 각 구간 안에서 사람별 오차를 먼저 평균했다. 한 사람의 좌우·기록이 다른 구간에 들어갈 수 있어 인원 열의 합은 175가 아닐 수 있다.

| 실제 GDI 구간 | 포함 사람 | 기록·좌우 값 | 사람별 MAE | 정확 구간 일치 |
|---|---:|---:|---:|---:|
{rows}

![점수대별 오차](03_error_by_band.png)

실제 70~80점 구간에서는 잘 맞는 편이지만, 낮은 점수는 높게, 높은 점수는 낮게 예측하는 양상이 보인다. 앞 구간의 실제 50~60점 MAE는 11.69점, 90~100점 MAE는 16.45점이다. 양 끝 구간의 사람 수가 적다는 점도 함께 고려해야 한다. 다음 개선 실험에서는 전체 평균 오차와 함께 이런 점수대별 편향이 줄어드는지 살펴볼 필요가 있다.

## 5. 구간으로 표시할 가능성

구간은 **[10k, 10(k+1))**로 고정했다. 예를 들어 70은 [70,80)에 들어간다. 0~100으로 자르거나 자료를 보고 경계를 이동하지 않았다. 정상·경증 등의 임상 이름도 붙이지 않았다.

앞 구간의 정확 일치율은 **{front['exact_bin_rate']*100:.1f}%**, 한 구간 이내 비율은 **{front['within_one_bin_rate']*100:.1f}%**다. 뒤 구간의 정확 일치율은 **{back['exact_bin_rate']*100:.1f}%**다. 앞뒤 예측이 같은 구간에 놓이는 비율은 **{front['front_back_same_bin_rate']*100:.1f}%**다. 모두 사람 동일 비중의 집계다.

구간화는 같은 구간 안의 차이를 감추는 표시 방식이다. 좌표 측정이나 원래 GDI 예측 오차를 줄인 결과는 아니다. 한 구간 이내 비율 역시 임상 허용오차가 아니다. 실제값에서 가장 가까운 10의 배수 경계까지의 거리를 1점 간격으로 나누어 `boundary_distance.csv`에 남겼다. 이 구분은 경계 효과를 살펴보기 위한 설명용이며 채택 기준이 아니다.

| 가장 가까운 경계까지 거리 | 포함 사람 | 기록·좌우 값 | 앞 구간 불일치율 |
|---|---:|---:|---:|
{boundary_table}

거리 구간은 하한 포함·상한 제외이며 마지막 구간은 5점도 포함한다. 한 사람은 여러 거리 구간에 들어갈 수 있다. 경계에 가까울수록 일관되게 오류가 커지는 단순한 관계는 보이지 않는다. 따라서 현재 오류를 경계 문제만으로 설명하지 않는다. 이 분석으로 경계 효과가 없다고 확정하는 것도 아니다.

![앞 구간의 혼동표](04_bin_confusion.png)

혼동표 칸은 406개 기록·좌우 값의 단순 개수다. 위 표의 사람가중 일치율과 집계 기준이 다르다. `confusion.csv`에는 같은 칸의 사람 동일 가중치도 함께 저장했다.

현재 보여줄 수 있는 것은 **관절 특징에서 GDI를 예측하고 고정 구간으로 표현하는 절차와 그 오류 양상**이다. 앱의 독립적인 보행능력 등급으로 사용할 수 있는지는 별도 판단이 필요하다.

## 6. 검증과 한계

- 175명·203기록·5묶음과 묶음별 학습140명/평가35명, 사람 겹침 0을 검사했다.
- 묶음별 표준화 평균·크기를 학습 기록에서 다시 계산했다. 저장 Ridge 계수의 alpha1 정규방정식과 학습 정답 대응을 검사했다. 새 모델을 맞춘 것이 아니라 기존 계수와 학습 자료의 관계를 검산했다.
- 저장된 평가 예측과 같은 묶음의 특징·계수로 재계산한 예측의 최대 차이는 {replay_error:.3g}점이다. 기존 앞·뒤 MAE와 앞뒤 차이를 재현했다.
- 구간 혼동표 합계406, 오류 유형의 사람가중 비율 합계1, 누락·무한값 없음과 원본 입력 지문의 실행 전후 일치를 검사했다.
- 보호된 기존37/38명·새223명·새520명 자료는 읽지 않았다. 새 학습·GPU·서버 접속·MotionBERT 재추론은 없다.
- 반복 탐색한 개발자료의 결과다. 독립 최종 성능, 임상 유효성, 고령자 가정 촬영 성능으로 일반화하지 않는다.
- 각 기록의 GDI 정답을 앞·뒤 짧은 구간에 공통으로 비교했다. 구간 자체의 별도 임상 정답이 있다는 뜻은 아니다.

## 7. 다시 실행하는 방법과 파일

Python 3.12와 NumPy·Matplotlib을 사용한다. 이 환경의 기존 plotting dependency를 읽기 전용으로 재사용한다. 모든 출력과 Matplotlib 캐시는 이 폴더 안에 둔다.

```powershell
& 'C:/Users/jun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 'D:/Projects/D_P/gdi-research-harness/studies/20261030-score-analysis/analyze.py'
```

- `summary.json`, `distribution.csv`, `metrics.csv`: 전체 집계와 방법.
- `score_bands.csv`, `confusion.csv`, `boundary_distance.csv`: 점수대·구간·경계 분석.
- `fold_audit.csv`, `verification.json`: 사람 분리·계수·예측·파일 점검.
- `01`~`04` PNG: 분포, 실제–예측 관계, 점수대별 오류, 구간 혼동표.
- `analyze.py`: 계산과 그림·보고서 생성 코드. `STATUS.md`: 현재 상태.
'''
    (OUT/'report.md').write_text(report, encoding='utf-8')
    after = {name: sha(BUNDLE/name) for name in names}
    assert before == after
    verification = {'passed': True, 'people': 175, 'records': 203, 'sides': 2, 'folds': audit,
                    'fold_people_disjoint': True, 'train_only_scalers_reproduced': True,
                    'ridge_alpha1_normal_equation_checked': True, 'baseline_replay_max_difference': replay_error,
                    'all_fold_predictions_covered': True, 'all_values_finite': True,
                    'confusion_count_per_model_window': 406, 'bin_probability_partition': True,
                    'inputs_unchanged': before == after, 'source_hashes': before,
                    'protected_data_read': False, 'new_training': False, 'new_pose_inference': False,
                    'python': sys.version, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__}
    dump('verification.json', verification)
    status('실행 중', '집계·수치 검산·그림4개·보고서 생성',
           '그림과 보고서의 최종 검토', '최종 검토 후 조율 채팅에서 결과 통합')
    print(json.dumps({'front': front, 'back': back, 'mean': mean, 'distribution': dist, 'replay_error': replay_error}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
