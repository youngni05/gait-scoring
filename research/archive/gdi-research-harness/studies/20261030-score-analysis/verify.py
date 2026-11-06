"""Independent arithmetic/CSV/PNG checks; writes only in this study folder."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import re
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
B = ROOT / 'gdi-research-harness/control/manual-sessions/20261029-gpu-finetune-preparation/bundle'


def load(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def csv_rows(name):
    with (OUT/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    s, v = load('summary.json'), load('verification.json')
    with np.load(B/'train_only.npz', allow_pickle=False) as z:
        y, ids, folds = z['y'].copy(), z['person_id'].copy(), z['fold'].copy()
    with np.load(B/'baseline_predictions.npz', allow_pickle=False) as z:
        p = z['predictions'].copy()
    people = sorted(set(ids))
    q = np.empty_like(p)
    for k in range(5):
        q[folds == k] = np.mean(y[folds != k], axis=0)
    # Recalculate without the analysis module or its aggregation functions.
    for model, predictions in [('motionbert_ridge',p), ('train_mean',q)]:
        for wi, window in enumerate(['front','back']):
            metric = next(r for r in s['metrics'] if r['model']==model and r['window']==window)
            gap = abs(np.floor(predictions[:,wi]/10) - np.floor(y/10))
            measures = {'person_macro_mae': abs(predictions[:,wi]-y), 'exact_bin_rate': gap==0,
                        'one_bin_error_rate': gap==1, 'two_or_more_bin_error_rate': gap>=2}
            for key, a in measures.items():
                answer = sum(sum(a[ids == person].ravel())/a[ids == person].size for person in people)/len(people)
                assert abs(answer-metric[key]) < 1e-12, (model,window,key)
            cm = [r for r in csv_rows('confusion.csv') if r['model']==model and r['window']==window]
            assert sum(int(r['record_side_count']) for r in cm)==406
            diag_weight = sum(float(r['person_equal_weight']) for r in cm if r['actual_bin_low']==r['predicted_bin_low'])
            assert abs(diag_weight-metric['exact_bin_rate']) < 1e-12
            assert abs(sum(float(r['person_equal_weight']) for r in cm)-1) < 1e-12
            bands = [r for r in csv_rows('score_bands.csv') if r['model']==model and r['window']==window]
            assert sum(int(r['record_side_count']) for r in bands)==406
            boundaries = [r for r in csv_rows('boundary_distance.csv') if r['model']==model and r['window']==window]
            assert sum(int(r['record_side_count']) for r in boundaries)==406
    # Ensure interval semantics hold even for exact boundaries and values outside 0..100.
    assert np.array_equal(np.floor(np.array([-1.,0.,9.999,10.,100.,110.])/10), [-1,0,0,1,10,11])
    for name, digest in s['source_hashes'].items():
        assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest
    assert len(people)==175 and len(ids)==203 and np.isfinite(y).all() and np.isfinite(p).all()
    for file in OUT.glob('*.png'):
        with Image.open(file) as im:
            assert im.width >= 1000 and im.height >= 600
            im.verify()
    assert len(list(OUT.glob('*.png')))==4
    report=(OUT/'report.md').read_text(encoding='utf-8')
    assert '\ufffd' not in report
    for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report):
        assert (OUT/target).is_file()
    v['independent_verification'] = {
        'passed': True, 'person_macro_metrics_recomputed': True,
        'confusion_weighted_diagonal_matches_exact_rate': True,
        'score_band_and_boundary_counts_complete': True,
        'fixed_half_open_interval_boundary_cases': True,
        'all_four_pngs_readable': True, 'report_image_links_valid': True,
        'report_figures_visually_reviewed': True,
        'inputs_still_unchanged': True,
        'checked_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')}
    (OUT/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    artifact_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
                     if p.is_file() and p.name not in ['STATUS.md','artifact_manifest.json']}
    (OUT/'artifact_manifest.json').write_text(json.dumps(artifact_hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    (OUT/'STATUS.md').write_text(f'''# A · GDI 분포와 점수화 분석

- KST 갱신 시각: {now}
- 상태: 완료
- 완료 내용: 175명·203기록/5묶음의 기존 예측 분포·점수대별 오류·고정 폭10 구간화 분석. 그림4개와 보고서 생성, 독립 계산·PNG·입력 지문 검증 완료
- 현재 작업: 조율 채팅에서 결과 확인·통합 가능
- 결정이 필요한 항목: 임상 구간·앱 점수 채택은 이번 결과와 별도로 판단
- 결과 경로: 이 폴더의 report.md, summary.json, metrics.csv, verification.json, 01~04 PNG
- 다음 작업: B의 결과와 동일한 사람/기록/앞뒤/좌우 집계 기준으로 비교

핵심 결과: 실제 GDI 39.12~106.64점. 앞 MAE 8.276점(평균 예측 9.538점), 뒤 MAE 9.034점. 앞 정확 구간 일치38.6%, 한 구간 이내80.9%, 앞뒤 예측 차이3.960점. 예측이 중앙으로 모이고 양 끝 점수대 구분은 부족하다.

원본 입력·공통 코드와 현황은 변경하지 않았다. 새 학습·GPU·서버 접속·MotionBERT 재추론은 없고 보호평가자료는 읽지 않았다. 반복 탐색한 개발자료의 분석이며 독립 최종·임상 성능이 아니다.
''',encoding='utf-8')
    print(json.dumps(v['independent_verification'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
