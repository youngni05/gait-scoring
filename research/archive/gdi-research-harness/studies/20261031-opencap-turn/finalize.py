"""Write the Korean study record from verified saved results; never retrain."""
from common import *
import platform
from collections import Counter
import sklearn, scipy, joblib

def main():
 checks={n:json.loads((HERE/n).read_text()) for n in ['input-verification.json','geometry-input-verification.json','results-verification.json']}
 assert all(x['passed'] for x in checks.values())
 assert checks['results-verification.json']['models_replayed']==1425
 cohorts=['event_matched','fixed6','event_flagged','geometry']
 summaries={c:json.loads((HERE/c/'summary.json').read_text()) for c in cohorts}
 s=summaries['geometry'];m=json.loads((HERE/'geometry/manifest.json').read_text());plan=json.loads((HERE/'geometry/splits.json').read_text());d=dict(np.load(HERE/'geometry/input.npz'))
 a=pd.read_csv(HERE/'geometry/person-results.csv');long=a[a.target>=20];duration=np.array([r['matched_input_seconds'] for r in m])
 links={r['file']:r for r in json.loads((HERE/'linkage-audit.json').read_text())}
 short_records=[r['id'] for r in m if links[r['turn_file']].get('recording_shorter_than_label_by_gt1s',False)]
 ix=int(np.flatnonzero(d['ids']=='sub-283')[0]);ceilings=[float(d['y'][f['train']].max()) for rep in plan for f in rep['splits'] if ix in f['test']]
 assert len(ceilings)==5
 extra={'geometry_matched_input_seconds':{'min':float(duration.min()),'median':float(np.median(duration)),'max':float(duration.max())},'geometry_task_counts':dict(Counter(r['task'] for r in m)),'geometry_recording_shorter_than_visit_label_by_gt1s':short_records,'sub283_training_target_max_per_partition':ceilings}
 save('additional-audit.json',extra)
 names={'OldWalk3Speed':'기존 구간 3초 속도','Walk3Speed':'구간 점검 후 3초 속도','WalkMatchedSpeed':'관찰시간을 맞춘 직선 속도','Walk3TurnTime':'직선 3초 + 회전 구간 시간','Walk3TurnShape':'직선 3초 + 회전 특징(시간 항목 제외)','Walk3Turn':'직선 3초 + 회전 특징 전체'}
 model_rows='\n'.join(f"| {label} | {s['models'][n]['all']['MAE']:.3f} | {s['models'][n]['ge20']['MAE']:.3f} | {s['models'][n]['ge20']['bias']:.3f} | {s['models'][n]['ge20']['underprediction']:.3f} |" for n,label in names.items())
 base=long[long.method=='WalkMatchedSpeed'].set_index('id');turn=long[long.method=='Walk3Turn'].set_index('id');old=long[long.method=='OldWalk3Speed'].set_index('id');clean=long[long.method=='Walk3Speed'].set_index('id')
 people_rows='\n'.join(f"| {pid} | {base.loc[pid,'target']:.2f} | {base.loc[pid,'prediction']:.2f} | {turn.loc[pid,'prediction']:.2f} | {base.loc[pid,'MAE']-turn.loc[pid,'MAE']:+.2f} |" for pid in sorted(base.index))
 labels={'event_matched':'각도로 찾은 전체 회전(주 비교)','fixed6':'고정 6초(별도 비교)','event_flagged':'각도 품질 기준 완화(민감도)','geometry':'이동 경로 기반(추가 보조 비교)'}
 cohort_rows='\n'.join(f"| {labels[c]} | {summaries[c]['models']['Walk3Turn']['all']['people']} / {summaries[c]['models']['Walk3Turn']['ge20']['people']} | {summaries[c]['models']['WalkMatchedSpeed']['all']['MAE']:.3f} → {summaries[c]['models']['Walk3Turn']['all']['MAE']:.3f} | {summaries[c]['models']['WalkMatchedSpeed']['ge20']['MAE']:.3f} → {summaries[c]['models']['Walk3Turn']['ge20']['MAE']:.3f} |" for c in cohorts)
 pair=s['comparisons']['Walk3Turn minus WalkMatchedSpeed / all'];tail=s['comparisons']['Walk3Turn minus WalkMatchedSpeed / ge20'];time_pair=s['comparisons']['Walk3Turn minus Walk3TurnTime / all']
 versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'joblib':joblib.__version__,'execution':'local Windows CPU; 4 independent fold processes; 1 RF thread each'}
 save('environment.json',versions)
 report=f'''# OpenCap 직선 보행·방향 전환 비교 — 2026-10-31

관찰시간과 사람 분할을 맞춘 비교에서 방향 전환 정보를 추가하면 실제 TUG 시간 예측 오차가 줄었다. 긴 TUG 9명을 모두 포함한 이동 경로 기반 보조 실험에서 전체 오차는 2.114초에서 1.118초로, 긴 사람들의 오차는 11.571초에서 7.592초로 줄었다. 다만 42.59초 사례의 예측은 19.88초로 여전히 크게 낮다. 회전 소요시간만 추가해도 전체 오차가 1.156초이므로 복잡한 움직임 특징의 독립적인 추가 효과는 아직 뚜렷하지 않다.

## 모델이 답하는 질문과 실제 점수 연결

질문은 “이 사람이 직선 보행과 방향 전환에서 보이는 움직임으로, 같은 방문에서 기록된 실제 TUG 시간을 얼마나 예측할 수 있는가?”이다. 입력은 OpenCap의 `.mot` 위치·관절 시계열에서 계산한 움직임 요약값, 정답은 공개 임상표 `participant_info.csv`의 `tft_tug`(초)이다. AI가 만든 점수를 정답으로 쓰지 않았다.

415명 공개자료 중 기존 실험의 298명·선택 방문을 고정했다. 그중 동일인·동일 방문의 TUG 파일이 있는 사람은 276명이며 356개 파일을 확보했다. 전부 ZIP CRC와 SHA256을 검산했다. 사람과 방문을 함께 연결한 임상값은 기존 정답과 일치한다. 최초 각도 탐지에서 353개 파일이 통과했고, 1개는 원본 자체의 불완전/비유한 마지막 행, 2개는 각도 기반 중간 회전 탐지 실패였다. 후자의 두 사례는 이동 경로에서는 회전이 확인된다.

다만 **임상표의 한 TUG 시간이 cone/line 중 어느 시행을 재었는지는 공개 파일만으로 확정하지 못했다.** 이동 경로 실험에서 사용한 과제는 cone {extra['geometry_task_counts'].get('cone',0)}명·line {extra['geometry_task_counts'].get('line',0)}명이다. {', '.join(short_records)}의 전체 시계열은 방문 임상시간보다 1초 이상 짧아 원본 촬영·시행 대응을 추가 확인해야 한다. 임상 정답에 맞춰 파일을 고르거나 이 차이를 이유로 정답을 수정하지 않았다. 정확한 시행 대응이 확인된 결과로 해석해서는 안 된다.

자료 출처: [공개 OpenCap 자료](https://zenodo.org/records/22309771). 원본과 임상표 해시는 `protocol.json`, 개별 연결은 `linkage-audit.json`에 기록했다.

## 회전 구간 품질과 추가 실험을 만든 이유

먼저 골반 회전각의 20→160도 변화 주변을 찾고, 시간 누락·위치 급변·각도 급변을 점검했다. 위치속도 6 m/s, 골반각 한 프레임 변화 30도, 기타 관절각 변화 45도를 상한으로 둔 경험적 선별 기준이다. 각도 기준 주 비교에는 227명, 긴 TUG는 4명만 남았다. 제외된 어려운 사례를 숨기지 않도록 품질 기준을 완화한 민감도 비교도 따로 수행했다.

긴 TUG 9명의 원본 수치 그래프를 모두 살펴보니 sub-266·sub-406은 골반각이 누적해서 돌아가더라도 위치 궤적에는 왕복 회전이 있었다. sub-283의 기존 직선 입력 초반에는 30 m/s를 넘는 위치 급변이 있었다. 이 입력 점검 후, 최초 모델 성능을 확인하기 전에 이동 경로 기반 보조 실험을 명시했다. 결과가 나쁜 사람만 다른 정답이나 방문으로 바꾸지 않았다.

보조 실험은 시작 위치에서 가장 멀리 간 방향을 찾고, 진행방향 최원점의 0.75 m 이내로 들어왔다 나가는 구간을 선택한다. 따라서 접근·이탈이 포함된 **회전 주변 구간**이며, 사람이 정답으로 표시한 순수 회전 구간은 아니다. 시간 간격 0.05초 이하, 왕복 이동 범위 1.5 m 이상, 출발 방향으로 복귀, 선택 구간의 위치속도 6 m/s 이하 등을 검사한다. 사용하지 않는 관절각 오류 때문에 위치 정보를 버리지는 않는다.

직선 보행은 기존 파일·방문을 유지하고, 기존 시작점이 기준을 만족하면 그대로 쓴다. 그렇지 않으면 같은 파일에서 조건을 만족하는 가장 이른 구간을 쓴다. 직선성은 평활화한 순이동거리/경로길이 0.85 이상으로 확인했다. 이 조건을 통과한 사람은 238명이며 긴 TUG 9명이 모두 포함된다. 시작점이 바뀐 사람은 10명이다. 같은 새 시작점으로 3초 모델과 긴 직선 모델을 함께 다시 학습해 구간 교체와 회전 추가 효과를 구분했다.

긴 9명의 최종 위치 궤적·선택 구간도 이미지로 확인했다. sub-283의 교체 구간에도 최고 위치속도 4.23 m/s가 남으므로 잡음이 모두 제거됐다고 할 수 없다. 원본 RGB와 독립적인 위치 정답은 확인하지 못했다. 이 작업은 수치의 타당성 점검이며 카메라 측정 정확도 검증은 아니다. 자세한 기록은 `geometry-review-notes.json`, 그림은 `geometry-quality-1.png`~`3.png`에 있다.

## 관찰시간과 사람 분리

회전 구간 길이를 D초라 하면 **직선만 3+D초**와 **직선 3초+회전 D초**를 비교했다. 이동 경로 실험의 총 입력시간은 {duration.min():.2f}~{duration.max():.2f}초, 중앙값 {np.median(duration):.2f}초이다. 늘어난 관찰시간 자체의 효과를 구분하기 위해 직선 3초도 함께 비교했다. 고정 길이 검증으로 직선 6초와 직선 3초+회전 중심 3초를 별도 비교했다. 고정 3초 회전창에는 느린 사람의 회전 전체가 담기지 않을 수 있다.

직선은 10m 보행 기록, 회전은 같은 방문의 별도 TUG 기록이다. 전체 기록으로 구간 위치를 찾은 뒤 선택한 움직임만 모델에 전달한다. 따라서 위 시간은 **모델 입력의 합계**이며 실제 필요한 촬영시간이나 연속 한 영상의 길이가 아니다. 원본 OpenCap 처리도 전체 기록을 이용한다. TUG 전체 소요시간·기록 길이·구간의 절대 시각·진단·나이는 모델에 넣지 않았다.

기존 5가지 사람 분할을 그대로 제한 적용하고, 각 분할의 5개 평가 묶음에서 같은 사람이 학습과 평가에 섞이지 않게 했다. 모델 초기값 3개를 바꿔 총 15회 반복했다. 각 비교군 안에서는 모델들이 같은 사람·정답·평가 묶음을 쓴다. Random Forest 128개 나무, 잎의 최소 자료 수 2 또는 5를 학습 사람 내부의 3묶음 검증으로 선택했다. 표준화는 학습 사람으로만 계산했다. 모델별 전체 관절·특징 수는 다르므로 아래 결과만으로 회전 위치 하나의 순수 인과효과를 증명하지 않는다.

## 긴 TUG 9명을 포함한 238명 결과

단위는 초이다. 평균 절대오차는 각 15회 실행의 오차를 평균한 값이다. 평균 예측을 먼저 만든 뒤 오차를 계산한 값이 아니다. `편향=예측-실제`이므로 음수면 과소예측이다. 과소예측량은 `max(실제-예측, 0)`의 평균으로, 과대예측이 이를 상쇄하지 않는다. 20초 기준은 이번 분석의 묶음 기준이며 임상 판정선이 아니다.

| 입력 | 전체 평균 절대오차 | 20초 이상 9명 오차 | 9명 편향 | 9명 과소예측량 |
|---|---:|---:|---:|---:|
{model_rows}

관찰시간을 맞춘 직선 대비 회전 추가의 전체 오차 변화는 {pair['MAE_difference']:.3f}초이며, 사람 단위 재표집 95% 구간은 [{pair['conditional_person_bootstrap95'][0]:.3f}, {pair['conditional_person_bootstrap95'][1]:.3f}]초이다. 긴 9명의 변화는 {tail['MAE_difference']:.3f}초, 구간은 [{tail['conditional_person_bootstrap95'][0]:.3f}, {tail['conditional_person_bootstrap95'][1]:.3f}]초이다. 전체·긴 사람 평균 모두 15회 전부 개선 방향이었다. 개인별로는 전체 187명 개선·51명 악화, 긴 사람 8명 개선·1명 악화였다.

이 구간은 이미 학습한 모델의 사람별 오차를 재표집한 조건부 범위이다. 매번 다시 학습하거나 독립 외부 집단을 검사한 구간이 아니다. 15회도 서로 독립된 15개 연구가 아니며, 긴 사람이 9명이라는 한계는 그대로다.

회전 시간만 추가한 모델과 회전 특징 전체 모델의 차이는 {time_pair['MAE_difference']:.3f}초, 조건부 95% 구간 [{time_pair['conditional_person_bootstrap95'][0]:.3f}, {time_pair['conditional_person_bootstrap95'][1]:.3f}]초로 0을 포함한다. 복잡한 모양 정보를 더 넣어야 유리하다고 결론 내리기 어렵다. 시간 항목을 제외한 모델에도 속도·경로길이처럼 시간과 연관된 정보가 남는다. 이를 “시간과 무관한 균형 능력만 학습했다”는 증거로 쓰면 안 된다.

## 원래 정한 비교와 민감도 결과도 함께 보기

표 안의 화살표는 각 행의 동일인·동일 입력시간에서 직선 속도 모델→회전 추가 모델이다. 행 사이에는 사람과 구간 정의가 달라 우열을 직접 비교하지 않는다.

| 비교 | 전체/긴 사람 수 | 전체 오차 | 긴 사람 오차 |
|---|---:|---:|---:|
{cohort_rows}

각도 기반 주 비교의 관찰시간을 맞춘 전체 관절 요약 모델은 1.799초, 고정 6초에서는 2.267초였다. 속도 모델보다 분명히 좋아지지 않았으며 회전 추가의 개선을 설명하지 못했다. 품질 기준 완화 결과는 문제가 있는 관절 자료를 정당화하는 근거가 아니다.

## 긴 사람에게 실제로 무엇이 달라졌나

예측 열은 15회 평균이다. 마지막 열은 각 실행의 절대오차를 평균한 뒤 두 모델을 뺀 값으로, 양수면 개선이다.

| 사람 | 실제 TUG | 같은 시간 직선 예측 | 회전 추가 예측 | 오차 감소 |
|---|---:|---:|---:|---:|
{people_rows}

가장 어려운 sub-283(42.59초)은 동일한 238명 학습 조건에서 기존 3초 구간 {old.loc['sub-283','prediction']:.2f}초, 구간 점검 후 3초 {clean.loc['sub-283','prediction']:.2f}초, 관찰시간을 맞춘 직선 {base.loc['sub-283','prediction']:.2f}초, 회전 추가 {turn.loc['sub-283','prediction']:.2f}초였다. 입력 이상과 회전 단서 누락이 각각 오차와 관련된다는 증거지만 개인의 임상 원인을 확정하지는 못한다.

이 사람을 평가할 때 학습 정답의 최댓값은 분할에 따라 {min(ceilings):.2f}~{max(ceilings):.2f}초였다. 현재 평균을 내는 Random Forest는 학습 정답 범위 밖으로 예측하지 못한다. 따라서 42.59초처럼 학습 범위를 넘는 사람의 큰 과소예측은 입력을 늘리는 것만으로 해결되기 어렵다. 추가로 일어서기·착석 등 관찰하지 않은 TUG 단계의 영향도 따로 확인해야 한다.

## 이 결과가 의미하는 범위

직선 보행을 조금 더 길게 보는 것보다 방향 전환 주변의 시간과 움직임을 보는 편이 TUG 예측에 유용하다는 개발자료 근거가 생겼다. 특히 다음 후보는 “회전 소요시간+간단한 속도·경로 지표”이다. 복잡한 관절 모델의 필요성은 현재 결과로 확인되지 않았다.

회전은 예측 대상인 TUG 과제의 일부에서 가져왔다. 따라서 일반 생활 속 왕복 보행으로 독립 평가 점수를 추론한 검증과는 의미가 다르다. 과제 지시·속도·환경도 직선 기록과 다를 수 있다. SPPB 균형·기립 점수, 단일 카메라 RGB에서의 측정, 새로운 사람 집단에서의 성능, 10~30초 연속 촬영 방식의 실용성은 이번에 검증하지 않았다. 다음 실험의 우선순위는 정확한 계시 시행 연결, 독립 왕복 보행과 TUG의 대응, 긴 TUG 사람 확대, 기립·착석 단서 추가이다.

## 구현·재현·검증

`acquire.py`는 ZIP 일부 다운로드와 무결성 확인, `prepare.py`·`geometry.py`는 구간 선택 및 특징 생성, `train.py`·`train_geometry.py`는 공통 사람 분할 학습, `analyze.py`는 실행별·사람별 결과 집계를 담당한다. 이동 경로 방식을 추가한 이유는 각도 오류로 실제 왕복 회전까지 제외되는 문제를 줄이기 위해서다. 기존 실험 폴더와 기존 방문 선택은 변경하지 않았다.

검증 결과는 모두 통과했다. 356개 다운로드의 CRC/SHA, 실제 정답 연결, 기존 내부·외부 사람 분할 일치, 비교 입력시간 일치, 다른 계산 방식으로 만든 특징 일치를 확인했다. 저장 모델 **1,425개**를 다시 불러와 예측을 재현했고 최대 차이는 3.56e-15초 미만이었다. 학습 집합만의 표준화·나무 학습 정답, 내부 설정 선택, 실행별 지표와 재표집 범위도 독립 계산으로 점검했다. 이는 구현 검증이며 임상 검증은 아니다.

이미 확보한 입력에서 재현하는 순서는 `verify_inputs.py` → `verify_geometry.py` → 필요 시 `train.py`와 `train_geometry.py` → `analyze.py` → `verify_results.py` → `plot.py` → `finalize.py`이다. 학습은 Windows CPU에서 수행했다. 사용 Python은 `C:/Users/jun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`이며 기존 프로젝트 의존성 경로를 `common.py`에서 추가한다. 실제 버전은 `environment.json`에 기록했다. 새로운 패키지 설치나 외부 연산 업로드는 없었다.

핵심 산출물은 `geometry/summary.json`, `geometry/person-results.csv`, `geometry/long-people.csv`, `turn-comparison.png`이다. 초기 세 비교 결과도 각 폴더에 남겨두었다. 검증 근거는 `input-verification.json`, `geometry-input-verification.json`, `results-verification.json`, 최종 파일 목록과 해시는 `artifact-manifest.json`이다.
'''
 (HERE/'STATUS.md').write_text(report,encoding='utf8')
 for p in HERE.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
 for c in cohorts:
  completion=json.loads((HERE/c/'completion.json').read_text())
  assert completion['complete'] and sha(HERE/c/'input.npz')==completion['input_sha256']
  assert sha(HERE/'train.py')==completion['code_sha256']
  assert len(list((HERE/c/'models').glob('*.joblib')))==completion['models']
 assert all((HERE/f'geometry-quality-{k}.png').is_file() for k in [1,2,3])
 assert (HERE/'turn-comparison.png').is_file()
 assert '\ufffd' not in report
 files=[p for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json']
 save('artifact-manifest.json',{'files':{p.relative_to(HERE).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},'verification_passed':True,'models':1425,'code_compiled':len(list(HERE.glob('*.py')))})
 print(json.dumps({'finished':True,'files':len(files),'additional_audit':extra,'long_metrics':{n:s['models'][n]['ge20'] for n in ['WalkMatchedSpeed','Walk3Turn']},'environment':versions},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
