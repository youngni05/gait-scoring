"""Close this manual study with readable results and immutable evidence hashes."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import json
from audit_labels import HERE,H,ROOT,load,sha,save

def main():
    assert not (HERE/'study_receipt.json').exists()
    r=load(HERE/'transfer_result.json');v=load(HERE/'independent_verification.json')
    meeting=H/'reports/20261027-team-meeting'
    checks=load(meeting/'document_checks.json')
    assert r['verified'] and v['verified']
    assert checks['pages']==8 and checks['visual_review'].startswith('passed')
    assert sha(meeting/'GDI_실험정리_팀미팅_20261027.pdf')==checks['pdf_sha256']
    m=r['frozen124']['person_mae'];b=r['old_train_mean_baseline']['person_mae'];d=r['person_mae_reduction_vs_trainmean'];ci=r['paired_bootstrap95']
    now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    text=f'''

## 2단계 완료: 실제 결과

| 같은 새 개발 520명·861건에서 비교 | 사람별 평균 절대 오차 |
|---|---:|
| 기존 학습 175명·203건의 좌우 평균만 예측 | {b:.6f} |
| 기존 고정 124장 모델 | {m:.6f} |
| 단순 평균 대비 오차 감소 | {d:.6f} |

작을수록 좋은 수치다. 단순 평균 대비 감소량의 95% 재표집 범위는 {ci[0]:.6f}~{ci[1]:.6f}다. 평균 예측보다 개선된 사람은 {r['people_improved']}명, 악화된 사람은 {r['people_worsened']}명, 같은 사람은 {r['people_equal']}명이다. 520명을 같은 비중으로 평가했고, 좌우와 반복 기록은 사람 안에서 먼저 평균했다. 정확도 백분율이나 진단 일치율이 아니다.

이번 수치는 **새 개발 집단에서 고정 모델과 단순 평균을 비교한 결과**다. 기존 검증 37명 수치와 직접 빼지 않는다. 새 집단에 243장 입력이 없으므로 프레임을 줄이면 더 정확해진다는 가설을 직접 검증한 실험이 아니다. 같은 배포처의 다른 사람이며 외부 기관·고령자 가정 촬영·임상 유용성 검증이 아니다. 개발 520명이라는 역할도 그대로 유지한다.

## 무엇을 검증했는가

- 기존 검증 37명의 고정 모델 오차 7.2882633764가 재현됐다.
- 새 개발 520명·861건 전부에서 누락 없는 예측을 생성했다. 새 학습·파인튜닝·표준화 재학습·설정 탐색은 0회다.
- 입력 1,239건과 새 개발 특징 861건의 SHA-256을 확인했다. 개발 정답 861건은 원본과 독립적으로 다시 연결했다.
- 예측 CSV를 별도 계산으로 다시 검산했고, 다른 행렬 계산 순서로도 모델 예측이 재현됐다.
- 학습·검증·새 개발·최종 평가 예약 사람의 겹침은 0이다. 최종 예약 223명은 숫자 정답·특징·예측·오차를 사용하지 않았다.
- 기존 기준·정답·사람 분리·운영 설정·앱·원본·과거 봉인 연구는 수정하지 않았다.

## 종료 이유와 다음 목표

기존 실험 정리, 새 정답 연결, 개발 전원 특징 추출, 고정 모델 진단, 독립 재검산을 한 단위로 마쳤다. 시간이 남았다는 이유로 같은 결과를 반복하거나 설정을 맞추지 않고 종료한다.

다음 목표는 개발 자료를 더 학습시키는 효과를 공정하게 비교하는 계획을 고정하는 것이다. 새 520명 일부를 예측 대상에서 빼놓고, 기존 175명만 학습한 경우와 새 개발 학습분을 보탠 경우를 같은 대상에서 비교하는 방안을 검토한다. 최종 예약 223명은 이 개발 선택을 끝내고 방법을 고정한 뒤 평가한다. 이 다음 비교는 아직 실행하지 않았다.

## 팀 미팅 문서와 보존

`reports/20261027-team-meeting/GDI_실험정리_팀미팅_20261027.pdf`의 8쪽은 기존 실험이다. 사용자가 먼저 검토할 수 있도록 공유한 PDF를 그대로 보존했다. 이번 후속 결과는 본 보고서로 별도 제공한다. PDF와 같은 폴더에 수정 가능한 한국어 원문과 근거 파일 지문을 보존했다. 개인별 정답·예측·좌표는 공유용 문서에 넣지 않았다.

완료 시각: {now}. 자동 배정기의 과거 대기 상태는 수정하지 않았고, 이 작업은 사용자의 직접 요청에 따른 수동 연구다.
'''
    p=HERE/'report.md';p.write_text(p.read_text(encoding='utf-8')+text,encoding='utf-8')
    review=H/'RESEARCH_REVIEW.md';previous=review.read_text(encoding='utf-8')
    (HERE/'prior_review.md').write_text(previous,encoding='utf-8')
    overview=f'''# 보행 점수 연구 — 최신 결과

갱신: {now}.

**기존 실험을 팀 미팅용 8쪽으로 정리했고, 새 개발 520명·861건의 정답 연결과 고정 모델 적용·검산을 마쳤다.** 새 개발 전원의 좌우 GDI가 유일하고 유효하게 연결됐다. 기존 175명으로 만든 124장 모델의 사람별 오차는 {m:.4f}, 기존 학습 평균만 예측한 오차는 {b:.4f}다. 같은 새 개발 집단에서의 비교이며 최종 평가나 243장 대 124장 우월성의 확인이 아니다.

목적은 보행 움직임에서 GDI를 예측할 수 있는지와 개선 가능한 부분을 찾는 것이다. 관절 좌표 → 고정 AI 특징 → 기존 Ridge 예측 → 사람별 오차 비교를 유지했다. 새 학습·파인튜닝·설정 탐색을 하지 않았고, 데이터가 늘었다는 이유로 사람 분리·임상 기준·정답·평가 지표를 바꾸지 않았다.

최종 평가 예약 223명·378건은 정답 행 존재만 확인했다. 숫자 유효성·분포·특징·예측·오차는 분석하지 않았다. 최종 평가 성능은 아직 모른다. 기존 평가 38명도 이번 작업에는 사용하지 않았다.

이번 실행은 정리→연결→특징→고정 모델 진단→독립 검산까지 한 단위를 끝내 종료했다. 다음은 새 개발 자료를 학습에 보탰을 때의 효과를 같은 예측 대상에서 비교하는 계획을 고정하는 것이다. 그 다음 실험은 아직 수행하지 않았다.

[팀 미팅 원문](reports/20261027-team-meeting/GDI_실험정리_팀미팅_20261027.md), [실제 결과·실패·한계](studies/20261027-gdi-linkage/report.md), [정답 연결 계획](studies/20261027-gdi-linkage/plan.md), [고정 모델 적용 계획](studies/20261027-gdi-linkage/transfer_plan.md).

자동 감독 프로그램·설정·과거 증거·원본·기존 앱은 수정하지 않았다. 기존 클라우드 이전 ZIP에는 이번 결과가 아직 추가되지 않았다. 아래는 당시 상태를 보존한 이전 기록이다.

## 이전 연구 단계 기록

'''+previous+'\n<!-- session:manual-20261027-gdi-linkage -->\n'
    review.write_text(overview,encoding='utf-8')
    (HERE/'review_snapshot.md').write_text(overview,encoding='utf-8')
    save('response.json',{'status':'completed','kind':'user_requested_manual_research','finished':now,
        'completed':['team meeting report','frozen cohort GDI linkage','development520 frozen model transfer','independent verification'],
        'result':{'development_people':520,'development_records':861,'frozen124_person_mae':m,'trainmean_person_mae':b},
        'artifacts':['report.md','review_snapshot.md','transfer_result.json','independent_verification.json',
                     '../../reports/20261027-team-meeting/GDI_실험정리_팀미팅_20261027.pdf'],
        'stop_reason':'Completed the predeclared coherent research unit; no repeated tuning or final-test access.',
        'next_goal':'Predeclare fair training-expansion comparison; keep reserved final223 protected.',
        'automatic_supervisor_modified':False})
    paths=[p for p in HERE.rglob('*') if p.is_file()]
    paths += [meeting/'GDI_실험정리_팀미팅_20261027.pdf',meeting/'GDI_실험정리_팀미팅_20261027.md',
              meeting/'document_checks.json',meeting/'evidence_index.json',meeting/'build_report.py']
    artifacts=[{'path':p.relative_to(H).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)]
    save('study_receipt.json',{'kind':'manual_study','finished':now,'status':'completed','artifacts':artifacts,
                             'note':'No automatic claim or legacy receipt edited.'})
    assert all(sha(H/r['path'])==r['sha256'] for r in artifacts)
    print(json.dumps({'status':'completed','artifacts_verified':len(artifacts),'frozen124_person_mae':m,'trainmean_person_mae':b}))

if __name__=='__main__':main()
