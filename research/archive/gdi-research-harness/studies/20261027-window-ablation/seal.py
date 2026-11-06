"""Seal the verified manual study; do not alter automatic-control evidence."""
import json
from datetime import datetime,timezone,timedelta
from prepare import HERE,H,load,save,sha,require

require(not (HERE/'study_receipt.json').exists(),'Study already sealed')
require(load(HERE/'independent_verification.json')['verified'],'Missing verification')
require((HERE/'report.md').is_file(),'Missing readable report')
review=H/'RESEARCH_REVIEW.md'
old=review.read_text(encoding='utf-8')
save(HERE/'previous_review_pointer.json',{'sha256':sha(review),'historical_session':'20261027T065618Z-8aab63'})
stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M')
intro=f'''# 보행 점수 연구 — 최신 사용자 요청 결과

갱신: {stamp}(한국 시간).

**원인 비교를 마쳤고, 사용자 선택에 따라 새 후보 743명을 개발 520명·최종 평가 예약 223명으로 정답 개봉 전에 분리했다.** 준비 입력은 각각 861건·378건이다. 유효 GDI 정답을 연결한 평가 인원이 223명이라는 뜻은 아니다. 새 정답은 아직 열지 않았다.

기존 검증 37명과 내부 175명에서 좌표 조정 효과와 짧은 입력의 개선 크기가 달랐다. ‘짧으면 항상 정확하다’고 결론내리지 않는다. 243장 전체로 좌표 크기를 구한 앞 124장 진단 조건은 실제 124장만 있는 새 영상에 바로 쓸 수 없다.

[이번 계획](studies/20261027-window-ablation/plan.md), [실제 결과·실패·검증과 평가 확대](studies/20261027-window-ablation/report.md), [새 사람 분리 계획](studies/20261027-window-ablation/new_cohort_plan.md)에 기록했다. 다음은 고정된 목록을 유지한 정답 연결·누락 확인 절차 준비다.

아래 자동 연구 단계의 ‘두 안 선택 대기’는 당시 기록이다. 이번에 개발용·최종 평가용 분리 방향은 선택됐다. 기존 자동 감독 프로그램·설정·완료 증거는 수정하지 않았다. 앞서 만든 클라우드 이전 ZIP에도 이번 연구 결과와 새 분리 목록은 아직 반영하지 않았다.

## 이전 자동 연구 단계 기록

'''
text=intro+old
review.write_text(text,encoding='utf-8')
(HERE/'review_snapshot.md').write_text(text,encoding='utf-8')
save(HERE/'study_summary.json',{'completed':True,'old_validation_people':37,'internal_people':175,
    'new_development_candidate_people':520,'new_final_reserved_people':223,
    'new_final_GDI_labels_confirmed':False,'new_final_evaluation_executed':False,
    'failed_preparation_preserved':True,'independent_verification_passed':True,
    'automatic_scheduler_state_changed':False,'cloud_bundle_refreshed':False})
artifacts=[{'path':p.relative_to(H).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='study_receipt.json']
save(HERE/'study_receipt.json',{'kind':'manual_user_requested_study','finished':datetime.now(timezone.utc).isoformat(),
    'status':'completed','artifacts':artifacts,
    'note':'Direct user-requested study, separate from the waiting automatic supervisor.'})
require(review.read_bytes()==(HERE/'review_snapshot.md').read_bytes(),'Snapshot mismatch')
print(json.dumps({'status':'completed','artifact_count':len(artifacts),'reserved_final_people':223,
                  'new_final_evaluation_executed':False},ensure_ascii=False))
