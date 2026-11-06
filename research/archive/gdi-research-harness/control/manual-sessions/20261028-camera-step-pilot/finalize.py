import json,hashlib
from pathlib import Path
S=Path(__file__).parent;H=S.parents[2]
def write(name,text): (S/name).write_text(text.strip()+'\n',encoding='utf-8')
for name in ['APP_DEVELOPMENT.md','AI_RESEARCH.md','RESEARCH_REVIEW.md']:
 (S/('before_'+name)).write_bytes((H/name).read_bytes())
app='''# 앱 개발 현재 상태 — 카메라 연결 준비 완료

2026-10-28 직접 요청. app-development/camera-preparation에 관리 앱의 현재 소스·설정과 기존 카메라·관절 추정 모듈을 준비했다. 원본95개 파일이 변하지 않았음을 확인했다. 새 Git 작업트리나 독립 배포본은 아니다.

전체 웹 타입 검사와 기존 검사132개가 통과했다. 처음에는 Supabase 의존성 경로 때문에 타입 검사와 테스트 수집이 실패했고, 사본 검사 설정에서 기존 설치 경로를 연결해 해결했다. 새 검사 기준을 만들거나 완화하지 않았다.

준비된 흐름: 카메라 → MediaPipe → 프레임 시각·관절 좌표 또는 미검출. prepared-camera/index.ts가 화면의 연결 지점이다. 아직 화면 연결·실제 카메라 실행·원본 영상 녹화·보행 기록 저장은 하지 않았다. 모의 측정과 SPPB 저장 흐름은 사본에 남아 있다.

다음 목표: 사본의 MediaPipe 의존성·잠금 파일·WASM 자산을 준비하고, 모의 측정과 구분되는 미리보기·관절 표시 화면 연결을 진행한다. 실제 사용자 카메라 사용·서버 저장은 별도 확인 범위다. 대상자 연결·영상 보관·측정값 저장 정책은 미정이다.

[이번 앱 준비와 검사](control/manual-sessions/20261028-camera-step-pilot/app_report.md)
'''
ai='''# AI 연구 현재 상태 — 실제 공개 영상에서 짧은 입력 실험 수행

2026-10-28. TNT15의 한 사람·한 걷기를 동시에 찍은 두 영상(각10.22초)을 확보하고 MediaPipe로 추론했다. 별도의2초·4초 입력6회도 실행했다. 같은4초에서 단순 발 간격 기반 검출은 카메라03/06 각각 후보5개/2개였다. 이를 실제 걸음 수라고 확정하지 않는다.

짧은 구간 끝에서 후보가 사라지는 현상을 확인했다. 전체 좌표를 먼저 자른 대조와 짧은 원본 입력의 새 추론이6조건 모두1프레임 이내로 일치해, 이번 차이는 주로 검출기의 구간 경계에서 설명됐다. 중간 관측 누락을 가로지르는 후보 간격을 실제 걸음 간격으로 표시하면 안 된다.

정답 대비 정확도는 아직 미완료다. 예측 전 원본에서 AI가 판독한 두 접촉은 임시 참고이며 사람·전문가·제공자 정답이 아니다. 이를 임상 성능이나 정확도100%로 바꾸지 않는다. 한 사람의 한 시도는 일반화·고령자·재촬영 안정성을 입증하지 않는다. Python실행과 웹앱실행도 다르다.

다음 목표: 확보 영상의 발 접촉 주석을 사람이 확인할 수 있게 준비하거나 제공자 사건 정답이 있는 전신 영상 자료를 확보한다. 발별 접촉 검출 선행연구와 필요한 시점·주석 조건을 비교한다. 정답 없이 이 한 영상에 임계값을 반복 맞추지 않는다. 새 제품 측정 거절·부분 결과 기준은 정하지 않는다. 기존 평가38명·최종223명은 사용하지 않았다.

[실험 방법·실제 수치·한계](control/manual-sessions/20261028-camera-step-pilot/ai_report.md)
'''
for name,txt,snapshot in [('APP_DEVELOPMENT.md',app,'app_status_snapshot.md'),('AI_RESEARCH.md',ai,'ai_status_snapshot.md')]:
 (H/name).write_text(txt,encoding='utf-8');write(snapshot,txt)
summary='''# 최신 진행: 카메라 연결 준비와 실제 짧은 영상 실험 — 2026-10-28

**앱의 연결 부품을 별도 작업본에 준비했고, 연구는 실제 공개 영상에서 걸음 후보를 추출했다. 다만 정답 대비 정확도는 아직 확인하지 못했다.**

| 흐름 | 이번에 실제로 한 일 | 다음 작업 |
|---|---|---|
| 앱 개발 | 별도 관리 앱 소스 사본과 카메라·관절 추정 모듈 준비. 웹 타입 검사·기존132개 검사 통과 | 의존성·WASM 준비 후 미리보기·관절 표시 화면 연결 |
| AI 연구 | 한 걷기의 두 공개 영상 확보, 전체 추론과2초·4초 입력6회 비교 | 발 접촉 정답 확인과 시점·구간 경계에 강한 검출 방법 비교 |

같은4초 입력에서 카메라별 걸음 후보는5개/2개였다. 짧게 자르면 구간 끝의 후보를 확인할 이후 프레임이 부족해 후보 하나가 사라졌다. 전체 좌표를 먼저 자른 대조에서도 같은 현상이 나와, 이번 차이를 모델 학습 부족만으로 설명할 수 없다. 사람 검출·발 관측 가능성·실제 접촉 시각 정확성은 각각 다르다.

AI가 원본 화면에서 임시로 표시한 두 접촉과 비교는 했지만, 이를 제공자·사람 정답으로 취급하지 않는다. 현 단계에서 정확도100%, 임상 성능, 고령자 재촬영 안정성을 주장할 수 없다. 영상은 연구용으로만 보관하고 앱에 포함하지 않았다.

[앱 상세](control/manual-sessions/20261028-camera-step-pilot/app_report.md) · [연구 상세](control/manual-sessions/20261028-camera-step-pilot/ai_report.md)

이번 직접 요청은 수동 실행 기록으로 남겼다. 원본 앱·GDI 평가 자료·예약 설정·감독 프로그램은 변경하지 않았다. 종료 이유는 연결 준비와 첫 실제 추출·경계 대조 단위를 마쳤고, 정확도 확정에는 접촉 정답이 더 필요하기 때문이다. 다음 자동 실행은 이 최신 결과를 읽고 이전 준비·같은 영상 추론을 반복하지 않는다.

---

'''
old=(S/'before_RESEARCH_REVIEW.md').read_text(encoding='utf-8-sig')
review=summary+old.rstrip()+'\n\n<!-- manual-session:20261028-camera-step-pilot -->\n'
(H/'RESEARCH_REVIEW.md').write_text(review,encoding='utf-8');write('review_snapshot.md',review)
write('report.md','''# 이번 실행 요약

앱은 별도 소스 작업본에 카메라·관절 추정 부품을 준비했다. 전체 웹 타입 검사와 기존132개 검사가 통과했다. 화면 실행·영상 녹화·서버 저장은 아직이다.

연구는 TNT15 한 사람의 동기화된 두 영상에서 실제 관절 추정과2초·4초 입력 실험을 수행했다. 같은4초에서 걸음 후보가5개와2개로 나왔고,2초 구간 끝의 후보가 사라지는 문제를 후처리 경계 대조로 확인했다. 정답 주석이 없어 정확도는 아직 확정하지 못했다. AI 임시 접촉2건은 정답으로 승격하지 않는다.

종료 이유: 앱 연결 준비 단위, 공개 영상 확보·실제 추론·짧은 입력·경계 원인 대조를 마쳤다. 다음 정확도 비교에는 사람이 확인한 발 접촉 정답 또는 제공자 주석이 필요하다. 동일 영상에 임계값을 반복 맞추지 않는다. 원본95개 해시·영상 해시·프레임 시각을 검증했다. 예약 설정과 감독 프로그램은 변경하지 않았다. 직접 요청 수동 실행으로 기록하며 자동 claim/finish를 위조하지 않는다.
''')
write('sources.json',json.dumps([
 {'url':'https://www.tnt.uni-hannover.de/project/TNT15/','status':'two synchronized RGB files acquired and inspected','use':'research only; cite von Marcard et al. 2016','ground_truth':'no provided foot-contact annotation acquired'},
 {'url':'https://cecas.clemson.edu/~ahoover/pedometer/','status':'HTML and reference images acquired; zip HEAD 11714615325 bytes; videos and event labels not acquired','limitation':'follow-camera close legs view; app full-body pipeline mismatch'},
 {'url':'https://github.com/AVAuco/healthgait','status':'metadata inspected only; RGB/event labels not acquired'},
 {'url':'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task','status':'official model downloaded; SHA256 recorded in HTTP metadata'}],ensure_ascii=False,indent=2))
write('failures.json',json.dumps([
 {'step':'initial video inspection','error':'cv2 unavailable while pip install still running','resolution':'waited for installation; same inspection succeeded'},
 {'step':'app initial typecheck and test collection','error':'Supabase dependency path unresolved in copy','resolution':'local config points to existing installation; typecheck and remaining 5 tests pass'},
 {'step':'Clemson web-tool fetch','error':'timeout','resolution':'ordinary HTTP page succeeded; no authorization bypass'},
 {'step':'old Pedometer Drive links','error':'labels404; video sign-in landing','resolution':'did not bypass or download through restricted links'}],ensure_ascii=False,indent=2))
artifacts=[str(p.relative_to(H)).replace('\\','/') for p in S.iterdir() if p.is_file() and p.name not in ['response.json','receipt.json']]
response=dict(status='completed',summary='앱 연결 준비 및 실제 공개 영상 추출·짧은 입력·경계 대조 완료. 정답 대비 정확도는 미완료.',artifacts=artifacts,checks=['웹 타입 검사와 기존132개 검사 통과','공개 영상2개 CRC/SHA256 검증','전체1022프레임·짧은 입력800프레임 실제 추론','6조건 경계 대조와 타임스탬프 검사','원본95개 해시 보존'],next_goal='[앱 개발] 의존성·WASM 및 미리보기 연결. [AI 연구] 접촉 정답 확인·발별 사건 검출법 비교.',why_next='기능 연결과 실제 측정 정확도를 각각 검증해야 한다. 관측 누락·구간 경계를 처리하지 않고 점수화하면 흔들림을 그대로 전달한다.',needs_user_reason='정확도 확정에 쓸 발 접촉 주석의 사람 검토와 제품 측정·저장 기준은 미정. 독립적인 연결 준비와 자료 조사는 계속 가능.')
write('response.json',json.dumps(response,ensure_ascii=False,indent=2))
for a,b in [('APP_DEVELOPMENT.md','app_status_snapshot.md'),('AI_RESEARCH.md','ai_status_snapshot.md'),('RESEARCH_REVIEW.md','review_snapshot.md')]:assert (H/a).read_bytes()==(S/b).read_bytes()
receipt=dict(mode='manual_user_requested_experiment',automatic_session_claimed=False,status='completed',completed_scope='camera module preparation and public-video exploratory extraction, not validated accuracy',status_snapshots_equal=True,artifacts=[dict(path=str(p.relative_to(H)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(S.iterdir()) if p.is_file() and p.name!='receipt.json'])
write('receipt.json',json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(dict(artifacts=len(receipt['artifacts']),snapshots_equal=True,accuracy_validated=False)))
