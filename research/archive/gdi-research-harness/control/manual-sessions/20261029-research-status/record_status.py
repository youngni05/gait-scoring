"""Record verified artifacts and explicit limits; do not execute research jobs."""
import hashlib,json
from datetime import datetime,timedelta,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
HARNESS=HERE.parents[2]
DATA=Path('D:/gait-research-data/MoVi')
BASE=DATA/'experiments'/'walking-expansion'
HIP=DATA/'experiments'/'hip-definition'
MANUAL=HERE.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def link(label,p):return f'[{label}]({p.as_posix()})'

completion=read(BASE/'completion.json');baseline=read(BASE/'results.json');verification=read(BASE/'verification.json')
assert completion['results_sha256']==sha(BASE/'results.json')==verification['results_sha256']
assert completion['verification_sha256']==sha(BASE/'verification.json')
assert len(baseline['results'])==len(verification['checks'])==240
assert len(completion['subjects'])==30 and completion['raw_files']==64
hip=read(HIP/'analysis.json');origin=read(HIP/'reference-origin-audit.json')
export=read(HIP/'export-audit-verification.json');human=read(HIP/'marker-review'/'manifest.json')
assert export['export_fields_audit_sha256']==sha(HIP/'export-fields-audit.json')
assert export['global_reference_reads']==30 and export['calibration_centroid_checks']==29
assert [r['id'] for r in human['records'] if r['human_status']=='reviewed_qualitative']==[1,2,3]
gpu=read(MANUAL/'20261029-gpu-preparation'/'preparation.json')
assert gpu['people']==30 and gpu['videos']==60 and gpu['evaluation_video_frames']==16382
gdi_folder=MANUAL/'20261029-gpu-finetune-preparation'
gdi=read(gdi_folder/'preparation.json');receipt=read(gdi_folder/'receipt.json')
assert gdi['real_data_optimizer_steps']==receipt['actual_gdi_training_steps']==0
assert not gdi['actual_gpu_used'] and not receipt['gpu_used']
assert (gdi['people'],gdi['records'],gdi['folds'])==(175,203,5)
assert sha(gdi_folder/'motionbert_gpu_preparation.zip')==receipt['zip_sha256']
contact=MANUAL/'20261028-contact-evaluation';contact_diagnostic=read(contact/'comparison-diagnostic.json')
reviews=[read(contact/'reviewed'/f'contact-{c}-review-from-chat.json') for c in ['03','06']]
assert [len(r['events']) for r in reviews]==[4,3]
assert not contact_diagnostic['accuracy_computed']
total_bytes=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
now=datetime.now(timezone(timedelta(hours=9)))
assert now.date().isoformat()=='2026-10-29','Use the client date for a new record.'
automations=[]
for p in Path('C:/Users/jun/.codex/automations').glob('*/automation.toml'):
    text=p.read_text(encoding='utf-8')
    if p.parent.name=='movi' or '보행 앱 개발 · AI 연구 병행' in text:
        fields={line.split('=',1)[0].strip():line.split('=',1)[1].strip().strip('"')
                for line in text.splitlines() if line.startswith(('name =','status ='))}
        automations.append({'id':p.parent.name,**fields})
assert len(automations)==2 and all(x['status']=='PAUSED' for x in automations)
evidence=[BASE/'completion.json',BASE/'results.json',BASE/'verification.json',BASE/'patterns.json',
 HIP/'analysis.json',HIP/'reference-origin-audit.json',HIP/'verification.json',HIP/'export-fields-audit.json',
 HIP/'export-audit-verification.json',HIP/'marker-review'/'manifest.json',
 MANUAL/'20261029-gpu-preparation'/'preparation.json',MANUAL/'20261029-gpu-preparation'/'frame-inventory.json',
 MANUAL/'20261029-gpu-preparation'/'local-framework-check.json',gdi_folder/'preparation.json',gdi_folder/'receipt.json',
 gdi_folder/'report.md',gdi_folder/'bundle'/'README.md',contact/'comparison-diagnostic.json',
 contact/'reviewed'/'contact-03-review-from-chat.json',contact/'reviewed'/'contact-06-review-from-chat.json',
 HARNESS/'control'/'desktop-sessions'/'20261029T091602Z-e46435'/'ai_report.md']
snapshot={'recorded_at_kst':now.isoformat(),'scope':'Research state: primary MoVi measurement validation, with separately sourced TNT15 contact and GDI prediction/preparation records.',
 'objective':'좁은 공간·단일 구도 일반 카메라에서 보행과 회전의 움직임을 의미 있게 측정한다. GDI 예측만을 목표로 고정하지 않으며 최종 1~2개 지표를 집중 개선할 계획이다.',
 'current_conclusion':'고관절 차이를 전부 AI 검출 실패로 볼 근거가 부족하다. 관절 대표점 대응과 영상/기준 좌표 정렬을 먼저 확인한다. 정의 차이의 인과 비율은 미확정이다.',
 'movi':{'people':30,'videos':60,'conditions':240,'raw_files':64,'raw_bytes':completion['raw_bytes'],
  'current_total_data_bytes':total_bytes,'storage_budget_bytes':21_000_000_000,'completion':completion,
  'protocol':{'full_widths':[800,400,200],'heavy_widths':[800],'cameras':['PG1','PG2'],
   'window':'provider walking flags30, inclusive','warmup_frames':30,'inference':'CPU, cached predictions preserved',
   'angle':'180 minus hip-knee-ankle included angle; not clinical joint-coordinate-system flexion',
   'aggregation':'person first; do not count frames/views as independent people'},
  'hip_diagnostic_conditions':120,'source_audit_people':30,'calibration_centroid_checks':29,
  'human_reviews':human,'fine_tuning_performed':False},
 'tnt15':{'people':1,'views':2,'contact_annotation_records':[4,3],
  'reference_status':'one human reviewer; not adjudicated sensor truth','accuracy_computed':False,
  'remaining':'Camera06 frames210-230 follow-up and event-matching rules; boundary/gap handling comparison not yet agreed.'},
 'gdi':{'source_kind':'existing saved reports/receipt inspected, no training rerun for this status record',
  'development_people':175,'records':203,'folds':5,'latest_comparison':'Ridge versus Kernel Ridge; no clear replacement evidence, Ridge retained',
  'gpu_preparation':gdi,'gpu_receipt_path':str(gdi_folder/'receipt.json'),
  'limits':'Existing 2D-pose/GDI prediction research; not RGB coordinate or angle accuracy validation, not independent final clinical validation.'},
 'gpu_preparation':gpu,'automations':automations,'new_research_experiments_executed_for_this_record':0,
 'next_actions':[{'priority':1,'action':'Check observable-marker video/reference alignment, especially subject2 PG1 tentative left offset. Use clear observations; do not fit offsets to pose predictions.'},
  {'priority':2,'action':'Confirm original Visual3D model/export and joint-point mapping if source becomes available; do not adopt the approximate CODA reconstruction as corrected truth.'},
  {'priority':3,'action':'Agree 1-2 target metrics and person-level evaluation before selecting a MoVi adaptation model.'},
  {'priority':4,'action':'Before a GPU window, report purpose, model, data, VRAM/time pilot and deliverables. Keep the separate GDI development split and MoVi diagnostic cohort distinct.'}],
 'evidence':[{'path':str(p),'sha256':sha(p)} for p in evidence]}
assert total_bytes<snapshot['movi']['storage_budget_bytes']
(HERE/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

result_page=MANUAL/'20261028-movi-totalcapture-acquisition'/'experiment-results.html'
marker_page=MANUAL/'20261029-movi-expansion'/'hip-definition'/'marker-review.html'
full=completion['cohort_comparison']['all30'];hf=hip['summary']['full']['equal_person_view_means']
rf=origin['summary']['full']['equal_person_view_means']
block=f'''<!-- research-status-20261029-start -->
# 2026-10-29 연구 진행 기록

작성 기준: {now.strftime('%Y-%m-%d %H:%M')} KST. 저장된 결과·검증 파일·사용자 판독을 대조해 정리했다. 이번 기록 작업에서 새 추론·학습·임계값 조정은 하지 않았다. 아래는 현재 상태이며, 이 뒤의 문단은 기존 연구의 경과 기록이다.

**현재 결론은 고관절의 큰 좌표 차이를 전부 AI의 검출 실패로 볼 수 없다는 것이다. 관절 대표점의 대응과 영상–정밀 좌표의 정렬을 먼저 확인해야 하며, 정의 차이가 오차를 얼마나 설명하는지는 아직 확정하지 못했다.**

연구 목적은 좁은 공간에서 한 구도의 일반 카메라만으로 보행·회전 움직임을 최대한 정확하게 측정하고, 의미 있는 보행 분석을 제공하는 것이다. 고령자나 GDI 점수 예측만으로 범위를 고정하지 않는다. 회전은 시간뿐 아니라 몸통·골반·다리 등 관절 움직임을 함께 볼 대상이다. 최종적으로 1~2개 지표를 집중 개선할 계획이나, 대상 지표와 능력 점수화 방식은 아직 확정하지 않았다.

**어디까지 진행했는가**

| 연구 흐름 | 실제 완료 범위 | 남은 부분 |
|---|---|---|
| MoVi 자료 확보 | 30명·두 시점 영상 60개, 기준 TAR 2개·보정 자료·안내문을 합친 원본 64파일 검증 | 새 참여자 추가 다운로드는 이번 범위에 없음 |
| MoVi 반복 비교 | Full 폭 800/400/200, Heavy 폭 800 × 두 시점 × 30명 = 240조건 저장·검증 완료 | 가정·고령자·회전 전체의 임상 정확도 검증은 아님 |
| 고관절 진단 | 기존 800폭 Full/Heavy 120조건 재사용, 위치 편향·각도 민감도·공개 코드·MAT 필드 대조 | 원래 Visual3D 설정·내보내기 코드와 정확한 해부학적 대응 미확인 |
| 피부 마커 사람 검토 | 준비한 8장 중 3장 판독 기록, 관측 불가와 의심을 분리 | 나머지 5장 미검토, 3번 장면의 왼쪽 변위 의심 미확정 |
| TNT15 발 접촉 | 한 사람·두 카메라의 사람 1차 주석과 검출 후보 첫 대조, 짧은 구간 경계 제한 확인 | 사건 대응 규칙·06번 210–230프레임 후속 검토, 정확도 계산 미완료 |
| GDI 점수 예측 | 별도 개발자료 175명·203기록의 Ridge–Kernel Ridge 비교 기록 확인, 기존 Ridge 유지 | 이 결과는 RGB 관절 측정·고령자 임상 성능 검증이 아님 |
| GPU 사전 준비 | MoVi 자료 인덱스·환경 점검 도구 준비. 별도 GDI 부분학습 묶음과 CPU 기술검사 기록도 존재 | 실제 GPU 검사·실제 GDI 추가 학습은 0회. 환경·학습 예산 등 확정 대기 |

**MoVi에서 사람과 시점이 달라도 반복된 현상**

- 제공된 기준 좌표와 비교한 3D 무릎 세 점 굽힘각의 평균 절대 차이는 Full **{full['full_person_mean_3d']:.2f}°**, Heavy **{full['heavy_person_mean_3d']:.2f}°**다. 사람별로 먼저 집계한 값이며 임상적 관절 각도 오차나 보행 능력 점수가 아니다.
- PG2의 차이가 더 큰 사람은 **29/30명**이었다. 양 발목이 두 시점 모두 화면 안인 공통 프레임에서도 같은 경향이었다. 카메라 이름만으로 촬영 각도 하나가 원인이라고 단정하지 않는다.
- 적어도 한 발목이 화면 밖에 나가는 구간이 있었던 사람은 **23/30명**, 고관절 중 하나가 위치 차이 상위 3개 관절에 든 영상은 **60/60개**였다. 가림·화면 잘림·관절 정의·기준 정렬을 함께 검토해야 한다.
- Heavy가 제공된 기준에 대한 3D 각도 차이를 줄인 영상은 **57/60개**였다. 이 우위는 비교 기준에 의존하므로 해부학적 정확성의 우위로 확정하지 않는다.

자료는 `D:/gait-research-data/MoVi`에 있다. 현재 저장량은 **{total_bytes/1e9:.3f} GB**(십진 단위), 원본은 **{completion['raw_bytes']/1e9:.3f} GB**다. 기준 TAR에는 다른 참여자의 좌표도 포함되지만 영상으로 반복 검증한 인원은 30명이다. 원본을 유지하며 보기용 MP4/GIF는 추가 생성하지 않았다.

**고관절 진단에서 확인한 것과 확인하지 못한 것**

30명 모두 기존 비교 입력이 공식 예제가 안내한 전역 관절 좌표와 정확히 일치했다. MoVi는 Bell–Brand 방식으로 추정한 관절 중심이며, MediaPipe의 고관절 지점과 정확히 같은 해부학적 점인지는 확인되지 않았다. 정적 골반 마커 정보가 있는 29명에서는 Bell 계수 형태와 맞는 수치 관계를 확인했다. 골반 너비를 제공된 고관절 간격에서 역산한 관계 검산이므로 원래 생성 설정을 독립적으로 복원한 것은 아니다. 49번의 정적 골반 정보는 이 검산에 사용할 수 없었다.

2·31번의 저장된 골반 원점에는 고정된 이동 차이가 있었다. 이를 다른 고관절 정의의 증거로 보지 않는다. MAT의 저장 골반 원점이 논문에서 사용한 해부학적 CODA 원점과 같은지도 미확인이다. 공식 저장소의 작은 텍스트 소스 11개와 Python 노트북·MATLAB Live Script의 입력 코드까지 조사했지만 원래 Visual3D 모델·MAT 내보내기 스크립트를 찾지 못했다.

고관절만 제공된 기준점으로 대입한 진단에서는 Full의 화면상 2D 각도 차이가 **{hf['full_2d_knee_mae_deg']:.2f}→{hf['reference_hip_substitution_2d_knee_mae_deg']:.2f}°**였다. 문서식 근사 고관절을 참고 기준으로 쓰면 같은 예측의 위치 차이는 **{rf['provided_hip_distance_common_px']:.2f}→{rf['document_coda_hip_distance_common_px']:.2f}픽셀**, 3D 각도 차이는 **{rf['provided_reference_3d_mae_common_deg']:.2f}→{rf['document_coda_reference_3d_mae_common_deg']:.2f}°**로 달라졌다. **두 계산 모두 실제 모델 개선 성적이 아니다. 원래 정답을 복원하거나 정의 차이의 원인 비율을 입증한 결과도 아니다. 기존 기준·예측 240조건은 유지한다.**

| 검토 화면 | 사용자 판독 |
|---|---|
| 1번: 참여자 1·PG1 | 1·5·6·7·8번 마커는 겹쳐 보임. 2·3·4번 판독 불가 |
| 2번: 참여자 1·PG2 | 2·6·8번은 겹쳐 보임. 나머지 판독 불가 |
| 3번: 참여자 2·PG1 | 대부분 판독 불가. 7번 고리와 전체 배치가 조금 왼쪽이라는 의심은 미확정 |

이는 보이는 피부 마커의 정성적 대조다. 관측 불가를 위치 오류로 세지 않으며, 몸속 고관절 중심이나 픽셀 정확도를 확정하지 않는다.

**GDI 점수화·GPU 준비는 별도 흐름이다**

기존 GDI 연구는 추출된 2D 관절 자료에서 MotionBERT 특징과 회귀 모델로 점수를 예측한다. 개발자료 175명·203기록의 Kernel Ridge 비교는 일부 구간 개선과 악화가 섞여 명확한 교체 근거가 없었다. 기존 Ridge를 유지한다. 같은 영상의 앞뒤 비교를 재촬영·시점 변화 안정성으로 해석하지 않는다.

`20261029-gpu-finetune-preparation`에는 이 GDI 개발자료의 출력부·마지막 특징변환·마지막 Transformer 단계 부분학습 비교 묶음이 있다. 저장된 기록상 기준 예측 재현과 가상 자료의 업데이트·중단 재개 등 CPU 기술검사는 완료됐으나 **실제 GDI 추가 학습 0 step, GPU 사용 0회**다. 실제 자료 학습 예산은 미확정이다. 이 준비를 MoVi 영상 추정기 파인튜닝 완료나 성능 개선으로 세지 않는다.

별도 `20261029-gpu-preparation`에는 MoVi 30명·60영상·16,382 평가 프레임 인덱스와 환경 확인 도구가 있다. MoVi용 학습 모델·정답·사람별 분할은 미확정이며 실제 학습 실행기는 아직 없다. 앞으로 GPU가 필요한 작업은 **목적·모델·자료·메모리/시간·산출물을 먼저 설명하고 사용자가 연결하는 시간에 묶어 실행**한다. 작은 실행으로 자원을 측정한 뒤 전체 예산을 정한다. 현재 30명은 모두 오류 탐색에 썼으므로 최종 독립 평가 설계가 별도로 필요하다.

**현재 우선순위와 재개 조건**

1. MoVi 영상–정밀 좌표 정렬과 고관절 대표점 대응을 먼저 확인한다. 참여자 2·PG1의 변위 의심은 선명한 관측에서 추가 검토할 대상이다. 예측에 맞춰 시간·좌표를 조절하지 않는다.
2. 원래 Visual3D 모델·내보내기 코드를 확보할 수 있으면 생성·좌표계 연결을 확인한다. 근사 CODA 좌표를 새 정답으로 채택하지 않는다.
3. 집중할 1~2개 지표와 사람 단위 평가 설계를 정한 뒤 MoVi 개선 방법을 선택한다. 속도·회전 전체 관절 움직임·실제 가정·임상 능력 점수의 검증은 아직 완료하지 않았다.
4. GDI 부분학습은 별도 준비 묶음의 조건을 유지하고, GPU 환경·자원 측정·학습 예산 확정 후 진행한다. MoVi의 정렬 검증을 GDI 모델 성능으로, GDI 학습 결과를 RGB 측정 정확도로 대신하지 않는다.

현재 `MoVi 30명 확대 검증`과 `보행 앱 개발 · AI 연구 병행` 예약은 모두 **일시중지**다. 이번 기록 요청으로 재개·새 예약·외부 공유를 하지 않았다.

주요 재개 위치: {link('MoVi 상세 결과',result_page)} · {link('피부 마커 검토 화면',marker_page)} · {link('GDI 부분학습 준비 안내',gdi_folder/'bundle'/'README.md')} · {link('근거 파일·SHA256·전체 상태 스냅샷',HERE/'snapshot.json')}
<!-- research-status-20261029-end -->'''

brief=f'''<!-- research-status-20261029-start -->
# 2026-10-29 현재 연구 상태 요약

**MoVi 30명·60영상·240조건 비교와 고관절 진단을 마쳤지만, 고관절 차이의 원인은 미확정이다. 모델 보정·파인튜닝보다 관절 대표점 대응과 영상–정밀 좌표 정렬 확인이 우선이다.**

현재 연구 목표는 좁은 공간에서 단일 카메라로 보행·회전 움직임을 의미 있게 측정하고 최종 1~2개 지표를 집중 개선하는 것이다. GDI 예측 연구는 병행하되 측정 정확도 검증과 구분한다.

- MoVi: 제공된 기준에 대한 Full/Heavy 3D 세 점 각도 차이는 13.99°/11.78°다. 임상 각도나 능력 점수가 아니다. 30명 기준 좌표 읽기는 검증했고, 29명 골반 필드 관계와 공개 코드를 대조했다. 원래 Visual3D 모델·내보내기 설정은 미확인이다.
- 사람이 8개 검토 화면 중 3개를 판독했다. 관측 불가와 참여자 2·PG1의 왼쪽 변위 의심을 확정 오류와 구분해 보존했다.
- 고관절 참고 기준을 바꾼 17.9→8.1픽셀 진단은 실제 개선 성과나 원래 정답 복원으로 보고하지 않는다.
- TNT15: 한 사람·두 시점의 발 접촉 1차 주석·기술 대조까지 완료, 사건 대응 규칙·06번 210–230프레임 후속 검토와 정확도 평가는 남아 있다.
- GDI: 별도 175명·203기록의 Ridge/Kernel Ridge 비교는 기존 Ridge 유지로 정리됐다. 부분학습 실행 묶음과 CPU 기술검사 기록은 있지만 실제 GDI 추가 학습·GPU 사용은 아직 0회다.
- GPU: MoVi 자료 인덱스·환경 점검과 GDI 부분학습 준비를 구분한다. GPU 연결 전 목적·자료·시간/메모리·산출물을 먼저 알리고, 환경과 예산을 확정한 뒤 묶어서 진행한다.
- 두 연구 예약은 현재 일시중지이며 이번 작업은 기록만 갱신했다.

{link('통합 연구 진행 기록',HARNESS/'AI_RESEARCH.md')} · {link('수치·검증·자료 경로 스냅샷',HERE/'snapshot.json')}
<!-- research-status-20261029-end -->'''

prior_hashes={}
for name,text in [('AI_RESEARCH.md',block),('RESEARCH_REVIEW.md',brief)]:
    p=HARNESS/name;original=p.read_text(encoding='utf-8');prior_hashes[name]=sha(p)
    start='<!-- research-status-20261029-start -->';end='<!-- research-status-20261029-end -->'
    remainder=original
    if start in original:
        lo=original.index(start);hi=original.index(end,lo)+len(end)
        remainder=(original[:lo]+original[hi:]).lstrip('\n')
        if remainder.startswith('---\n'):remainder=remainder[4:].lstrip('\n')
    updated=text+'\n\n---\n\n'+remainder
    assert updated.endswith(remainder) and updated.count(start)==1
    p.write_text(updated,encoding='utf-8')
snapshot['record_documents']={name:{'before_sha256':prior_hashes[name],'after_sha256':sha(HARNESS/name),'prior_history_preserved':True}
                              for name in prior_hashes}
(HERE/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'recorded_at_kst':now.isoformat(),'people':30,'conditions':240,'human_reviewed_cases':3,
                  'gdi_actual_training_steps':0,'gpu_used':False,'documents':list(prior_hashes),'source_artifacts':len(evidence)}))
