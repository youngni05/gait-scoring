"""Prepare an inventory and a gated experiment plan; execute no GPU/research jobs."""
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('D:/gait-research-data/MoVi')
BASE=ROOT/'experiments'/'walking-expansion'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
baseline=read(BASE/'results.json');audit=read(ROOT/'experiments'/'hip-definition'/'export-audit-verification.json')
assert audit['baseline_results_sha256_unchanged']==sha(BASE/'results.json')
manifest_path=ROOT/'control'/'add20'/'experiment-manifest.json';manifest=read(manifest_path)
human=read(ROOT/'experiments'/'hip-definition'/'marker-review'/'manifest.json')
rows=[r for r in baseline['results'] if r['variant']=='full' and r['input_width']==800]
assert len(rows)==60 and len({r['subject'] for r in rows})==30
clips=[]
for row in rows:
    subject,camera=row['subject'],row['camera'];start,end=row['video_frames']
    video=ROOT/'raw'/f'F_{camera}_Subject_{subject}_L.avi'
    reference=ROOT/'reference'/f'F_v3d_Subject_{subject}.mat'
    assert video.is_file() and reference.is_file()
    clips.append({'subject':subject,'camera':camera,'video':str(video),'video_bytes':video.stat().st_size,
      'video_sha256_from_verified_baseline':row['video_sha256'],'reference':str(reference),
      'reference_sha256':row['reference']['sha256'],'evaluation_video_frames_one_based_inclusive':[start,end],
      'existing_baseline_context_frames_one_based_inclusive':[start-30,start-1],
      'video_fps':30,'mocap_fps':120,'mocap_frame_one_based_formula':'(video_frame_one_based-1)*4+1',
      'train_val_test_assignment':None,'use_status':'previously_explored; not a frozen held-out test set',
      'missing_reference_policy':'Preserve original markerGaps mask; do not interpolate into ground truth.'})
assert len({(x['subject'],x['camera']) for x in clips})==60
plan={'prepared_date_kst':'2026-10-29','status':'cpu_preparation_only_no_training_runner',
 'user_request':'GPU가 필요한 실험은 목적·준비 자료·사용량을 미리 설명하고, 사용자가 GPU를 연결하는 시간에 묶어서 실행한다.',
 'data_manifest':str(manifest_path),'data_manifest_sha256':sha(manifest_path),
 'baseline_results_sha256':sha(BASE/'results.json'),'data_location':str(ROOT),
 'people':30,'videos':60,'original_video_bytes':sum(x['video_bytes'] for x in clips),
 'evaluation_video_frames':sum(x['evaluation_video_frames_one_based_inclusive'][1]-x['evaluation_video_frames_one_based_inclusive'][0]+1 for x in clips),
 'no_video_or_mat_copied':True,'no_upload_or_model_download':True,'no_new_schedule_created':True,
 'gpu_environment':{'provider':None,'model':None,'vram_bytes':None,'status':'awaiting_user_information'},
 'preconditions':{'reference_loader':'verified_for_30_people','anatomical_joint_mapping':'unresolved',
   'video_reference_alignment':'partial_qualitative_review; tentative left-offset concern in subject 2 PG1',
   'human_reviewed_cases':[r['id'] for r in human['records'] if r['human_status']=='reviewed_qualitative'],
   'trainable_model_and_original_training_weights':'not_selected',
   'person_disjoint_split':'not_agreed; keep both cameras and all frames of each person together',
   'held_out_test':'All current 30 people were used for exploratory analysis; discuss independent final validation before training.'},
 'cpu_work_before_gpu':['Resolve reference/observable-marker alignment and target joint definitions',
   'Choose one improvement question and confirm a trainable model/weight format and usage conditions',
   'Agree person-level train/validation/test design and target metrics before fitting',
   'Prepare decoder/labels with missing masks and exercise data loading on CPU',
   'Prepare resumable checkpoints, fixed configs, output paths and logs once the model is selected'],
 'gpu_queue':[
  {'id':'GPU-00','purpose':'Environment and data-free GPU forward/backward check','status':'tool_prepared_gpu_not_verified',
   'command':'python gpu_preflight.py --tensor-check --out gpu-check-001.json','research_data_required':False},
  {'id':'GPU-01','purpose':'Selected-model single-batch correctness, VRAM and runtime measurement',
   'status':'waiting_for_model_and_gpu','vram_estimate':None,'time_estimate':None,
   'note':'Report measured pilot usage and a full-run estimate before launching the full batch.'},
  {'id':'GPU-02','purpose':'Frozen selected-model baseline inference with cached outputs',
   'status':'waiting_for_model_reference_and_evaluation_agreement','note':'Reuse existing MediaPipe caches; do not rerun 240 unchanged CPU conditions merely to use a GPU.'},
  {'id':'GPU-03','purpose':'One agreed minimal adaptation versus the frozen baseline, then held-out evaluation',
   'status':'proposal_not_approved_or_implemented','method':None,'seeds':None,
   'note':'Choose head training, partial fine-tuning or compatible LoRA based on the actual model. Do not assume an inference .task is a trainable PyTorch checkpoint.'}],
 'before_gpu_window_notify':['Question and expected decision','Model/version and trainable parameter scope',
   'Data and reference definition; person split; metrics and missing-data policy',
   'GPU model/VRAM and pilot-measured runtime/memory; total budget',
   'Files/bytes to transfer if remote; agree destination and data-use conditions',
   'Commands, checkpoint/resume behavior and exact deliverables'],
 'sources':{'pytorch_transfer_learning':'https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html',
            'pytorch_cuda':'https://docs.pytorch.org/docs/stable/notes/cuda.html',
            'mediapipe_pose':'https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker'}}
for name,obj in [('frame-inventory.json',{'purpose':'Index only; not ready-to-train labels or an agreed data split.','clips':clips}),('preparation.json',plan)]:
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
section=f'''<!-- gpu-preparation-start --><section id="gpu-preparation"><p class="lead"><b>GPU 실험 사전 준비: 환경 확인 도구와 자료 목록을 준비했다.</b></p>
<p>앞으로 GPU가 필요한 실험은 목적·자료·시간과 메모리·산출물을 미리 설명하고, 사용자가 GPU를 연결하는 시간에 묶어 실행한다. 현재는 학습 대상 모델·정답·사람별 자료 분할이 미확정이므로 학습 실행 코드가 완성된 상태는 아니다.</p>
<p>현재 MediaPipe 반복 실험은 CPU로 실행했다. 좌표 검산과 작은 회귀 모델은 CPU에서도 진행할 수 있다. 큰 영상 모델의 추론·미세조정은 GPU 사용 후보이며, 필요한 VRAM과 시간은 모델 선정 후 작은 실행으로 측정한다.</p>
<table><tr><th>연결 후 순서</th><th>목적</th><th>현재 준비 상태</th></tr><tr><td>환경 확인</td><td>GPU 감지·작은 행렬 계산과 역전파 확인</td><td>도구 준비, GPU 실물 검증 대기</td></tr><tr><td>작은 실행</td><td>선택한 모델의 메모리·소요 시간 측정</td><td>모델·GPU 확정 대기</td></tr><tr><td>비교 실험 묶음</td><td>고정 모델의 추론→최소 범위 미세조정→평가</td><td>정답·모델·사람별 분할 합의 후 구현</td></tr></table>
<p>기존 30명·60개 영상의 평가 구간과 기준 좌표 파일을 연결한 목록을 만들었다. 원본 복사·클라우드 전송·새 모델 다운로드는 하지 않았다. 현재 30명은 모두 오류 탐색에 사용했으므로 최종 평가의 독립성을 별도로 설계해야 한다. 같은 사람의 두 카메라와 여러 프레임을 학습·평가 양쪽으로 나누지 않는다.</p>
<p><a href="{(HERE/'preparation.json').as_uri()}">준비 기록·실험 후보</a> · <a href="{(HERE/'gpu_preflight.py').as_uri()}">환경 확인 도구</a> · <a href="{(HERE/'frame-inventory.json').as_uri()}">60개 영상의 프레임 목록</a></p></section><!-- gpu-preparation-end -->'''
report=HERE.parent/'20261028-movi-totalcapture-acquisition'/'experiment-results.html'
html=report.read_text(encoding='utf-8');start='<!-- gpu-preparation-start -->';end='<!-- gpu-preparation-end -->'
if start in html:
    lo=html.index(start);hi=html.index(end,lo)+len(end);html=html[:lo]+section+html[hi:]
else:html=html.replace('<!-- export-source-audit-start -->',section+'\n<!-- export-source-audit-start -->',1)
assert html.count(start)==1;report.write_text(html,encoding='utf-8')
print(json.dumps({'people':30,'videos':60,'evaluation_video_frames':plan['evaluation_video_frames'],
                  'status':plan['status'],'new_gpu_training_jobs':0}))
