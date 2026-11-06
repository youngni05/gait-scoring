from pathlib import Path
import json,hashlib,zipfile,datetime
H=Path('C:/Users/jun/Desktop/D_P/gdi-research-harness');S=H/'control/manual-sessions/20261029-gpu-finetune-preparation';B=S/'bundle';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
for name,digest in load(S/'sources.json').items():assert sha(H.parent/name)==digest
assert load(S/'cpu_integration_v2/integration_checks.json')['real_data_optimizer_steps']==0
assert load(S/'aggregation_checks.json')['status']=='passed'
manifest={p.relative_to(B).as_posix():sha(p) for p in B.rglob('*') if p.is_file() and p.name!='manifest.json'};save(B/'manifest.json',manifest)
zpath=S/'motionbert_gpu_preparation.zip'
with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
 for p in B.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(B).as_posix())
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None
 for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
 assert set(z.namelist())==set(manifest)|{'manifest.json'}
report='''# GPU 부분학습 사전 준비 결과\n\n사용자 직접 요청에 따른 수동 준비만 완료했다. 자동 claim은 cooldown_wait였으며 자동 실행/운영 설정을 변경하지 않았다. 실제 연구 학습이나 GPU 접속은 하지 않았다. 자동 배정 세션이 없으므로 desktop_resume --finish를 호출하지 않는다.\n\n준비: 기존 기준 + 출력부만 학습 + 마지막 특징변환/출력부 + 마지막Transformer단계/특징변환/출력부 비교. 기존175명/203기록/5fold와 앞뒤124프레임만 포함했다. 보호37/38/223 및 새520명은 접근하지 않았다. 211개 원천 지문을 다시 검사했다.\n\n검사: 원입력으로5fold×앞뒤2구간의 기준 예측10개를 재현했으며 최대차이0이었다. 가상입력에서 지정한 부분만 갱신되고 연속2step과중단재개1+1step이세방식모두같았다. 전체학습루프의가상자료통합검사, 완료작업재학습생략, 잘못된재개/초안설정/사람연결훼손/누락fold 거부를 검사했다. 결과집계는 기존기준예측을복제한테스트자료로만검사했고 새모델성능으로사용하지않는다.\n\n새 실패와 수정: 첫 통합검사에서 torch 버전이 문자열의 하위형이라 안전한 체크포인트 읽기에 실패했다. 순수 문자열로 저장하도록 수정했고 weights_only=True를 유지한 재검사는 통과했다. 실패 기록을 보존했다.\n\n한계: 실제 GDI 학습0step, GPU검사0회. 가상입력 업데이트는 기술검사이며 GDI 성능 결과가 아니다. 학습률·반복수·배치·weight decay 미정이라 실제학습초안은 실행되지 않는다. CPU가상검사만으로 CUDA최고메모리·속도·수치동일성을 주장하지 않는다. 같은영상앞뒤는 재촬영/시점안정성이 아니다. 과거개발자료의반복탐색이며 독립최종평가가 아니다.\n\n다음: GPU모델/VRAM/OS/접속방식을 받은 뒤 가상124프레임으로 자원검사, 기준재현, 동일학습예산 사전확정 후 실행. 최대3방법×5fold×3seed=45작업이며 평가결과로시드/학습예산을선택하지않는다. 전체모델학습·일관성손실·새임상경계는 보류한다. 앱 코드 변경없음.\n\n종료: 사용자 요청한 로컬 준비 묶음과 CPU검사 완료. 수동준비 경과시간은 문서작성을 포함해30분목표를초과했다(시작09:31UTC, 종료시각은receipt참조). 추가실험은중단하고증거보존만수행했다. 최종 확인 사용률50%. 예약 제한을 변경하거나 별도 자동배정으로 기록하지 않았다.\n'''
(S/'report.md').write_text(report,encoding='utf-8');(S/'ai_report.md').write_text(report,encoding='utf-8');(S/'app_report.md').write_text('이번 직접 요청은 GPU 사전 준비다. 앱 작업은 수행하지 않았으며 기존 앱·앱 정책·운영 설정은 변경하지 않았다.\n',encoding='utf-8')
header='''# 최신 준비 — GPU에서 비교할 부분학습 묶음\n\n2026-10-29 사용자 직접 요청으로 로컬 사전 준비와 CPU 기술검사를 마쳤다. **실제 GDI 학습과 GPU 검사는 아직 하지 않았다.** 기존175명/203기록/5fold와 앞뒤124프레임만 묶었고 보호평가자료는 포함하지 않았다. 기존 기준에 출력부만 추가학습하는 대조, 마지막 특징변환 학습, 마지막Transformer단계까지의 학습을 비교할 코드다. 출력부는 기존Ridge로초기화하는선형층이며 학습후Ridge해를다시계산한것은아니다.\n\n원입력 기준점수10개 재현차이0, 가상입력3방식의 지정범위갱신/연속실행과저장재개동일성/전체루프검사 통과. 체크포인트 버전문자열 문제를수정했고안전한읽기를유지했다. 성능개선이나재촬영안정성을입증한것은아니다.\n\n학습률·반복·배치·weight decay는미정이다. GPU모델/VRAM/접속방식확인→가상입력자원검사→예산사전확정→실제비교 순서다. 자동실행은 GPU연결이나실제학습을임의시작하지않는다. 앱개발범위·TNT15/공개자료검토대기는유지한다. 수동준비시간이30분목표를넘어추가실험없이정리했다. 자동배정기록과혼동하지않는다.\n\n[실험 구성과 실행 안내](control/manual-sessions/20261029-gpu-finetune-preparation/bundle/README.md) · [검사·한계·종료 기록](control/manual-sessions/20261029-gpu-finetune-preparation/report.md)\n\n---\n\n'''
for name,snap in [('AI_RESEARCH.md','ai_status_snapshot.md'),('RESEARCH_REVIEW.md','review_snapshot.md')]:
 p=H/name;old=p.read_text(encoding='utf-8-sig');new=header+old
 if name=='RESEARCH_REVIEW.md':new=new.rstrip()+'\n\n<!-- session:manual-20261029-gpu-finetune-preparation -->\n'
 p.write_text(new,encoding='utf-8');(S/snap).write_text(new,encoding='utf-8');assert sha(p)==sha(S/snap)
p=H/'APP_DEVELOPMENT.md';(S/'app_status_snapshot.md').write_bytes(p.read_bytes());assert sha(p)==sha(S/'app_status_snapshot.md')
artifacts=[p for p in S.iterdir() if p.is_file()];artifacts += [B/'README.md',B/'manifest.json',S/'cpu_v2_check/replay.json',S/'cpu_v2_smoke/checks.json',S/'cpu_integration_v2/integration_checks.json']
response={'status':'completed','summary':'GPU 사전 준비와 CPU 가상자료 검증만 완료. 실제 GDI 학습/GPU 검사는 미실행.','artifacts':[p.relative_to(H).as_posix() for p in artifacts],'checks':['211 source hashes unchanged','baseline replay 10 scores max error 0','3 variants scope and exact resume checks passed','synthetic full-loop and aggregation checks passed','ZIP CRC and SHA256 verified','state snapshots equal'],'next_goal':'[앱 개발] 기존 범위 유지. [AI 연구] GPU 환경 확인 후 가상 입력 자원검사와 학습예산 확정.','why_next':'[앱 개발] 이번 요청 범위 아님. [AI 연구] CUDA 실행 가능성과 계산량을 확인하기 전 실제 학습을 시작하지 않음.','needs_user_reason':'GPU 모델·VRAM·운영체제·접속 방식 확인 필요. 실제 학습 설정은 이후 확정.'};save(S/'response.json',response)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();save(S/'receipt.json',{'kind':'manual_preparation_not_autorun','finished_at_utc':now,'actual_gdi_training_steps':0,'gpu_used':False,'zip_bytes':zpath.stat().st_size,'zip_sha256':sha(zpath),'artifacts':[{'path':p.relative_to(H).as_posix(),'sha256':sha(p)} for p in artifacts+[S/'response.json']]})
print(json.dumps({'status':'preparation_completed','zip':str(zpath),'zip_MiB':round(zpath.stat().st_size/2**20,2),'files':len(manifest)+1,'snapshot_checks':True,'real_training_steps':0,'finished_at_utc':now}))
