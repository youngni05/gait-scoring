from pathlib import Path
import json,datetime,hashlib
HERE=Path(__file__).resolve().parent
CONTROL=HERE.parents[1]/'control/research-loop/threads/01a18b70-7da2-7430-a223-d4b314940cb3'
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
if __name__=='__main__':
    registry=CONTROL/'registry.json';q=json.loads(registry.read_text());assert q['thread_id']=='01a18b70-7da2-7430-a223-d4b314940cb3'
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt=json.loads((HERE/'main-process.json').read_text());pilot=json.loads((HERE/'pilot-verification.json').read_text());assert pilot['passed'] and pilot['gate_replays']==8
    old=next(x for x in q['experiments'] if x['id']=='cadence-straight-control-20261101');old['result_note']='14명·28사건·224주기 독립 검산 통과. 직선 사건 가용12/14명,RGB3D 차이 PG1 3.00/PG2 1.82걸음/분. 직선·회전 공동6명의 기준3D차이1.34대23.96,6명 모두 개선. 공동10명 사전목표 미달. 전문가 접촉 정답이 아니다. 후속 영상만 회전 거부 진단 실행.'
    item=dict(id='camera-straight-gate-20261101',kind='local_cpu_rgb_only_observability_gate',local=str(HERE),large='D:/gait-research-data/D_P/camera-straight-gate-20261101',jobs=[],actual_pid=receipt['pid'],launcher_pid=receipt['launcher_pid'],process_receipt=str(HERE/'main-process.json'),process_exit_marker='main-process-exit.json',pipeline_exit_marker='pipeline-exit.json',launch_command='python -X utf8 launch.py -> study.py main;analyze.py;verify.py main',protocol=str(HERE/'protocol.json'),expected_tasks=14,expected_people=14,expected_clips=56,expected_gate_replays=56,expected_metric_rows=2,expected_paired_rows=2,expected_proxy_rows=8,expected_new_model_fits=0,completion_marker='main-completion.json',verification_marker='main-verification.json',report='README.md',review_state='pending',reported_digest=None,state='running',pilot_verified=True,audit='같은14명 직선/회전2초×2카메라. RGB어깨/골반 좌우벡터x/z각도p95-p05<=30도,가시성>=.5/전프레임유한을고정. 광학기하는정답조건표시용뿐RGB입력아님. 독립각도펼침/분위수/가시성,56클립SHA/프레임/연결 및2지표2시점짝8대용값비교검산. 고정1e-10,판정완전일치. 원래사람/점수구간/5seed유지,새모델학습0. 직선유지80%/회전거부80%를각카메라에서공동판정,실패하면작은문턱격자금지. 통과못함=측정불가이며낮은점수아님.')
    observation={'checked_at':now}
    for name,key in [('main-completion.json','completion'),('main-process-exit.json','main_exit'),('pipeline-exit.json','pipeline_exit'),('main-verification.json','verification')]:
        if (HERE/name).exists():observation[key]=json.loads((HERE/name).read_text())
    if observation.get('verification',{}).get('passed') and observation.get('pipeline_exit',{}).get('exit_code')==0:item['state']='completed_verified_pending_report'
    item['last_observation']=observation;item['observed_completed_tasks']=observation.get('completion',{}).get('tasks',0)
    assert not any(x['id']==item['id'] for x in q['experiments']);q['experiments'].append(item);q.update(current_experiment_id=item['id'],updated_at=now,last_checked_at=now);save(registry,q)
    prompt=CONTROL/'automation-prompt.txt';text=prompt.read_text();first=text.index('現在') if '現在' in text else text.index('현재 연구는');end=text.index('미완료 회수·학습·분석·검산은')
    context='''현재 연구는 전용 등록부의 current_experiment_id에 해당하는 실험이다. 목표는 고정휴대폰/바닥표시없음/키1회입력으로 핵심값+실제측정가능추가값의 SPPB/TUG4구간 적중률을 개선하는 것이다. 기존 점수·속도·오차학습·영상주기·참조진단은 완료검산보고됐으므로 재학습/중복보고하지 않는다. cadence-reference-audit27명4초의 사건가용23/27,기준주기와공동19명차이20.00걸음/분,직선기하0/27였다. 후속 cadence-straight-control의 같은14명 직선/회전2초28사건224주기 검산에서는 직선 사건가용12/14,RGB3D차이3.00/1.82걸음/분;두조건공동6명기준3D차이직선1.34/회전23.96으로6명모두줄었지만공동10명목표미달이었다. 둘다힘판/전문가접촉정답없고 실제cadence정확도/SPPB/TUG정확도 아니다. 원래27명과14명표본차이/2초품질91.53spm구조적선택효과를보존한다.

지금20261101-camera-straight-gate는 영상만으로 회전 구간을 거부할 수 있는지 보는 관측품질 진단이다. 같은14명의 과거 고정직선/회전2초×PG1/PG2 56클립을쓴다. RGB MediaPipe local3D 어깨11/12·골반23/24 좌우벡터의x/z각도p95-p05최댓값<=30도,영상가시성>=.5,전프레임유한을고정했다.30도는이전기하기준에서옮겼고RGB결과로선택하지않았다. 광학기하는평가조건표시용이며RGB입력에없다. 실제해부학적방향정확도는아니다. 새학습/추론/5seed없음,기존점수모델사람/구간/5seed유지. 파일럿2명8클립의독립순차각도펼침/분위수/가시성검산을통과한후본pipeline을실행했다. 실제PID/완료파일을확인해중복실행하지않는다. main14작업56클립·종료0·56독립판정재계산·2지표/2시점짝/8대용값비교행을검산한뒤report.py를실행한다. 원본/배열/프레임/사람/조건/SHA,판정완전일치·각도1e-10을유지한다. 각카메라의직선유지율과회전오통과율,양시점공동가용/불일치,선별전후사건대용값차이와모든분모를보고한다. 통과자차이만으로전체정확도향상주장금지. 직선80%유지·회전80%거부는탐색공학목표이며임상기준아니다. 실패해도문턱을바꾸거나유리한사람만남기지않는다. 관측실패는측정불가,낮은점수아니다. 현재참조·임상정답한계가남았으므로무한작은대용값격자를피하고독립접촉정답/실제추가정보확보의가치있는다음가설을우선한다. 기존 ForcePose인증서만료/BioCV신청/OpenCap403은해결된것처럼말하지않는다. 각항목최신프로토콜기대수를따른다.

'''
    prompt.write_text(text[:first]+context+text[end:],encoding='utf8')
    print(json.dumps({'registered':item['id'],'pid':receipt['pid'],'launcher_pid':receipt['launcher_pid'],'state':item['state'],'prompt':str(prompt)},ensure_ascii=False))
