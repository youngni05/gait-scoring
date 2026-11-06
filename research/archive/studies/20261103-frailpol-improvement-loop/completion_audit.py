"""Closeout: unchanged evidence, full-person metrics and fresh original-model replay."""
from pathlib import Path
import json,hashlib,datetime,time,sys
import numpy as np,openpyxl
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop')
B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103')
G=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn')
GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104')
M=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop')
S=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def main():
    bridge=read(H/'current-label-bridge.json');assert bridge['status']=='passed'
    for p,key in [(S/'raw/participants.xlsx','source_workbook_sha256'),(S/'cohorts.json','source_cohort_sha256'),(GB/'data.npz','current_data_sha256'),(GB/'native-motion/data.npz','native_data_sha256')]:assert sha(p)==bridge[key]
    d=np.load(GB/'data.npz');ids=d['ids'];y=d['y'];fold=d['fold'];assert len(ids)==len(set(ids))==594 and sum(y>=20)==9
    rows=list(openpyxl.load_workbook(S/'raw/participants.xlsx',read_only=True,data_only=True).active.values);headers=[str(x).strip() for x in rows[0]];byid={int(r[0]):dict(zip(headers,r)) for r in rows[1:] if isinstance(r[0],(int,float))};assert len(byid)==771 and np.array_equal(y,[byid[int(i)]['TUG time (s)'] for i in ids])
    source=read(S/'cohorts.json');assert len(source)==668;eligible=[r for r in source if r['status']=='ok' and r['tug'] is not None];assert ids.tolist()==[r['id'] for r in eligible] and fold.tolist()==[r['fold'] for r in eligible]
    folds=[]
    for f in range(5):
        tr=ids[fold!=f];te=ids[fold==f];assert not set(tr)&set(te) and set(tr)|set(te)==set(ids);folds.append(dict(fold=f,training_people=len(tr),test_people=len(te)))
    artifacts={};manifest_count=0
    for i in range(1,13):
        root=H/f'round{i:02d}';v=read(root/'verification.json');assert v['status']=='passed';assert sha(root/'comparison.json')==v['comparison_sha256']
        review=H/f'review{i}.py';key='review_code_sha256' if 'review_code_sha256' in v else 'review_sha256';assert sha(review)==v[key]
        assert read(root/'protocol.json')['code_sha256']==sha(H/f'round{i}.py')
        receipt_paths=[root/'receipt-speed.json',root/'receipt-bilateral.json'] if i==6 else [root/'run-receipt.json']
        for rp in receipt_paths:
            receipt=read(rp);assert sha(root/'protocol.json')==receipt['protocol_sha256']
            for p,h in receipt['hashes'].items():assert sha(p)==h;manifest_count+=1
            artifacts[str(rp)]=sha(rp)
        uncertainty=root/'uncertainty.json' if i>=7 else H/'uncertainty.json'
        for p in [root/'protocol.json',root/'verification.json',root/'comparison.json',uncertainty]:artifacts[str(p)]=sha(p)
    for name in ['local-directml','native-motion','manual-controls','norm-aux']:
        root=G/name;v=read(root/'verification.json');assert v['status']==('partially_passed_with_auxiliary_logit_failures' if name=='norm-aux' else 'passed')
        if name=='norm-aux':assert v['failed_conditions']==['symnorm_aux_tail4'] and len(v['passed_conditions'])==4
        for p in [root/'verification.json',root/'comparison.json']:artifacts[str(p)]=sha(p)
    for path in [G/'data-verification.json',G/'native-motion/input-verification.json']:
        assert read(path)['status']=='passed';artifacts[str(path)]=sha(path)
    path=G/'norm-input/verification.json';norm=read(path);assert norm['status']=='passed_input_invariance_only_no_model_fitted' and norm['people']==594 and norm['left_right_swap_exact'] and not norm['TUG_labels_accessed'];assert norm['output_sha256']==sha(GB/'norm-input/inputs.npz') and norm['protocol_sha256']==sha(G/'norm-input/protocol.json');artifacts[str(path)]=sha(path)
    print('Original labels, frozen cohort, prior protocols and artifact hashes checked:',manifest_count,flush=True)
    deadline=time.monotonic()+1200
    while '--wait-for-replay' in sys.argv and not (H/'final-model-replay/verification.json').exists():
        assert time.monotonic()<deadline,'Final replay receipt still absent after bounded wait'
        time.sleep(10)
    replay=read(H/'final-model-replay/verification.json');assert replay['status']=='passed' and replay['final_base_models_replayed']==245 and replay['gate_models_replayed']==60 and replay['conditions_rebuilt']==24
    assert replay['code_sha256']==sha(H/'final_model_replay.py') and replay['fresh_heldout_sha256']==sha(B/'final-model-replay/fresh-heldout.json')
    assert replay['source_hashes_sha256']==sha(H/'final-model-replay/source-model-hashes.json')
    for p,h in read(H/'final-model-replay/source-model-hashes.json').items():assert sha(p)==h
    scoreboard=read(H/'scoreboard.json');assert scoreboard['verified_conditions']==134 and not scoreboard['pending_studies'] and scoreboard['code_sha256']==sha(H/'scoreboard.py')
    expected=['round12/combined/config1/guarded','round12/combined/config3/guarded'];assert scoreboard['numeric_goal_hits']==expected and replay['goal_meeting_configurations']==[k.removeprefix('round12/') for k in expected]
    assert read(M/'first4s-verification.json')['status']=='passed';assert read(M/'motion-pretrain/verification.json')['status']=='failed_pretraining_reconstruction_cpu64_tolerance'
    monitor=read(G/'native-motion/local-monitor-state.json');assert len(monitor['completed_models'])==3 and monitor['interval_seconds'] is None and monitor['next_not_before_utc'] is None
    for p in [H/'current-label-bridge.json',H/'scoreboard.json',H/'final-model-replay/verification.json',M/'first4s-verification.json',M/'motion-pretrain/verification.json']:artifacts[str(p)]=sha(p)
    result=dict(status='passed_research_development_goal',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),people=594,slow_people=9,outer_folds=folds,numeric_goal_hits=expected,verified_conditions=134,preserved_ineligible_norm_aux_conditions=['symnorm_aux_tail4'],original_training_artifact_hashes_checked=manifest_count,evidence_hashes=artifacts,goal_metrics={k:scoreboard['conditions'][k] for k in expected},source_model_replay=dict(base=245,gates=60,policies=24),selection_scope='Source candidate/epoch selection was inner-training-only. All four round12 fixed configurations and both policies were predeclared and reported; no production winner selected from outer outcomes. Architecture exploration reused outer folds: development evidence only, not untouched validation.',coverage='Original668 IMU cohort;72 technical failures and2 missingTUG remain unavailable,not reclassified. Extra103 metadata have no cohort gait input. All594 available people retained in every reported condition.',limits=['Research point-estimate goals only;conditional intervals cross target cutoffs.','Only9 peopleTUG>=20;not externally or clinically validated.','No originalCP-GDI/SPPB labels inFRAILPOL/Toronto.','MediaPipe14 exploratory: no established gain;failed reconstruction replay not adopted.','Sensor input0.69..3.98s;not proof of camera orMediaPipe measurement accuracy.','WholeTUG timing used only for outcome-blind localization,not an input predictor.','No unrelated jobs or schedules changed. Allactual localGPUtrainingandreviewsfinished;remoteactualdatatransferstillawaitsseparateexplicitapproval.'],code_sha256=sha(Path(__file__)))
    save(H/'completion-audit.json',result)
    previous=H/'STATE-pre-final-closeout.json'
    if not previous.exists():previous.write_bytes((H/'STATE.json').read_bytes())
    state=dict(status='research_development_goal_verified',owner_thread='01a1a1c2-655b-79a3-87bc-e325af86028e',updated_utc=result['completed_utc'],goal=dict(mae_max=.9,p95_absolute_error_max=3,tug20plus_mae_max=3,people=594),sessions={},next='Improvement loop closed at the existing numerical development criteria. No new training automatically queued. Both fixed qualifying configurations reported; fresh external/person validation and camera-specific evidence remain outside the achieved claim.',completion_audit=str(H/'completion-audit.json'),numeric_goal_hits=expected,verified_conditions=134,historical_state=str(previous),all_results=str(H/'scoreboard.json'),large_root=str(B),mediapipe_scope=dict(people=14,status='exploratory_no_established_gain',verified_comparison=str(M/'first4s-comparison.json'),pretraining_failed_not_adopted=str(M/'motion-pretrain/verification.json')),school_actual_data_transfer='pending_explicit_user_approval_no_upload_or_real_job_submitted',unrelated_jobs_and_automations='unchanged')
    state['round11']=dict(status='passed',models=60,policies=24,goal_met=False);state['round12']=dict(status='passed_all60gates_and245source_final_models_replayed',qualifying_configs=expected,all_policies_reported=24)
    state['native_gpu_closeout']=dict(status='all3conditions_training_and195checkpoint_reviews_finished',unrelated_jobs_untouched=True)
    save(H/'STATE.json',state);print(json.dumps({k:result[k] for k in ['status','people','slow_people','numeric_goal_hits','verified_conditions','original_training_artifact_hashes_checked']},indent=2))
if __name__=='__main__':main()
