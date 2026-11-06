from pathlib import Path
import json,datetime
import numpy as np
import sensor_loop as s
from verify_batch import delta
HERE=s.HERE;HEAVY=s.HEAVY
def main():
    checks=['sensor-verification.json','projection-verification.json','refinement-verification.json','extension-verification.json']
    for name in checks:
        v=json.loads((HERE/name).read_text());assert v['passed']
        result=name.replace('verification','results');assert s.sha(HERE/result)==v['results_sha256']
    source=np.load(s.SOURCE);y=source['y'];ids=source['ids'];score=json.loads(Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop/scoreboard.json').read_text())
    ref=score['conditions']['round07/speed/rbf_ridge/tail4'];path=Path(ref['source_heldout']);assert s.sha(path)==ref['source_heldout_sha256']
    records=json.loads(path.read_text());selected=[r for r in records if r['condition']+('/'+r['kind'] if 'kind' in r else '')==ref['condition']]
    lookup={int(r['id']):r for r in selected};assert len(selected)==len(lookup)==594 and set(lookup)==set(ids)
    baseline=np.array([lookup[int(i)]['prediction'] for i in ids]);assert all(lookup[int(i)]['actual']==t for i,t in zip(ids,y))
    new=np.load(HEAVY/'refinement/predictions.npz');comp={}
    for name in ['metric12__mae_tail1','metric12__mae_tail4','metric12__ridge10','timing7__mae_tail1']:
        comp[name]=delta(abs(new[name]-y),abs(baseline-y))
    s.save(HERE/'descriptive-strong-speed-comparison.json',{'reference':'historical speed winner1.038194671; descriptive, source and candidate selected after searches',
      'reference_sha256':s.sha(path),'comparisons':comp,'limits':'Conditional bootstrap, reused594 people, not corrected for multiple development comparisons.'})
    sensor=json.loads((HERE/'sensor-results.json').read_text());projection=json.loads((HERE/'projection-results.json').read_text());refine=json.loads((HERE/'refinement-results.json').read_text());extension=json.loads((HERE/'extension-results.json').read_text())
    inventory=[
      {'name':'GSTRIDE V3','published_people':163,'processed_gait_people':158,'usable_shortwalk_realTUG_people':142,'severe_TUG20plus':32,
       'source':'https://zenodo.org/records/17052815','status':'downloaded_CRC_SHA_verified_and_analyzed','input':'nominal camera-computable quantities estimated by onefootIMU; noRGB',
       'role':'separate first-pass external test, followed by clearly separate local/pooled development; do not claim phone validation'},
      {'name':'HealthAndGait','people':398,'videos':1564,'source':'https://zenodo.org/records/14039922','status':'primary_source_refreshed_not_added_to_TUG_training',
       'role':'Potential2D feature measurement comparison against OptoGait/MuscleLAB, noTUGtarget. Requires full measurement/clip linkage audit; not new clinical outcome validation'},
      {'name':'TCAI-FallNet 2026','people':688,'source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC13306775/','status':'rawvideo_and_keypoints_not_public_not_downloaded',
       'role':'Method reference only. Author request required; no message sent'},
      {'name':'WorldSpacedHMR2026','people':52,'paired_sensor_people':30,'source':'https://arxiv.org/html/2604.11961v1','status':'paper_source_read_no_public_repository_verified',
       'role':'Measurement-method reference, not an acquired independent test set'},
      {'name':'GIST_TUG2022','people':88,'source':'https://www.mdpi.com/1424-8220/22/17/6323','status':'paper_reports_no_applicable_data_sharing_no_new_download',
       'role':'Manual subtask event annotation reference, not available test data'}]
    s.save(HERE/'data-source-refresh.json',inventory)
    q=projection['geometry'];candidate=refine['summary']['metric12/mae_tail4'];diff=refine['comparisons']['metric12/mae_tail4']['previous_metric12']
    lines=['평면 영상 호환 TUG 모델 개선 및 투영 점검 — 실행 결과 기록','',
      '1. 투영 검산: 같은 OpenCap297명, 같은3초 구간, 같은25개 사람 분할. 원래64특징 재생성 오차0.',
      f"속도 손실: 절대상대오차95백분위 {q['speed_abs_relative_error_p95']*100:.3f}%. 무릎 평균각 차이 중앙값 {q['knee_angle_mean_abs_difference_median_degrees']:.3f}도.",
      '투영 비교 전체 MAE(초):']
    for key,r in projection['summary'].items():
        if key.endswith('/features'):lines.append(f"  {key}: {r['MAE_repeated']:.6f}")
    lines+=['이 범위의 투영 차이는1.0초와1.8초의 차이를 설명하지 못했다. 원래3D자체오차/실제카메라가림/MediaPipe측정은 검증하지 못했다.','',
      '2. FRAILPOL594명: 허용1/3/7/12특징만 사용한 새12학습구성과 가상측정오차5/10%학습, 기존강한특징모델과의 재결합을 비교했다.',
      f"이전12특징0.982086초. 새 사전 주분석0.990781초. 탐색상 최저 후보{candidate['MAE']:.6f}초.",
      f"최저 후보의 이전 대비차이 {diff['delta_MAE']:.6f}초,95%구간 {diff['CI95']}. 확정 개선으로 채택하지 않는다.",
      f"고TUG9명 MAE {candidate['high_MAE']:.6f}초. 센서측정가능성/결측패턴 진단대조군은{sensor['summary']['availability_diagnostic/stacked']['MAE']:.6f}초로 해석에 주의가 필요하다.",
      '모든 값은센서추정치를 사용했다. 실제 평면영상 추출성능이 아니다.','',
      '3. 새 GSTRIDE:163명중 처리파일158명, 유효한임상TUG와<=4초보행을연결한142명(고TUG32명). 기존594명과합쳐736명 개발비교.',
      f"새142명에FRAILPOL모델 무보정 적용: 속도{extension['external/speed']['MAE']:.6f}초,기본3특징{extension['external/core3']['MAE']:.6f}초.",
      f"새142명 내부사람분리: 속도{extension['local_GSTRIDE/speed']['MAE']:.6f}초,기본3특징{extension['local_GSTRIDE/core3']['MAE']:.6f}초.",
      f"자료결합 후 새142명: 기본3특징{extension['pooled_GSTRIDE/core3']['MAE']:.6f}초,기존594명:{extension['pooled_FRAILPOL/core3']['MAE']:.6f}초.",
      '자료를추가한효과의구간은0을포함했다. 서로다른모집단/보행과제/한발대양발/이벤트정의차이를확인해야하며,현재무조건합치는방법을채택하지않는다.',
      '4.688명영상연구는원본영상과키포인트비공개. 새로확보한대규모휴대폰영상+실제TUG검증세트로기록하지않는다.','',
      '검산: 새 센서특징모델1200개+메타50개,투영모델300개,개선결합75개,자료확장모델960개+메타20개 검산통과.']
    (HERE/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state={'status':'completed_verified_two_sensor_batches_projection_audit_and_new_cohort_extension',
      'completed_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
      'verification_hashes':{n:s.sha(HERE/n) for n in checks},'prior_metric12_reference_retained':True,
      'new_best_observed_candidate_MAE':candidate['MAE'],'adopted_as_confirmed_improvement':False,
      'actual_new_RGB_TUG_cohort_added':False,'new_sensor_proxy_people_added':142,'new_source_severe_people':32,
      'no_new_GPU_jobs':True,'no_author_messages_sent':True,'no_ongoing_training_from_this_batch':True,
      'limits':['Repeateddevelopment,conditionalCIsnotsearchadjusted','Syntheticprojectionnotactualphonevalidation','Newcohortonefoot_vs_bilateral_event/taskshift','Sensor-derivedvaluesnotactualcameraextraction']}
    s.save(HERE/'STATE.json',state)
    names=['sensor-protocol.json','projection-protocol.json','refinement-protocol.json','extension-protocol.json','sensor-results.json','projection-results.json','refinement-results.json','extension-results.json','RESULTS.txt','STATE.json','data-source-refresh.json']+checks
    s.save(HERE/'artifact-manifest.json',{n:s.sha(HERE/n) for n in names})
    print('BATCH_CLOSED_VERIFIED',flush=True)
if __name__=='__main__':main()
