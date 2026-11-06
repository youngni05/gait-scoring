"""Record the real submitted pose job in this chat's existing registry."""
from pathlib import Path
import argparse,datetime,json
ROOT=Path('D:/Projects/D_P')
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
THREAD='01a1906b-7581-7403-8c8e-797c78676dbe'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('job');a=ap.parse_args();assert a.job.isdigit()
    reg=ROOT/'gdi-research-harness/studies/20261031-cnn-high-gdi/watch/registry.json'
    data=json.loads(reg.read_text(encoding='utf8'));assert data['thread_id']==THREAD and data['automation_id']=='gdi'
    item={'id':'gdi-tug-transfer-toronto12','kind':'public_video_pose_inference_then_authorized_cpu_comparison',
          'local':'gdi-research-harness/studies/20261031-gdi-tug-transfer','artifact_root':str(OUT),
          'remote':'/abr/RESEARCH_USER/gdi-tug-transfer-20261031','jobs':[a.job],'preserve_allocations':['32844'],
          'expected_fits':0,'expected_outer_folds':0,'expected_people':12,'expected_frames':1488,
          'summary_path':'progress.json','marker_path':'complete.json','review_state':'pending','reported_digest':None,
          'audit':'Read new study README.md and protocol.json. Pose job is NOT TUG training. Verify1488frames/48chunks, input/remote hashes, official CPU first5 coordinate equivalence<0.01. Resume receive_pose.py without duplicate consumer. Run cohort.py and visually audit12 tracks before quality-final.json and export_inputs.py. Then execute approved compare.py scalar and scratch on CPU, same retained people/intervals/outer folds; original training and final cohorts untouched. Audit raw labels, train-only normalization/inner selection, all model replays and independent person metrics. Report A/B/C/D and random-feature control; do not call this clinical validation. Keep pending until the authorized comparison or failure is reported. No additional GPU jobs or new model tuning from monitor.'}
    existing=[e for e in data['experiments'] if e['id']==item['id']]
    if existing:assert existing[0]['jobs']==[a.job]
    else:data['experiments'].append(item)
    reg.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    receipt=json.loads((OUT/'package-receipt.json').read_text())
    receipt.update({'job':a.job,'thread_id':THREAD,'registry':str(reg),'submitted_at':datetime.datetime.now().astimezone().isoformat(),
                    'remote_archive_sha256_verified':True,'phase':'submitted_pose_forward','comparison_training_started':False})
    (OUT/'submission.json').write_text(json.dumps(receipt,indent=2))
    (OUT/'status.json').write_text(json.dumps({'phase':'GPU BODY25 submitted; TUG training not started','job':a.job,
        'next':'Retrieve/decode/QA; export frozen features; authorized CPU A/B/C/D comparisons; independent audit',
        'native_cpu_124_run':'Interrupted by operator for runtime after first5 official reference completed; partial reference files retained',
        'consumer_session':None},indent=2))
    print('registered',a.job,len(data['experiments']))
if __name__=='__main__':main()
