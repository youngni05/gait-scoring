"""Independent file/identity audit; no fitting, no synthetic GDI labels."""
from pathlib import Path
import csv,json,hashlib,subprocess,sys,ast,datetime
import numpy as np
ROOT=Path('D:/Projects/D_P');HERE=ROOT/'gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20'
DATA=Path('D:/gait-research-data/D_P/synthgait19k-review/sagittal20');REMOTE='/abr/RESEARCH_USER/synthgait-pose20-20261101'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST';SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST];SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=120)
    assert p.returncode==0,(p.stdout,p.stderr)
    return p.stdout
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
    registry=json.loads((ROOT/'gdi-research-harness/control/gpu-watch/registry.json').read_text(encoding='utf-8-sig'))
    entry=next(e for e in registry['experiments'] if e['id']=='synthgait-sagittal20-body25')
    assert registry['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a' and entry['jobs']==['33604'] and entry['remote']==REMOTE
    receipt=[];received=DATA/'received_meta';received.mkdir(exist_ok=True)
    for name in ['full/index.json','full/complete.json','full/equivalence.json','logs/worker-33604.log','gpu_pose_worker.py','pose_torch.py','pose_jobs.json','pose.slurm']:
        server=command(SSH+['sha256sum '+REMOTE+'/'+name]).split()[0];dest=received/name.replace('/','_')
        command(SCP+[HOST+':'+REMOTE+'/'+name,str(dest)]);assert sha(dest)==server
        receipt.append({'remote':REMOTE+'/'+name,'local':str(dest),'server_sha256':server,'local_sha256':sha(dest)})
    accounting=command(SSH+['sacct -n -P -j 33604 --format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])
    assert any(line.startswith('33604|COMPLETED|0:0|') for line in accounting.splitlines())
    summary=json.loads((received/'full_complete.json').read_text());assert summary['frames']==1231 and len(summary['records'])==20
    maps=json.loads((DATA/'received_maps/receipt.json').read_text())
    json_hashes={n:v for row in maps.values() for n,v in row['json_sha256'].items()}
    for name,item in maps.items():assert sha(DATA/'received_maps'/name)==item['server_sha256']==item['local_sha256']
    selected={row['filename']:int(row['person_index']) for row in csv.DictReader((HERE/'target-selection.csv').open(encoding='utf-8'))}
    sources=json.loads((HERE/'download-manifest.json').read_text(encoding='utf-8'))
    features=json.loads((HERE/'feature-verification.selected.json').read_text());audited=0
    names=['cadence','walking_speed','step_length_mean','step_width_mean','stoop_posture_mean','arm_swing']
    for source,feature in zip(sources['records'],features['records']):
        assert source['sample_id']==feature['sample_id']
        for file in source['files'].values():assert sha(Path(file['path']))==file['sha256']
        with np.load(source['files']['.npz']['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['features'],[source['metadata']['gait_features'][n] for n in names])
            for side in ['left','right']:np.testing.assert_array_equal(z[side+'_heel_strikes'],source['metadata'][side+'_heel_strikes'])
        path=Path(feature['features_path']);assert sha(path)==feature['features_sha256']
        with np.load(path,allow_pickle=False) as z:
            assert np.isfinite(z['x_native']).all() and z['x_native'].shape==(2,feature['frames'],12)
            assert not any(k in z.files for k in ['y','GDI','target','left_GDI','right_GDI'])
            np.testing.assert_array_equal(z['time_seconds'],np.arange(feature['frames'])/16.)
            for i,p in enumerate(sorted((DATA/'body25'/source['sample_id']).glob('*_keypoints.json'))):
                assert sha(p)==json_hashes[p.name]
                people=json.loads(p.read_text())['people'];index=selected.get(p.name,0 if len(people)==1 else -1)
                expected=np.array(people[index]['pose_keypoints_2d']).reshape(25,3) if index>=0 else np.full((25,3),np.nan)
                np.testing.assert_array_equal(z['raw_body25'][i],expected);audited+=1
    assert audited==1231 and features['finite_feature_records']==20 and features['native124_eligible_records']==0
    quality=[r['sample_id'] for r in features['records'] if r['max_required_joint_missing_fraction']>.25]
    timing=json.loads((HERE/'timing-diagnostic.json').read_text())
    side=json.loads((HERE/'side-mapping-check.json').read_text());assert side['passed']
    real=json.loads((HERE/'real-frame-side-diagnostic.json').read_text())
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    sys.path.insert(0,str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'))
    from infer import verify_bundle
    verify_bundle()
    result={'artifact_audit_passed':True,'checked_at':datetime.datetime.now().astimezone().isoformat(),
        'verified_source_triplets':20,'verified_native_frames':audited,'verified_finite_feature_records':20,'body25_json_hashes_checked':len(json_hashes),
        'gpu_job':'33604','accounting':accounting,'optimizer_steps':0,'server_local_sha256_matched':True,
        'receipt':receipt,'chunk_receipt_sha256':sha(DATA/'received_maps/receipt.json'),
        'feature_report_sha256':sha(HERE/'feature-verification.selected.json'),
        'sample_replay':json.loads((HERE/'sample-replay.json').read_text()),'side_mapping':side,'real_side_diagnostic':real,
        'whole_clip_required_joint_missing_over25pct':quality,'whole_clip_missing_check_is_not_124frame_eligibility':True,
        'time_diagnostic':{'outside_frame_range_records':features['contact_outside_video_records'],
            'outside_duration_by_more_than_one_frame_records':timing['records_with_events_beyond_video_by_more_than_one_frame'],
            'maximum_excess_after_duration_seconds':timing['max_excess_seconds']},
        'pretraining_pilot_ready':False,'native124_eligible_records':0,'matched_physical_time_to_gdi_verified':False,
        'blocking_reasons':features['blocking_reasons']+['Two clips exceed25% required-joint missingness over full native clip; finite imputation is not quality assurance.'],
        'protected_final_data_used':False,'existing_training_changed':False,
        'initial_decoder_failure_preserved':True,'initial_unique_only_exclusion_preserved':True}
    save(HERE/'final-verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['receipt','accounting','sample_replay','side_mapping','real_side_diagnostic']},indent=2))
if __name__=='__main__':main()
