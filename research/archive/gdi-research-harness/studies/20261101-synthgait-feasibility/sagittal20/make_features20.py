"""Reuse frozen feature functions at native timestamps, refuse fabricated 124 frames."""
from pathlib import Path
import sys,json,hashlib,warnings,argparse
import numpy as np
ROOT=Path('D:/Projects/D_P');HERE=ROOT/'gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20'
DATA=Path('D:/gait-research-data/D_P/synthgait19k-review/sagittal20');BUNDLE=ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
sys.path.insert(0,str(BUNDLE))
import preprocess
from infer import verify_bundle

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compute(raw,ns):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        p=ns['process_video_and_add_cols'](raw.reshape(len(raw),75),swap_orientation=False)
    return np.stack([p[:,preprocess.LEFT],p[:,preprocess.RIGHT]]).astype(np.float32)

def check_sides(ns):
    # Deliberately asymmetric, fully observed artificial skeleton: verifies indexing,
    # not anatomical correctness of the detector on real video.
    raw=np.ones((124,25,3));raw[:,:,0]=100+np.arange(25)*7;raw[:,:,1]=200+np.arange(25)*3;raw[:,:,2]=.9
    for j,xy in {9:(320,180),10:(330,240),11:(345,290),12:(300,180),13:(290,235),14:(275,285),19:(255,288),22:(365,295)}.items():raw[:,j,:2]=xy
    x=compute(raw,ns);center=(raw[0,9,:2]+raw[0,12,:2])/2
    scale=(np.linalg.norm(raw[0,9,:2]-raw[0,10,:2])+np.linalg.norm(raw[0,12,:2]-raw[0,13,:2]))/2
    for s,joints in enumerate([[14,13,12,19],[11,10,9,22]]):
        expected=((raw[0,joints,:2]-center)/scale).reshape(-1)
        np.testing.assert_allclose(x[s,0,:8],expected,rtol=0,atol=1e-6)
    pairs=[(2,5),(3,6),(4,7),(9,12),(10,13),(11,14),(15,16),(17,18),(19,22),(20,23),(21,24)]
    swapped=raw.copy()
    for a,b in pairs:swapped[:,a]=raw[:,b];swapped[:,b]=raw[:,a]
    err=float(np.max(np.abs(compute(swapped,ns)-x[::-1])));assert err<1e-6
    reflected=raw.copy();reflected[:,:,0]=831-raw[:,:,0]
    expected=x.copy();expected[:,:,[0,2,4,6,11]]*=-1
    mirrorerr=float(np.max(np.abs(compute(reflected,ns)-expected)));assert mirrorerr<1e-6
    return {'passed':True,'sample_kind':'artificial asymmetric skeleton, not video ground truth','left_joints':[14,13,12,19],
        'right_joints':[11,10,9,22],'left_right_swap_max_error':err,'horizontal_reflection_max_error':mirrorerr,
        'automatic_direction_flip':False,'automatic_left_right_swap':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selected',action='store_true');args=parser.parse_args()
    verify_bundle();ns=preprocess.source_functions();side=check_sides(ns);save(HERE/'side-mapping-check.json',side)
    original=json.loads((HERE/'download-manifest.json').read_text(encoding='utf-8'))
    if args.selected:
        selection=json.loads((HERE/'target-selection-review.json').read_text(encoding='utf-8'));assert selection['visually_reviewed_all_frames']
    pose=json.loads((HERE/('pose-verification.selected.json' if args.selected else 'pose-verification.json')).read_text(encoding='utf-8'));assert pose['complete']
    out=DATA/('features_selected' if args.selected else 'features_native');out.mkdir(exist_ok=True);rows=[];identity=[]
    for row,pr in zip(original['records'],pose['records']):
        assert row['sample_id']==pr['sample_id']
        for f in row['files'].values():assert sha(Path(f['path']))==f['sha256']
        with np.load(pr['raw_path'],allow_pickle=False) as z:raw=z['raw'];time=z['time_seconds'];counts=z['detected_people'];frames=z['frame']
        n=len(raw);assert np.array_equal(frames,np.arange(n)) and np.array_equal(time,np.arange(n)/16)
        x=compute(raw,ns);assert x.shape==(2,n,12)
        observed=(np.isfinite(raw[:,:,:2]).all(axis=2)&(raw[:,:,:2]!=0).all(axis=2)&(raw[:,:,2]>0))
        required=observed[:,preprocess.REQUIRED]
        strikes={}
        for side_name in ['left','right']:
            source=np.array(row['metadata'][side_name+'_heel_strikes'],np.int64);seconds=source/30.
            nearest=np.rint(seconds*16).astype(np.int64);inside=(nearest>=0)&(nearest<n)
            # Do not clamp out-of-video events to an invented boundary contact.
            strikes[side_name]={'source_annotation_indices':source.tolist(),'seconds':seconds.tolist(),
                'nearest_video_indices':nearest.tolist(),'within_frame_range':inside.tolist()}
            assert np.array_equal(source,np.rint(seconds*30).astype(np.int64))
            if inside.any():assert np.max(np.abs(nearest[inside]/16-seconds[inside]))<=1/32+1e-12
        cadence=float(row['metadata']['gait_features']['cadence'])
        dest=out/(row['sample_id']+'.npz')
        np.savez_compressed(dest,x_native=x,raw_body25=raw,observed_joint_mask=observed,frame=frames,time_seconds=time,
                            fps=np.array(16.),cadence=np.array(cadence),split=np.array(row['split']),walk_id=np.array(row['walk_id']))
        finite=np.isfinite(x);finite_ratio=float(finite.mean());reason=None
        try:
            # The original fixed-window extractor must explicitly refuse short video.
            preprocess.extract_windows(raw.reshape(n,75))
            raise AssertionError('Expected original window extractor to reject short input')
        except ValueError as error:reason=str(error)
        report={'sample_id':row['sample_id'],'walk_id':row['walk_id'],'split':row['split'],'frames':n,'features_shape':list(x.shape),
            'features_sha256':sha(dest),'features_path':str(dest),'all_features_finite':bool(finite.all()),'finite_fraction':finite_ratio,
            'required_joint_missing_fraction':float(1-required.mean()),'max_required_joint_missing_fraction':float(1-required.mean(axis=0).min()),
            'left_toe_observed_fraction':float(observed[:,19].mean()),'right_toe_observed_fraction':float(observed[:,22].mean()),
            'unique_person_frames':int((counts==1).sum()),'explicitly_selected_frames':pr.get('explicitly_selected_frames',0),
            'source_fps':16,'annotation_fps':30,'native_124_frames_available':False,
            'original_extractor_rejection':reason,'heel_strikes':strikes,
            'events_outside_video_frame_range':sum(len(s['within_frame_range'])-sum(s['within_frame_range']) for s in strikes.values()),
            'resampling_performed':False,'gdi_label_created':False}
        rows.append(report);identity.append((row['walk_id'],row['split']))
        print(json.dumps({'sample_id':row['sample_id'],'frames':n,'finite_fraction':finite_ratio,'outside_events':report['events_outside_video_frame_range']}),flush=True)
    train={k for k,s in identity if s=='train'};valid={k for k,s in identity if s=='validation'};assert len(train)==16 and len(valid)==4 and not train&valid
    result={'conversion_complete':True,'records':rows,'total_frames':sum(r['frames'] for r in rows),'side_order':['left','right'],
        'feature_names':preprocess.FEATURE_NAMES,'same_fixed_feature_code':True,'same_body25_schema':True,
        'same_physical_time_as_gdi_verified':False,'gdi_training_json_fps':None,'native124_eligible_records':0,
        'finite_feature_records':sum(r['all_features_finite'] for r in rows),'contact_outside_video_records':sum(r['events_outside_video_frame_range']>0 for r in rows),
        'split_walk_disjoint':True,'subject_disjoint_verifiable':False,'resampling_performed':False,'gdi_label_created':False,
        'pretraining_gpu_pilot_ready':False,'blocking_reasons':['All 20 native videos have 25..81 frames; original CNN needs124.',
            'Actual FPS/time mapping of existing GDI JSON is unverified.','30Hz heel-strike origin/frame correspondence requires further validation.'],
        'evidence':{'side_mapping_check_sha256':sha(HERE/'side-mapping-check.json'),'sample_replay_sha256':sha(HERE/'sample-replay.json'),
            'preprocess_sha256':sha(BUNDLE/'preprocess.py'),'source_utility_sha256':sha(BUNDLE/'sources/video_process_utils.py'),
            'source_feature_notebook_sha256':sha(BUNDLE/'sources/process_raw_videos.ipynb')}}
    save(HERE/('feature-verification.selected.json' if args.selected else 'feature-verification.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','evidence','feature_names']},indent=2))
if __name__=='__main__':main()
