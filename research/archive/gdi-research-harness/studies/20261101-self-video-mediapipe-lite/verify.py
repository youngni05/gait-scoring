"""Independently verify unchanged inputs/model and recompute the comparisons."""
import math
import statistics
import warnings
from common import *

def main():
    full_preserve=read(HERE/'full-preservation.json')
    for name,digest in full_preserve.items():assert sha(FULL/name)==digest,('Full artifact changed',name)
    assert read(FULL/'verification.json')['passed']
    result=read(HERE/'results.json');before=read(OLD/'results.json');windows=read(HERE/'windows.json');original=read(OLD/'windows.json')
    extraction=read(HERE/'extraction.json');prep=module('verify_feature_source',OLD/'prepare_features.py');ns=prep.source_functions()
    assert sha(HERE/'protocol.json')==extraction['protocol_sha256']==result['protocol_sha256']
    assert sha(HERE/'windows.json')==result['windows_sha256']
    assert sha(BIG/'features.npz')==result['features_sha256']==read(HERE/'preparation.json')['features_sha256']
    assert sha(BIG/'predictions.npz')==result['predictions_sha256']
    assert sha(TASK)==read(HERE/'model-receipt.json')['sha256']==extraction['task_sha256']
    assert [(w['clip'],w['kind'],w['start'],w['end']) for w in windows]==[(w['clip'],w['kind'],w['start'],w['end']) for w in original]
    x=np.load(BIG/'features.npz')['x'];mapping_error=feature_error=pose_error=0.
    pairs=[(23,12),(25,13),(27,14),(31,19),(24,9),(26,10),(28,11),(32,22)]
    for row in read(OLD/'inventory.json'):
        clip=row['id'];assert sha(row['file'])==row['sha256'];z=np.load(BIG/(clip+'-pose.npz'));full=np.load(FULL_BIG/(clip+'-pose.npz'))
        r=next(r for r in extraction['runs'] if r['clip']==clip);lm=z['landmarks'];raw=z['raw']
        assert sha(BIG/(clip+'-pose.npz'))==r['output_sha256'] and lm.shape==(row['frames'],33,5)
        assert np.array_equal(z['source_ms'],full['source_ms']) and np.array_equal(z['task_ms'],full['task_ms'])
        assert np.all(np.diff(z['task_ms'])>0)
        assert int((z['counts']==1).sum())==r['detected'] and int((z['counts']==0).sum())==r['missing']
        assert abs(float(z['inference_seconds'].sum())-r['inference_seconds'])<1e-9
        for src,dst in pairs:
            for axis,scale in [(0,row['width']),(1,row['height'])]:
                expected=lm[:,src,axis]*scale
                np.testing.assert_array_equal(np.isnan(expected),np.isnan(raw[:,3*dst+axis]))
                mapping_error=max(mapping_error,float(np.nanmax(np.abs(expected-raw[:,3*dst+axis]))))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);proc=ns['process_video_and_add_cols'](raw,swap_orientation=False)
        for i,w in enumerate(windows):
            if w['clip']!=clip:continue
            s=w['start'];e=w['end']+1
            expected=np.array([proc[s:e,prep.LEFT],proc[s:e,prep.RIGHT]],dtype=np.float32)
            feature_error=max(feature_error,float(np.abs(x[i]-expected).max()))
            missing=float(ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](raw[s:e]),prep.REQUIRED))
            assert missing<=.25 and missing==w['missing'] and np.isfinite(expected).all()
        metrics=next(c for c in read(HERE/'full-lite-pose.json')['clips'] if c['clip']==clip)['metrics']
        for name,j in JOINTS.items():
            a=full['raw'].reshape(-1,25,3)[:,j,:2];b=raw.reshape(-1,25,3)[:,j,:2]
            good=np.isfinite(a).all(1)&np.isfinite(b).all(1)&(a!=0).all(1)&(b!=0).all(1)
            distances=np.sqrt(((a[good]-b[good])**2).sum(1))
            pose_error=max(pose_error,abs(float(np.median(distances))-metrics['joints'][name]['pixel_distance']['median']))
    assert mapping_error==feature_error==pose_error==0
    assert result['members']==before['members'] and result['optimizer_steps']==0 and len(result['members'])==54
    for member in result['members']:assert sha(member['checkpoint'])==member['checkpoint_sha256']
    with np.load(BIG/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    selection=read(HERE.parent/'20261030-gpu-motionbert-study/mean-difference692/analysis/selection.json')['folds'];rebuilt=[]
    for f in range(5):
        parts={}
        for family in ['CNN','STT']:
            keys=[k for k in pred if k.startswith('%s_%d_'%(family,f)) and not k.endswith('_bag')]
            assert len(keys)==(4 if family=='CNN' and f==0 else 5)
            parts[family]=sum(pred[k] for k in keys)/len(keys)
            np.testing.assert_allclose(parts[family],pred['%s_%d_bag'%(family,f)],rtol=0,atol=1e-12)
        sf=selection[f];base=sf['cnn_weight']*parts['CNN']+(1-sf['cnn_weight'])*parts['STT'];avg=(base[:,0]+base[:,1])/2
        gap=(1-sf['fraction'])*(base[:,0]-base[:,1])+sf['fraction']*(pred['bilateral_%d'%f][:,0]-pred['bilateral_%d'%f][:,1])
        v=np.array([avg+gap/2,avg-gap/2]).T;np.testing.assert_allclose(v,pred['best_%d'%f],rtol=0,atol=1e-12);rebuilt.append(v)
    np.testing.assert_allclose(sum(rebuilt)/5,pred['best'],rtol=0,atol=1e-12)
    summary_error=0
    for kind,models in result['summary'].items():
        for name,stats in models.items():
            means={}
            for clip in ['clip0','clip1']:
                ids=[i for i,w in enumerate(windows) if w['clip']==clip and w['kind']==kind]
                means[clip]=math.fsum(float(pred[name][i,j]) for i in ids for j in (0,1))/(2*len(ids))
                summary_error=max(summary_error,abs(means[clip]-stats[clip]['mean']))
            summary_error=max(summary_error,abs(means['clip1']-means['clip0']-stats['normal_minus_abnormal']))
    assert summary_error<1e-10
    bench=read(HERE/'benchmark.json');assert len(bench['runs'])==8
    for r in bench['runs']:
        assert r['frames']==len(r['per_frame_seconds'])==50 and r['detected_frames']+r['missing_frames']==50
        assert abs(math.fsum(r['per_frame_seconds'])-r['step_seconds'])<1e-9
    for variant in ['Full','Lite']:
        values=[r['step_seconds'] for r in bench['runs'] if r['variant']==variant]
        assert len(values)==4 and statistics.median(values)==bench['summary'][variant]['median_seconds']
    for clip,hashes in bench['input_frame_hashes'].items():assert hashes==read(FULL/'benchmark-inputs.json')['rows'][0 if clip=='clip0' else 1]['all_frame_pixel_sha256'][:25]
    report=dict(passed=True,original_full_artifacts_unchanged=len(full_preserve),original_videos_sha256_verified=2,pose_frames=2382,windows=len(windows),
        mapping_max_error=mapping_error,features_max_error=feature_error,pose_median_recalculation_max_error=pose_error,
        checkpoint_calibration_and_combination_unchanged=True,checkpoint_sha256_verified=54,summary_recalculation_max_error=summary_error,
        independently_rebuilt_ensemble=True,benchmark_passes_verified=8,identical_full_lite_timestamps=True,optimizer_steps=0,
        results_sha256=sha(HERE/'results.json'),predictions_sha256=sha(BIG/'predictions.npz'),task_sha256=sha(TASK),
        no_ground_truth=True,smartphone_tested=False)
    save(HERE/'verification.json',report);print(report,flush=True)

if __name__=='__main__':main()
