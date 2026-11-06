"""Independent adapter, unchanged-model, result arithmetic and timing checks."""
import math
import warnings
from common import *
import cv2

def main():
    result=read(HERE/'results.json');before=read(OLD/'results.json');windows=read(HERE/'windows.json');oldwindows=read(OLD/'windows.json')
    extraction=read(HERE/'extraction.json');prep=module('independent_preprocessing',OLD/'prepare_features.py');ns=prep.source_functions()
    assert sha(HERE/'protocol.json')==extraction['protocol_sha256']==result['protocol_sha256']
    assert sha(HERE/'windows.json')==result['windows_sha256']
    assert sha(BIG/'features.npz')==result['features_sha256']==read(HERE/'preparation.json')['features_sha256']
    assert sha(BIG/'predictions.npz')==result['predictions_sha256']
    assert sha(TASK)==extraction['task_sha256']
    assert [(w['clip'],w['kind'],w['start'],w['end']) for w in windows]==[(w['clip'],w['kind'],w['start'],w['end']) for w in oldwindows]
    x=np.load(BIG/'features.npz')['x'];feature_error=0;mapping_error=0;pixel_metric_error=0;decoded_error=0
    inverse_pairs=[(23,12),(25,13),(27,14),(31,19),(24,9),(26,10),(28,11),(32,22)]
    decoder=read(HERE/'benchmark-decode.json')
    for row in read(OLD/'inventory.json'):
        clip=row['id'];assert sha(row['file'])==row['sha256']
        run=next(r for r in extraction['runs'] if r['clip']==clip)
        pinned=next(r for r in read(HERE/'benchmark-inputs.json')['rows'] if r['clip']==clip)
        cap=cv2.VideoCapture(row['file'])
        for digest in pinned['all_frame_pixel_sha256']:
            ok,image=cap.read();assert ok and hashlib.sha256(image.tobytes()).hexdigest()==digest
        assert not cap.read()[0];cap.release()
        path=BIG/(clip+'-pose.npz');assert sha(path)==run['output_sha256'];z=np.load(path);raw=z['raw'];lm=z['landmarks']
        assert lm.shape==(row['frames'],33,5) and raw.shape==(row['frames'],75)
        assert np.all(np.diff(z['task_ms'])>0) and np.all(np.diff(z['source_ms'])>0)
        assert int(z['counts'].sum())==run['detected'] and int((z['counts']==0).sum())==run['missing']
        assert abs(float(z['inference_seconds'].sum())-run['inference_seconds'])<1e-9
        for mpj,opj in inverse_pairs:
            for axis,scale in [(0,row['width']),(1,row['height'])]:
                expected=lm[:,mpj,axis]*scale
                np.testing.assert_array_equal(np.isnan(expected),np.isnan(raw[:,3*opj+axis]))
                mapping_error=max(mapping_error,float(np.nanmax(np.abs(expected-raw[:,3*opj+axis]))))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            proc=ns['process_video_and_add_cols'](raw,swap_orientation=False)
        for i,w in enumerate(windows):
            if w['clip']!=clip:continue
            start=w['start'];end=w['end']+1
            expected=np.array([proc[start:end,prep.LEFT],proc[start:end,prep.RIGHT]],dtype=np.float32)
            feature_error=max(feature_error,float(np.max(np.abs(x[i]-expected))))
            missing=float(ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](raw[start:end]),prep.REQUIRED))
            assert missing<=.25 and missing==w['missing'] and np.isfinite(expected).all()
        op=np.load(OLD_BIG/(clip+'-raw.npz'))['raw'].reshape(-1,25,3)
        comp=next(c for c in read(HERE/'pose-comparison.json')['clips'] if c['clip']==clip)
        for name,j in JOINTS.items():
            a=op[:,j,:2];b=raw.reshape(-1,25,3)[:,j,:2]
            good=np.isfinite(a).all(1)&np.isfinite(b).all(1)&(a!=0).all(1)&(b!=0).all(1)
            distances=np.sqrt(((a[good]-b[good])**2).sum(1))
            pixel_metric_error=max(pixel_metric_error,abs(float(np.median(distances))-comp['joints'][name]['pixel_distance']['median']))
        fresh=next(c for c in decoder['rows'] if c['clip']==clip)
        hm=np.load(BIG/(clip+'-benchmark-heatmaps.npz'))['heatmaps']
        prior_hm=np.load(OLD_BIG/'received'/(clip+'_000000_000024.npz'))['heatmaps']
        assert float(np.max(np.abs(hm-prior_hm)))<1e-4
        for frame,kps in enumerate(fresh['keypoints']):
            source=read(OLD_BIG/'body25'/clip/('%s_%012d_keypoints.json'%(clip,frame)))['people']
            assert len(source)==len(kps)
            for actual,person in zip(kps,source):decoded_error=max(decoded_error,float(np.abs(np.asarray(actual).reshape(-1)-np.asarray(person['pose_keypoints_2d'])).max()))
    assert mapping_error==feature_error==pixel_metric_error==0
    assert decoded_error<.01,decoded_error
    assert result['members']==before['members'],'Checkpoints/calibration/replay proofs changed'
    assert len(result['members'])==54 and result['optimizer_steps']==0
    for member in result['members']:assert sha(member['checkpoint'])==member['checkpoint_sha256']
    with np.load(BIG/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    selection=read(HERE.parent/'20261030-gpu-motionbert-study/mean-difference692/analysis/selection.json')['folds']
    rebuilt=[]
    for f in range(5):
        parts={}
        for family in ['CNN','STT']:
            keys=[k for k in pred if k.startswith('%s_%d_'%(family,f)) and not k.endswith('_bag')]
            assert len(keys)==(4 if family=='CNN' and f==0 else 5)
            parts[family]=sum(pred[k] for k in keys)/len(keys)
            np.testing.assert_allclose(parts[family],pred['%s_%d_bag'%(family,f)],rtol=0,atol=1e-12)
        sf=selection[f];base=sf['cnn_weight']*parts['CNN']+(1-sf['cnn_weight'])*parts['STT']
        mean=(base[:,0]+base[:,1])/2
        diff=(1-sf['fraction'])*(base[:,0]-base[:,1])+sf['fraction']*(pred['bilateral_%d'%f][:,0]-pred['bilateral_%d'%f][:,1])
        v=np.array([mean+diff/2,mean-diff/2]).T
        np.testing.assert_allclose(v,pred['best_%d'%f],rtol=0,atol=1e-12);rebuilt.append(v)
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
    for engine in ['openpose','mediapipe']:
        b=read(HERE/('benchmark-'+engine+'.json'))
        assert len(b['per_frame_engine_seconds'])==b['frames']==50
        assert abs(math.fsum(b['per_frame_engine_seconds'])-b['engine_seconds'])<1e-9
        assert abs(b['engine_fps']-50/b['engine_seconds'])<1e-9
    assert decoder['frames']==50 and len(decoder['per_frame_seconds'])==50
    report=dict(passed=True,video_hashes_verified=2,pose_frames_verified=2382,windows_verified=len(windows),
        mapping_max_error=mapping_error,feature_reconstruction_max_error=feature_error,pose_median_recalculation_max_error=pixel_metric_error,
        checkpoints_and_calibration_identical=True,checkpoints_sha256_verified=54,ensemble_independently_rebuilt=True,summary_recalculation_max_error=summary_error,
        fresh_openpose_cpu_vs_original_gpu_decoded_coordinate_and_confidence_max_error=decoded_error,
        timing_frame_counts_and_sums_verified=True,results_sha256=sha(HERE/'results.json'),optimizer_steps=0,
        decoded_video_pixels_identical_across_opencv_versions=2382,
        limitations=['No joint/GDI ground truth','One person; repeated windows are not independent participants','No smartphone measurement'])
    save(HERE/'verification.json',report);print(report,flush=True)

if __name__=='__main__':main()
