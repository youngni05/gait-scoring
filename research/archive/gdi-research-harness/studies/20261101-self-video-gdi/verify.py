"""Independent source integrity, window mapping and ensemble arithmetic checks."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BIG=Path('D:/gait-research-data/D_P/self-video-gdi-20261101')
sys.path[:0]=[str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'),str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle')]
import numpy as np
from preprocess import source_functions,LEFT,RIGHT,REQUIRED
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=read(HERE/'results.json');inventory=read(HERE/'inventory.json');windows=read(HERE/'windows.json');q=read(HERE/'quality.json')
    assert sha(HERE/'windows.json')==result['windows_sha256'] and sha(HERE/'protocol.json')==result['protocol_sha256']
    assert sha(BIG/'features.npz')==result['features_sha256'] and sha(BIG/'predictions.npz')==result['predictions_sha256']
    pose=read(HERE/'pose-complete.json');receipt=read(BIG/'received/receipt.json');assert sha(BIG/'received/receipt.json')==pose['receipt_sha256']
    assert len(receipt)==pose['chunks']==97 and sum(len(v['frames']) for v in receipt.values())==pose['frames']==2382
    filechecks=0
    for name,row in receipt.items():
        assert sha(BIG/'received'/name)==row['sha256']
        for filename,digest in row['json_sha256'].items():
            clip=filename.split('_')[0];assert sha(BIG/'body25'/clip/filename)==digest;filechecks+=1
    for row in inventory:assert sha(Path(row['file']))==row['sha256']
    x=np.load(BIG/'features.npz')['x'];features_error=0
    ns=source_functions()
    for row in inventory:
        clip=row['id'];raw=np.load(BIG/(clip+'-raw.npz'))['raw'];assert len(raw)==row['frames']
        processed=ns['process_video_and_add_cols'](raw,swap_orientation=False)
        for i,w in enumerate(windows):
            if w['clip']!=clip:continue
            a=w['start'];b=w['end']+1;assert b-a==124
            expect=np.array([processed[a:b,LEFT],processed[a:b,RIGHT]],dtype=np.float32)
            features_error=max(features_error,float(np.abs(x[i]-expect).max()))
            miss=float(ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](raw[a:b]),REQUIRED))
            assert miss<=.25 and abs(miss-w['missing'])<1e-12
    assert features_error==0 and len(windows)==32
    with np.load(BIG/'predictions.npz') as z:pred={k:z[k].copy() for k in z.files}
    S=HERE.parent/'20261030-gpu-motionbert-study';selection=read(S/'mean-difference692/analysis/selection.json')['folds']
    rebuilt=[];delta_error=0
    for f in range(5):
        parts={}
        for family in ['CNN','STT']:
            keys=[k for k in pred if k.startswith('%s_%d_'%(family,f)) and not k.endswith('_bag')]
            expected_members=4 if family=='CNN' and f==0 else 5;assert len(keys)==expected_members
            v=sum((pred[k] for k in keys))/len(keys);np.testing.assert_allclose(v,pred['%s_%d_bag'%(family,f)],atol=1e-12,rtol=0);parts[family]=v
        sf=selection[f];v=[]
        for i in range(len(windows)):
            l=sf['cnn_weight']*parts['CNN'][i,0]+(1-sf['cnn_weight'])*parts['STT'][i,0]
            r=sf['cnn_weight']*parts['CNN'][i,1]+(1-sf['cnn_weight'])*parts['STT'][i,1]
            gap=(1-sf['fraction'])*(l-r)+sf['fraction']*(pred['bilateral_%d'%f][i,0]-pred['bilateral_%d'%f][i,1])
            v.append([(l+r+gap)/2,(l+r-gap)/2])
        v=np.asarray(v);np.testing.assert_allclose(v,pred['best_%d'%f],atol=1e-12,rtol=0);rebuilt.append(v)
    np.testing.assert_allclose(np.mean(rebuilt,axis=0),pred['best'],atol=1e-12,rtol=0)
    for kind,models in result['summary'].items():
        for name,stats in models.items():
            means={}
            for clip in ['clip0','clip1']:
                ix=[i for i,w in enumerate(windows) if w['clip']==clip and w['kind']==kind]
                # Explicit scalar summation is separate from prediction-script aggregation.
                value=sum(float(pred[name][i,side]) for i in ix for side in [0,1])/(2*len(ix));means[clip]=value
                delta_error=max(delta_error,abs(value-stats[clip]['mean']))
            delta_error=max(delta_error,abs(means['clip1']-means['clip0']-stats['normal_minus_abnormal']))
    assert delta_error<1e-10
    for m in result['members']:assert sha(Path(m['checkpoint']))==m['checkpoint_sha256'] and m['replay_max_error']<1e-4
    assert len(result['members'])==54 and result['optimizer_steps']==0
    assert max(r['max_error'] for r in read(HERE/'gpu-equivalence.json'))<1e-4
    sample=read(HERE/'sample-replay/comparison.json');assert sample['passed'] and sample['feature_max_error']==0 and sample['prediction_max_error']==0
    report=dict(passed=True,original_videos_unchanged=2,heatmap_chunks_sha256_verified=97,body25_json_sha256_verified=filechecks,
                feature_windows=32,feature_reconstruction_max=features_error,ensemble_rebuilt=True,summary_recalculation_max=delta_error,
                model_checkpoints_sha256_verified=54,checkpoint_replay_max=max(m['replay_max_error'] for m in result['members']),
                replay_tolerance=1e-4,GPU_heatmap_max=max(r['max_error'] for r in read(HERE/'gpu-equivalence.json')),
                results_sha256=sha(HERE/'results.json'),predictions_sha256=sha(BIG/'predictions.npz'),
                original_sample_replay='sample-replay/comparison.json',single_person=True,true_GDI_unavailable=True,
                caption_extension_executed=False,caption_extension_reason='Both predefined analyses show higher normal scores in all five best-ensemble fold predictors; conditional fallback not triggered',
                confidence_exception='User approved; strict failure and official CPU reproduction preserved',optimizer_steps=0)
    (HERE/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
