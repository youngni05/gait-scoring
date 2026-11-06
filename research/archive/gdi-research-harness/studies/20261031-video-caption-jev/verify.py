"""Verify frame/feature/label isolation and recompute all reported prediction errors."""
import json
import hashlib
import sys
from bootstrap import HERE,ROOT,BIG,sha
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np

def save(n,v):(HERE/n).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')

def prepare_checks():
    sys.path.insert(0,str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
    import cv2
    d=json.loads((HERE/'numeric_features.json').read_text());rows=json.loads((HERE/'video_preparation.json').read_text())
    originals={v['person']:v for v in json.loads((HERE/'acquisition.json').read_text())['videos']}
    assert len(rows)==12==len(d['people'])==len(set(d['people']))
    assert [r['person'] for r in rows]==d['people']
    for i,r in enumerate(rows):
        pid=r['person'];x=np.load(BIG/'frames'/f'pose{pid:02d}.npy',allow_pickle=False)
        images=np.load(BIG/'frames'/f'person{pid:02d}.npy',allow_pickle=False)
        assert x.shape==(124,17,3) and images.shape[0]==9
        assert np.isfinite(x).all() and ((x[:,:,2]>=0)&(x[:,:,2]<=1)).all()
        expected=np.rint((r['start_seconds']+np.arange(124)/30)*r['fps']).astype(int)
        np.testing.assert_array_equal(expected,r['native_frame_indices'])
        source=__import__('pathlib').Path(originals[pid]['path'])
        assert sha(source)==originals[pid]['sha256']==r['video_sha256']
        cap=cv2.VideoCapture(str(source))
        for j,k in enumerate(r['sampled_frames']):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(expected[k]));ok,bgr=cap.read();assert ok
            h,w=bgr.shape[:2]
            rgb=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(round(w*768/h),768))
            np.testing.assert_array_equal(rgb,images[j])
        cap.release()
        flat=x.reshape(124,51)
        f=np.concatenate([np.mean(flat,axis=0),np.std(flat,axis=0),np.mean(np.abs(flat[1:]-flat[:-1]),axis=0)])
        np.testing.assert_allclose(f,d['X'][i],atol=1e-12)
        assert sha(__import__('pathlib').Path(r['clip_path']))==r['clip_sha256']
    save('preparation_verification.json',{'status':'passed','people':12,'same_time_window':True,'feature_vectors_recomputed':12,
                                        'clinical_labels_read':False,'raw_video_frames_independently_redecoded':108,
                                        'numeric_sha256':sha(HERE/'numeric_features.json')})
    print('12 people: same-time input and numerical feature recomputation passed.',flush=True)

def result_checks():
    from jev_and_compare import requests,metrics
    captions=[json.loads(s) for s in (HERE/'captions.jsonl').read_text().splitlines()]
    assert len(captions)==36 and all(c['clinical_labels_read'] is False for c in captions)
    for c in captions:
        assert c['frames_sha256']==sha(BIG/'frames'/f"person{c['person']:02d}.npy")
        assert c['protocol_sha256']==sha(HERE/'protocol.json')
    response=[json.loads(s) for s in (HERE/'jev_responses.jsonl').read_text().splitlines()]
    lookup={(r['person'],r['variant']):r for r in response};assert len(lookup)==60
    for r in requests():
        v=lookup[r['person'],r['variant']]
        assert v['payload_sha256']==hashlib.sha256(json.dumps(r['payload'],sort_keys=True).encode()).hexdigest()
        assert 'target' not in r['payload']['state'] and 'person' not in r['payload']['state']
    p=json.loads((HERE/'predictions.json').read_text());result=json.loads((HERE/'results.json').read_text())
    y=np.array(p['target']);fold=np.array(p['fold']);assert len(y)==12 and set(fold)==set(range(5))
    source=HERE.parent/'20261030-tug-toronto-timeseries/input.npz'
    original=dict(np.load(source,allow_pickle=False))
    assert sha(source)==result['label_source_sha256']
    for i,pid in enumerate(p['people']):
        mask=np.array([int(str(v).replace('OAW',''))==pid for v in original['ids']])
        assert mask.any()
        np.testing.assert_allclose(original['y'][mask],y[i],rtol=0,atol=0)
        np.testing.assert_array_equal(original['folds'][mask],np.repeat(fold[i],sum(mask)))
    for name,values in p['predictions'].items():
        pred=np.array(values);assert np.isfinite(pred).all()
        err=pred-y
        assert abs(float(sum(abs(err))/12)-result['metrics'][name]['MAE'])<1e-10
        assert abs(float((sum(err**2)/12)**.5)-result['metrics'][name]['RMSE'])<1e-10
    for pred,score in zip(p['shuffle_Ridge_predictions'],result['shuffle_Ridge_MAE']):
        assert abs(np.mean(abs(np.array(pred)-y))-score)<1e-10
    for name,r in result['paired'].items():
        family=name.split('_')[0]
        delta=abs(np.array(p['predictions'][name])-y)-abs(np.array(p['predictions'][family+'_numeric'])-y)
        assert abs(delta.mean()-r['difference'])<1e-10
        samples=np.random.default_rng(20261031).integers(0,len(y),size=(10000,len(y)))
        np.testing.assert_allclose(np.quantile(delta[samples].mean(1),[.025,.975]),r['conditional_bootstrap95'],atol=1e-12)
    # Independently solve standardized Ridge in sample space, without the
    # pipeline/model factory used during fitting. Check both primary views.
    features=np.array(json.loads((HERE/'numeric_features.json').read_text())['X'])
    from jev_and_compare import TRAITS,CHOICES
    focused=np.array([[lookup[pid,'focused']['response']['answers'][q]['probabilities'][c]
                       for q in TRAITS for c in CHOICES] for pid in p['people']])
    for name,xx in [('Ridge_numeric',features),('Ridge_numeric_plus_Jev_focused',np.column_stack([features,focused]))]:
        replay=np.zeros(12)
        for f in range(5):
            tr=fold!=f;te=fold==f
            mu=xx[tr].mean(0);scale=xx[tr].std(0);scale[scale==0]=1
            a=(xx[tr]-mu)/scale;b=(xx[te]-mu)/scale
            alpha=np.linalg.solve(a@a.T+100*np.eye(sum(tr)),y[tr]-y[tr].mean())
            replay[te]=b@a.T@alpha+y[tr].mean()
        np.testing.assert_allclose(replay,p['predictions'][name],rtol=0,atol=1e-9)
    save('verification.json',{'status':'passed','captions':36,'requests':60,'prediction_methods':len(p['predictions']),
                              'shuffle_recomputations':50,'same_window_checks':True,'caption_input_fingerprints_match':True,
                              'Jev_payload_fingerprints_match':True,'reported_metrics_recomputed':True,
                              'clinical_label_and_fold_mapping_rechecked':True,'bootstrap_intervals_recomputed':True,
                              'independent_Ridge_dual_solve_primary_views':True,'independent_RF_retraining':False,
                              'results_sha256':sha(HERE/'results.json'),
                              'code_sha256':{name:sha(HERE/name) for name in ['bootstrap.py','prepare_video.py','caption.py','jev_and_compare.py','diagnostics.py','verify.py']}})
    print((HERE/'verification.json').read_text(),flush=True)

if __name__=='__main__':
    prepare_checks()
    if '--results' in sys.argv:result_checks()
