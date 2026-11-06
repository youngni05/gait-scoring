"""Independently verify source hashes, unchanged feature contract and score arithmetic."""
import sys,json,hashlib,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OLD=HERE.parent/'20261101-self-video-gdi';BIG=Path('D:/gait-research-data/D_P/self-video-test01-20261101')
sys.path[:0]=[str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'),str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle')]
import numpy as np
from preprocess import source_functions,LEFT,RIGHT,REQUIRED,extract_windows
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    r=read(HERE/'results.json');inventory=read(HERE/'inventory.json');windows=read(HERE/'windows.json');quality=read(HERE/'quality.json');reference=read(OLD/'results.json')
    labels=read(HERE/'labels.json');assert set(labels['normal_numbers']).isdisjoint(labels['abnormal_numbers']) and set(labels['normal_numbers']+labels['abnormal_numbers'])==set(range(1,14))
    for row in r['records']:assert row['condition']==('normal' if row['number'] in labels['normal_numbers'] else 'abnormal')
    assert len(inventory)==13 and {v['number'] for v in inventory}==set(range(1,14))
    assert r['model_members']==reference['members'] and len(r['model_members'])==54
    for member in r['model_members']:assert sha(Path(member['checkpoint']))==member['checkpoint_sha256']
    assert r['reference_results_sha256']==sha(OLD/'results.json') and r['protocol_sha256']==sha(HERE/'protocol.json')
    assert r['windows_sha256']==sha(HERE/'windows.json') and r['features_sha256']==sha(BIG/'features.npz') and r['predictions_sha256']==sha(BIG/'predictions.npz')
    complete=read(HERE/'pose-complete.json');receipt=read(BIG/'received/receipt.json');expected=sum(v['processed_frames'] for v in inventory)
    assert expected==6634==complete['frames']==sum(len(v['frames']) for v in receipt.values())
    assert len(receipt)==complete['chunks'] and sha(BIG/'received/receipt.json')==complete['receipt_sha256']
    json_count=0
    for name,item in receipt.items():
        assert sha(BIG/'received'/name)==item['sha256']
        for filename,digest in item['json_sha256'].items():
            clip=filename.split('_')[0];assert sha(BIG/'body25'/clip/filename)==digest;json_count+=1
    x=np.load(BIG/'features.npz')['x'];assert x.shape==(len(windows),2,124,12) and np.isfinite(x).all();max_error=0
    ns=source_functions()
    for row in inventory:
        assert sha(Path(row['file']))==row['sha256'];clip=row['id'];raw=np.load(BIG/(clip+'-raw.npz'))['raw'];source=np.full(raw.shape,np.nan)
        paths=sorted((BIG/'body25'/clip).glob('*keypoints.json'));assert len(paths)==row['processed_frames']
        for i,path in enumerate(paths):
            assert int(path.stem.split('_')[-2])==i;people=read(path)['people']
            if len(people)==1:source[i]=people[0]['pose_keypoints_2d']
        np.testing.assert_array_equal(raw,source)
        q=next(v for v in quality if v['id']==clip)
        if q['status']=='prepared':
            e=extract_windows(raw);assert e['window_start'].tolist()==q['starts'];proc=ns['process_video_and_add_cols'](raw[:e['processed_frames']],swap_orientation=False)
            for i,w in enumerate(windows):
                if w['id']!=clip:continue
                a=w['start'];b=w['end']+1;assert b-a==124 and a in range(0,376,31)
                expect=np.asarray([proc[a:b,LEFT],proc[a:b,RIGHT]],dtype=np.float32);max_error=max(max_error,float(np.abs(x[i]-expect).max()))
                miss=float(ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](raw[a:b]),REQUIRED));assert miss<=.25 and miss==w['missing_fraction']
        else:
            try:extract_windows(raw)
            except ValueError:pass
            else:raise AssertionError('Excluded video actually eligible')
    assert max_error==0
    with np.load(BIG/'predictions.npz') as z:p={k:z[k] for k in z.files}
    selections=read(OLD.parent/'20261030-gpu-motionbert-study/mean-difference692/analysis/selection.json')['folds'];ensemble_error=0
    for f,s in enumerate(selections):
        parts={}
        for family in ['CNN','STT']:
            keys=[k for k in p if k.startswith('%s_%d_'%(family,f)) and not k.endswith('_bag')]
            assert len(keys)==(4 if family=='CNN' and f==0 else 5)
            parts[family]=sum(p[k] for k in keys)/len(keys);np.testing.assert_allclose(parts[family],p['%s_%d_bag'%(family,f)],atol=1e-12,rtol=0)
        rebuilt=[]
        for i in range(len(x)):
            l=s['cnn_weight']*parts['CNN'][i,0]+(1-s['cnn_weight'])*parts['STT'][i,0]
            rr=s['cnn_weight']*parts['CNN'][i,1]+(1-s['cnn_weight'])*parts['STT'][i,1]
            d=(1-s['fraction'])*(l-rr)+s['fraction']*(p['bilateral_%d'%f][i,0]-p['bilateral_%d'%f][i,1]);rebuilt.append([(l+rr+d)/2,(l+rr-d)/2])
        ensemble_error=max(ensemble_error,float(np.abs(np.asarray(rebuilt)-p['best_%d'%f]).max()))
    np.testing.assert_allclose(sum(p['best_%d'%f] for f in range(5))/5,p['best'],atol=1e-12,rtol=0);assert ensemble_error<1e-12
    summary_error=0
    for row in r['records']:
        ix=[i for i,w in enumerate(windows) if w['id']==row['id']]
        if not ix:assert row['score'] is None;continue
        value=sum(float(p['best'][i,j]) for i in ix for j in [0,1])/(2*len(ix));summary_error=max(summary_error,abs(value-row['score']))
    normal=[v['score'] for v in r['records'] if v['condition']=='normal' and v['score'] is not None];abnormal=[v['score'] for v in r['records'] if v['condition']=='abnormal' and v['score'] is not None]
    assert abs(sum(normal)/len(normal)-r['group']['normal_mean'])<1e-12 and abs(sum(abnormal)/len(abnormal)-r['group']['abnormal_mean'])<1e-12
    csvrows=list(csv.DictReader((HERE/'video-scores.csv').open(encoding='utf-8-sig')));assert len(csvrows)==13
    for a,b in zip(csvrows,r['records']):
        assert int(a['number'])==b['number']
        if b['score'] is not None:assert abs(float(a['score'])-b['score'])<1e-12
    assert r['previous_replay_max_error']<1e-4 and r['optimizer_steps']==0 and r['labels_used_for_inference'] is False
    report=dict(passed=True,source_videos=13,source_videos_unchanged=True,body25_json_sha256_verified=json_count,heatmap_chunks_sha256_verified=len(receipt),feature_windows=len(x),feature_reconstruction_max=max_error,ensemble_rebuild_max=ensemble_error,summary_recalculation_max=summary_error,checkpoints_identical_to_previous=54,previous_full_prediction_replay_max=r['previous_replay_max_error'],GPU_probe_max_error=max(v['max_error'] for v in read(HERE/'gpu-equivalence.json')),results_sha256=sha(HERE/'results.json'),optimizer_steps=0,unit='video; no participant-independence inference')
    assert report['GPU_probe_max_error']<1e-4
    (HERE/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
