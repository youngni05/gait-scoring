"""Independent replay from standing/walking arrays, native marker geometry, and units."""
from pathlib import Path
import sys,json,collections,hashlib
import numpy as np
from scipy.io import loadmat
from scipy.stats import linregress
import experiment as e

def close(a,b):
    if a is None or b is None:assert a is None and b is None,(a,b)
    else:np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)

def lengths(foot,events,fs):
    steps=[];strides=[]
    for a,b in zip(events,events[1:]):
        if a['side']!=b['side'] and .2<=(b['index']-a['index'])/fs<=1.5:
            steps.append(foot[b['index'],b['side']]-foot[b['index'],1-b['side']])
    for side in [0,1]:
        ix=[v['index'] for v in events if v['side']==side]
        for a,b in zip(ix,ix[1:]):
            if .8<=(b-a)/fs<=3:strides.append(foot[b,side]-foot[a,side])
    return {'step_length':float(np.mean(steps)) if len(steps)>=2 else None,'stride_length':float(np.mean(strides)) if strides else None}

def run():
    p=e.read(e.H/'protocol.json');r=e.read(e.H/'comparison.json');inf=e.read(e.H/'inference-completion.json')
    assert e.sha(e.H/'experiment.py')==p['code_sha256']==r['code_sha256']==inf['code_sha256']
    assert r['protocol_sha256']==e.sha(e.H/'protocol.json') and r['inference_sha256']==e.sha(e.H/'inference-completion.json')
    for path,digest in p['sources'].items():assert e.sha(path)==digest,path
    for group in ['arrays','receipts']:
        assert len(inf[group])==28
        for path,digest in inf[group].items():assert e.sha(path)==digest,path
    oldver=e.read(e.OLD/'verification.json');assert oldver['status']=='passed' and oldver['measurements_sha256']==e.sha(e.OLD/'measurements.json')
    oldmanifest=e.ROOT/'experiments/walking-expansion/results.json'
    assert e.sha(oldmanifest)==e.read(e.OLD/'protocol.json')['cached_pose_manifest_sha256']
    original_extracts=e.read(oldmanifest)['results']
    old={x['key']:x for x in e.read(e.OLD/'measurements.json')['records']};cal={x['key']:x for x in r['calibrations']};by={(x['key'],x['metric']):x for x in r['pairs']};rebuilt=[];failures=[]
    for item in e.primary():
        key=item['key'];sid=item['subject'];camera=item['camera'];saved=cal[key];previous=old[key]
        assert e.sha(item['array'])==item['sha256'];d=np.load(item['array']);standpath=e.B/f's{sid:02d}_{camera}.npz';s=np.load(standpath)
        receipt=e.read(standpath.with_suffix('.json'));assert receipt['model_sha256']==e.sha(e.MODEL) and receipt['video_sha256']==p['sources'][receipt['video']]
        original=next(q for q in original_extracts if q['subject']==sid and q['camera']==camera and q['variant']=='heavy' and q['input_width']==800)
        assert original['model_sha256']==receipt['model_sha256'] and original['video_sha256']==receipt['video_sha256']
        np.testing.assert_array_equal(s['video_frame_one_based'],np.arange(1,91));assert s['landmarks'].shape==(90,33,5)
        meta=loadmat(e.ROOT/'reference'/f'F_v3d_Subject_{sid}.mat',simplify_cells=True)[f'Subject_{sid}_F'];height=float(meta['subject']['height'])/100;close(height,saved['height']);assert int(meta['move']['flags30'][:,0].min())>90
        lm=s['landmarks'][30:];spans=(lm[:,:,1].max(1)-lm[:,:,1].min(1))*600
        valid=bool(np.isfinite(lm).all() and np.all(s['pose_counts'][30:]>0) and np.all(spans>0) and np.percentile(lm[:,[23,24],3],50)>=.5 and all(np.percentile(lm[:,j,3],50)>=.5 for j in [29,30]))
        assert valid==saved['standing_quality'];ss=height/np.percentile(spans,50) if valid else None;close(ss,saved['standing_scale'])
        if not valid:failures.append(key)
        xy=d['lm'][:,:,:2]*[800,600];hip=xy[:,[23,24]].mean(1);ws=height/np.percentile(xy[:,:,1].max(1)-xy[:,:,1].min(1),50);close(ws,previous['scale_metres_per_pixel']);close(ws,saved['walking_scale']);close(ss/ws if ss else None,saved['scale_ratio'])
        t=np.arange(len(hip))/30;pixel_velocity=np.sqrt(sum(linregress(t,hip[:,a]).slope**2 for a in [0,1]));direction=1 if hip[-1,0]>=hip[0,0] else -1;pixel_lengths=lengths(xy[:,[29,30],0]*direction,previous['predicted_events'],30)
        pelvis=d['pelvis120'][:,:2];ev,vectors=np.linalg.eigh(np.cov(pelvis.T));axis=vectors[:,-1]
        if np.dot(axis,pelvis[-1]-pelvis[0])<0:axis=-axis
        ref=lengths(d['heels120'][:,:,:2]@axis,previous['reference_events'],120);ref['speed']=np.sqrt(sum(linregress(np.arange(len(pelvis))/120,pelvis[:,a]).slope**2 for a in [0,1]))
        pixels={'speed':pixel_velocity,**pixel_lengths}
        for metric in ['speed','step_length','stride_length']:
            q=by[(key,metric)];close(q['reference'],ref[metric]);base=pixels[metric]*ws if pixels[metric] is not None else None
            if (metric=='speed' and not previous['speed_quality']) or (metric!='speed' and (not previous['foot_quality'] or (base is not None and base<=0))):base=None
            close(base,q['baseline']);new=pixels[metric]*ss if base is not None and valid else None;close(new,q['standing']);assert q['paired_available']==(new is not None and ref[metric] is not None)
            rebuilt.append({**q,'baseline':base,'standing':new,'reference':float(ref[metric]) if ref[metric] is not None else None})
    for camera in ['PG1','PG2']:
        for metric in ['speed','step_length','stride_length']:
            a=[q for q in rebuilt if q['camera']==camera and q['metric']==metric and q['paired_available']];summary=r['summary'][camera][metric];assert summary['paired_people']==len(a) and summary['registered_people']==14
            old_errors=[q['baseline']-q['reference'] for q in a];new_errors=[q['standing']-q['reference'] for q in a];delta=np.abs(new_errors)-np.abs(old_errors)
            close(float(np.mean(abs(np.array(old_errors)))) if a else None,summary['baseline_MAE']);close(float(np.mean(abs(np.array(new_errors)))) if a else None,summary['standing_MAE']);close(float(np.mean(delta)) if a else None,summary['MAE_change_standing_minus_baseline'])
            if len(a)>=2:
                rng=np.random.default_rng(20261105);idx=rng.integers(0,len(a),(10000,len(a)));ci=np.percentile(np.mean(delta[idx],axis=1),[2.5,97.5]);close(ci,summary['paired_change_CI95'])
            assert summary['improved_people']==int((delta<0).sum()) and summary['worse_people']==int((delta>0).sum())
    verdict={'status':'passed','source_hash_files':len(p['sources']),'new_arrays_and_receipts_hashed':56,'same_primary_people':14,'same_primary_windows_views':28,'independent_measurement_rows':len(rebuilt),'standing_failure_keys':failures,'standing_frames_before_first_motion_verified':True,'reference_not_used_for_calibration':True,'fixed_walking_coordinates_events_quality_and_availability':True,'native_geometry_and_pixel_metric_replay':True,'tolerance':1e-10,'comparison_sha256':e.sha(e.H/'comparison.json'),'protocol_sha256':e.sha(e.H/'protocol.json'),'review_code_sha256':e.sha(__file__)}
    e.save(e.H/'verification.json',verdict);print(json.dumps(verdict,indent=2))

if __name__=='__main__':run()
