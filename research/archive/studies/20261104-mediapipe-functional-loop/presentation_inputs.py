"""Masked four-second inputs on released-video time; not original-clock calibration."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104');sys.path.insert(0,str(H))
from pose_representations import build,JOINTS
PAIRS=[(11,12),(23,24),(25,26),(27,28),(29,30),(31,32)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def resample(coords,valid,times,provider,start_ms):
    assert len(coords)==len(valid)==len(times)==len(provider) and np.all(np.diff(times)>0)
    grid=start_ms+np.arange(120,dtype=float)*1000/30;out=np.zeros((120,33,5),float);mask=np.zeros((120,33),bool);src=np.full(120,-1,int);typical=float(np.median(np.diff(times))) if len(times)>1 else 0.
    for k,t in enumerate(grid):
        right=int(np.searchsorted(times,t,side='left'))
        exact=right if right<len(times) and abs(times[right]-t)<=1e-6 else right-1 if right>0 and abs(times[right-1]-t)<=1e-6 else -1
        if exact>=0 and start_ms-1e-6<=times[exact]<=start_ms+4000+1e-6:
            out[k]=coords[exact];mask[k]=valid[exact];src[k]=provider[exact];continue
        left=right-1
        if left<0 or right>=len(times) or times[left]<start_ms-1e-6 or times[right]>start_ms+4000+1e-6:continue
        if times[right]-times[left]>1.5*typical or provider[left]<0 or provider[right]!=provider[left]:continue
        good=valid[left]&valid[right];a=(t-times[left])/(times[right]-times[left]);out[k,:,:3]=coords[left,:,:3]*(1-a)+coords[right,:,:3]*a;out[k,:,3:]=np.minimum(coords[left,:,3:],coords[right,:,3:]);mask[k]=good;src[k]=provider[left]
    # Missing detector points are never bridged. Zero placeholders always carry false masks.
    out[~mask]=0.;return out,mask,grid,src
def symmetric(coords,valid,scale,width,height):
    p=coords[:,:,:2]*[width,height];hip=p[:,[23,24]].mean(axis=1);v=(p-hip[:,None,:])/scale;values=[];masks=[]
    for l,r in PAIRS:
        okay=valid[:,l]&valid[:,r]&valid[:,23]&valid[:,24];values.extend([np.where(okay[:,None],(v[:,l]+v[:,r])/2,0),np.where(okay[:,None],np.abs(v[:,l]-v[:,r]),0)]);masks.extend([okay,okay])
    return np.stack(values,axis=1).reshape(len(coords),-1).astype(np.float32),np.stack(masks,axis=1)
def selftest():
    checks=[]
    for fps in [28,29,30]:
        t=np.arange(4*fps+1)*1000/fps;c=np.zeros((len(t),33,5));base=np.linspace(.1,.7,33);c[:,:,0]=base[None,:]+t[:,None]/1000*.025;c[:,:,1]=base[None,:]+t[:,None]/1000*.01;c[:,:,2]=base[None,:]+t[:,None]/1000*.005;c[:,:,3:]=1;v=np.ones((len(t),33),bool);p=np.zeros(len(t),int);r,m,g,q=resample(c,v,t,p,0)
        assert m.all();expected=base[None,:]+g[:,None]/1000*.025;assert np.max(np.abs(r[:,:,0]-expected))<1e-12
        # One detector failure must stay missing on neighboring resampled times.
        bad=v.copy();bad[2*fps]=False;pp=p.copy();pp[2*fps]=-1;_,mm,_,_=resample(c,bad,t,pp,0);assert not mm[60].any() and mm[0].all()
        keep=(t<=1500)|(t>=2000);_,gap,gg,_=resample(c[keep],v[keep],t[keep],p[keep],0);assert not gap[(gg>1530)&(gg<1970)].any()
        b=build(r,m,g,q,1080,1920);a,am=symmetric(r,m,float(b['scale_pixels']),1080,1920);sw=r.copy();sv=m.copy()
        for l,rr in PAIRS:sw[:,[l,rr]]=sw[:,[rr,l]];sv[:,[l,rr]]=sv[:,[rr,l]]
        aa,aam=symmetric(sw,sv,float(b['scale_pixels']),1080,1920);assert np.array_equal(a,aa) and np.array_equal(am,aam)
        checks.append(dict(native_fps=fps,source_frames=len(t),model_samples=120,seconds=4,linear_signal_max_error=float(np.max(np.abs(r[:,:,0]-expected))),missing_detector_and_long_gap_not_imputed=True,symmetric_input_left_right_swap_invariant=True))
    save(H/'presentation-resampling-selftest.json',dict(status='passed',code_sha256=sha(Path(__file__)),data='Analytical linear trajectories at28/29/30fps; artificial detector failures and missing-time intervals, no clinical labels.',checks=checks,limits='Released-time arithmetic only. Original capture clock and anatomical accuracy unverified.'));print('released-clock analytical tests passed',flush=True)
def main():
    assert json.loads((H/'presentation-resampling-selftest.json').read_text())['code_sha256']==sha(Path(__file__));batch=json.loads((H/'batch14-receipt.json').read_text());camera=json.loads((H/'camera14-metadata.json').read_text());meta={r['person_id']:r for r in camera['records']};assert batch['people']==14;protocol=dict(code_sha256=sha(Path(__file__)),base_input_library_sha256=sha(H/'pose_representations.py'),batch_protocol_sha256=batch['protocol_sha256'],variants=['image_xy','pelvis_stable_xy','relative_mp_xyz','bones_velocity','pair_mean_absdiff'],sampling='120sampleson4secondsreleasedpresentationtime at30Hz. Exact observations or interpolationonly between adjacentvalidsame-provider sourceframes withgap<=1.5typicalframeinterval. No detector-gap/provider-transition/large-time-gap filling. Source points stayinside4s support. Raw14FPS28/29/30are nottimewarped. Originalcaptureclock unverified.',selection='Previously fixed4000ms nonoverlap sourcewindows, thensame80percentall6joint/200msmaxgap on30Hzmask. Allfive representations sharecombinedwindowset. Additional common_full_frame_eligible flag forpaired full-versus-crop availability. No clinical labels read. No supervised fit. Later actor/walking/clock review still mandatory.',model_input_allowlist=['image_xy','pelvis_stable_xy','relative_mp_xyz','bones_velocity','pair_mean_absdiff'],metadata_not_model_inputs=['person_id','start_ms','absolute_source_frame','total_source_duration','number_of_windows','provider'],limits='Four-second samples selectedfromlongerfiles,not4secondtotalacquisition. All14 retained incoverage, peoplewithoutqualifiedwindowsmarkedunmeasurable,neverlow score. RelativeMPznotmetres; symmetricinputloseslaterality,appropriateonlyforglobalfunctional targets,notleft/rightGDI.')
    if(H/'presentation-input-protocol.json').exists():assert json.loads((H/'presentation-input-protocol.json').read_text())==protocol
    else:save(H/'presentation-input-protocol.json',protocol)
    arrays={};persons=[];starts=[];paired=[];coverage=[]
    for rec in batch['receipts']:
      assert sha(rec['output'])==rec['output_sha256'];pid=rec['person_id'];d=np.load(rec['output']);cm=meta[pid];accepted=0
      for window in rec['windows_combined']:
        if not window['operationally_usable']:continue
        r,v,t,provider=resample(d['landmarks'],d['valid'],d['source_ms'],d['provider'],window['start_ms']);six=np.all(v[:,[23,24,25,26,27,28]],axis=1);run=0;longest=0
        for good in six:run=0 if good else run+1;longest=max(longest,run)
        if six.mean()<.8 or longest>6:continue
        try:a=build(r,v,t,provider,cm['width'],cm['height'])
        except AssertionError:continue
        pair,pm=symmetric(r,v,float(a['scale_pixels']),cm['width'],cm['height']);a['pair_mean_absdiff']=pair;a['pair_mask']=pm
        _,fv,_,_=resample(d['full_landmarks'],d['full_valid'],d['source_ms'],np.where(d['provider']==0,0,-1),window['start_ms']);fsix=np.all(fv[:,[23,24,25,26,27,28]],axis=1);rr=0;ll=0
        for good in fsix:rr=0 if good else rr+1;ll=max(ll,rr)
        paired.append(bool(fsix.mean()>=.8 and ll<=6))
        for key,value in a.items():assert np.isfinite(value).all();arrays.setdefault(key,[]).append(value)
        persons.append(pid);starts.append(window['start_ms']);accepted+=1
      coverage.append(dict(person_id=pid,windows=accepted,status='inputs_prepared_pending_actor_walking_clock_review' if accepted else '측정 불가: no qualified4swindow'))
    if persons:np.savez_compressed(B/'bottom14-inputs.npz',person_ids=np.array(persons),window_start_ms=np.array(starts),common_full_frame_eligible=np.array(paired),**{k:np.stack(v) for k,v in arrays.items()})
    save(H/'presentation-input-verification.json',dict(status='input_arithmetic_passed_pending_actor_walking_clock_review',people=14,people_with_windows=sum(r['windows']>0 for r in coverage),windows=len(persons),matched_full_frame_windows=sum(paired),shapes={k:list(np.stack(v).shape) for k,v in arrays.items()},coverage=coverage,clinical_model_fitted=False,output_sha256=sha(B/'bottom14-inputs.npz') if persons else None));print('14personinput preparation complete',len(persons),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');args=ap.parse_args();selftest() if args.selftest else main()
