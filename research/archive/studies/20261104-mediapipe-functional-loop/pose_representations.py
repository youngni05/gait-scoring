"""MediaPipe-native masked motion inputs; no clinical target is loaded here."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104')
JOINTS=[11,12,23,24,25,26,27,28,29,30,31,32]
EDGES=[(11,23),(12,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(28,30),(30,32),(11,12),(23,24)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def build(coords,valid,times,provider,width,height):
    p=coords[:,:,:2]*np.array([width,height]);j=valid[:,JOINTS];hipok=valid[:,23]&valid[:,24];hip=np.mean(p[:,[23,24]],axis=1);scale=[]
    for a,b,c in [(23,25,27),(24,26,28)]:
        s=np.linalg.norm(p[:,a]-p[:,b],axis=1)+np.linalg.norm(p[:,b]-p[:,c],axis=1);s[~(valid[:,a]&valid[:,b]&valid[:,c])]=np.nan;scale.append(s)
    values=np.array(scale);good=values[np.isfinite(values)&(values>5)];assert len(good)>0,'unmeasurable_scale';s=float(np.median(good));mask=j&hipok[:,None]
    xy=(p[:,JOINTS]-hip[:,None,:])/s;xyz=np.concatenate([xy,((coords[:,JOINTS,2]-np.mean(coords[:,[23,24],2],axis=1)[:,None])*width/s)[:,:,None]],axis=-1)
    bone=[];bone_mask=[]
    for a,b in EDGES:bone.append((p[:,b]-p[:,a])/s);bone_mask.append(valid[:,a]&valid[:,b])
    bone=np.stack(bone,axis=1);bm=np.stack(bone_mask,axis=1);velocity=np.zeros_like(xy);vm=np.zeros_like(mask);dt=np.diff(times)/1000
    okay=(dt>0)&(dt<=.05)&(provider[1:]==provider[:-1])&(provider[1:]>=0);vm[1:]=mask[1:]&mask[:-1]&okay[:,None];velocity[1:]=(xy[1:]-xy[:-1])/dt[:,None,None]
    def masked(x,m):return np.where(m[:,:,None],np.nan_to_num(x,nan=0.,posinf=0.,neginf=0.),0).reshape(len(x),-1).astype(np.float32)
    return dict(image_xy=masked(p[:,JOINTS]/height,j),pelvis_stable_xy=masked(xy,mask),relative_mp_xyz=masked(xyz,mask),bones_velocity=np.column_stack([masked(xy,mask),masked(bone,bm),masked(velocity,vm)]),joint_mask=j,pelvis_mask=mask,bones_velocity_mask=np.column_stack([mask,bm,vm]),scale_pixels=np.array(s))
def main():
    source=B/'OAW06-bottom-mediapipe-crop.npz';d=np.load(source);meta=json.loads((H/'video-metadata.json').read_text());quality=json.loads((H/'crop-verification.json').read_text());proto=dict(code_sha256=sha(Path(__file__)),source_sha256=sha(source),joint_indices=JOINTS,edges=EDGES,variants=['image xy/aspect ratio baseline','pelvis centered stable per4secondmedian observed leg scale','relative MPxyz with z treated as model-relative, not metres','pelvis xy + bone vectors + physical-time velocity'],masking='Keep explicit confidence masks. Invalid coordinates zero only with false mask; never interpreted as normal coordinates/low score. Velocity invalid across missing frames,provider transitions or time gaps>.05s. No invented joint interpolation.',windows='Previously fixed120frame nonoverlapping starts and>=80percentall6joints,maxbadgap6. Same eligible windows for all representations. Onepilotparticipant; no clinical labels used.',verification='Independent geometry identities fortranslation and resolution scaling; timestamp shift leaves outputs unchanged; all finite masked arrays; output traceable to source frames/provider. This verifies input arithmetic, not anatomical accuracy or prediction performance.')
    if(H/'representation-protocol.json').exists():assert json.loads((H/'representation-protocol.json').read_text())==proto
    else:save(H/'representation-protocol.json',proto)
    result={};starts=[];checks=[]
    for w in quality['windows_after']:
        if not w['usable']:continue
        lo=w['start_frame'];hi=w['end_exclusive'];c=d['landmarks'][lo:hi];v=d['valid'][lo:hi];t=d['source_ms'][lo:hi];pr=d['provider'][lo:hi];a=build(c,v,t,pr,meta['width'],meta['height']);shift=c.copy();shift[:,:,0]+=.1;shift[:,:,1]+=.05;b=build(shift,v,t,pr,meta['width'],meta['height']);scaled=build(c,v,t+1000,pr,meta['width']*2,meta['height']*2)
        for k in ['pelvis_stable_xy','relative_mp_xyz','bones_velocity']:assert np.allclose(a[k],b[k],atol=1e-6,rtol=1e-6)
        for k in ['image_xy','pelvis_stable_xy','relative_mp_xyz','bones_velocity']:assert np.isfinite(a[k]).all() and np.allclose(a[k],scaled[k],atol=1e-6,rtol=1e-6)
        assert not np.array_equal(a['image_xy'],b['image_xy'])
        for k,value in a.items():result.setdefault(k,[]).append(value)
        starts.append(lo);checks.append(dict(start_frame=lo,scale_pixels=float(a['scale_pixels'])))
    assert len(starts)>0;np.savez_compressed(B/'OAW06-input-variants.npz',start_frames=np.array(starts),**{k:np.stack(v) for k,v in result.items()});save(H/'representation-verification.json',dict(status='passed_input_arithmetic_only',person_count=1,windows=len(starts),shapes={k:list(np.stack(v).shape) for k,v in result.items()},checks=checks,output_sha256=sha(B/'OAW06-input-variants.npz'),protocol_sha256=sha(H/'representation-protocol.json'),clinical_accuracy_tested=False));print('real MediaPipe input variants verified',len(starts),flush=True)
if __name__=='__main__':main()
