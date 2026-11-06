"""Matched-person geometry audit; synthetic cameras are not phone validation."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys,hashlib,importlib.util
import numpy as np,joblib
from scipy.signal import savgol_filter
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
HERE=Path(__file__).resolve().parent
HEAVY=Path('D:/gait-research-data/D_P/planar-tug-improvement-20261104')
OLD=Path('D:/Projects/D_P/studies/20261104-opencap-camera-tug')
RAW=Path('D:/gait-research-data/D_P/opencap-camera-tug-20261104')
spec=importlib.util.spec_from_file_location('frozen_camera',OLD/'experiment.py');cam=importlib.util.module_from_spec(spec);spec.loader.exec_module(cam)
VIEWS=['orthographic','geometry3d_reference','perspective3m','perspective5m','yaw20_perspective5m','depth3d_reference']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def from_pose(pose,t):
    reader=cam.readtrc
    try:
        cam.readtrc=lambda p:(t,pose,cam.LANDMARKS)
        return cam.extract('memory',float(t[0]))
    finally:cam.readtrc=reader
def features(raw,t,view):
    sm=savgol_filter(raw,7,2,axis=0);lookup={n:sm[:,i] for i,n in enumerate(cam.LANDMARKS)}
    pelvis=lookup['midHip'];direction=(pelvis[-1]-pelvis[0])[[0,2]];direction/=np.linalg.norm(direction)
    transverse=np.array([-direction[1],direction[0]])
    if view.startswith('perspective') or view.startswith('yaw'):
        distance=3. if view=='perspective3m' else 5.;yaw=np.radians(20 if view.startswith('yaw') else 0)
        center=(pelvis[0]+pelvis[-1])/2
        xx=(raw[:,:,[0,2]]-center[[0,2]])@direction
        depth=(raw[:,:,[0,2]]-center[[0,2]])@transverse
        xc=np.cos(yaw)*xx+np.sin(yaw)*depth;zc=distance-np.sin(yaw)*xx+np.cos(yaw)*depth
        assert (zc>1).all()
        synthetic=np.zeros_like(raw);synthetic[:,:,0]=distance*xc/zc;synthetic[:,:,1]=1+distance*(raw[:,:,1]-1)/zc
        return from_pose(synthetic,t)[0]
    f,qa=from_pose(raw,t)
    if view=='orthographic':return f
    leg=np.median(np.concatenate([np.linalg.norm(lookup[s+'Hip']-lookup[s+'Knee'],axis=1)+np.linalg.norm(lookup[s+'Knee']-lookup[s+'Ankle'],axis=1) for s in ['L','R']]))
    ratio=qa['projected_leg_length_m']/leg
    for n in f:
        if n.startswith(('LAnkle','RAnkle','LWrist','RWrist','pelvis_bob','pelvis_vertical_speed','ankle_separation')):f[n]*=ratio
    def summary(n,a):f[n+'_mean']=float(a.mean());f[n+'_std']=float(a.std());f[n+'_range']=float(np.ptp(np.percentile(a,[5,95])))
    def angle(a,b):return np.degrees(np.arccos(np.clip(np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12),-1,1)))
    for side in ['L','R']:
        summary(side+'_knee_flexion_2d',180-angle(lookup[side+'Hip']-lookup[side+'Knee'],lookup[side+'Ankle']-lookup[side+'Knee']))
        thigh=lookup[side+'Knee']-lookup[side+'Hip'];mag=angle(thigh,np.tile([0,-1,0],(len(t),1)))
        summary(side+'_thigh_inclination',mag*np.sign(thigh[:,[0,2]]@direction))
    shoulder=(lookup['LShoulder']+lookup['RShoulder'])/2-pelvis
    summary('trunk_lean',angle(shoulder,np.tile([0,1,0],(len(t),1)))*np.sign(shoulder[:,[0,2]]@direction))
    velocity=np.linalg.norm(np.diff(pelvis[:,[0,2]],axis=0),axis=1)*30
    f.update(speed_mean=float(velocity.mean()),speed_std=float(velocity.std()),speed_p90_minus_p10=float(np.ptp(np.percentile(velocity,[10,90]))),speed_acceleration_rms=float(np.sqrt(np.mean((np.diff(velocity)*30)**2))))
    if view=='depth3d_reference':
        for n in ['midHip','LAnkle','RAnkle','LWrist','RWrist']:
            dep=(lookup[n]-pelvis)[:,[0,2]]@transverse/leg
            if n=='midHip':dep=pelvis[:,[0,2]]@transverse/leg
            f[n+'_depth_std']=float(dep.std());f[n+'_depth_range']=float(np.ptp(np.percentile(dep,[5,95])))
    return f
def freeze():
    assert not (HERE/'projection-protocol.json').exists()
    save(HERE/'projection-protocol.json',{'code_sha256':sha(__file__),'old_code_sha256':sha(OLD/'experiment.py'),
      'source_features_sha256':sha(RAW/'features.npz'),'source_audit_sha256':sha(OLD/'source-audit.json'),'source_splits_sha256':sha(OLD/'effective-splits.json'),
      'people':297,'views':VIEWS,'window':'exactly same frozen3s30Hz keypoint windows, no relabeling or exclusion',
      'original_agreement':'rebuild original64 features from cached raw20-keypoint3D windows before any new fitting; tolerance1e-8',
      'geometry3d':'same named descriptors, but knee/thigh/trunk angles and leg scale from3D geometry; speed is full horizontal path. Diagnostic unavailable from an uncalibrated monocular plane',
      'depth3d':'geometry reference plus10 transverse trajectory summaries, diagnostic only',
      'synthetic_camera':'fixed pinhole camera3m/5m from plane, height1m, yaw0/20degrees; metric scale known at reference plane; no lens distortion/noise/occlusion, not a reconstruction of original phone cameras',
      'model':'same fixedET128 minleaf10 across all views; train-only median imputation; input own mean speed+inverse speed versus all descriptors+inverse speed',
      'evaluation':'same25 frozen outer folds; no hyperparameter selection; MAE and severe9 errors; paired person bootstrap; distinguish projection loss from raw3D quality',
      'sanity':'translation/yaw invariance and2x scale response; projected segment length <=3D; angle and speed differences',
      'limits':'Projection direction derived from3D per person is ideal alignment, not deployable phone orientation estimation. No original RGB or pose visibility/confidence. Failure to improve does not prove equivalence.'})
def prepare():
    p=json.loads((HERE/'projection-protocol.json').read_text());assert sha(__file__)==p['code_sha256']
    d=np.load(RAW/'features.npz');names=d['names'].tolist();audit=json.loads((OLD/'source-audit.json').read_text())
    arrays={v:[] for v in VIEWS};poses=[];times=[];qalog=[];maxold=0.;invariant=[]
    for i,a in enumerate(audit):
        t,pose,ln=cam.readtrc(RAW/'raw'/a['member']);grid=a['start_time']+np.arange(91)/30
        raw=np.stack([np.stack([np.interp(grid,t,pose[:,ln.index(n),k]) for k in range(3)],1) for n in cam.LANDMARKS],1)
        poses.append(raw);times.append(grid)
        orig=features(raw,grid,'orthographic');vals=np.array([orig[n] for n in names])
        np.testing.assert_allclose(vals,d['X'][i],atol=1e-8,rtol=0,equal_nan=True)
        finite=np.isfinite(vals);maxold=max(maxold,float(abs(vals[finite]-d['X'][i,finite]).max()))
        allf={v:features(raw,grid,v) for v in VIEWS}
        for view,f in allf.items():arrays[view].append([f[n] for n in f])
        sm=savgol_filter(raw,7,2,axis=0);look={n:sm[:,j] for j,n in enumerate(cam.LANDMARKS)}
        leg3=np.median(np.r_[*[np.linalg.norm(look[s+'Hip']-look[s+'Knee'],axis=1)+np.linalg.norm(look[s+'Knee']-look[s+'Ankle'],axis=1) for s in ['L','R']]])
        _,oldqa=from_pose(raw,grid)
        qalog.append({'id':a['id'],'projected_leg_over3D':float(oldqa['projected_leg_length_m']/leg3),
          'speed2D':orig['speed_mean'],'speed3D':allf['geometry3d_reference']['speed_mean'],
          'knee_mean_difference_degrees':float(np.mean([abs(orig[s+'_knee_flexion_2d_mean']-allf['geometry3d_reference'][s+'_knee_flexion_2d_mean']) for s in ['L','R']]))})
        if i<3:
            translated=features(raw+np.array([10,2,4]),grid,'orthographic');ang=np.pi/4
            rot=np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])
            rotated=features(raw@rot.T,grid,'orthographic');scaled=features(raw*.8,grid,'orthographic')
            maxinvariance=0.;maxscale=0.
            for n in names:
                if not np.isfinite(orig[n]):continue
                maxinvariance=max(maxinvariance,abs(translated[n]-orig[n]),abs(rotated[n]-orig[n]))
                expected=orig[n]*(.8 if n.startswith('speed_') else 1)
                maxscale=max(maxscale,abs(scaled[n]-expected))
            assert maxinvariance<1e-7 and maxscale<1e-7
            invariant.append({'id':a['id'],'translation_yaw_maxdifference':maxinvariance,'scale_maxdifference':maxscale})
    (HEAVY/'projection').mkdir(exist_ok=True)
    np.savez_compressed(HEAVY/'projection/input.npz',y=d['y'],ids=d['ids'],**{v:np.array(a) for v,a in arrays.items()})
    np.savez_compressed(HEAVY/'projection/raw-window-cache.npz',poses=np.stack(poses),times=np.stack(times))
    save(HERE/'projection-geometry-audit.json',{'original_feature_max_difference':maxold,'invariance_checks':invariant,'people':qalog,
      'view_feature_names':{v:list(features(poses[0],times[0],v)) for v in VIEWS},'input_sha256':sha(HEAVY/'projection/input.npz'),
      'raw_cache_sha256':sha(HEAVY/'projection/raw-window-cache.npz')})
    print('PROJECTION_PREPARED',len(audit),flush=True)
def train():
    d=np.load(HEAVY/'projection/input.npz');splits=json.loads((OLD/'effective-splits.json').read_text());y=d['y'];records=[]
    for v in VIEWS:
      for arm in ['speed','features']:
        X=d[v][:,:1] if arm=='speed' else d[v];X=np.column_stack([X,1/np.maximum(X[:,0],.1)])
        pred=np.full((5,len(y)),np.nan)
        for rep in splits:
          for fold in rep['splits']:
            tr=np.array(fold['train']);te=np.array(fold['test']);rr=rep['repeat'];ff=fold['fold']
            imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X[tr]);m=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=10,max_features=1.,random_state=20261104,n_jobs=1).fit(imp.transform(X[tr]),y[tr])
            p=m.predict(imp.transform(X[te]));pred[rr,te]=p;path=HEAVY/'projection'/f'{v}-{arm}-r{rr}-f{ff}.joblib';joblib.dump({'imputer':imp,'model':m},path,compress=3)
            records.append({'view':v,'arm':arm,'repeat':rr,'fold':ff,'train':tr.tolist(),'test':te.tolist(),'model':path.name,'sha256':sha(path),'prediction':p.tolist()})
        save(HERE/'projection-receipt.json',records);print('PROJECTION_FITTED',v,arm,flush=True)
    print('PROJECTION_MODELS_COMPLETE',len(records),flush=True)
if __name__=='__main__':{'freeze':freeze,'prepare':prepare,'train':train}[sys.argv[1]]()
