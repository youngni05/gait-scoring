"""Frozen landmark-only, synthetic side-view short-walk TUG comparison."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
from pathlib import Path
import csv, hashlib, json, sys, time
import joblib
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor

HERE=Path('D:/Projects/D_P/studies/20261104-opencap-camera-tug')
HEAVY=Path('D:/gait-research-data/D_P/opencap-camera-tug-20261104')
OLD=Path('D:/Projects/D_P/gdi-research-harness/studies/20261030-opencap415-speed-stability')
LABEL=Path('D:/gait-research-data/D_P/camera-tug-data-audit-20261104/new_opencap_csv.txt')
LANDMARKS=['Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist','midHip','RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel']
VIEWS=['speed_only','speed_dynamics','landmarks_only','speed_plus_landmarks']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def readtrc(p):
    lines=Path(p).read_text().splitlines()
    names=[v.strip() for v in lines[3].split('\t')[2:] if v.strip()]
    a=np.loadtxt(p,skiprows=6)
    assert a.shape[1]==2+3*len(names)
    assert lines[2].split()[4]=='m'
    return a[:,1],a[:,2:].reshape(len(a),len(names),3),names

def walking_start(p):
    """First >=3s active span from actual pose-landmark translation, no labels."""
    t,a,names=readtrc(p)
    pelvis=a[:,names.index('midHip')]
    grid=t[0]+np.arange(int(np.floor((t[-1]-t[0])*30))+1)/30
    pelvis=np.stack([np.interp(grid,t,pelvis[:,d]) for d in range(3)],axis=1)
    pelvis=savgol_filter(pelvis,7,2,axis=0)
    t=grid
    # Half-second net travel rejects stationary frame-to-frame tracking jitter.
    previous=np.maximum(0,np.arange(len(t))-15)
    travel=pelvis[:,[0,2]]-pelvis[previous][:,[0,2]]
    v=np.linalg.norm(travel,axis=1)/np.maximum(t-t[previous],1/30)
    dt=float(np.median(np.diff(t)));width=max(1,round(.25/dt))
    smooth=np.convolve(np.pad(v,(width//2,width-1-width//2),mode='edge'),np.ones(width)/width,mode='valid')
    mask=smooth>.2
    for i in range(1,len(mask)-1):
        if not mask[i] and mask[i-1]:
            end=i
            while end<len(mask) and not mask[end]:end+=1
            if end<len(mask) and t[end]-t[i]<=.25:mask[i:end]=True
    changes=np.diff(np.r_[False,mask,False].astype(int))
    spans=[(int(s),int(en),float(t[en]-t[s])) for s,en in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)-1)]
    valid=[(s,en,dur) for s,en,dur in spans if dur>=3-1e-6]
    return (float(t[valid[0][0]]) if valid else None),{'longest_active_span_seconds':max([x[2] for x in spans],default=0.),'movement_speed_threshold_mps':.2}

def extract(p,start):
    t,a,names=readtrc(p)
    assert np.all(np.diff(t)>0) and start>=t[0]-1e-6 and start+3<=t[-1]+1e-5
    tt=start+np.arange(91)/30
    # Only original pose keypoints, never LSTM-augmented anatomical markers.
    xyz=np.stack([np.stack([np.interp(tt,t,a[:,names.index(n),d]) for d in range(3)],1) for n in LANDMARKS],1)
    assert np.isfinite(xyz).all()
    xyz=savgol_filter(xyz,7,2,axis=0)
    pelvis=xyz[:,LANDMARKS.index('midHip')]
    direction=(pelvis[-1]-pelvis[0])[[0,2]]
    distance=np.linalg.norm(direction); assert distance>.1
    direction/=distance
    # Ideal, calibrated orthographic lateral projection. Discard transverse depth.
    q=np.stack([xyz[:,:,[0,2]]@direction,xyz[:,:,1]],axis=-1)
    q[:,:,0]-=q[0,LANDMARKS.index('midHip'),0]
    z={n:q[:,k] for k,n in enumerate(LANDMARKS)}
    leg=np.median(np.concatenate([np.linalg.norm(z[s+'Hip']-z[s+'Knee'],axis=1)+np.linalg.norm(z[s+'Knee']-z[s+'Ankle'],axis=1) for s in ['L','R']]))
    assert .2<leg<1.7
    v=np.diff(z['midHip'][:,0])*30
    f={'speed_mean':float(np.mean(np.abs(v))),'speed_std':float(np.std(v)),
       'speed_p90_minus_p10':float(np.ptp(np.percentile(v,[10,90]))),
       'speed_acceleration_rms':float(np.sqrt(np.mean((np.diff(v)*30)**2)))}
    z={n:(x-z['midHip'])/leg for n,x in z.items() if n!='midHip'} | {'midHip':z['midHip']/leg}
    def summary(name,x):
        f[name+'_mean']=float(np.mean(x));f[name+'_std']=float(np.std(x));f[name+'_range']=float(np.ptp(np.percentile(x,[5,95])))
    def corr(x,y):
        return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-8 and np.std(y)>1e-8 else np.nan
    def angle(u,v):
        return np.degrees(np.arccos(np.clip(np.sum(u*v,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-12),-1,1)))
    py=z['midHip'][:,1]
    summary('pelvis_bob',py-py.mean())
    f['pelvis_bob_mean']=0.0  # Mathematical zero: do not fit floating-point residue.
    f['pelvis_vertical_speed_rms']=float(np.sqrt(np.mean((np.diff(py)*30)**2)))
    shoulder=(z['LShoulder']+z['RShoulder'])/2
    trunk=np.degrees(np.arctan2(shoulder[:,0],shoulder[:,1]));summary('trunk_lean',trunk)
    knee={}
    for side in ['L','R']:
        for joint in ['Ankle','Wrist']:
            for axis in [0,1]:summary(side+joint+str(axis),z[side+joint][:,axis])
        knee[side]=180-angle(z[side+'Hip']-z[side+'Knee'],z[side+'Ankle']-z[side+'Knee'])
        summary(side+'_knee_flexion_2d',knee[side])
        thigh=z[side+'Knee']-z[side+'Hip']
        summary(side+'_thigh_inclination',np.degrees(np.arctan2(thigh[:,0],-thigh[:,1])))
    d=z['LAnkle'][:,0]-z['RAnkle'][:,0]
    summary('ankle_separation',d)
    f['ankle_antiphase_corr']=corr(z['LAnkle'][:,0],z['RAnkle'][:,0])
    f['knee_left_right_corr']=corr(knee['L'],knee['R'])
    f['arm_left_right_corr']=corr(z['LWrist'][:,0],z['RWrist'][:,0])
    f['left_arm_opposite_leg_corr']=corr(z['LWrist'][:,0],z['RAnkle'][:,0])
    f['right_arm_opposite_leg_corr']=corr(z['RWrist'][:,0],z['LAnkle'][:,0])
    for joint in ['Ankle','Wrist']:
        l=np.std(z['L'+joint][:,0]);r=np.std(z['R'+joint][:,0])
        f[joint+'_amplitude_asymmetry']=float(abs(l-r)/(l+r+1e-12))
    peaks=[]
    for sign in [1,-1]:
        pp,_=find_peaks(sign*d,distance=6,prominence=.1)
        peaks.extend([(int(i),sign) for i in pp])
    peaks.sort()
    intervals=np.diff([i/30 for i,s in peaks])
    # Extrema are timing proxies, not clinically labelled foot contacts.
    usable=len(peaks)>=3 and all(peaks[k][1]!=peaks[k+1][1] for k in range(len(peaks)-1))
    f['extrema_step_time_mean']=float(np.mean(intervals)) if usable else np.nan
    f['extrema_step_time_cv']=float(np.std(intervals)/np.mean(intervals)) if usable else np.nan
    f['extrema_cadence']=float(60/np.mean(intervals)) if usable else np.nan
    f['extrema_timing_available']=float(usable)
    centered=d-d.mean();ac=np.correlate(centered,centered,mode='full')[90:]/(np.dot(centered,centered)+1e-12)
    aps,_=find_peaks(ac[12:61]); lag=12+aps[np.argmax(ac[12+aps])] if len(aps) else None
    f['autocorr_cadence']=float(3600/lag) if lag is not None else np.nan
    f['autocorr_regularity']=float(ac[lag]) if lag is not None else np.nan
    f['autocorr_available']=float(lag is not None)
    return f,{'start_time':float(start),'end_time':float(start+3),'source_start':float(t[0]),'source_end':float(t[-1]),
              'projected_displacement_m':float(distance),'projected_leg_length_m':float(leg),'extrema_count':len(peaks),'timing_available':bool(usable)}

def candidates():
    out={}
    for alpha in [1.,100.]:out['ridge_'+str(alpha)]=Ridge(alpha=alpha)
    for C in [1.,10.,100.]:out['svr_'+str(C)]=SVR(C=C,epsilon=.1,gamma='scale')
    for leaf in [3,10]:out['et_'+str(leaf)]=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=leaf,max_features=1.,random_state=20261104,n_jobs=1)
    for leaves in [5,15]:out['lgb_'+str(leaves)]=LGBMRegressor(n_estimators=160,num_leaves=leaves,max_depth=-1,learning_rate=.03,min_child_samples=15,reg_lambda=5.,verbosity=-1,random_state=20261104,n_jobs=1)
    return {name:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),est) for name,est in out.items()}

def freeze():
    HERE.mkdir(parents=True,exist_ok=True);HEAVY.mkdir(parents=True,exist_ok=True)
    assert not (HERE/'protocol.json').exists()
    splits=json.loads((OLD/'splits.json').read_text())
    save(HERE/'splits.json',splits)
    protocol={'question':'Does a fixed three-second, ideal calibrated lateral landmark view add TUG prediction information beyond speed?',
      'target':'same-person, same-visit tft_tug; separate normal 10m walk, not TUG trial duration',
      'expected_people':298,'seconds':3,'cohort_manifest':str(OLD/'manifest.csv'),'manifest_sha256':sha(OLD/'manifest.csv'),
      'labels':str(LABEL),'labels_sha256':sha(LABEL),'splits_sha256':sha(HERE/'splits.json'),
      'source':'https://zenodo.org/records/22309771','source_representation':'post-augmentation TRC, but select only 20 original pose landmarks',
      'phone_scope':'synthetic orthographic lateral view from multi-camera 3D; not an actual phone camera or MediaPipe accuracy test; scale calibration required',
      'window':'reuse old first 3 seconds and start time, not reselect with outcome; resample 30Hz; Savitzky-Golay7 polynomial2 within window',
      'landmarks':LANDMARKS,'projection':'vertical Y plus horizontal travel direction from window endpoints; discard transverse component; ideal alignment',
      'normalization':'body-relative displacement / median projected leg length within clip; speed in metres/sec from calibration',
      'forbidden_predictors':['age','sex','diagnosis','body-height metadata','clinical 10m time','TUG motion','record duration','OpenSim angles','augmented anatomical markers','transverse depth','other clinical scores'],
      'views':{'speed_only':['speed_mean'],'speed_dynamics':['speed_mean','speed_std','speed_p90_minus_p10','speed_acceleration_rms'],
        'landmarks_only':'all derived features except four speed quantities','speed_plus_landmarks':'all features'},
      'feature_families':['speed dynamics','vertical pelvis motion','sagittal trunk lean','body-relative ankle/wrist motion','2D knee flexion/thigh inclination','bilateral and arm-leg correlations','amplitude asymmetry','extrema timing and autocorrelation proxies'],
      'timing_note':'No labelled heel strikes; three-second interval variation is a noisy proxy; unavailable is missing, never assumed zero',
      'models':{n:str(p.steps[-1][1]) for n,p in candidates().items()},
      'selection':'one model/hyperparameter per outer fold/view by minimum mean absolute inner OOF error; same pool all views; train-only imputation/scaling; no target transformation',
      'evaluation':'reuse all five old person KFold partitions, five folds each; original repeat0 primary; other4 sensitivity; same298 in every arm',
      'subgroup':'age>=65, CTL, TUG<12/12-20/>=20 descriptive only; no subgroup tuning',
      'uncertainty':'10000 paired person bootstraps, seed20261104; report feature minus baseline MAE; repeated partitions conditional, not independent',
      'decision':'credible development gain requires paired95% upper bound<0 against both speed baselines; no clinical or single-phone accuracy claim',
      'prior_reuse':'people, labels and splits already used for development; source official heldout split not untouched; no new independent external validation',
      'failure_rule':'halt on missing source, CRC failure, nonfinite keypoint or insufficient 3s coverage; no silent cohort change',
      'code_sha256':sha(__file__)}
    save(HERE/'protocol.json',protocol); print('FROZEN',flush=True)

def prepare():
    protocol=json.loads((HERE/'protocol.json').read_text());assert sha(__file__)==protocol['code_sha256']
    rows=list(csv.DictReader((OLD/'manifest.csv').open())); assert len(rows)==298
    source=np.load(OLD/'input.npz',allow_pickle=False)
    ids=source['ids'];assert list(ids)==[r['id'] for r in rows]
    labels={(r['subid'],int(r['visit'])):r for r in csv.DictReader(LABEL.open())}
    assert sha(LABEL)==protocol['labels_sha256']
    assert json.loads((HERE/'acquisition.json').read_text())['complete']
    fs=[];audit=[];meta=[];selection=[];retained=[]
    for i,row in enumerate(rows):
        member=row['file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc')
        start,qa=walking_start(HEAVY/'raw'/member)
        selection.append({'id':row['id'],'original_index':i,'old_start':float(row['start_time']),'new_start':start,
                          'included':start is not None,'reason':'first continuous 3s pose-pelvis walking span' if start is not None else 'no continuous 3s pose-pelvis speed>0.2m/s span',**qa})
        if start is None:continue
        f,q=extract(HEAVY/'raw'/member,start);fs.append(f);retained.append(i)
        lab=labels[(row['id'],int(row['visit']))];assert abs(float(lab['tft_tug'])-float(source['y'][i]))<1e-5
        audit.append({'id':row['id'],'visit':int(row['visit']),'member':member,'sha256':sha(HEAVY/'raw'/member),**q})
        meta.append({'id':row['id'],'age':float(lab['age']),'diagnosis':lab['diag'],'visit':int(row['visit']),'tug':float(lab['tft_tug']),'released_split':lab['split']})
    retained=np.array(retained);ids=ids[retained];names=list(fs[0]);X=np.array([[f[n] for n in names] for f in fs]);assert not np.isinf(X).any()
    views={'speed_only':[names.index('speed_mean')],'speed_dynamics':list(range(4)),
           'landmarks_only':list(range(4,len(names))),'speed_plus_landmarks':list(range(len(names)))}
    np.savez_compressed(HEAVY/'features.npz',X=X,y=source['y'][retained].astype(float),ids=ids,names=np.array(names))
    save(HERE/'window-selection.json',selection)
    indexmap={old:new for new,old in enumerate(retained.tolist())}
    splits=json.loads((OLD/'splits.json').read_text())
    for repeat in splits:
      for fold in repeat['splits']:
        for key in ['train','test']:fold[key]=[indexmap[i] for i in fold[key] if i in indexmap]
        for inner in fold['inner']:
          for key in ['train','validation']:inner[key]=[indexmap[i] for i in inner[key] if i in indexmap]
    save(HERE/'effective-splits.json',splits)
    save(HERE/'feature-schema.json',{'names':names,'views':views,'missing_per_feature':dict(zip(names,np.isnan(X).sum(0).tolist()))})
    save(HERE/'source-audit.json',audit);save(HERE/'metadata.json',meta)
    save(HERE/'preparation.json',{'features_sha256':sha(HEAVY/'features.npz'),'protocol_sha256':sha(HERE/'protocol.json'),
      'people':len(ids),'features':len(names),'age65plus':sum(x['age']>=65 for x in meta),
      'TUG20plus':sum(x['tug']>=20 for x in meta),'timing_available':sum(x['timing_available'] for x in audit),
      'excluded_people':len(rows)-len(ids),'effective_splits_sha256':sha(HERE/'effective-splits.json')})
    print('PREPARED',len(ids),len(names),flush=True)

def train():
    prep=json.loads((HERE/'preparation.json').read_text());assert sha(HEAVY/'features.npz')==prep['features_sha256']
    assert sha(HERE/'protocol.json')==prep['protocol_sha256']
    d=np.load(HEAVY/'features.npz');X=d['X'];y=d['y'];ids=d['ids']
    schema=json.loads((HERE/'feature-schema.json').read_text());splits=json.loads((HERE/'effective-splits.json').read_text())
    pool=candidates();out=[];pred=[];(HEAVY/'models').mkdir(exist_ok=True)
    for repeat in splits:
      rr=repeat['repeat']
      for fold in repeat['splits']:
        ff=fold['fold'];tr=np.array(fold['train']);te=np.array(fold['test'])
        assert not set(ids[tr])&set(ids[te]);assert len(set(ids))==len(ids)
        for view in VIEWS:
          xx=X[:,schema['views'][view]];losses={};innerpred={}
          for name,template in pool.items():
            pp=np.full(len(y),np.nan)
            for inf,inner in enumerate(fold['inner']):
              it=np.array(inner['train']);iv=np.array(inner['validation'])
              assert set(it)<=set(tr) and set(iv)<=set(tr) and not set(it)&set(iv)
              m=clone(template).fit(xx[it],y[it]);pp[iv]=m.predict(xx[iv])
            assert np.isfinite(pp[tr]).all();losses[name]=float(np.mean(np.abs(pp[tr]-y[tr])));innerpred[name]=pp[tr]
          best=min(losses,key=losses.get);model=clone(pool[best]).fit(xx[tr],y[tr]);pp=model.predict(xx[te])
          stem=f'r{rr}-f{ff}-{view}';path=HEAVY/'models'/(stem+'.joblib');joblib.dump(model,path)
          np.savez_compressed(HEAVY/'models'/(stem+'-inner.npz'),**innerpred)
          out.append({'repeat':rr,'fold':ff,'view':view,'selected':best,'inner_MAE':losses,'model_sha256':sha(path),
                      'train':tr.tolist(),'test':te.tolist()})
          for i,p in zip(te,pp):pred.append({'repeat':rr,'fold':ff,'view':view,'index':int(i),'id':str(ids[i]),'true':float(y[i]),'pred':float(p)})
        print('FITTED',rr,ff,flush=True)
        save(HERE/'selection.json',out);save(HEAVY/'predictions.json',pred)
    print('TRAINING_COMPLETE',len(out),'MODELS',len(pred),'PREDICTIONS',flush=True)

if __name__=='__main__':
    {'freeze':freeze,'prepare':prepare,'train':train}[sys.argv[1]]()
