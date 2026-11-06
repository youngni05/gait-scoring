"""Outcome-blind fixed-duration image-plane gait input from actual AlphaPose files."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np
from scipy.signal import savgol_filter,detrend
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;B=Path('D:/gait-research-data/D_P/mmu-frip-evaluation-20261105');S=B/'source/MMU-FRiP'
JOINTS=[0,5,6,7,8,9,10,11,12,13,14,15,16,18]
N=60;FPS=30
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def freeze():
    assert not(H/'input-protocol.json').exists()
    save(H/'input-protocol.json',dict(code_sha256=sha(__file__),source_version=3,
      target='Clinical CSV JH-FRAT total score0..31; never per-frame JSON detection score. Age/sex/ethnicity/fallhistory/scorecomponents/wholeTUGduration excluded from inputs.',
      modality='Primary three arms all use the same actual side-view2DAlphaPose26 coordinates,30fps perprovider; no3Dprojection or reference3D. Bodyheight/camerametrics absent =>m/sunmeasurable; speed is neck-to-midankle-size-normalized image-plane pelvis horizontal OLSspeed(bodylength/s). No independentphysicalspeedtruth.',
      selection='One60frame2secwindow perperson. ConsecutiveframeIDs,single detection/frame. SavitzkyGolay7/order2coordinates for localization. Candidate required joints confidence>=.3 in>=90%jointframes;medianverticalneck-midanklespan>20px; pelvis horizontal net/span>=.1;monotonicityabs(net)/sum(abssteps)>=.85;pelvisverticalp95-p5/span<=.35. Group consecutive eligible startindices having same direction; choose longestcluster,earliesttie,centerstart. Neitherfastestnorlabel-selected. No candidate=>unmeasurable,notnormalzero.',
      representation='14joint2D hip-centred with fixed windowmedianneck-midanklescale,directioncanonicalxflip. Window coordinates smooth7/order2; no temporal stretching,padding,fulltrialduration or qualitymask feature. Speed same allarms. Features=knee/hip/elbowanglemean/std/p5/p95,trunklean,ankleseparation,verticalpelvis,spectralanklediff,perjointvelocityRMS; relative2Ddescriptors,notclinicallycalibratedspatiotemporalmeasurements.',
      split='All125sortedpersonIDs stratified5foldseed20261105 onJHFRATbins<6/6..13/>=14beforequalityexclusion. Allthreeconditions share folds. Report>=65subset heldout fromsamefits. No independentclinicalvalidation claim.',
      QA='SourceZIPCRC/SHA,all125raw2DandclinicalIDmapping;pilotfirstP001/P003and12fixedIDsoutcomeblind;reviewselectedskeletoncontactsheetsbeforemodeling. Featureformulasindependentrebuilt; sourceboundaryinvariance andexactwindowreplay.',
      resources='Pinned498MBuncompressedsource,archive<1GB,CPUonlysmall125personsequencelearning,heavycache/modelsunderDdata; noexternalGPUtransfer.',
      targets={'MAE_points_max':1,'within2_min':.8,'relative_improvement_over_speed_min':.2,'scope':'engineeringexploratorytargets,notestablishedclinicalMDC;alland65plusrequired'},
      quality_failure_policy='Retainall125incoverageaudit;three-armaccuracyoncommonmeasurablepeople. No thresholdsrelaxedafteraccuracyresults. Allageandscorebandexclusionsreported.'))
def load_pose(pid):
    p=S/'Raw/2D'/f'{pid}_2D'/f'{pid}_2D.json';r=json.loads(p.read_text());frames=[int(x['image_id'].split('.')[0]) for x in r]
    if len(frames)!=len(set(frames)):raise ValueError('ambiguous_multiple_detections_perframe')
    order=np.argsort(frames);frames=np.array(frames)[order];a=np.array([r[i]['keypoints'] for i in order],float).reshape(-1,26,3)
    assert np.isfinite(a).all();return frames,a
def select(frames,a):
    if len(a)<N:raise ValueError('less_than_2sec')
    xy=savgol_filter(a[:,:,:2],7,2,axis=0);hip=xy[:,[11,12]].mean(1);span=abs(xy[:,18,1]-xy[:,[15,16],1].mean(1));candidates=[]
    for start in range(len(a)-N+1):
        sl=slice(start,start+N);hh=float(np.median(span[sl]));p=hip[sl];net=p[-1,0]-p[0,0];path=np.abs(np.diff(p[:,0])).sum()
        if np.any(np.diff(frames[sl])!=1) or hh<=20:continue
        if np.mean(a[sl][:,JOINTS,2]>=.3)<.9:continue
        if abs(net)/hh<.1 or abs(net)/max(path,1e-12)<.85:continue
        if np.ptp(np.quantile(p[:,1],[.05,.95]))/hh>.35:continue
        candidates.append((start,1 if net>=0 else -1))
    if not candidates:raise ValueError('no_eligible_2sec_straight_image_motion')
    clusters=[]
    for start,direction in candidates:
        if clusters and start==clusters[-1][-1][0]+1 and direction==clusters[-1][-1][1]:clusters[-1].append((start,direction))
        else:clusters.append([(start,direction)])
    chosen=max(clusters,key=lambda v:(len(v),-v[0][0]));start,direction=chosen[(len(chosen)-1)//2]
    return start,direction,len(chosen)
def angle(a,b,c):
    u=a-b;v=c-b;return np.rad2deg(np.arctan2(abs(u[:,0]*v[:,1]-u[:,1]*v[:,0]),np.sum(u*v,axis=1)))
def transform(window,direction):
    xy=savgol_filter(window[:,:,:2],7,2,axis=0);hip=xy[:,[11,12]].mean(1);scale=float(np.median(abs(xy[:,18,1]-xy[:,[15,16],1].mean(1))))
    if scale<=20:raise ValueError('invalid_local_scale')
    t=np.arange(N)/FPS;t-=t.mean();speed=float(abs(t@hip[:,0]/(t@t))/scale)
    p=(xy-hip[:,None,:])/scale;p[:,:,0]*=direction;F={'speed':speed}
    def stats(name,x):
        F[name+'_mean']=float(np.mean(x));F[name+'_std']=float(np.std(x));F[name+'_p05']=float(np.quantile(x,.05));F[name+'_p95']=float(np.quantile(x,.95))
    for side,sh,el,wr,h,k,ank in [('left',5,7,9,11,13,15),('right',6,8,10,12,14,16)]:
        stats(side+'_knee',angle(p[:,h],p[:,k],p[:,ank]));stats(side+'_hip',angle(p[:,sh],p[:,h],p[:,k]));stats(side+'_elbow',angle(p[:,sh],p[:,el],p[:,wr]))
    shoulder=p[:,[5,6]].mean(1);stats('trunk_lean',np.rad2deg(np.arctan2(shoulder[:,0],-shoulder[:,1])))
    stats('ankle_forward_separation',p[:,15,0]-p[:,16,0]);stats('ankle_vertical_separation',p[:,15,1]-p[:,16,1]);stats('pelvis_bob',(hip[:,1]-hip[:,1].mean())/scale)
    vel=np.diff(p[:,JOINTS],axis=0)*FPS
    for j,idx in enumerate(JOINTS):F[f'joint{idx}_velocity_rms']=float(np.sqrt(np.mean(np.sum(vel[:,j]**2,axis=1))))
    wave=detrend(p[:,15,0]-p[:,16,0]);freq=np.arange(.5,3.01,.25);power=np.array([abs(np.sum(wave*np.exp(-2j*np.pi*f*np.arange(N)/FPS)))**2 for f in freq]);F['ankle_periodicity_peak_hz']=float(freq[np.argmax(power)]);F['ankle_periodicity_concentration']=float(power.max()/max(power.sum(),1e-12))
    seq=p[:,JOINTS].reshape(N,-1).T.astype(np.float32)
    return F,seq,scale
def main(pilot=False):
    prot=json.loads((H/'input-protocol.json').read_text());assert prot['code_sha256']==sha(__file__)
    rows=sorted(csv.DictReader((S/'Extracted/final_gait_feature.csv').open(encoding='utf-8-sig')),key=lambda r:r['ID']);assert len(rows)==125
    ids=np.array([r['ID'] for r in rows]);y=np.array([float(r['score']) for r in rows]);age=np.array([float(r['age']) for r in rows]);fold=np.zeros(len(rows),int)
    for f,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261105).split(ids,np.digitize(y,[6,14]))):fold[te]=f
    audits=[];features=[];seqs=[];keep=[];windows=[]
    for i,pid in enumerate(ids):
        if pilot and i>=12:break
        try:
            frames,a=load_pose(pid);start,direction,cluster=select(frames,a);window=a[start:start+N];F,seq,scale=transform(window,direction)
            assert np.isfinite(seq).all() and np.isfinite(list(F.values())).all()
            audits.append(dict(id=pid,status='ok',frame_start=int(frames[start]),frame_end=int(frames[start+N-1]),source_frames=len(a),direction=direction,eligible_cluster_starts=cluster,scale_pixels=scale,speed=F['speed'],age=float(age[i]),score=float(y[i])))
            features.append(list(F.values()));seqs.append(seq);keep.append(i);windows.append(window)
        except Exception as e:audits.append(dict(id=pid,status='unmeasurable',reason=str(e),age=float(age[i]),score=float(y[i])))
    out=B/('pilot.npz' if pilot else 'input.npz');keep=np.array(keep)
    np.savez_compressed(out,X=np.array(features),sequence=np.array(seqs),windows=np.array(windows),ids=ids[keep],y=y[keep],age=age[keep],fold=fold[keep],feature_names=np.array(list(F)),all_ids=ids,all_y=y,all_age=age,all_fold=fold)
    save(H/('pilot-input-audit.json' if pilot else 'input-audit.json'),{'registered_people':125,'evaluated_for_quality':len(audits),'measurable':len(keep),'older65_measurable':int(sum(age[keep]>=65)),'feature_names':list(F),'feature_count':len(F),'records':audits,'input_sha256':sha(out),'code_sha256':sha(__file__)})
    print('PREPARED',len(keep),'OF',len(audits),'OLDER',int(sum(age[keep]>=65)),'FEATURES',len(F),flush=True)
if __name__=='__main__':{'freeze':freeze,'pilot':lambda:main(True),'full':main}[sys.argv[1]]()
