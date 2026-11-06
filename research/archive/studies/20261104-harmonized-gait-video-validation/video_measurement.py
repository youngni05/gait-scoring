"""Actual RGB-derived MediaPipe replay and explicitly non-concurrent table audit."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104')
CAD=Path('D:/Projects/D_P/gdi-research-harness/studies/20261101-camera-cadence')
STORE=Path('D:/gait-research-data/D_P/camera-cadence-20261101')
HEIGHT=Path('D:/gait-research-data/D_P/camera-height-speed-20261101')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def freeze():
    assert not(HERE/'video-protocol.json').exists()
    save(HERE/'video-protocol.json',dict(code_sha256=sha(__file__),
      purpose='Separate actualRGB measurement error from syntheticprojection and clinicalTUGprediction.',
      MoVi='27previouslyfrozenpeople,pairedPG1/PG2,firstsame4sec windows from priorcadence availability; sourceRGB-onlyMediaPipeHeavy800pixel2D,30fps,knownheight. Referenceindependentlyderived provider3Djointcoordinates in matched actualframes. No new pose inference or calibration. PG1/PG2separate; no accuracy-basedclipselection.',
      speed='height/median33landmarkverticalspan in pixels times OLSimagepelvis2Dslope. Referencehorizontal3DhipmidpointOLSslope in m/s. Recompute prior14straight2sec as separatematchedspeedcheck,notpoolwith27.',
      cycle='Candidate unchanged historical spectralankledifference estimate .4..1.8Hz,qualityR2>=.5,>=1.5cycles,frequencyaway.02bounds. Period=120/cadence. Independentreference algorithm: 3Dfootforwardexcursionrelativepelvispeaks with distance>=.6s,prominence20percentPTP,consecutiveinterval.6..3s. Forwardaxis=PCAhorizontalpelvis. Atleastonevalidsamefootpair. Notexpert/forceplateHSlabels; kinematiccycleproxy.',
      stride='Candidateimageplanespeed*estimatedcycle; reference3Dhorizontalankleendpointdisplacementbetweensamefootforwardpeaks,meanovervalidpairs. Kinematicphase-matchedcyclelength,notvalidatedheelcontactstridelength. No independentlyvalidatedstep-length algorithm inourMediaPipepipeline: clinicalstep/HSstride measurementstatus=unmeasurable.',
      HealthGait='398IDmatchedreleasedauthorAlphaPose-derivedestimates versusOptoGait/MuscleLAB. UGS/FGSseparate,completepositivepairspermetric,nooutliertrimming. Cycle=120/cadence (derivedmean-equivalentperiod,notmeanindividualstridetime). Notconcurrent:paperexplicitlysaysmeasurementsfirst,videoslater. Thereforecross-passdiscrepancy,notisolatedvideoextractionMAE. Adults19..64,no65+,noTUG/SPPB. RawRGBnotreleased,soownMediaPipere-inferenceunavailable.',
      units='speedm/s;stepandstridecm;cadencesteps/min;cycletimeseconds. Neverinferfullcycleas60/cadence.',
      metrics='count,coverage,MAEorbona-fidediscrepancy,bias,p95,Pearsonr,5000personbootstrap95CI. No thresholdoptimization;allresultsretained. Failedquality=>unmeasurable,notlowscore.',
      sources={str(p):sha(p) for p in [CAD/'availability.json',CAD/'protocol.json',HERE/'healthgait-tables-acquisition.json']},
      scope='No realphonecapturetest,elderlycamaccuracyorTUGend-to-endclaim; synchronizedclinicalfootcontacttruthstillmissing.' ))
def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);err=p-y;rng=np.random.default_rng(20261104)
    bs=np.mean(np.abs(err)[rng.integers(0,len(y),(5000,len(y)))],axis=1)
    return dict(n=len(y),mae=float(np.abs(err).mean()),bias=float(err.mean()),p95=float(np.quantile(abs(err),.95)),mae_ci95=np.quantile(bs,[.025,.975]).tolist(),r=float(np.corrcoef(y,p)[0,1]) if len(y)>1 and y.std()>0 and p.std()>0 else None)
def slope(v):
    t=np.arange(len(v))/30;t-=t.mean();return t@v/(t@t)
def kin_reference(ref):
    ref=ref/1000;hip=ref[:,[6,7]].mean(1);xy=hip[:,:2];_,_,vv=np.linalg.svd(xy-xy.mean(0),full_matrices=False);axis=vv[0]
    periods=[];lengths=[];events=[]
    for j in [10,11]:
        wave=(ref[:,j,:2]-xy)@axis
        peaks,_=find_peaks(wave,distance=18,prominence=.2*np.ptp(wave))
        for a,b in zip(peaks[:-1],peaks[1:]):
            dt=(b-a)/30
            if .6<=dt<=3:
                periods.append(dt);lengths.append(float(np.linalg.norm(ref[b,j,:2]-ref[a,j,:2])));events.append([j,int(a),int(b)])
    speed=float(np.linalg.norm(slope(xy)))
    return {'speed':speed,'cycle':float(np.mean(periods)) if periods else None,'stride':float(np.mean(lengths)) if lengths else None,'events':events}
def run():
    proto=json.loads((HERE/'video-protocol.json').read_text());assert proto['code_sha256']==sha(__file__)
    for p,h in proto['sources'].items():assert sha(p)==h
    b=BIG/'healthgait/tables';a=pd.read_csv(b/'gait_parameters.csv').set_index('ID');e=pd.read_csv(b/'gait_parameters_estimation.csv').set_index('ID');m=pd.read_csv(b/'patients_measures.csv').set_index('ID')
    assert len(a)==len(e)==len(m)==398 and a.index.is_unique and set(a.index)==set(e.index)==set(m.index)
    table=[];paired=[]
    for mode in ['UGS','FGS']:
      for label,field,unit in [('speed','Velocity','m/s'),('step','Step','cm'),('stride','Stride','cm'),('cadence','Cadence','steps/min'),('cycle','Cadence','s')]:
        y=a[field+'_'+mode].to_numpy();p=e.reindex(a.index)[field+'_'+mode].to_numpy();mask=np.isfinite(y)&np.isfinite(p)&(y>0)&(p>0)
        if label=='cycle':y=120/y;p=120/p
        table.append(dict(mode=mode,metric=label,unit=unit,interpretation='different-walking-pass discrepancy, not isolated measurement error',**metrics(y[mask],p[mask])))
        paired.extend(dict(dataset='HealthGait',id=pid,condition=mode,metric=label,reference=float(yy),prediction=float(pp)) for pid,yy,pp in zip(a.index[mask],y[mask],p[mask]))
    sys.path.insert(0,str(CAD));import study as cadence
    rows=[];estimates=[];sources={}
    availability=json.loads((CAD/'availability.json').read_text())
    for sid in availability['selected_people']:
      path=STORE/f's{sid:02d}-4sec.npz';receipt=json.loads(path.with_suffix('.json').read_text());assert sha(path)==receipt['array_sha256'];sources[str(path)]=sha(path)
      z=np.load(path);meta=Path(f'D:/gait-research-data/MoVi/reference/F_v3d_Subject_{sid}.mat');height=float(loadmat(meta,simplify_cells=True)[f'Subject_{sid}_F']['subject']['height'])/100
      np.testing.assert_allclose(z['PG1_reference'],z['PG2_reference'],rtol=0,atol=0)
      truth=kin_reference(z['PG1_reference'])
      for cam in ['PG1','PG2']:
        lm=z[cam+'_lm'];xy=lm[:,:,:2]*[800,600];hip=xy[:,[23,24]].mean(1);scale=height/np.median(np.ptp(xy[:,:,1],axis=1));speed=float(np.linalg.norm(slope(hip))*scale)
        period=cadence.estimate(xy[:,27]-xy[:,28]);cycle=120/period['cadence'] if period['quality'] else None
        rec=dict(subject=sid,camera=cam,reference=truth,speed=speed,cycle=cycle,stride=speed*cycle if cycle is not None else None,period_quality=period)
        estimates.append(rec)
        for field in ['speed','cycle','stride']:
            if truth[field] is not None and rec[field] is not None:paired.append(dict(dataset='MoVi',id=str(sid),condition=cam,metric=field,reference=truth[field],prediction=rec[field]))
    df=pd.DataFrame(paired);df.to_csv(HERE/'measurement-pairs.csv',index=False)
    for cam in ['PG1','PG2']:
      for field,unit in [('speed','m/s'),('cycle','s'),('stride','m')]:
        q=df[(df.dataset=='MoVi')&(df.condition==cam)&(df.metric==field)]
        rows.append(dict(camera=cam,metric=field,unit=unit,eligible_people=27,interpretation='synchronized 3D pelvis reference' if field=='speed' else 'kinematic reference proxy; no force/expert HS truth',**metrics(q.reference,q.prediction)))
    straight=[]
    old=pd.read_csv('D:/Projects/D_P/gdi-research-harness/studies/20261101-camera-height-speed/main-clips.csv')
    for cam in ['PG1','PG2']:
      q=old[(old.condition=='straight')&(old.camera==cam)];ys=[];ps=[]
      for _,r in q.iterrows():
        path=HEIGHT/(r['key']+'.npz');assert sha(path)==r['array_sha256'];z=np.load(path)
        xy=z['lm'][:,:,:2]*[800,600];hip=xy[:,[23,24]].mean(1);pred=float(np.linalg.norm(slope(hip))*float(z['height'])/np.median(np.ptp(xy[:,:,1],axis=1)))
        ref=z['reference'][:,[6,7]].mean(1)/1000;target=float(np.linalg.norm(slope(ref[:,:2])))
        np.testing.assert_allclose([pred,target],[r['plane_ols'],r['truth_ols']],rtol=0,atol=1e-12);ys.append(target);ps.append(pred)
      straight.append(dict(camera=cam,metric='speed',condition='previously_frozen_straight_2seconds',**metrics(ys,ps)))
    save(HERE/'video-results.json',dict(healthgait=table,healthgait_age={'min':int(m.Age.min()),'max':int(m.Age.max()),'age65plus':int((m.Age>=65).sum())},MoVi=rows,MoVi_straight_speed=straight,MoVi_estimates=estimates,
      source_hashes=sources,pairs_sha256=sha(HERE/'measurement-pairs.csv'),
      clinical_HS_stride_or_step_accuracy='unmeasurable: no synchronized independent footcontact groundtruth in current ownMediaPipecohort',
      phone_accuracy='unverified; laboratory cameras and cached RGB detections'))
    print(json.dumps({'HealthGait':table,'MoVi':rows,'straight':straight},indent=2),flush=True)
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
