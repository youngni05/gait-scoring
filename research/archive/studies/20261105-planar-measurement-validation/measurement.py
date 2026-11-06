"""Actual RGB-derived MediaPipe measurements versus independent physical MoCap markers."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter,find_peaks
from scipy.optimize import linear_sum_assignment
H=Path(__file__).resolve().parent;B=Path('D:/gait-research-data/D_P/planar-measurement-validation-20261105')
ROOT=Path('D:/gait-research-data/MoVi');POSE=ROOT/'experiments/walking-expansion'
PREVIOUS=Path('D:/Projects/D_P/gdi-research-harness/studies/20261101-camera-height-speed/main-clips.csv')
CAD=Path('D:/Projects/D_P/gdi-research-harness/studies/20261101-camera-cadence/availability.json')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def slope(z,fs):
    t=np.arange(len(z))/fs;t-=t.mean();return t@z/(t@t)

def freeze():
    assert not(H/'protocol.json').exists()
    save(H/'protocol.json',{'code_sha256':sha(__file__),'previous_window_table_sha256':sha(PREVIOUS),'cached_pose_manifest_sha256':sha(POSE/'results.json'),
      'question':'Howaccuratelydoactual2DMediaPipevideomeasuresspeed,steplength,steptimeandbilateraldifferencesagainstindependent120Hzopticalphysicalmarkers?',
      'primary':'Samepreviouslyfrozen14people,pairedPG1near-side/PG2oblique2secondstraightwindows.Noaccuracy-drivenclipselection. Reportall14includingunmeasurable.',
      'additional':'All30cachedwalkingpeople,earliestcommonconsecutive4secondblocks,nonoverlap,fromfirstcommonframe. Referencegeometryfilterisvalidation-only:netdistance>.5m,net/path>=.9,firstPCAvariance>=.98;sameacceptedwindowsbothcameras. No inferred deployablecameraqualityclassifier.',
      'source':'ActualRGBMediaPipeHeavy800x600,30Hz.Imagecoordinatesonly.NofeedingprojectedMoCapintopredictor.Knownsubjectheightusedforprimaryscale;height/median33landmarkverticalspan.Nooutcome-basedmetriccalibration.',
      'reference':'Native120HzMoViFmarkerLocation,LHEE/RHEE,pelviscentroidLFWT/LBWT/RFWT/RBWT.Publishedframeconventionindex=(videoframeonebased-1)*4.ReplaycachedLHIP/RHIP/LKJC/RKJC/LAJC/RAJCtoverifyframeandunits.Missingphysicalmarkersrejectreference,nogapinterpolation.',
      'events':'Zeni-inspiredkinematicHS:forwardheelpositionrelativepelvislocalmax;SG7/order2at30HzandSG29/order2at120Hz;peakdistance.4s,prominence=max(.02m,.15range).ForwardreferenceaxisPCAofpelvis,signnetdirection;predictedaxisimageX,signownhiptrend. Notforceplateorexpertcontacttruth.',
      'measures':'Speed=normOLShorizontalpelvisvelocitym/s(referenceworldXY,predimageXY*scale);steptime=consecutiveopposite-sideHStimedifference.2..1.5s;steplength=forwardheelsseparationatHS;stridelength=sameheelpositionadvanceatnextsamefootHS.8..3s;eventpairmatchingseparatepertolerance.15s,nooffsetfitting.',
      'symmetry':'AbsoluteLmean-Rmeandividedbybilateralmean*100percent,separatelength/time;atleast2validstepspersideforreferenceandprediction.Otherwindowmeasures>=2validsteps;stride>=1validsamefootpair.',
      'inputquality':'Nomissinglandmarks;pelvismedianvisibility>=.5forspeed;bothheelmedianvisibility>=.5andimagehorizontalnetprogress>=.1mforsteplength/time/stride/asymmetry. Fail=>unmeasurablewithreason,notzeroorlowhealthscore. Rawunfiltered-speeddiagnosticretained.',
      'metrics':'Coverage,person-equalMAE,bias,95percentLoA,p95AE;10000personbootstrap. Step/strideinmetres,steptimeseconds,symmetrypercentagepoints.Eventprecision/recall/F1withsamefootmatching. Keeppersonandwindowcountsseparate.',
      'repeat_or_change':'Onlysame-continuous-recordnonoverlapping4secondblocks:comparepredictedchangeagainstmarker-referencechange,personclusterederrors. Noclaimseparatecapturetestretest,MDCorlongitudinaldecline. AnnotatedchangepopulationisnotavailableincurrentMoVi.',
      'verification':'SourceSHA,exactvideo/nativeframeindices,cachedjointreplay,independentOLS/geometry/eventmatching/step-tableformulas,independentmetricrebuild.Allfailuresretained.Nomodeltraining,nofittedtimeoffset,noautomaticthresholdrelaxation.',
      'scope':'LaboratoryRGB,mostlyyoungadults;notgeneralelderly/realphonehardwarevalidation.Marker-eventuncertaintyremains.No clinicalfootcontactgoldorTUG/SPPB prediction inthisstudy.',
      'storage':'Smallcode/tables/researchrecordinD:/Projects/D_P/studies/20261105-planar-measurement-validation;arraysunderD:/gait-research-data/D_P/planar-measurement-validation-20261105;expected<30MB,noGPU/noexternaltransfer.'})

def reference(sid):
    path=ROOT/'reference'/f'F_v3d_Subject_{sid}.mat';mat=loadmat(path,simplify_cells=True)[f'Subject_{sid}_F'];return path,mat

def prepare():
    prot=read(H/'protocol.json');assert prot['code_sha256']==sha(__file__) and prot['previous_window_table_sha256']==sha(PREVIOUS)
    B.mkdir(parents=True,exist_ok=True);records=[];people=[];prior=list(csv.DictReader(PREVIOUS.open()));manifest=read(POSE/'results.json')['results'];availability=read(CAD)['people'];sourcehash={}
    for person in availability:
        sid=person['subject'];refpath,mat=reference(sid);move=mat['move'];names=[value if isinstance(value,str) else '' for value in move['markerName']];age=float(mat['subject']['age']);height=float(mat['subject']['height'])/100;people.append({'subject':sid,'age':age,'height':height});sourcehash[str(refpath)]=sha(refpath)
        arrays={}
        for camera in ['PG1','PG2']:
            path=POSE/f'subject-{sid:02d}'/'heavy'/f'{camera}_800.npz';comparison=path.with_name(camera+'_800_comparison.npz');assert sha(path)==next(r for r in manifest if r['subject']==sid and r['camera']==camera and r['variant']=='heavy' and r['input_width']==800)['prediction_sha256']
            assert sha(comparison)==person['files'][str(comparison)];sourcehash[str(path)]=sha(path);sourcehash[str(comparison)]=sha(comparison);arrays[camera]=(np.load(path),np.load(comparison))
        common=np.intersect1d(arrays['PG1'][1]['frames'],arrays['PG2'][1]['frames']);windows=[]
        for row in prior:
            if int(row['subject'])==sid and row['condition']=='straight' and row['camera']=='PG1':windows.append(('primary2s',int(row['first_frame']),int(row['last_frame'])))
        if len(common):
            for n,group in [(120,'extra4s'),(60,'extra2s')]:
                for start in range(int(common[0]),int(common[-1])-n+2,n):
                    frames=np.arange(start,start+n)
                    if np.isin(frames,common).all():windows.append((group,start,start+n-1))
        for group,start,end in windows:
            frames=np.arange(start,end+1);native=np.arange((start-1)*4,(end-1)*4+1);assert native[-1]<len(move['markerLocation'])
            markerids=[names.index(n) for n in ['LFWT','LBWT','RFWT','RBWT','LHEE','RHEE']];raw=move['markerLocation'][native][:,markerids]/1000;gaps=move['markerGaps'][native][:,markerids].astype(bool);reason=None
            if gaps.any() or not np.isfinite(raw).all():reason='missing_physical_reference_marker'
            pelvis=raw[:,:4].mean(1);heel=raw[:,4:];horizontal=pelvis[:,:2];travel=horizontal[-1]-horizontal[0];net=float(np.linalg.norm(travel));pathlength=float(np.linalg.norm(np.diff(horizontal[::12],axis=0),axis=1).sum());singular=np.linalg.svd(horizontal-horizontal.mean(0),compute_uv=False);linearity=float(singular[0]**2/max(sum(singular**2),1e-12));ratio=net/max(pathlength,1e-12)
            geometry=net>.5 and ratio>=.9 and linearity>=.98
            if group.startswith('extra') and not geometry:reason=reason or 'reference_geometry_not_straight'
            for camera,(pose,cmp) in arrays.items():
                idx=np.searchsorted(pose['video_frame_one_based'],frames);np.testing.assert_array_equal(pose['video_frame_one_based'][idx],frames);lm=pose['landmarks'][idx];counts=pose['pose_counts'][idx];cidx=np.searchsorted(cmp['frames'],frames);np.testing.assert_array_equal(cmp['frames'][cidx],frames)
                # Native marker-to-cached virtual-centre replay establishes frame convention independently.
                for key,j in [('LHIP',6),('RHIP',7),('LKJC',8),('RKJC',9),('LAJC',10),('RAJC',11)]:
                    expected=move['markerLocation'][(frames-1)*4,names.index(key)];good=~move['markerGaps'][(frames-1)*4,names.index(key)].astype(bool);np.testing.assert_allclose(expected[good],cmp['reference_3d'][cidx,j][good],rtol=0,atol=0)
                key=f's{sid:02d}_{camera}_{group}_{start}';dest=B/(key+'.npz');np.savez_compressed(dest,frames=frames,native_index=native,lm=lm,pose_counts=counts,pelvis120=pelvis,heels120=heel,height=np.array(height))
                records.append({'key':key,'subject':sid,'age':age,'camera':camera,'group':group,'first_frame':start,'last_frame':end,'reference_status':'ok' if reason is None else reason,'reference_net_path_ratio':ratio,'reference_linearity':linearity,'array':str(dest),'sha256':sha(dest)})
        print('PREPARED_SUBJECT',sid,flush=True)
    assert len(people)==30 and sum(r['group']=='primary2s' for r in records)==28
    save(H/'input-audit.json',{'status':'passed','people':people,'source_hashes':sourcehash,'records':records,'counts':{'people':30,'primary_people':14,'registered_windows_views':len(records),'older65':sum(r['age']>=65 for r in people)},'code_sha256':sha(__file__),'native_marker_to_cached_joint_replay':'exact','native_fs':120,'video_fs':30})

def events(heel,pelvis,fs):
    n=len(heel);window=7 if fs==30 else 29;hh=savgol_filter(heel,window,2,axis=0);pp=savgol_filter(pelvis,window,2,axis=0);eventlist=[]
    for side in [0,1]:
        signal=hh[:,side]-pp;peaks,_=find_peaks(signal,distance=int(.4*fs),prominence=max(.02,.15*np.ptp(signal)))
        eventlist.extend({'side':side,'index':int(index),'time':float(index/fs)} for index in peaks)
    return sorted(eventlist,key=lambda r:r['time'])

def measures(heel,ev,fs):
    steps=[];strides=[]
    for previous,current in zip(ev[:-1],ev[1:]):
        side=current['side'];dt=(current['index']-previous['index'])/fs;index=current['index']
        if side!=previous['side'] and .2<=dt<=1.5:
            length=float(heel[index,side]-heel[index,1-side])
            steps.append({'side':side,'time':current['time'],'step_time':float(dt),'step_length':length})
    for side in [0,1]:
        same=[r for r in ev if r['side']==side]
        for a,b in zip(same[:-1],same[1:]):
            dt=(b['index']-a['index'])/fs
            if .8<=dt<=3:strides.append(float(heel[b['index'],side]-heel[a['index'],side]))
    out={'step_time':float(np.mean([r['step_time'] for r in steps])) if len(steps)>=2 else None,'step_length':float(np.mean([r['step_length'] for r in steps])) if len(steps)>=2 else None,'stride_length':float(np.mean(strides)) if strides else None}
    for field in ['step_time','step_length']:
        a=[r[field] for r in steps if r['side']==0];b=[r[field] for r in steps if r['side']==1]
        out[field+'_asymmetry']=float(100*abs(np.mean(a)-np.mean(b))/max((np.mean(a)+np.mean(b))/2,1e-6)) if len(a)>=2 and len(b)>=2 else None
    return out,steps

def match_events(pred,truth):
    errors=[];matches=0
    for side in [0,1]:
        a=np.array([r['time'] for r in pred if r['side']==side]);b=np.array([r['time'] for r in truth if r['side']==side])
        if not len(a) or not len(b):continue
        cost=abs(a[:,None]-b);ii,jj=linear_sum_assignment(cost)
        for i,j in zip(ii,jj):
            if cost[i,j]<=.15:errors.append(float(a[i]-b[j]));matches+=1
    return {'TP':matches,'FP':len(pred)-matches,'FN':len(truth)-matches,'timing_errors':errors}

def analyze():
    assert read(H/'protocol.json')['code_sha256']==sha(__file__);audit=read(H/'input-audit.json');rows=[]
    for item in audit['records']:
        assert sha(item['array'])==item['sha256'];d=np.load(item['array']);record={k:v for k,v in item.items() if k not in ['array','sha256']};record.update(reference={},prediction={},reason=None)
        if item['reference_status']!='ok':record['reason']=item['reference_status'];rows.append(record);continue
        lm=d['lm'];xy=lm[:,:,:2]*[800,600];hip=xy[:,[23,24]].mean(1);scale=float(d['height'])/float(np.median(np.ptp(xy[:,:,1],axis=1)));speed=float(np.linalg.norm(slope(hip,30))*scale)
        pelvis=d['pelvis120'][:,:2];heels=d['heels120'][:,:,:2];_,_,vv=np.linalg.svd(pelvis-pelvis.mean(0),full_matrices=False);axis=vv[0]
        if np.dot(axis,pelvis[-1]-pelvis[0])<0:axis=-axis
        forward=pelvis@axis;heel=heels@axis;truth_events=events(heel,forward,120);ref,truth_steps=measures(heel,truth_events,120);ref['speed']=float(np.linalg.norm(slope(pelvis,120)))
        direction=1 if hip[-1,0]>=hip[0,0] else -1;predheel=xy[:,[29,30],0]*scale*direction;predhip=hip[:,0]*scale*direction;pred_events=events(predheel,predhip,30);estimate,pred_steps=measures(predheel,pred_events,30);estimate['speed']=speed
        speedgood=np.isfinite(lm).all() and (d['pose_counts']>0).all() and np.median(lm[:,[23,24],3])>=.5
        footgood=speedgood and all(np.median(lm[:,index,3])>=.5 for index in [29,30]) and abs(hip[-1,0]-hip[0,0])*scale>=.1
        if not speedgood:estimate={key:None for key in estimate};record['reason']='pelvis_pose_quality'
        elif not footgood:
            for key in estimate:
                if key!='speed':estimate[key]=None
            record['reason']='heel_visibility_or_nonlateral_progress'
        # Invalid lengths remain a failed measurement, not an invented normal value.
        for key in ['step_length','stride_length']:
            if estimate[key] is not None and estimate[key]<=0:estimate[key]=None;record['reason']=record['reason'] or 'nonpositive_forward_length'
        record.update(reference=ref,prediction=estimate,raw_speed=speed,scale_metres_per_pixel=scale,reference_events=truth_events,predicted_events=pred_events,reference_steps=truth_steps,predicted_steps=pred_steps,event_match=match_events(pred_events,truth_events) if footgood else {'TP':0,'FP':0,'FN':len(truth_events),'timing_errors':[]},speed_quality=bool(speedgood),foot_quality=bool(footgood));rows.append(record)
    save(H/'measurements.json',{'records':rows,'input_audit_sha256':sha(H/'input-audit.json'),'code_sha256':sha(__file__),'marker_HS_gold_status':'kinematic_reference_not_forceplate_or_expert_contact'})
    print('ANALYZED',len(rows),flush=True)

if __name__=='__main__':{'freeze':freeze,'prepare':prepare,'analyze':analyze}[sys.argv[1]]()
