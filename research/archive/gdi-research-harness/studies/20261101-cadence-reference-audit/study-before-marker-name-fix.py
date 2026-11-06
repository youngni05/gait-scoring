"""Audit spectral cadence proxies using separately derived optical heel events."""
from pathlib import Path
import importlib.util,json,argparse,time,datetime
HERE=Path(__file__).resolve().parent;REFERENCE=HERE.parent/'20261101-camera-cadence'
spec=importlib.util.spec_from_file_location('cadence_reference',REFERENCE/'study.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
np,pd,sha,save=ref.np,ref.pd,ref.sha,ref.save
from scipy.io import loadmat
from scipy.signal import savgol_filter,find_peaks
STORE=Path('D:/gait-research-data/D_P/cadence-reference-audit-20261101')
NAMES=['LFWT','RFWT','LBWT','RBWT','LHEE','RHEE']
def tasks(mode):
    ids=json.loads((REFERENCE/'availability.json').read_text())['selected_people'];return ids if mode=='main' else ids[:2]
def derive(points,gaps,indices,start,end):
    front=points[:,[0,1],:2].mean(1);back=points[:,[2,3],:2].mean(1);axis=front-back;norm=np.linalg.norm(axis,axis=1);forward=np.divide(axis,norm[:,None],out=np.zeros_like(axis),where=norm[:,None]>1e-6)
    valid=~gaps.any(1)&np.isfinite(points).all((1,2))&(norm>1e-6);signals=np.stack([np.sum((points[:,k,:2]-back)*forward,axis=1) for k in [4,5]],1);smooth=np.full_like(signals,np.nan);events=[]
    edges=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(int)));runs=list(zip(edges[::2],edges[1::2]))
    for a,b in runs:
        if b-a<9:continue
        for side in range(2):
            signal=savgol_filter(signals[a:b,side],9,2,mode='interp');smooth[a:b,side]=signal;peak,_=find_peaks(signal,height=float(signal.mean()),distance=48)
            events.extend((int(indices[a+k]),'L' if side==0 else 'R') for k in peak if start<=indices[a+k]<=end)
    events=sorted(events,key=lambda x:(x[0],x[1]));delta=np.diff([e[0] for e in events])/120.;alt=float(np.mean([a[1]!=b[1] for a,b in zip(events[:-1],events[1:])])) if len(events)>1 else 0.;window=(indices>=start)&(indices<=end);coverage=float(valid[window].mean())
    cadence=float(60/np.median(delta)) if len(delta) and (delta>0).all() else None
    quality=bool(valid[window].all() and len(events)>=3 and alt>=.8 and (delta>=.2).all() and (delta<=1.5).all())
    if valid[window].all():
        yaw=np.unwrap(np.arctan2(forward[window,1],forward[window,0]));span=float(np.rad2deg(np.percentile(yaw,95)-np.percentile(yaw,5)));trajectory=back[window];distance=float(np.linalg.norm(trajectory[-1]-trajectory[0]));path=float(np.linalg.norm(np.diff(trajectory,axis=0),axis=1).sum());efficiency=distance/max(path,1e-12);speed=distance/((end-start)/120.);straight=bool(span<=30 and efficiency>=.95 and speed>=.2)
    else:span=efficiency=speed=None;straight=False
    return dict(cadence=cadence,quality=quality,events=[[i,s] for i,s in events],event_count=len(events),alternation=alt,marker_valid_fraction=coverage,heading_p90_span=span,path_efficiency=efficiency,net_speed=speed,straight=straight),dict(signals=signals,smoothed=smooth,valid=valid,forward=forward)
def prepare():
    files=[REFERENCE/n for n in ['study.py','protocol.json','availability.json','main-estimates.csv','main-verification.json','README.md']]
    save(HERE/'protocol.json',dict(code_sha256=sha(__file__),sources={str(p):sha(p) for p in files},
      hypothesis='Current4sec spectral reference may select a slow envelope or half-period. Audit it with opticalheel/pelvis event candidates derived without spectral fitting,not retuneRGBperiods.',
      method_source='https://pmc.ncbi.nlm.nih.gov/articles/PMC2384115/ (Zeni etal.2008).Coordinate principle: maxima of heel AP displacement relative to sacrum. Adaptation here: posteriorwaist midpoint instead of sacrum and instantaneous front-minus-back pelvic direction. This is not exact paper reproduction or new forceplate validation.',
      algorithm='Nativeoptical120Hz,physicalLFWT/RFWT/LBWT/RBWT/LHEE/RHEE;meters. Same4secvideo window,0.5seccontextclippedtoproviderwalking. Validruns only,no gap interpolation. SavitzkyGolay9framesdegree2;positive peaks above runmean;48sample samefoot refractory. MergeL/Rwithinwindow;cadence60/medianintereventseconds. Eventquality:allmarkersvalid,>=3events,alternation>=.8,allintervals.2..1.5sec. No parameter selection by period agreement.',
      mapping='Same27people4secwindows. Opticalindex=(video_onebased-1)*4;verifyflags120 andsame12exportedreferencejoints atallvideo timestamps. No timeoffsetsearch. Independent savedmarkers are never added toRGB estimator inputs.',
      evaluation='All27people retained in availability. Compare old4secReference3D/RGB2D/RGB3D toeventproxy,reportquality/sample counts andboth-available discrepancy. Separate heading<=30deg/path>=.95/speed>=.2 geometry subgroup,not select by agreement. Flaghalfproxy ifabs(spectral-event/2)<=5andabs(spectral-event)>5. Not true cadence MAE.',
      success='Reference-consistency diagnostic only: eventquality>=80%of27andoldreference/eventproxy discrepancy<=5steps/min among bothquality,all andstraightgroupsseparate. Failure means references disagree or lackcoverage,not which one isgroundtruth. No clinical standard.',
      limitations='No independent foot-contact groundtruth acquired. Markerheuristic is anotherproxy;turns,backwardwalk,gaps,abnormalfeet can invalidate adaptation. SameMoVi27people,notnewclinicalpeople;noSPPB/TUGlabels/no scoretraining/newseeds. Existingperson/bout/5seedscoreexperimentsunchanged.',
      external_access={'ForcePose':'Officialv1.1HTTPScertificateexpired viaWindowscurl;didnotdisablecertificateverification.Downloadnotacquired;codeMITnotdata-licenseproof.','BioCV':'Separateaccessrequestcontradicts immediate-noapplication condition,notacquired.','OpenCap_validation':'SimTKdownloadpage403,rawvideos/forcesnotacquired.'},
      expected=dict(main_tasks=27,people=27,pilot_tasks=2,comparison_rows=5,paired_rows=2,detail_rows=135,new_model_fits=0),
      verification='Source/code/marker/frameSHA,mappingwithoutoffset,independentAPsignals/smoothing+localmaxgreedyselection,exacteventindices/sides,independentcadenceandmetrics. Signals/cadence1e-10,events exact. Pilot2peoplethenmain.',
      resource='CPU1,estimate3-8min,noGPU/download/inference,newselectedarrays<10MBD,smallcode/tablesC.',storage=dict(small=str(HERE),large=str(STORE))))
def work(sid):
    dest=STORE/f's{sid:02d}.json'
    if dest.exists():
        q=json.loads(dest.read_text());assert q['code_sha256']==sha(__file__);return q
    source=ref.STORE/f's{sid:02d}-4sec.json';q=json.loads(source.read_text());assert sha(q['array_path'])==q['array_sha256'];z=np.load(q['array_path']);frames=z['PG1_frames'];path=Path('D:/gait-research-data/MoVi/reference')/f'F_v3d_Subject_{sid}.mat';m=loadmat(path,simplify_cells=True)[f'Subject_{sid}_F']['move'];names=list(m['markerName']);assert np.array_equal(m['flags120'].astype(int),(m['flags30'].astype(int)-1)*4+1)
    ix=(frames-1)*4;joints=['LSJC','RSJC','LEJC','REJC','LWJC','RWJC','LHIP','RHIP','LKJC','RKJC','LAJC','RAJC'];raw=m['markerLocation'][ix][:,[names.index(n) for n in joints]];np.testing.assert_array_equal(raw,z['PG1_reference'])
    start,end=int(ix[0]),int(ix[-1]);walk=list(m['motions_list']).index('walking');lo,hi=m['flags120'][walk].astype(int)-1;indices=np.arange(max(lo,start-60),min(hi,end+60)+1);cols=[names.index(n) for n in NAMES];points=m['markerLocation'][indices][:,cols]/1000.;gaps=m['markerGaps'][indices][:,cols].astype(bool)
    result,derived=derive(points,gaps,indices,start,end);arrays=STORE/f's{sid:02d}.npz';np.savez_compressed(arrays,points=points,gaps=gaps,indices=indices,frames=frames,source_columns=cols,**derived)
    out=dict(subject=sid,start=start,end=end,**result,sources={str(p):sha(p) for p in [path,source,Path(q['array_path'])]},array_path=str(arrays),array_sha256=sha(arrays),code_sha256=sha(__file__));save(dest,out);return out
def run(mode):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for q,h in protocol['sources'].items():assert sha(q)==h
    STORE.mkdir(parents=True,exist_ok=True);started=time.time();rows=[]
    for i,sid in enumerate(tasks(mode),1):
        q=work(sid);rows.append({k:v for k,v in q.items() if k not in ['sources','events']});print('HEEL_PROXY',i,sid,flush=True)
    pd.DataFrame(rows).to_csv(HERE/f'{mode}-events.csv',index=False);save(HERE/f'{mode}-completion.json',dict(tasks=len(rows),people=len(rows),new_model_fits=0,code_sha256=sha(__file__),seconds=time.time()-started))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    else:run(a.mode)
