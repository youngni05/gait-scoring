"""Actual SPPB labels + fixed 3.0s optical-marker walking windows; no RGB claim."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
import openpyxl
from sklearn.model_selection import KFold
import csv
import re
import json
import hashlib

RAW=HERE.parent/'20261030-sppb-timeseries/raw'
LABELS=ROOT/'outputs/gait-public-access-20261015/OlderAdults44/01_demography_sppb_mmse.xlsx'
MARKERS=['LASI','RASI','LPSI','RPSI','LKNE','RKNE','LANK','RANK','LHEE','RHEE','LTOE','RTOE']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value): (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def read_walk(path):
    rows=list(csv.reader(path.open(encoding='utf-8-sig'),delimiter='\t'))
    start=next(i for i,r in enumerate(rows) if r==['Trajectories'])
    fps=float(rows[start+1][0]);assert fps==100
    names=rows[start+2];columns=[]
    sid=int(re.fullmatch(r'Subject(\d+)_\d+\.csv',path.name).group(1))
    for name in MARKERS:
        choices=[j for j,v in enumerate(names) if v==f'Subject{sid:02d}:{name}']
        assert len(choices)==1,(path,name,choices)
        j=choices[0];assert rows[start+3][j:j+3]==['X','Y','Z'] and rows[start+4][j:j+3]==['mm']*3
        columns.extend([j,j+1,j+2])
    frames=[];points=[]
    for row in rows[start+5:]:
        if not row or not row[0].strip():break
        frames.append(int(row[0]))
        points.append([float(row[j]) if j<len(row) and row[j].strip() else np.nan for j in columns])
    return np.array(frames),np.array(points).reshape(-1,len(MARKERS),3),fps

def window(frames,points,fps):
    valid=np.isfinite(points).all((1,2))
    runs=[];first=None
    for i,ok in enumerate(valid):
        consecutive=i>0 and frames[i]==frames[i-1]+1
        if first is not None and (not ok or not consecutive):runs.append((first,i-1));first=None
        if ok and first is None:first=i
    if first is not None:runs.append((first,len(frames)-1))
    info={'source_frames':len(frames),'valid_fraction':float(valid.mean()),'fps':fps}
    if not runs:return None,{**info,'reason':'no_complete_marker_span'}
    a,b=max(runs,key=lambda q:frames[q[1]]-frames[q[0]])
    info.update({'run_start_row0':a,'run_end_row0':b,'run_start_frame':int(frames[a]),'run_end_frame':int(frames[b]),'valid_span_seconds':float((frames[b]-frames[a])/fps)})
    if (frames[b]-frames[a])/fps+1e-9<3.0:return None,{**info,'reason':'continuous_span_shorter_than3.0s'}
    # Compute all normalization and velocity from the selected 3.0s only.
    end=a+301
    p=points[a:end]/1000
    root=p[:,:4].mean(1)
    travel=root[-1]-root[0];travel[2]=0
    distance=float(np.linalg.norm(travel));assert distance>.1
    forward=travel/distance
    vertical=np.array([0.,0.,1.]);lateral=np.cross(forward,vertical)
    basis=np.stack([lateral,forward,vertical],axis=1)
    scale=float(np.median(np.linalg.norm(p[:,[6,7]]-root[:,None],axis=-1)))
    assert .3<scale<1.5,scale
    centered=(p-root[:,None])@basis/scale
    velocity=np.gradient(root,1/fps,axis=0)@basis
    values=np.concatenate([centered.reshape(len(p),-1),velocity],axis=1)
    times=(frames[a:end]-frames[a])/fps
    sample_times=np.arange(91)/30
    assert sample_times[-1]<=times[-1]+1e-9 and (np.diff(frames[a:end])==1).all()
    x=np.stack([np.interp(sample_times,times,v) for v in values.T]).astype(np.float32)
    return x,{**info,'leg_scale_m':scale,'travel_direction':forward.tolist(),'start_time_seconds':float(frames[a]/fps),'end_time_seconds':float(frames[a]/fps+3.0)}

def run():
    protocol={
        'declared_before_fitting':True,'question':'Can short walking marker trajectories predict actual clinical SPPB total for unseen people?',
        'source':'CaicedoRodriguez etal2020,10.17632/xgw6bg3g8h.1,CCBY4.0; optical Vicon100Hz, not markerless RGB',
        'labels':str(LABELS),'labels_sha256':sha(LABELS),'target':'column8 SPPB total as supplied; integral0..12, no sum of invented component ratings',
        'markers':MARKERS,'channels':'12markers3D root-relative/leg-scale coordinates(36) + pelvis velocity in meters/second(3)',
        'normalization':'within selected3.0s ONLY: pelvis center mean of4pelvicmarkers eachframe; median pelvis-to-ankle distance; horizontal axes from travel direction and verticalZ; no population normalization before split',
        'selection':'one3.0s window per CSV, earliest3.0s of longest complete contiguous 12marker span; tiesearliest; all selected before using labels; require100Hz consecutiveframeIDs and finite all36coordinates; no gaps,padding,extrapolation',
        'resampling':'linear100Hz to30Hz at91timepoints over3.0s within observed adjacent frames; velocity finite differences of raw pelvis positions',
        'analysis_unit':'person; average available trial predictions; trial labels repeat one person SPPB, not220 independent clinical labels',
        'split':'label-independent person5fold seed20261030; nested training-person3fold seed20261031; all trials sameperson grouped; no data augmentation/pretraining',
        'weight':'equal total training weight perperson; primary MAE over averaged perperson predictions',
        'cpu':'Mean,Median; Ridgealpha[1,100],RF128trees leaf[2,5]; inputchannel mean/std/mean absolute temporal difference only, standardize using trainingpeople',
        'gpu':'CNN/TCN3layers width24 kernel5,dilation[1,1,1]vs[1,2,4],dropout.2,AdamW lr.001 decay.001; trainperson-only standardizedweightedMSE; max80epochs patience12; median bestepoch of3innerfolds refit; oneseed20261030',
        'pilot':'actualCUDA arithmetic, realinner-training3epochs permodel; finite loss+prediction; checkpoint reload equality; time+peak allocated/reservedGPUmemory; nooutertesttraining',
        'limits':'44people max; SPPB5..12 in source, no0..4; no clinicalvalidation or video-based inference claim; exploratory developmentdataset',
        'models_rationale':'Ma2020 gait summary to small regression; Romeo2023 normalized skeletal temporalfeatures/convolution method; lightweightTCN inspired byTCAI-FallNet but neither paper proves superiority for this target/input',
        'no_clipping_or_rounding':True,'constant_scalar':'framecount91 for compatibility, same across all records; original duration excluded'}
    save('protocol.json',protocol)
    wb=openpyxl.load_workbook(LABELS,data_only=True)
    labels={}
    for row in wb.active.values:
        m=re.fullmatch(r'Subject\s*(\d+)',str(row[0] or '').strip())
        if not m:continue
        sid=int(m.group(1));assert sid not in labels
        score=row[7];assert isinstance(score,(int,float)) and int(score)==score and 0<=score<=12
        labels[sid]=int(score)
    wb.close();assert len(labels)==44
    meta=json.loads((ROOT/'outputs/gait-public-access-20261015/older44-metadata.json').read_text(encoding='utf-8-sig'))
    sources={f['filename']:f for f in meta['files'] if f['filename'].endswith('.csv')}
    assert len(sources)==220
    xs=[];manifest=[];rejected=[];raw_hashes={}
    for name,f in sorted(sources.items()):
        path=RAW/name;digest=sha(path);assert digest==f['content_details']['sha256_hash'],name
        raw_hashes[name]=digest
        sid=int(re.fullmatch(r'Subject(\d+)_\d+\.csv',name).group(1));assert sid in labels
        frames,p,fps=read_walk(path);x,info=window(frames,p,fps)
        info.update({'file':name,'id':str(sid),'target':labels[sid],'sha256':digest})
        if x is None:rejected.append(info);continue
        manifest.append({'sample':len(xs),**info});xs.append(x)
    df=pd.DataFrame(manifest);people=np.array(sorted(df.id.unique(),key=int));assert len(people)>12
    folds={}
    for fold,(_,te) in enumerate(KFold(5,shuffle=True,random_state=20261030).split(people)):folds.update({p:fold for p in people[te]})
    df['fold']=df.id.map(folds);df.to_csv(HERE/'manifest.csv',index=False)
    np.savez_compressed(HERE/'input.npz',x=np.array(xs),y=df.target.to_numpy(dtype=np.float32),ids=df.id.to_numpy(dtype=str),folds=df.fold.to_numpy(),frames=np.full(len(xs),91,dtype=np.float32))
    assert np.array(xs).shape==(len(df),39,91) and np.isfinite(xs).all()
    save('rejected.json',rejected)
    save('input-audit.json',{'input_sha256':sha(HERE/'input.npz'),'protocol_sha256':sha(HERE/'protocol.json'),'manifest_sha256':sha(HERE/'manifest.csv'),
        'source_files':len(sources),'source_people':len(labels),'eligible_records':len(df),'eligible_people':len(people),'rejected_records':len(rejected),'missing_people':sorted(set(map(str,labels))-set(people)),
        'trials_per_person':{str(k):int(v) for k,v in df.groupby('id').size().value_counts().sort_index().items()},
        'person_score_counts':{str(k):int(v) for k,v in df.drop_duplicates('id').target.value_counts().sort_index().items()},'raw_files_sha256':raw_hashes})
    old=HERE.parent/'20261030-tug-toronto-timeseries'
    for name in ['common.py','cpu.py','gpu.py','submit.slurm']:
        code=(old/name).read_text(encoding='utf8')
        if name=='gpu.py':code=code.replace('layers=[];cin=51','layers=[];cin=39').replace('TORONTO_TUG','OLDER44_SPPB')
        if name=='submit.slurm':code=code.replace('tug-toronto-ts','older44-sppb-ts')
        (HERE/name).write_text(code,encoding='utf8',newline='\n')
    print({k:v for k,v in json.loads((HERE/'input-audit.json').read_text()).items() if k!='raw_files_sha256'})

if __name__=='__main__':run()
