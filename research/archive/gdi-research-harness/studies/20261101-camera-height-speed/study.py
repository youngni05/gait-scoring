"""Deterministic measurement diagnostic; no clinical score model or new pose inference."""
from pathlib import Path
import sys, json, hashlib, datetime, argparse, io, tarfile, time, os
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
from scipy.io import loadmat
SOURCE=Path('D:/gait-research-data/MoVi')
PRIOR=SOURCE/'experiments/walking-expansion'
SELECTION=HERE.parent/'20261031-care-pd-rf-tcn/video-stability/straight-followup'
STORE=Path('D:/gait-research-data/D_P/camera-height-speed-20261101')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def clips(mode):
    rows=json.loads((SELECTION/'clips.json').read_text()); people=sorted(set(r['subject'] for r in rows))
    assert len(rows)==56 and len(people)==14
    return [r for r in rows if mode=='main' or r['subject'] in people[:2]]
def slope(x):
    t=np.arange(len(x))/30.;t=t-t.mean()
    return (t[:,None]*(x-x.mean(0))).sum(0)/(t*t).sum()
def measure(lm,reference,height,intrinsic):
    xy=lm[:,:,:2]*[800.,600.]; pelvis=xy[:,[23,24]].mean(1)
    span=np.ptp(xy[:,:,1],axis=1); assert np.isfinite(span).all() and (span>0).all()
    scale=height/np.median(span);dt=(len(xy)-1)/30.
    # Simple height-scaled image displacement: camera-plane proxy, not depth recovery.
    plane_net=np.linalg.norm(pelvis[-1]-pelvis[0])*scale/dt
    plane_ols=np.linalg.norm(slope(pelvis))*scale
    # Extra-intrinsic diagnostic, NOT an estimator requiring height alone.
    fx,fy=intrinsic[0,0],intrinsic[1,1];cx,cy=intrinsic[2,:2]
    depth=fy*height/span
    xyz=np.c_[(pelvis[:,0]-cx)*depth/fx,(pelvis[:,1]-cy)*depth/fy,depth]
    pinhole_ols=np.linalg.norm(slope(xyz))
    center=reference[:,[6,7]].mean(1)/1000.
    truth_net=np.linalg.norm(center[-1,:2]-center[0,:2])/dt
    truth_ols=np.linalg.norm(slope(center[:,:2]))
    return dict(plane_net=float(plane_net),plane_ols=float(plane_ols),pinhole_ols=float(pinhole_ols),truth_net=float(truth_net),truth_ols=float(truth_ols),span_cv=float(span.std()/span.mean()),min_visibility=float(lm[:,[0,23,24,27,28],3].min())),dict(lm=lm,reference=reference,height=np.array(height),intrinsic=intrinsic,pelvis_xy=pelvis,span=span,estimated_xyz=xyz)
def work(row):
    sid=row['subject'];cam=row['camera'];key=row['key'];dest=STORE/f'{key}.npz';done=STORE/f'{key}.json'
    if done.exists():
        q=json.loads(done.read_text());assert q['code_sha256']==sha(__file__);return q
    pose=PRIOR/f'subject-{sid:02d}'/'heavy'/f'{cam}_800.npz';refpath=pose.with_name(f'{cam}_800_comparison.npz');meta=SOURCE/'reference'/f'F_v3d_Subject_{sid}.mat'
    assert sha(pose)==row['pose_sha256']
    audit=json.loads((SELECTION/'availability.json').read_text());assert sha(refpath)==audit['source_sha256'][str(refpath)]
    z=np.load(pose);r=np.load(refpath);frames=np.arange(row['frames_one_based'][0],row['frames_one_based'][1]+1);assert len(frames)==60
    a=np.searchsorted(z['video_frame_one_based'],frames);b=np.searchsorted(r['frames'],frames)
    np.testing.assert_array_equal(z['video_frame_one_based'][a],frames);np.testing.assert_array_equal(r['frames'][b],frames)
    assert (z['pose_counts'][a]>0).all() and not r['reference_gap'][b].any()
    height=float(loadmat(meta,simplify_cells=True)[f'Subject_{sid}_F']['subject']['height'])/100.
    assert 1<height<2.3
    with tarfile.open(SOURCE/'raw/Camera Parameters.tar') as t:
        intrinsic=np.load(io.BytesIO(t.extractfile(f'Calib/cameraParams_{cam}.npz').read()))['IntrinsicMatrix']
    values,arrays=measure(z['landmarks'][a],r['reference_3d'][b],height,intrinsic)
    np.savez_compressed(dest,frames=frames,**arrays)
    result=dict(key=key,subject=sid,camera=cam,condition=row['condition'],**values,height=height,frames=60,first_frame=int(frames[0]),last_frame=int(frames[-1]),sources={str(p):sha(p) for p in [pose,refpath,meta]},array_path=str(dest),array_sha256=sha(dest),code_sha256=sha(__file__))
    save(done,result);return result
def prepare():
    protocol=dict(hypothesis='After limited sensor classifier gains, test whether height-scaled image translation preserves speed across fixed camera views, before claiming phone score accuracy.',
      user='Fixed phone,no floor marks,one height entry allowed.',
      inputs='Existing RGB-only MediaPipe Heavy image landmarks, user height from provider metadata. Reference 3D/geometry only for fixed selection and evaluation. No reference coordinates in estimator.',
      methods='PlaneNet: first-last image pelvis displacement*height/median33-landmark vertical span. PlaneOLS: linear time slope with same scale. PinholeOLS: depth from fy*height/span then 3D slope; uses actual extra camera intrinsics and is diagnostic only. No distortion/extrinsic correction. The landmark span misses top-of-head and can change with pose; it is a deliberately simple candidate, not a validated stature measurement.',
      truth='Provider reference hip-midpoint net/OLS horizontal translation m/s. Net method compared with net truth,OLS methods with OLS truth. Pinhole output includes vertical drift. This is not average stride speed as defined by Mobilise-D.',
      people='Same14 previously selected MoVi people,pairedPG1/PG2,straight2sec primary,turn2sec control,56clips3360frames. No assumed healthy score or clinical labels.',
      selection='Reuse existing geometry-frozen clips.json; no windows/thresholds chosen by this result. No new training or calibration; deterministic,so no new seed/fold count. Existing Mobilise-D5seeds/person folds remain untouched.',
      success='Before viewing these results: simple height-only PlaneOLS straight MAE<=0.10m/s and paired-camera absolute difference<=0.10m/s are exploratory engineering targets,not clinical standards. Report all methods/errors/coverage regardless of pass.',
      evaluation='Person-equal MAE,bias,median/p90 error,paired absolute/signed view difference,per-view values,height +/-2cm analytic sensitivity.14independent people,not56; no score accuracy or domain validation.',
      resources='CPU1,existing cached detections,no GPU/download,new NPZ<5MB D,small tables/protocol C,estimate2-5minutes.',
      expected=dict(pilot_clips=8,pilot_people=2,main_clips=56,main_people=14,frames=3360,model_fits=0,metric_rows=12,paired_rows=6),
      sources={str(p):sha(p) for p in [SELECTION/'clips.json',SELECTION/'availability.json',SOURCE/'raw/Camera Parameters.tar']},code_sha256=sha(__file__),storage=dict(small=str(HERE),large=str(STORE)))
    save(HERE/'protocol.json',protocol)
def run(mode):
    start=time.time();STORE.mkdir(parents=True,exist_ok=True);rows=[]
    for i,c in enumerate(clips(mode),1):rows.append(work(c));print('MEASURED',i,c['key'],flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['sources']} for r in rows]).to_csv(HERE/f'{mode}-clips.csv',index=False)
    save(HERE/f'{mode}-completion.json',dict(clips=len(rows),people=len(set(r['subject'] for r in rows)),frames=len(rows)*60,model_fits=0,code_sha256=sha(__file__),seconds=time.time()-start))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode)
