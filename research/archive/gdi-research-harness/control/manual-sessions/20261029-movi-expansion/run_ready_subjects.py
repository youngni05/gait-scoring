"""Repeat the frozen walking baseline on available MoVi subset files; no movies."""
import argparse
import hashlib
import io
import json
import msvcrt
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / '20261028-movi-totalcapture-acquisition'
POSE = HERE.parent / '20261028-camera-step-pilot'
sys.path.insert(0, str(POSE / 'deps'))
import cv2
import numpy as np
import mediapipe as mp
from scipy.io import loadmat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path('D:/gait-research-data/MoVi')
OUT = ROOT / 'experiments' / 'walking-expansion'
MAPPING = {'LSJC':11,'RSJC':12,'LEJC':13,'REJC':14,'LWJC':15,'RWJC':16,
           'LHIP':23,'RHIP':24,'LKJC':25,'RKJC':26,'LAJC':27,'RAJC':28}
NAMES, MP = list(MAPPING), list(MAPPING.values())
VERIFIED_ARCHIVES = set()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def stats(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return {'n':int(x.size), 'mean':float(x.mean()) if x.size else None,
            'median':float(np.median(x)) if x.size else None,
            'p90':float(np.percentile(x,90)) if x.size else None}


def angles(x):
    result = []
    for side in 'LR':
        p = x[:,[NAMES.index(side+n) for n in ['HIP','KJC','AJC']]]
        a,b = p[:,0]-p[:,1], p[:,2]-p[:,1]
        denom = np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        cos = np.divide(np.sum(a*b,axis=-1),denom,out=np.full_like(denom,np.nan),where=denom>1e-10)
        result.append(180-np.degrees(np.arccos(np.clip(cos,-1,1))))
    return np.stack(result,axis=1)


def project(points, intr, ext):
    r,t = ext['rotationMatrix'],ext['translationVector']
    camera = points @ r + t
    assert np.all(camera[...,2]>0)
    xy = camera[...,:2]/camera[...,2,None]
    r2 = np.sum(xy*xy,axis=-1); k1,k2 = intr['RadialDistortion']
    xy *= (1+k1*r2+k2*r2*r2)[...,None]
    uv = (np.concatenate([xy,np.ones(xy.shape[:-1]+(1,))],axis=-1) @ intr['IntrinsicMatrix'])[...,:2]-1
    other = cv2.projectPoints(points.reshape(-1,3),cv2.Rodrigues(r.T)[0],t,intr['IntrinsicMatrix'].T,np.array([k1,k2,0.,0.,0.]))[0]
    assert np.max(np.abs(uv.reshape(-1,2)-(other.reshape(-1,2)-1)))<1e-6
    return uv


def reference(subject, manifest):
    name = f'F_v3d_Subject_{subject}.mat'
    archive_name = 'F_Subjects_1_45.tar' if subject<=45 else 'F_Subjects_46_90.tar'
    archive = ROOT/'raw'/archive_name
    complete = archive.exists()
    if not complete: archive = archive.with_name(archive.name+'.part')
    if not archive.exists(): return None
    if complete and archive_name not in VERIFIED_ARCHIVES:
        assert digest(archive,'md5')==manifest[archive_name]['md5']
        VERIFIED_ARCHIVES.add(archive_name)
    # Only fully received members are read. Never extract arbitrary archive paths.
    try:
        with tarfile.open(archive,'r:') as tar:
            for member in tar:
                if Path(member.name).name != name: continue
                if not member.isfile() or member.offset_data+member.size>archive.stat().st_size: return None
                data = tar.extractfile(member).read()
                assert len(data)==member.size
                target = ROOT/'reference'/name
                if target.exists(): assert target.read_bytes()==data
                else: target.write_bytes(data)
                provenance = {'file':str(target),'sha256':hashlib.sha256(data).hexdigest(),
                              'archive':archive_name,'member':member.name,'offset':member.offset_data,'bytes':member.size,
                              'archive_md5_verified':complete,
                              'status':'verified_archive_member' if complete else 'provisional_complete_member_of_partial_archive'}
                dump(target.with_suffix('.provenance.json'),provenance)
                return loadmat(io.BytesIO(data),simplify_cells=True)[f'Subject_{subject}_F']['move'],provenance
    except tarfile.ReadError:
        return None
    return None


def measure(landmarks, world, refxy, ref3, gaps, size):
    xy = landmarks[:,MP,:2]*size
    xyz = world[:,MP,:3]
    distance = np.linalg.norm(xy-refxy,axis=-1)
    pred2,pred3,truth2,truth3 = angles(xy),angles(xyz),angles(refxy),angles(ref3)
    comparisons = {'pred2d_vs_ref2d':np.abs(pred2-truth2),
                   'ref2d_vs_ref3d_projection_difference':np.abs(truth2-truth3),
                   'pred2d_vs_ref3d':np.abs(pred2-truth3),'pred3d_vs_ref3d':np.abs(pred3-truth3)}
    vis = landmarks[:,MP,3]
    knee_vis = np.stack([np.min(vis[:,[NAMES.index(s+n) for n in ['HIP','KJC','AJC']]],axis=1) for s in 'LR'],axis=1)
    row = {'position_discrepancy_px':stats(distance),'position_visibility_ge_05':stats(distance[vis>=.5]),
           'reference_valid_joint_samples':int(np.sum(~gaps)),
           'joint_position_discrepancy_px':{n:stats(distance[:,j]) for j,n in enumerate(NAMES)},
           'knee_abs_angle_difference_deg':{k:stats(v) for k,v in comparisons.items()},
           'knee_visibility_ge_05':{k:stats(v[knee_vis>=.5]) for k,v in comparisons.items()}}
    return row, {'reference_xy':refxy,'predicted_xy':xy,'discrepancy_px':distance,'visibility':vis,
                 'reference_gap':gaps,'reference_3d_bend_deg':truth3,'reference_2d_bend_deg':truth2,
                 'predicted_2d_bend_deg':pred2,'predicted_3d_bend_deg':pred3}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--subjects',type=int,nargs='+',default=[1,13])
    parser.add_argument('--manifest',type=Path,default=ROOT/'control/subset-manifest.json');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_data=json.loads(args.manifest.read_text(encoding='utf-8'))
    manifest={e['name']:e for e in manifest_data['files']}
    assert set(args.subjects)<=set(manifest_data['selected_subjects'])
    calibration=ROOT/'raw'/'Camera Parameters.tar'
    assert digest(calibration,'md5')==manifest[calibration.name]['md5']
    models={'full':POSE/'pose_landmarker_full.task','heavy':PILOT/'heavy/pose_landmarker_heavy.task'}
    plan={'subjects':args.subjects,'selected_subjects':manifest_data['selected_subjects'],
          'manifest':str(args.manifest),'manifest_sha256':digest(args.manifest),
          'conditions':{'full':[800,400,200],'heavy':[800]},
          'interval':'Provider walking flags30, inclusive; each participant has own interval',
          'synchronization':'mocap=(video-1)*4+1; no shift search',
          'warmup_frames':30,'options':'VIDEO CPU; 3 poses; all thresholds .5; largest predicted bbox',
          'joint_mapping':MAPPING,'angle':'180 minus hip-knee-ankle included angle; not clinical JCS flexion',
          'model_sha256':{k:digest(v) for k,v in models.items()},'no_training':True,
          'turn_segments':'Not transferred from subject1; no new turn annotation or accuracy thresholds',
          'outputs':'Arrays, metrics and sparse JPG only; no GIF or MP4',
          'reference_integrity':'Complete members of partial archives are provisional until archive checksum succeeds'}
    dump(OUT/'plan.json',plan)
    for subject in args.subjects:
        if not any((ROOT/'raw'/f'F_{cam}_Subject_{subject}_L.avi').exists() for cam in ['PG1','PG2']):
            print('WAIT_VIDEO',subject,flush=True);continue
        available=reference(subject,manifest)
        if available is None: print('WAIT_REFERENCE',subject,flush=True); continue
        move,provenance=available
        motion=list(move['motions_list']).index('walking'); start,end=map(int,move['flags30'][motion])
        assert start>30 and np.array_equal(move['flags120'].astype(int),(move['flags30'].astype(int)-1)*4+1)
        frames=np.arange(start,end+1); mocap=(frames-1)*4
        ids=move['virtualMarkers'].astype(int)-1
        names=[str(move['markerName'][i]) for i in ids]; refids=[names.index(n) for n in NAMES]
        assert np.array_equal(move['virtualMarkerLocation'],move['markerLocation'][:,ids])
        raw3=move['virtualMarkerLocation'][mocap][:,refids].copy()
        gaps=move['markerGaps'][mocap][:,ids[refids]].astype(bool)
        ref3=raw3.copy(); ref3[gaps]=np.nan
        for cam in ['PG1','PG2']:
            video=ROOT/'raw'/f'F_{cam}_Subject_{subject}_L.avi'
            if not video.exists(): print('WAIT_VIDEO',subject,cam,flush=True); continue
            assert video.stat().st_size==manifest[video.name]['bytes'] and digest(video,'md5')==manifest[video.name]['md5']
            with tarfile.open(calibration) as tar:
                def npz(name):
                    with np.load(io.BytesIO(tar.extractfile('Calib/'+name+'.npz').read())) as z:return dict(z)
                refxy=project(raw3,npz('cameraParams_'+cam),npz('Extrinsics_'+cam));refxy[gaps]=np.nan
            cap=cv2.VideoCapture(str(video));assert cap.get(cv2.CAP_PROP_FPS)==30
            cap.set(cv2.CAP_PROP_POS_FRAMES,start-31);images=[]
            for _ in range(len(frames)+30):
                ok,image=cap.read();assert ok;images.append(image)
            cap.release();height,width=images[0].shape[:2];assert (width,height)==(800,600)
            for variant,widths in plan['conditions'].items():
                for input_width in widths:
                    folder=OUT/f'subject-{subject:02d}'/variant;folder.mkdir(parents=True,exist_ok=True)
                    target=folder/f'{cam}_{input_width}.npz'
                    if target.exists():
                        old=json.loads(target.with_suffix('.json').read_text())
                        assert old['prediction_sha256']==digest(target) and old['model_sha256']==plan['model_sha256'][variant]
                        assert old['video_sha256']==digest(video) and old['reference']['sha256']==provenance['sha256']
                        with np.load(target) as z:
                            assert np.array_equal(z['video_frame_one_based'],np.arange(start-30,end+1))
                            lm=z['landmarks'];world=z['world_landmarks'];counts=z['pose_counts']
                    elif subject==1:
                        # Reuse the frozen pilot outputs, checking the recorded hash, not rerunning identical inference.
                        source_dir=PILOT/('baseline' if variant=='full' else 'heavy');source=source_dir/target.name
                        assert json.loads((source_dir/'plan.json').read_text())['model_sha256']==plan['model_sha256'][variant]
                        record=next(r for r in json.loads((source_dir/'inference-summary.json').read_text()) if r['file']==target.name)
                        assert digest(source)==record['sha256']
                        with np.load(source) as z:
                            assert np.array_equal(z['video_frame_one_based'],np.arange(start-30,end+1))
                            lm=z['landmarks'];world=z['world_landmarks'];counts=z['pose_counts']
                    else:
                        lm=np.full((len(images),33,5),np.nan);world=np.full_like(lm,np.nan);counts=[]
                        options=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(models[variant]),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_poses=3,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
                        with vision.PoseLandmarker.create_from_options(options) as detector:
                            for i,image in enumerate(images):
                                if input_width!=width:image=cv2.resize(image,(input_width,int(input_width*height/width)),interpolation=cv2.INTER_AREA)
                                prediction=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),round(i*1000/30))
                                counts.append(len(prediction.pose_landmarks))
                                if not prediction.pose_landmarks:continue
                                index=max(range(counts[-1]),key=lambda k:np.ptp([p.x for p in prediction.pose_landmarks[k]])*np.ptp([p.y for p in prediction.pose_landmarks[k]]))
                                lm[i]=[[p.x,p.y,p.z,p.visibility,p.presence] for p in prediction.pose_landmarks[index]]
                                world[i]=[[p.x,p.y,p.z,p.visibility,p.presence] for p in prediction.pose_world_landmarks[index]]
                    if not target.exists():np.savez_compressed(target,video_frame_one_based=np.arange(start-30,end+1),landmarks=lm,world_landmarks=world,pose_counts=counts)
                    row,arrays=measure(lm[30:],world[30:],refxy,ref3,gaps,[width,height])
                    if subject==1:
                        original=json.loads((PILOT/('baseline' if variant=='full' else 'heavy')/'metrics.json').read_text())['results']
                        original=next(r for r in original if r['camera']==cam and r['input_width']==input_width)
                        assert abs(row['position_discrepancy_px']['mean']-original['position_discrepancy_px']['mean'])<1e-9
                        for k in row['knee_abs_angle_difference_deg']:assert abs(row['knee_abs_angle_difference_deg'][k]['mean']-original['knee_abs_angle_difference_deg'][k]['mean'])<1e-9
                    row.update(subject=subject,camera=cam,variant=variant,input_width=input_width,evaluation_frames=len(frames),evaluation_detected=int(np.sum(np.array(counts)[30:]>0)),video_frames=[start,end],reference=provenance,video_sha256=digest(video),prediction_sha256=digest(target),model_sha256=plan['model_sha256'][variant],reused_pilot=subject==1)
                    np.savez_compressed(folder/f'{cam}_{input_width}_comparison.npz',frames=frames,reference_3d=ref3,**arrays)
                    dump(target.with_suffix('.json'),row)
                    if variant=='full' and input_width==800:
                        perframe=np.nanmean(arrays['discrepancy_px'],axis=1)
                        samples=sorted(set([0,len(frames)//2,len(frames)-1,int(np.nanargmax(perframe))]))
                        tiles=[]
                        for i in samples:
                            canvas=images[i+30].copy()
                            for pts,color in [(refxy[i],(70,255,70)),(arrays['predicted_xy'][i],(230,80,230))]:
                                for a,b in [('LSJC','RSJC'),('LHIP','RHIP')]+[(s+a,s+b) for s in 'LR' for a,b in [('SJC','EJC'),('EJC','WJC'),('SJC','HIP'),('HIP','KJC'),('KJC','AJC')]]:
                                    p=pts[[NAMES.index(a),NAMES.index(b)]]
                                    if np.isfinite(p).all():cv2.line(canvas,tuple(np.rint(p[0]).astype(int)),tuple(np.rint(p[1]).astype(int)),color,2)
                            cv2.putText(canvas,f'S{subject} {cam} f{frames[i]} green=reference pink=model',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
                            tiles.append(cv2.resize(canvas,(600,450)))
                        assert cv2.imwrite(str(folder/f'{cam}_review.jpg'),np.concatenate(tiles,axis=0))
                    print(json.dumps({'subject':subject,'camera':cam,'variant':variant,'width':input_width,'frames':len(frames),'detected':row['evaluation_detected'],'position_px':row['position_discrepancy_px']['mean'],'knee3d_deg':row['knee_abs_angle_difference_deg']['pred3d_vs_ref3d']['mean']}),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('subject-*/*/PG[12]_*.json'))]
    dump(OUT/'results.json',{'updated_unix':time.time(),'results':rows,'unique_subjects':sorted(set(r['subject'] for r in rows)),
                           'limits':['Provisional archive integrity where indicated','No new turn ground truth','No speed or clinical angle validation','Descriptive comparison; no fitting or success threshold']})


if __name__=='__main__':
    # The OS releases this lock even when a process stops unexpectedly.
    with (ROOT/'control'/'experiment.lock').open('a+b') as lock:
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Another experiment process owns the lock; no duplicate run started.')
        main()
