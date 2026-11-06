"""Local MoVi overlay audit; no learned model, fitting, or offset search."""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '20261028-camera-step-pilot' / 'deps'))
import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat

OUT = ROOT / 'alignment'
OUT.mkdir(exist_ok=True)
acquisition = json.loads((ROOT / 'acquisition-status.json').read_text(encoding='utf-8'))
for item in acquisition['files']:
    assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256']
move = loadmat(ROOT / 'F_v3d_Subject_1.mat', simplify_cells=True)['Subject_1_F']['move']
motion = list(move['motions_list']).index('walking')
start, end = map(int, move['flags30'][motion])  # MATLAB one-based, inclusive.
video_indices = np.arange(start - 1, end)
mocap_indices = video_indices * 4
assert np.array_equal(move['flags120'].astype(int), (move['flags30'].astype(int)-1)*4+1)
names = [str(move['markerName'][int(i)-1]) for i in move['virtualMarkers']]
virtual_ids = move['virtualMarkers'].astype(int)-1
joint_gaps = move['markerGaps'][mocap_indices][:,virtual_ids].astype(bool)
assert np.array_equal(move['virtualMarkerLocation'], move['markerLocation'][:,virtual_ids])
links = [('HEAD','THORAX'),('THORAX','PELVIS'),('PELVIS','LHIP'),('PELVIS','RHIP')]
for side in 'LR':
    links += [('THORAX',side+'SJC'),(side+'SJC',side+'EJC'),(side+'EJC',side+'WJC'),
              (side+'WJC',side+'HJC'),(side+'HIP',side+'KJC'),(side+'KJC',side+'AJC'),(side+'AJC',side+'FJC')]

def project(points, intrinsics, extrinsics):
    # Legacy MATLAB worldToImage uses row-vector postmultiplication.
    r, t = extrinsics['rotationMatrix'], extrinsics['translationVector']
    camera = points @ r + t
    assert np.all(camera[...,2] > 0)
    xy = camera[...,:2] / camera[...,2,None]
    r2 = np.sum(xy*xy, axis=-1)
    k1,k2 = intrinsics['RadialDistortion']
    xy = xy * (1+k1*r2+k2*r2*r2)[...,None]
    homogeneous = np.concatenate([xy, np.ones(xy.shape[:-1]+(1,))], axis=-1)
    # MATLAB pixel centers start at 1, OpenCV array coordinates at 0.
    uv = (homogeneous @ intrinsics['IntrinsicMatrix'])[...,:2] - 1
    # Independent implementation check, not a measurement-accuracy check.
    rotation_vector = cv2.Rodrigues(r.T)[0]
    cv_uv = cv2.projectPoints(points.reshape(-1,3), rotation_vector, t,
                             intrinsics['IntrinsicMatrix'].T, np.array([k1,k2,0.,0.,0.]))[0]
    error = np.max(np.abs(uv.reshape(-1,2) - (cv_uv.reshape(-1,2)-1)))
    assert error < 1e-6
    return uv, float(error)

clips, results = [], {}
with tarfile.open(ROOT / 'Camera_Parameters.tar') as archive:
    for cam in ['PG1','PG2']:
        def read_npz(name):
            with np.load(io.BytesIO(archive.extractfile('Calib/'+name+'.npz').read()), allow_pickle=False) as data:
                return dict(data)
        intr = read_npz('cameraParams_'+cam)
        ext = read_npz('Extrinsics_'+cam)
        joints, error = project(move['virtualMarkerLocation'][mocap_indices], intr, ext)
        physical_ids = move['physicalMarkers'].astype(int)-1
        markers, _ = project(move['markerLocation'][mocap_indices][:,physical_ids], intr, ext)
        gaps = move['markerGaps'][mocap_indices][:,physical_ids]
        cap = cv2.VideoCapture(str(ROOT / ('F_'+cam+'_Subject_1_L.avi')))
        assert cap.get(cv2.CAP_PROP_FPS) == 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_indices[0]))
        frames = []
        for local, original in enumerate(video_indices):
            ok, frame = cap.read()
            assert ok, (cam, int(original))
            for a,b in links:
                ia,ib = names.index(a), names.index(b)
                if joint_gaps[local,ia] or joint_gaps[local,ib]:
                    continue
                color = (255,180,40) if b.startswith('L') else ((40,150,255) if b.startswith('R') else (255,255,255))
                cv2.line(frame, tuple(np.rint(joints[local,ia]).astype(int)), tuple(np.rint(joints[local,ib]).astype(int)), color, 2, cv2.LINE_AA)
            for xy,gap in zip(markers[local],gaps[local]):
                if not gap:
                    cv2.circle(frame, tuple(np.rint(xy).astype(int)), 3, (40,255,70), 1, cv2.LINE_AA)
            frame = np.concatenate([np.full((60,800,3),20,dtype=np.uint8),frame],axis=0)
            cv2.putText(frame,f'{cam} | video {original+1} | mocap {mocap_indices[local]+1}',(12,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            label = 'Reference: green=markers; blue=L; orange=R'
            missing = [names[j] for j in np.flatnonzero(joint_gaps[local])]
            if missing:
                label += ' | MISSING: '+','.join(missing)
            cv2.putText(frame,label,(12,45),cv2.FONT_HERSHEY_SIMPLEX,.48,(220,220,220),1,cv2.LINE_AA)
            frames.append(frame)
        cap.release()
        joints[joint_gaps] = np.nan
        np.savez_compressed(OUT / (cam+'_walking_projection.npz'), video_frame_one_based=video_indices+1,
                            mocap_frame_one_based=mocap_indices+1, joint_names=np.array(names), joints_xy_zero_based=joints, joint_gaps=joint_gaps,
                            physical_marker_indices_zero_based=physical_ids, markers_xy_zero_based=markers, marker_gaps=gaps)
        clips.append(frames)
        results[cam] = {'decoded_walking_frames':len(frames),'projection_implementation_max_difference_px':error,
                        'valid_projected_joint_points_in_image_fraction':float(np.mean(((joints[...,0]>=0)&(joints[...,0]<800)&(joints[...,1]>=0)&(joints[...,1]<600))[~joint_gaps]))}

sample_offsets = [0,30,60,90,120,176]
pairs = [np.concatenate([clips[0][i],clips[1][i]],axis=1) for i in range(len(video_indices))]
sheet = np.concatenate([cv2.resize(pairs[i],(1200,495)) for i in sample_offsets],axis=0)
assert cv2.imwrite(str(OUT/'walking_contactsheet.jpg'),sheet)
for i in sample_offsets:
    assert cv2.imwrite(str(OUT/f'frame_{start+i}.jpg'),pairs[i])
gif_frames = [Image.fromarray(cv2.cvtColor(cv2.resize(frame,(960,396)),cv2.COLOR_BGR2RGB)) for frame in pairs[::2]]
gif_frames[0].save(OUT/'walking_reference_overlay.gif',save_all=True,append_images=gif_frames[1:],duration=67,loop=0)
report = {'scope':'Subject 1, F round, walking; two views; reference projection only',
          'video_frames_one_based_inclusive':[start,end],'sampled_mocap_frames_one_based_inclusive':[int(mocap_indices[0]+1),int(mocap_indices[-1]+1)],
          'sources':['https://github.com/saeed1262/MoVi-Toolbox/blob/master/MoCap/worldToVideo.m',
                     'https://github.com/saeed1262/MoVi-Toolbox/blob/master/MoCap/single_videos.m',
                     'https://www.mathworks.com/help/vision/ref/worldtoimage.html'],
          'method':'virtualMarkerLocation and physical markerLocation; supplied NPZ intrinsics/extrinsics; row-vector world@R+t; radial distortion; MATLAB-to-OpenCV pixel origin conversion',
          'assumptions':['NPZ supplies two radial terms, no tangential terms; tangential distortion set to zero; serialized MATLAB camera object not fully decoded',
                         'Native spatial units retained for points and translation; no metric speed calculated'],
          'no_fitted_parameters':True,'offset_search_performed':False,'ai_pose_inference_performed':False,
          'reference_missingness':{'field':'markerGaps indexed by virtualMarkers; jointsGaps_v3d refers to a different field',
                                  'left_elbow_mocap_frames_one_based':(np.flatnonzero(move['markerGaps'][5360:6065,virtual_ids[names.index('LEJC')]])+5361).tolist(),
                                  'left_elbow_video_frames_one_based':(video_indices[joint_gaps[:,names.index('LEJC')]]+1).tolist(),
                                  'handling':'Omit flagged joint and connected lines; exported coordinates are NaN; no interpolation'},
          'camera_results':results,'visual_review':'pending','limits':['Projection arithmetic agreement is not measured video alignment error',
          'No independent manual pixel annotations or subframe synchronization validation','One participant and one short movement do not establish generalization']}
(OUT/'alignment-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
