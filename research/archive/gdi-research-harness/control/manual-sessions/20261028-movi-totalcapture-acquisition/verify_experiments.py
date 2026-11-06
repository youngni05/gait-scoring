"""Independent saved-output checks before reporting this local pilot."""
import hashlib
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'20261028-camera-step-pilot'/'deps'))
import numpy as np
from PIL import Image

def bend_scalar(h,k,a):
    u=[float(x-y) for x,y in zip(h,k)];v=[float(x-y) for x,y in zip(a,k)]
    denom=math.sqrt(sum(t*t for t in u)*sum(t*t for t in v))
    if denom<=1e-10:return float('nan')
    return 180-math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(u,v))/denom))))

assert abs(bend_scalar([0,1,0],[0,0,0],[0,-1,0]))<1e-8
assert abs(bend_scalar([0,1,0],[0,0,0],[1,0,0])-90)<1e-8
assert abs(bend_scalar([3,5,2],[3,3,2],[5,3,2])-90)<1e-8
checked=[]
for folder in ['baseline','heavy']:
    out=ROOT/folder
    inference=json.loads((out/'inference-summary.json').read_text())
    report=json.loads((out/'metrics.json').read_text())
    for run,row in zip(inference,report['results'],strict=True):
        file=out/run['file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==run['sha256']
        with np.load(file,allow_pickle=False) as p:
            assert p['landmarks'].shape==p['world_landmarks'].shape==(207,33,5)
            assert p['video_frame_one_based'].tolist()==list(range(1311,1518))
            assert int((p['pose_counts'][30:]>0).sum())==run['evaluation_detected']==177
        cam=run['camera'];width=run['width']
        with np.load(out/f'{cam}_{width}_comparison.npz',allow_pickle=False) as c:
            assert c['frames'].tolist()==list(range(1341,1518))
            assert c['reference_gap'].sum()==3
            assert np.isnan(c['reference_xy']).all(axis=-1).sum()==3
            d=np.sqrt(np.sum((c['predicted_xy']-c['reference_xy'])**2,axis=-1))
            assert np.isfinite(d).sum()==2121
            assert abs(float(np.nanmean(d))-row['position_discrepancy_px']['mean'])<1e-10
            for key,pred,ref in [('pred2d_vs_ref2d','predicted_2d_bend_deg','reference_2d_bend_deg'),
                                 ('pred2d_vs_ref3d','predicted_2d_bend_deg','reference_3d_bend_deg'),
                                 ('pred3d_vs_ref3d','predicted_3d_bend_deg','reference_3d_bend_deg')]:
                e=np.abs(c[pred]-c[ref]);assert np.isfinite(e).sum()==354
                assert abs(float(e.mean())-row['knee_abs_angle_difference_deg'][key]['mean'])<1e-10
            # Sample scalar recalculations use Python math rather than NumPy's vector formula.
            for frame in [0,50,90,176]:
                for side,indices in enumerate([[6,8,10],[7,9,11]]):
                    actual=bend_scalar(*c['predicted_xy'][frame,indices])
                    assert abs(actual-c['predicted_2d_bend_deg'][frame,side])<1e-7
        checked.append({'folder':folder,'camera':cam,'width':width,'samples':2121,'angle_samples':354})
    with Image.open(out/'model_vs_reference.gif') as im:
        assert im.n_frames==89
        for i in range(im.n_frames):im.seek(i);im.load()
    with Image.open(out/'diagnostic-curves.png') as im:im.verify()
full=json.loads((ROOT/'baseline/metrics.json').read_text())['results']
heavy=json.loads((ROOT/'heavy/metrics.json').read_text())['results']
for cam in ['PG1','PG2']:
    selected=[r for r in full+heavy if r['camera']==cam]
    assert len(set(r['knee_abs_angle_difference_deg']['ref2d_vs_ref3d_projection_difference']['mean'] for r in selected))==1
record={'status':'saved_outputs_verified','runs_checked':checked,'total_evaluated_frame_outputs':177*len(checked),
        'unique_subjects':1,'unique_motion_frames_per_view':177,
        'checks':['Input and output frame indices','Prediction file SHA256','Reference gap masks and sample counts',
                  'Position and angle summary recomputation','Known geometric angles and scalar formula cross-check',
                  'Reference-only projection comparison unchanged across models/resolutions','Both GIFs fully decoded; plot images readable'],
        'not_proven':['Clinical angle validity','Subframe synchronization','Real-distance robustness','Unseen-person generalization']}
(ROOT/'experiment-verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
