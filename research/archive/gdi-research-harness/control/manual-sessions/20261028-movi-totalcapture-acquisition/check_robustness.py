"""Fixed postprocessing and alignment sensitivity; no selected best offset."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'20261028-camera-step-pilot'/'deps'))
import numpy as np
OUT=ROOT/'baseline'
plan={'smoothing':'coordinatewise median over fixed 5 frames; trailing and centered variants; no missing-value interpolation',
      'common_smoothing_frames_one_based':[1341,1515],'centered_future_frames':2,
      'alignment_offsets_video_frames':[-1,0,1],'common_alignment_frames_one_based':[1342,1516],
      'turn_window_variants':[[1401,1441],[1411,1451],[1421,1461]],
      'purpose':'Sensitivity checks on the same pilot, not validation on new participants or selection of tuned settings'}
(OUT/'robustness-plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
MAPPING={'LSJC':11,'RSJC':12,'LEJC':13,'REJC':14,'LWJC':15,'RWJC':16,'LHIP':23,'RHIP':24,'LKJC':25,'RKJC':26,'LAJC':27,'RAJC':28}
names=list(MAPPING);ids=list(MAPPING.values())

def knee(x):
    output=[]
    for side in 'LR':
        h,k,a=[names.index(side+s) for s in ['HIP','KJC','AJC']]
        u=x[:,h]-x[:,k]; v=x[:,a]-x[:,k]
        # atan2 form provides an independent implementation of geometric angle.
        dot=np.sum(u*v,axis=-1)
        cross=np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0]) if x.shape[-1]==2 else np.linalg.norm(np.cross(u,v),axis=-1)
        bend=180-np.degrees(np.arctan2(cross,dot))
        bend[(np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1))<1e-10]=np.nan
        output.append(bend)
    return np.stack(output,axis=1)

results=[]
for cam in ['PG1','PG2']:
    p=np.load(OUT/f'{cam}_800.npz',allow_pickle=False)
    c=np.load(OUT/f'{cam}_800_comparison.npz',allow_pickle=False)
    xy=p['landmarks'][:,ids,:2]*[800,600]
    xyz=p['world_landmarks'][:,ids,:3]
    assert np.allclose(knee(xy[30:]),c['predicted_2d_bend_deg'],atol=1e-6,equal_nan=True)
    assert np.allclose(knee(xyz[30:]),c['predicted_3d_bend_deg'],atol=1e-6,equal_nan=True)
    def variants(x):
        m=np.median(np.lib.stride_tricks.sliding_window_view(x,5,axis=0),axis=-1)
        centered=np.full_like(x,np.nan); centered[2:-2]=m
        trailing=np.full_like(x,np.nan); trailing[4:]=m
        return {'raw':x[30:205],'centered_median5':centered[30:205],'trailing_median5':trailing[30:205]}
    vv=variants(xyz);smooth=[]
    for label,pred in variants(xy).items():
        e=np.linalg.norm(pred-c['reference_xy'][:175],axis=-1)
        smooth.append({'method':label,'frames':175,'position_mean_px':float(np.nanmean(e)),
                       'position_p90_px':float(np.nanpercentile(e,90)),
                       'knee_2d_vs_reference_2d_mae_deg':float(np.nanmean(np.abs(knee(pred)-c['reference_2d_bend_deg'][:175]))),
                       'knee_2d_vs_reference_3d_mae_deg':float(np.nanmean(np.abs(knee(pred)-c['reference_3d_bend_deg'][:175]))),
                       'knee_3d_vs_reference_3d_mae_deg':float(np.nanmean(np.abs(knee(vv[label])-c['reference_3d_bend_deg'][:175])))})
    offsets=[]
    for shift in [-1,0,1]:
        ref_slice=slice(1+shift,176+shift)
        offsets.append({'reference_offset_video_frames':shift,'frames':175,
                        'position_mean_px':float(np.nanmean(np.linalg.norm(c['predicted_xy'][1:176]-c['reference_xy'][ref_slice],axis=-1))),
                        'knee_2d_vs_reference_3d_mae_deg':float(np.nanmean(np.abs(c['predicted_2d_bend_deg'][1:176]-c['reference_3d_bend_deg'][ref_slice]))),
                        'knee_3d_vs_reference_3d_mae_deg':float(np.nanmean(np.abs(c['predicted_3d_bend_deg'][1:176]-c['reference_3d_bend_deg'][ref_slice])))})
    windows=[]
    for start,end in plan['turn_window_variants']:
        mask=(c['frames']>=start)&(c['frames']<=end)
        windows.append({'window':[start,end],'inside_position_mean_px':float(np.nanmean(c['discrepancy_px'][mask])),
                        'outside_position_mean_px':float(np.nanmean(c['discrepancy_px'][~mask]))})
    results.append({'camera':cam,'smoothing':smooth,'alignment_sensitivity':offsets,'turn_window_sensitivity':windows})
(OUT/'robustness.json').write_text(json.dumps({'plan':plan,'results':results},indent=2)+'\n',encoding='utf-8')
print(json.dumps(results,indent=2))
