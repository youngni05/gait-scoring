"""Descriptive diagnostics, not a clinical score or validated benchmark."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'20261028-camera-step-pilot'/'deps'))
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.io import loadmat
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=['full','heavy'],default='full')
variant=parser.parse_args().variant
OUT=ROOT/('baseline' if variant=='full' else 'heavy')
WIDTHS=[800,400,200] if variant=='full' else [800]
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
MAPPING={'LSJC':11,'RSJC':12,'LEJC':13,'REJC':14,'LWJC':15,'RWJC':16,
         'LHIP':23,'RHIP':24,'LKJC':25,'RKJC':26,'LAJC':27,'RAJC':28}
NAMES=list(MAPPING); MP=list(MAPPING.values())
FRAMES=np.arange(1341,1518)
WINDOW=(FRAMES>=1411)&(FRAMES<=1451)
move=loadmat(ROOT/'F_v3d_Subject_1.mat',simplify_cells=True)['Subject_1_F']['move']
ref_names=[str(move['markerName'][int(i)-1]) for i in move['virtualMarkers']]
REF=[ref_names.index(n) for n in NAMES]
ref3d=move['virtualMarkerLocation'][(FRAMES-1)*4][:,REF]
ref_gap=move['markerGaps'][(FRAMES-1)*4][:,move['virtualMarkers'].astype(int)[REF]-1].astype(bool)
ref3d[ref_gap]=np.nan

def bend(x):
    a=x[:,0]-x[:,1]; b=x[:,2]-x[:,1]
    denom=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    cosine=np.divide(np.sum(a*b,axis=-1),denom,out=np.full_like(denom,np.nan),where=denom>1e-10)
    return 180-np.degrees(np.arccos(np.clip(cosine,-1,1)))

def stats(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return {'n':int(x.size),'mean':float(x.mean()) if len(x) else None,
            'median':float(np.median(x)) if len(x) else None,'p90':float(np.percentile(x,90)) if len(x) else None}

def angles(x):
    return np.stack([bend(x[:,[NAMES.index(s+n) for n in ['HIP','KJC','AJC']]]) for s in 'LR'],axis=1)

truth_angles=angles(ref3d)
all_results=[]; arrays={}
for cam in ['PG1','PG2']:
    ref=np.load(ROOT/'alignment'/f'{cam}_walking_projection.npz',allow_pickle=False)
    assert np.array_equal(ref['video_frame_one_based'],FRAMES)
    refxy=ref['joints_xy_zero_based'][:,REF]
    ref2_angles=angles(refxy)
    for width in WIDTHS:
        p=np.load(OUT/f'{cam}_{width}.npz',allow_pickle=False)
        assert np.array_equal(p['video_frame_one_based'][30:],FRAMES)
        lm=p['landmarks'][30:]; xyz=p['world_landmarks'][30:,:,0:3][:,MP]
        xy=lm[:,MP,:2]*[800,600]; vis=lm[:,MP,3]
        distance=np.linalg.norm(xy-refxy,axis=2)
        pred2_angles=angles(xy); pred3_angles=angles(xyz)
        knee_vis=np.stack([np.min(vis[:,[NAMES.index(s+n) for n in ['HIP','KJC','AJC']]],axis=1) for s in 'LR'],axis=1)
        comparisons={'pred2d_vs_ref2d':np.abs(pred2_angles-ref2_angles),
                     'ref2d_vs_ref3d_projection_difference':np.abs(ref2_angles-truth_angles),
                     'pred2d_vs_ref3d':np.abs(pred2_angles-truth_angles),
                     'pred3d_vs_ref3d':np.abs(pred3_angles-truth_angles)}
        row={'camera':cam,'input_width':width,'position_discrepancy_px':stats(distance),
             'position_visibility_ge_05':stats(distance[vis>=.5]),
             'reference_valid_joint_samples':int(np.sum(~ref_gap)),
             'joint_position_discrepancy_px':{name:stats(distance[:,j]) for j,name in enumerate(NAMES)},
             'knee_abs_angle_difference_deg':{k:stats(v) for k,v in comparisons.items()},
             'knee_visibility_ge_05':{k:stats(v[knee_vis>=.5]) for k,v in comparisons.items()},
             'visual_turn_window_1411_1451':{'position':stats(distance[WINDOW]),'knees':{k:stats(v[WINDOW]) for k,v in comparisons.items()}},
             'outside_visual_turn_window':{'position':stats(distance[~WINDOW]),'knees':{k:stats(v[~WINDOW]) for k,v in comparisons.items()}},
             'worst_mean_joint_discrepancy_frame':int(FRAMES[np.nanargmax(np.nanmean(distance,axis=1))])}
        all_results.append(row)
        arrays[(cam,width)]={'xy':xy,'refxy':refxy,'distance':distance,'vis':vis,'pred2':pred2_angles,'pred3':pred3_angles,'ref2':ref2_angles}
        np.savez_compressed(OUT/f'{cam}_{width}_comparison.npz',frames=FRAMES,reference_xy=refxy,predicted_xy=xy,
                            discrepancy_px=distance,visibility=vis,reference_gap=ref_gap,
                            reference_3d_bend_deg=truth_angles,reference_2d_bend_deg=ref2_angles,
                            predicted_2d_bend_deg=pred2_angles,predicted_3d_bend_deg=pred3_angles)

report={'model_variant':variant,'scope':'One subject; two views; 177 frames each; 12 corresponding joints; descriptive exploratory results',
        'joint_mapping':MAPPING,'pixel_units':'All predictions mapped back to original 800x600 image coordinates',
        'angle_definition':'180 minus included angle of hip-knee-ankle; 0 = straight; not clinical joint-coordinate-system flexion',
        'turn_window':'1411-1451 selected visually before model error inspection; provisional, not independently annotated',
        'interpretation_limits':['Corresponding landmark definitions are approximate, especially hip and shoulder centers',
            'Camera calibration, synchronization and reference errors contribute to discrepancy',
            'Projection differences are not additive to estimation errors','Frames are repeated observations of one person, not independent participants',
            'Confidence is not ground-truth visibility; high-confidence filtering can select easier observations',
            'No gait-speed validation, clinical scoring, fine-tuning or generalization claim'],
        'results':all_results}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
for col,cam in enumerate(['PG1','PG2']):
    ax=axes[0,col]
    for width,color in [(800,'#0072B2'),(400,'#E69F00'),(200,'#CC79A7')]:
        if width not in WIDTHS:continue
        ax.plot(FRAMES,np.nanmean(arrays[(cam,width)]['distance'],axis=1),label=f'{width}×{int(width*.75)} 입력',color=color,lw=1.3)
    ax.set_title(f'{cam}: 관절 12곳의 평균 위치 차이'); ax.set_ylabel('기준 좌표와 거리 (원본 픽셀)');ax.legend(fontsize=8)
    ax=axes[1,col]; a=arrays[(cam,800)]
    for values,label,color in [(truth_angles[:,0],'정밀 측정 3D','#222222'),(a['ref2'][:,0],'정밀 좌표를 화면에 투영한 2D','#009E73'),
                               (a['pred2'][:,0],'MediaPipe 2D','#D55E00'),(a['pred3'][:,0],'MediaPipe 3D','#0072B2')]:
        ax.plot(FRAMES,values,label=label,color=color,lw=1.25)
    ax.set_title(f'{cam}: 왼쪽 무릎의 세 점 굽힘각 (원본 입력)');ax.set_ylabel('각도 (도)');ax.set_xlabel('원본 영상 프레임 (1부터)');ax.legend(fontsize=8)
for ax in axes.flat:
    ax.axvspan(1411,1451,color='#888888',alpha=.12);ax.grid(alpha=.2)
fig.suptitle(f'MoVi 1명 예비 진단 · MediaPipe {variant.title()} · 회색: 방향전환 주변 임시 관찰 구간\n검출 성공 여부와 위치·각도의 정확성은 별도이며, 임상 각도 검증 결과가 아님',fontsize=13)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'diagnostic-curves.png',dpi=160);plt.close(fig)

# Same frames in both views; green reference, magenta model. Missing reference omitted.
links=[('LSJC','RSJC'),('LSJC','LHIP'),('RSJC','RHIP'),('LHIP','RHIP')]
for s in 'LR':links += [(s+'SJC',s+'EJC'),(s+'EJC',s+'WJC'),(s+'HIP',s+'KJC'),(s+'KJC',s+'AJC')]
clips=[]
for cam in ['PG1','PG2']:
    cap=cv2.VideoCapture(str(ROOT/f'F_{cam}_Subject_1_L.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,1340);clip=[]
    for i,f in enumerate(FRAMES):
        ok,bgr=cap.read();assert ok
        for key,color in [('refxy',(70,255,70)),('xy',(230,80,230))]:
            pts=arrays[(cam,800)][key][i]
            for a,b in links:
                points=pts[[NAMES.index(a),NAMES.index(b)]]
                if np.isfinite(points).all():cv2.line(bgr,tuple(np.rint(points[0]).astype(int)),tuple(np.rint(points[1]).astype(int)),color,2,cv2.LINE_AA)
            for point in pts:
                if np.isfinite(point).all():cv2.circle(bgr,tuple(np.rint(point).astype(int)),3,color,1,cv2.LINE_AA)
        bgr=np.concatenate([np.full((50,800,3),20,np.uint8),bgr])
        cv2.putText(bgr,f'{cam} | frame {f} | green=reference, magenta=MediaPipe',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        err=np.nanmean(arrays[(cam,800)]['distance'][i]);cv2.putText(bgr,f'Mean 12-joint discrepancy: {err:.1f} original pixels',(10,43),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
        clip.append(bgr)
    cap.release();clips.append(clip)
pairs=[np.concatenate([clips[0][i],clips[1][i]],axis=1) for i in range(177)]
worst=sorted(set([1341,1431,1517]+[r['worst_mean_joint_discrepancy_frame'] for r in all_results if r['input_width']==800]))
for f in worst:assert cv2.imwrite(str(OUT/f'comparison_frame_{f}.jpg'),pairs[f-1341])
gif=[Image.fromarray(cv2.cvtColor(cv2.resize(f,(800,325)),cv2.COLOR_BGR2RGB)) for f in pairs[::2]]
gif[0].save(OUT/'model_vs_reference.gif',save_all=True,append_images=gif[1:],duration=67,loop=0)
print(json.dumps([{k:r[k] for k in ['camera','input_width','position_discrepancy_px','knee_abs_angle_difference_deg','worst_mean_joint_discrepancy_frame']} for r in all_results],indent=2))
