"""No-fit hip-definition audit and geometric sensitivity, preserving baseline scores."""
import hashlib
import json
import sys
from pathlib import Path

CODE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(CODE))
from run_ready_subjects import ROOT,OUT as BASE,NAMES,MP,np,loadmat,digest,angles
OUT=ROOT/'experiments'/'hip-definition'
OUT.mkdir(exist_ok=True)
HIP=[NAMES.index('LHIP'),NAMES.index('RHIP')]
KNEE=[NAMES.index('LKJC'),NAMES.index('RKJC')]
SOURCES={
 'movi':'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0253157',
 'visual3d':'https://wiki.has-motion.com/doku.php?id=visual3d:documentation:modeling:segments:hip_joint_landmarks',
 'mediapipe':'https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker',
 'model_card':'https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf',
 'ghum_landmark_paper':'https://arxiv.org/abs/2206.11678'}

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def avg(x):
    x=np.asarray(x);v=x[np.isfinite(x)]
    return float(v.mean()) if v.size else None

rows=json.loads((BASE/'results.json').read_text(encoding='utf-8'))['results']
rows=[r for r in rows if r['input_width']==800]
people=sorted({r['subject'] for r in rows})
assert len(people)==30 and len(rows)==120
plan={'people':people,'variants':['full','heavy'],'cameras':['PG1','PG2'],
 'source_results_sha256':digest(BASE/'results.json'),'sources':SOURCES,
 'question':'How much of hip discrepancy is systematic, and how sensitive are 2D knee angles to hip locations?',
 'no_fitted_correction':True,'no_new_inference':True,'baseline_replaced':False,
 'diagnostics':['Per-joint mean vector and exact squared-error bias/variance identity',
                'Exact two-hip midpoint/span error identity on common finite hip pairs',
                'Hip-only hybrid: predicted hips with reference knees/ankles',
                'Reference-hip substitution: reference hips with predicted knees/ankles (oracle diagnostic, not deployable)',
                'Dimensionless hip-width/thigh-length ratio; no 3D coordinate alignment',
                'MoVi CODA/Bell coefficient compatibility using root pelvis frame and reference hip span'],
 'aggregation':'Compute within each person/camera/model first, then equal-weight persons; no frame pooling',
 'limits':['Bias fraction is squared-error energy, NOT proportion caused by definition mismatch',
           'Hip substitution isolates combined hip-location effects, not their anatomical or model cause',
           '2D angle sensitivity cannot establish the cause of 3D knee-angle error',
           'Public MediaPipe documents do not provide an exact anatomical mapping to Bell-Brand centers']}
dump(OUT/'plan.json',plan)

metadata=[];bell=[];results=[]
for subject in people:
    ref_path=ROOT/'reference'/f'F_v3d_Subject_{subject}.mat'
    move=loadmat(ref_path,simplify_cells=True)[f'Subject_{subject}_F']['move']
    names=[n if isinstance(n,str) else '' for n in move['markerName']]
    definitions={n:{'index_zero_based':names.index(n),'description':str(move['markerDescription'][names.index(n)]),
                    'type':str(move['markerType'][names.index(n)])}
                 for n in ['LHIP','RHIP','LHJC','RHJC','LFWT','RFWT','LBWT','RBWT']}
    assert definitions['LHIP']['description']=='Left Hip Joint'
    assert definitions['RHIP']['description']=='Right Hip Joint'
    assert definitions['LHJC']['description']=='Left Hand Joint'
    assert definitions['RHJC']['description']=='Right Hand Joint'
    metadata.append({'subject':subject,'file_sha256':digest(ref_path),'definitions':definitions})
    subset=[r for r in rows if r['subject']==subject]
    start,end=subset[0]['video_frames'];mocap=(np.arange(start,end+1)-1)*4
    both_views_ankles_inside=np.ones(len(mocap),dtype=bool)
    for cam in ['PG1','PG2']:
        with np.load(BASE/f'subject-{subject:02d}'/'full'/f'{cam}_800_comparison.npz') as z:
            ankle=z['reference_xy'][:,[NAMES.index('LAJC'),NAMES.index('RAJC')]]
            both_views_ankles_inside &= (np.isfinite(ankle)&(ankle>=0)&(ankle<[800,600])).all(axis=(1,2))
    pelvis=list(move['segmentName']).index('RPV')
    assert int(move['segmentParent'][pelvis])==0
    transform=move['jointsAffine_v3d'][mocap,pelvis]
    rot=transform[:,:3,:3];origin=transform[:,:3,3]
    hi=[names.index(n) for n in ['LHIP','RHIP']]
    points=move['markerLocation'][mocap][:,hi]
    good=~move['markerGaps'][mocap][:,hi].astype(bool).any(axis=1)
    good &= ~move['jointsGaps_v3d'][mocap,pelvis].astype(bool)
    local=np.einsum('nhi,nji->nhj',points-origin[:,None,:],rot)
    # R is used with row vectors in these files. The known reference hip span
    # supplies the effective ASIS metric for a coefficient-compatibility check;
    # this is not an independent anatomical validation or prediction fitting.
    width=np.linalg.norm(points[:,0]-points[:,1],axis=1)/.72
    expected=np.array([[-.36,-.19,-.3],[.36,-.19,-.3]])[None,:,:]*width[:,None,None]
    normalized=np.linalg.norm(local-expected,axis=-1)/width[:,None]
    bell.append({'subject':subject,'valid_frames':int(good.sum()),
                 'normalized_coefficient_residual_mean':avg(normalized[good]),
                 'normalized_coefficient_residual_max':float(np.max(normalized[good])),
                 'effective_width_relative_sd':float(np.std(width[good])/np.mean(width[good])),
                 'rotation_convention':'world = local @ R + translation; inverse uses R transpose',
                 'width_metric':'Reference hip span /0.72; coefficient-compatibility audit only'})
    for row in subset:
        camera,variant=row['camera'],row['variant']
        folder=BASE/f'subject-{subject:02d}'/variant
        prediction=folder/f'{camera}_800.npz';assert digest(prediction)==row['prediction_sha256']
        with np.load(folder/f'{camera}_800_comparison.npz') as z:d=dict(z)
        with np.load(prediction) as z:
            xy=z['landmarks'][30:,MP,:2]*[800,600]
            world=z['world_landmarks'][30:,MP,:3]
        assert np.array_equal(xy,d['predicted_xy'],equal_nan=True)
        ref=d['reference_xy'];e=xy-ref;eh=e[:,HIP]
        valid=np.isfinite(eh).all(axis=-1);count=valid.sum(axis=0)
        mu=np.nanmean(np.where(valid[...,None],eh,np.nan),axis=0)
        mse=avg(np.sum(eh*eh,axis=-1))
        bias=float(np.sum(count*np.sum(mu*mu,axis=-1))/count.sum())
        variable=avg(np.sum((eh-mu[None,:,:])**2,axis=-1))
        assert abs(mse-bias-variable)<1e-7
        common=valid.all(axis=1)
        mid=(xy[:,HIP].mean(axis=1)-ref[:,HIP].mean(axis=1))[common]
        span=((xy[:,HIP[0]]-xy[:,HIP[1]])-(ref[:,HIP[0]]-ref[:,HIP[1]]))[common]
        pair_mse=avg(np.sum(eh[common]**2,axis=-1))
        midpoint_energy=avg(np.sum(mid*mid,axis=-1));span_energy=avg(np.sum(span*span,axis=-1)/4)
        assert abs(pair_mse-midpoint_energy-span_energy)<1e-7
        hip_only=ref.copy();hip_only[:,HIP]=xy[:,HIP]
        oracle=xy.copy();oracle[:,HIP]=ref[:,HIP]
        truth=angles(ref);full=angles(xy);only=angles(hip_only);oracle_angle=angles(oracle)
        common_angle=np.isfinite(truth)&np.isfinite(full)&np.isfinite(only)&np.isfinite(oracle_angle)
        for test in [truth,full,only,oracle_angle]:assert test.shape==(len(ref),2)
        def ratio(x):
            width=np.linalg.norm(x[:,HIP[0]]-x[:,HIP[1]],axis=-1)
            thigh=np.linalg.norm(x[:,HIP]-x[:,KNEE],axis=-1).mean(axis=1)
            return width/thigh
        transform_test=d['reference_3d'] @ np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])*7.3+[100.,-50.,25.]
        difference=np.abs(angles(transform_test)-d['reference_3d_bend_deg'])
        assert np.nanmax(difference)<1e-5
        joint_bias={}
        for j,name in enumerate(NAMES):
            ej=e[:,j];goodj=np.isfinite(ej).all(axis=1);meanj=ej[goodj].mean(axis=0)
            joint_bias[name]={'mean_xy_px':meanj.tolist(),'mean_distance_px':avg(np.linalg.norm(ej,axis=-1)),
                              'mse_px2':avg(np.sum(ej*ej,axis=-1)),'mean_bias_energy_px2':float(np.sum(meanj*meanj))}
        down=ref[:,HIP].mean(axis=1)-ref[:,[NAMES.index('LSJC'),NAMES.index('RSJC')]].mean(axis=1)
        down=down/np.linalg.norm(down,axis=-1,keepdims=True)
        modelratio=ratio(world);refratio=ratio(d['reference_3d'])
        hip_mae=avg(np.linalg.norm(eh,axis=-1))
        previous=np.mean([row['joint_position_discrepancy_px'][n]['mean'] for n in ['LHIP','RHIP']])
        # Normally both hips have equal valid counts; use point-weighted identity always.
        if count[0]==count[1]:assert abs(hip_mae-previous)<1e-9
        visible_hip=(d['visibility'][:,HIP]>=.5)&valid
        visible_triplets=np.stack([(d['visibility'][:,[h,k,a]]>=.5).all(axis=1)
                                    for h,k,a in [(6,8,10),(7,9,11)]],axis=1)&common_angle
        restricted_angle=common_angle&both_views_ankles_inside[:,None]
        results.append({'subject':subject,'camera':camera,'variant':variant,'hip_point_samples':int(count.sum()),
            'hip_mean_distance_px':hip_mae,'hip_rmse_px':float(np.sqrt(mse)),
            'hip_mean_xy_by_side_px':mu.tolist(),'hip_mean_delta_y_px':avg(eh[:,:,1]),
            'bias_energy_fraction':bias/mse,'bias_energy_px2':bias,'total_mse_px2':mse,
            'fluctuation_energy_px2':variable,'midpoint_energy_fraction':midpoint_energy/pair_mse,
            'span_energy_fraction':span_energy/pair_mse,
            'along_reference_trunk_down_px':avg(np.sum(eh*down[:,None,:],axis=-1)),
            'angle_valid_samples':int(common_angle.sum()),
            'full_2d_knee_mae_deg':avg(np.abs(full-truth)[common_angle]),
            'hip_only_2d_knee_mae_deg':avg(np.abs(only-truth)[common_angle]),
            'reference_hip_substitution_2d_knee_mae_deg':avg(np.abs(oracle_angle-truth)[common_angle]),
            'unchanged_full_3d_knee_mae_deg':row['knee_abs_angle_difference_deg']['pred3d_vs_ref3d']['mean'],
            'model_3d_hipwidth_thigh_ratio':avg(modelratio),'reference_3d_hipwidth_thigh_ratio':avg(refratio),
            'visibility_05':{'hip_retained_fraction':float(visible_hip.sum()/valid.sum()),
                'hip_mean_distance_px':avg(np.linalg.norm(eh,axis=-1)[visible_hip]),
                'hip_mean_delta_y_px':avg(eh[:,:,1][visible_hip]),
                'angle_retained_fraction':float(visible_triplets.sum()/common_angle.sum()),
                'full_2d_knee_mae_deg':avg(np.abs(full-truth)[visible_triplets]),
                'reference_hip_substitution_2d_knee_mae_deg':avg(np.abs(oracle_angle-truth)[visible_triplets])},
            'both_views_ankles_inside':{'frames':int(both_views_ankles_inside.sum()),
                'angle_samples':int(restricted_angle.sum()),
                'full_2d_knee_mae_deg':avg(np.abs(full-truth)[restricted_angle]),
                'reference_hip_substitution_2d_knee_mae_deg':avg(np.abs(oracle_angle-truth)[restricted_angle])},
            'joint_bias':joint_bias,'prediction_sha256':row['prediction_sha256'],
            'comparison_sha256':digest(folder/f'{camera}_800_comparison.npz')})
    print('Audited subject',subject,flush=True)

numeric=['hip_mean_distance_px','hip_rmse_px','hip_mean_delta_y_px','bias_energy_fraction','midpoint_energy_fraction',
         'along_reference_trunk_down_px','full_2d_knee_mae_deg','hip_only_2d_knee_mae_deg',
         'reference_hip_substitution_2d_knee_mae_deg','unchanged_full_3d_knee_mae_deg',
         'model_3d_hipwidth_thigh_ratio','reference_3d_hipwidth_thigh_ratio']
summary={}
for variant in ['full','heavy']:
    subset=[r for r in results if r['variant']==variant]
    summary[variant]={'runs':len(subset),'people':len(people),
        'equal_person_view_means':{key:avg([r[key] for r in subset]) for key in numeric},
        'mean_delta_y_positive_runs':sum(r['hip_mean_delta_y_px']>0 for r in subset),
        'per_camera':{cam:{key:avg([r[key] for r in subset if r['camera']==cam]) for key in numeric} for cam in ['PG1','PG2']},
        'joint_control_mean_delta_y_px':{name:avg([r['joint_bias'][name]['mean_xy_px'][1] for r in subset]) for name in NAMES}}
    for restricted in ['visibility_05','both_views_ankles_inside']:
        summary[variant][restricted]={key:avg([r[restricted][key] for r in subset]) for key in subset[0][restricted]}
report={'plan':plan,'summary':summary,'bell_compatibility':bell,'metadata':metadata,'results':results,
        'conclusion_boundary':'Systematic discrepancy and hip-location sensitivity can be quantified. Causal percentage due to anatomical definition mismatch is not identifiable from these data/public definitions.'}
dump(OUT/'analysis.json',report)
print(json.dumps(summary,indent=2))
