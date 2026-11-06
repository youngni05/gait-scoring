"""Sensitivity to a documented CODA construction, never a replacement gold standard."""
import io
import json
import sys
import tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from run_ready_subjects import ROOT,OUT as BASE,np,loadmat,digest,angles,project,NAMES
OUT=ROOT/'experiments'/'hip-definition'
def mean(x):
    x=np.asarray(x);v=x[np.isfinite(x)]
    return float(v.mean()) if v.size else None
def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
original=json.loads((OUT/'analysis.json').read_text())
plan={'source_analysis_sha256':digest(OUT/'analysis.json'),
 'purpose':'Audit pelvic-frame origin and report reference-definition sensitivity; do not select whichever reference gives smaller errors.',
 'documented_formula':'CODA: origin=mid-ASIS; x toward right ASIS; z normal to ASIS/PSIS plane; y=z cross x; HJC=(+/-0.36W,-0.19W,-0.30W).',
 'width':'W=per-person median observed LFWT/RFWT distance over valid walking frames. Static lcData has unusable zero width for subject 49, so the same observed-marker procedure is used for every person. The original model metric is not recovered. No parameter is optimized against predictions.',
 'marker_radius':'Unknown: no anatomical marker-radius correction is made. This is explicitly an approximate reconstruction.',
 'missing_original_details':['Visual3D model file and its subject-specific ASIS_Distance override',
                             'Marker radius / anatomical offset used during original processing',
                             'Complete exporter mapping from Visual3D segment frames to MAT joint transforms'],
 'second_counterfactual':'Translate the supplied hips by mid-ASIS minus stored pelvis-transform translation; leave their spacing/orientation unchanged.',
 'no_new_gold_standard':True,'no_predictions_modified':True,
 'sources':{'coda':'https://www.has-motion.com/wiki/doku.php?id=visual3d:documentation:modeling:segments:coda_pelvis',
            'hip_formula':original['plan']['sources']['visual3d'],'movi':original['plan']['sources']['movi']}}
dump(OUT/'reference-origin-plan.json',plan)
with tarfile.open(ROOT/'raw'/'Camera Parameters.tar') as tar:
    def get(name):
        with np.load(io.BytesIO(tar.extractfile('Calib/'+name+'.npz').read())) as z:return dict(z)
    calibration={c:(get('cameraParams_'+c),get('Extrinsics_'+c)) for c in ['PG1','PG2']}
records=[];audits=[]
for subject in original['plan']['people']:
    m=loadmat(ROOT/'reference'/f'F_v3d_Subject_{subject}.mat',simplify_cells=True)[f'Subject_{subject}_F']['move']
    names=[n if isinstance(n,str) else '' for n in m['markerName']]
    with np.load(BASE/f'subject-{subject:02d}'/'full'/'PG1_800_comparison.npz') as z:frames=z['frames']
    indices=(frames-1)*4;allpoints=m['markerLocation'][indices]
    assert np.allclose(allpoints[:,np.asarray(m['virtualMarkers'],dtype=int)-1],m['virtualMarkerLocation'][indices],equal_nan=True)
    pi=[names.index(n) for n in ['LFWT','RFWT','LBWT','RBWT']]
    markers=allpoints[:,pi];good=~m['markerGaps'][indices][:,pi].astype(bool).any(axis=1)
    anterior=markers[:,:2].mean(axis=1);posterior=markers[:,2:].mean(axis=1)
    x=unit(markers[:,1]-markers[:,0]);z=unit(np.cross(x,anterior-posterior));y=np.cross(z,x)
    assert np.allclose(np.sum(x*y,axis=1),0,atol=1e-10)
    width=float(np.median(np.linalg.norm(markers[:,0]-markers[:,1],axis=1)[good]))
    assert np.isfinite(width) and width>0
    candidate=np.stack([anterior+sign*.36*width*x-.19*width*y-.3*width*z for sign in [-1,1]],axis=1)
    candidate[~good]=np.nan
    root=list(m['segmentName']).index('RPV');T=m['jointsAffine_v3d'][indices,root]
    root_good=good&~m['jointsGaps_v3d'][indices,root].astype(bool)
    hips=allpoints[:,[names.index('LHIP'),names.index('RHIP')]]
    relocated=hips+(anterior-T[:,:3,3])[:,None,:];relocated[~root_good]=np.nan
    virtual_pelvis=allpoints[:,names.index('PELVIS')]
    audits.append({'subject':subject,'valid_marker_frames':int(good.sum()),
        'pelvis_virtual_vs_four_marker_mean_relative_width':mean(np.linalg.norm(virtual_pelvis-markers.mean(axis=1),axis=1)[good]/width),
        'stored_pelvis_origin_vs_mid_asis_relative_width':mean(np.linalg.norm(T[:,:3,3]-anterior,axis=1)[root_good]/width),
        'stored_origin_vs_four_marker_mean_relative_width':mean(np.linalg.norm(T[:,:3,3]-markers.mean(axis=1),axis=1)[root_good]/width),
        'provided_vs_document_coda_hips_relative_width':mean(np.linalg.norm(hips-candidate,axis=-1)[good]/width),
        'candidate_vs_origin_relocation_relative_width':mean(np.linalg.norm(relocated-candidate,axis=-1)/width)})
    for camera in ['PG1','PG2']:
        projected=np.full(candidate.shape[:-1]+(2,),np.nan)
        finite_candidate=np.isfinite(candidate).all(axis=-1)
        projected[finite_candidate]=project(candidate[finite_candidate],*calibration[camera])
        for variant in ['full','heavy']:
            path=BASE/f'subject-{subject:02d}'/variant/f'{camera}_800_comparison.npz'
            with np.load(path) as z:d=dict(z)
            old3=d['reference_3d'];new3=old3.copy();new3[:,[6,7]]=candidate
            shifted=old3.copy();shifted[:,[6,7]]=relocated
            old_angle=angles(old3);new_angle=angles(new3);shift_angle=angles(shifted)
            model3=d['predicted_3d_bend_deg']
            common=np.isfinite(old_angle)&np.isfinite(new_angle)&np.isfinite(model3)
            root_common=common&np.isfinite(shift_angle)
            old2=d['reference_xy'];new2=old2.copy();new2[:,[6,7]]=projected
            model2=d['predicted_xy'];new2angle=angles(new2);old2angle=angles(old2);pred2angle=angles(model2)
            common2=np.isfinite(new2angle)&np.isfinite(old2angle)&np.isfinite(pred2angle)
            old_hip=old2[:,[6,7]];pred_hip=model2[:,[6,7]]
            hip_common=np.isfinite(projected).all(axis=-1)&np.isfinite(old_hip).all(axis=-1)&np.isfinite(pred_hip).all(axis=-1)
            records.append({'subject':subject,'camera':camera,'variant':variant,
                'angle_samples':int(common.sum()),
                'provided_reference_3d_mae_common_deg':mean(np.abs(model3-old_angle)[common]),
                'document_coda_reference_3d_mae_common_deg':mean(np.abs(model3-new_angle)[common]),
                'reference_3d_angle_change_deg':mean(np.abs(new_angle-old_angle)[common]),
                'provided_reference_3d_mae_root_common_deg':mean(np.abs(model3-old_angle)[root_common]),
                'origin_relocation_reference_3d_mae_common_deg':mean(np.abs(model3-shift_angle)[root_common]),
                'provided_reference_2d_mae_common_deg':mean(np.abs(pred2angle-old2angle)[common2]),
                'document_coda_reference_2d_mae_common_deg':mean(np.abs(pred2angle-new2angle)[common2]),
                'provided_hip_distance_common_px':mean(np.linalg.norm(pred_hip-old_hip,axis=-1)[hip_common]),
                'document_coda_hip_distance_common_px':mean(np.linalg.norm(pred_hip-projected,axis=-1)[hip_common])})
    print('Origin audit',subject,flush=True)
keys=[k for k in records[0] if k not in ['subject','camera','variant','angle_samples']]
summary={}
for variant in ['full','heavy']:
    rr=[r for r in records if r['variant']==variant]
    summary[variant]={'people':30,'runs':60,'angle_samples':sum(r['angle_samples'] for r in rr),
        'equal_person_view_means':{k:mean([r[k] for r in rr]) for k in keys},
        'coda_reference_3d_error_smaller_runs':sum(r['document_coda_reference_3d_mae_common_deg']<r['provided_reference_3d_mae_common_deg'] for r in rr),
        'coda_reference_hip_distance_smaller_runs':sum(r['document_coda_hip_distance_common_px']<r['provided_hip_distance_common_px'] for r in rr),
        'per_camera':{c:{k:mean([r[k] for r in rr if r['camera']==c]) for k in keys} for c in ['PG1','PG2']}}
output={'plan':plan,'origin_audits':audits,'results':records,'summary':summary,
        'interpretation':'Observable reference-construction sensitivity. Neither the alternative construction nor smaller error proves anatomical correctness, and none of this is an evaluated deployable improvement.'}
dump(OUT/'reference-origin-audit.json',output)
print(json.dumps(summary,indent=2))
