"""Audit stored fields and code provenance without altering any reference or prediction."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from run_ready_subjects import ROOT,OUT as BASE,np,loadmat,digest,NAMES
HERE=Path(__file__).resolve().parent;OUT=ROOT/'experiments'/'hip-definition'
previous=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
manifest=json.loads((HERE/'public-source'/'manifest.json').read_text(encoding='utf-8'))
inputs_path=HERE/'public-source'/'tutorial-inputs-manifest.json'
supplementary=json.loads(inputs_path.read_text(encoding='utf-8')) if inputs_path.exists() else None
def stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return {'mean':float(x.mean()),'max':float(x.max()),'n':int(x.size)} if x.size else None
rows=[]
for subject in previous['plan']['people']:
    path=ROOT/'reference'/f'F_v3d_Subject_{subject}.mat'
    loaded=loadmat(path,simplify_cells=True);record=loaded[f'Subject_{subject}_F'];m=record['move']
    assert digest(path)==next(r['file_sha256'] for r in previous['metadata'] if r['subject']==subject)
    names=[n if isinstance(n,str) else '' for n in m['markerName']]
    with np.load(BASE/f'subject-{subject:02d}'/'full'/'PG1_800_comparison.npz') as z:frames=z['frames']
    f=(frames-1)*4;pi=[names.index(n) for n in ['LFWT','RFWT','LBWT','RBWT']]
    hi=[names.index(n) for n in ['LHIP','RHIP']];root=list(m['segmentName']).index('RPV')
    marker=m['markerLocation'][f][:,pi];hip=m['markerLocation'][f][:,hi]
    T=m['jointsAffine_v3d'][f,root];R=T[:,:3,:3];origin=T[:,:3,3]
    valid=~m['markerGaps'][f][:,pi+hi].astype(bool).any(axis=1)
    valid &= ~m['jointsGaps_v3d'][f,root].astype(bool)
    w=np.linalg.norm(hip[:,0]-hip[:,1],axis=-1)/.72
    expected=np.array([[-.36,-.19,-.3],[.36,-.19,-.3]])[None,:,:]*w[:,None,None]
    local=np.einsum('nhi,nji->nhj',hip-origin[:,None,:],R)
    coefficients=local/w[:,None,None]
    vnames=[names[int(i)-1] for i in m['virtualMarkers']]
    raw=m['virtualMarkerLocation'][f][:,[vnames.index(n) for n in NAMES]].copy()
    gap=m['markerGaps'][f][:,[names.index(n) for n in NAMES]].astype(bool)
    raw[gap]=np.nan
    for camera in ['PG1','PG2']:
        with np.load(BASE/f'subject-{subject:02d}'/'full'/f'{camera}_800_comparison.npz') as z:
            assert np.array_equal(z['reference_3d'],raw,equal_nan=True)
    virtual_pelvis=m['markerLocation'][f,names.index('PELVIS')]
    hip_mid=m['markerLocation'][f,names.index('MHIP')]
    lc=m['lcData'][pi];lc_usable=bool(np.isfinite(lc).all() and np.linalg.norm(lc[0]-lc[1])>0)
    calibration=None
    if lc_usable:
        # Infer the translation consistent with the observed four-marker centroid,
        # stored local marker centroid, and stored rotation. This is a field
        # consistency diagnostic, not a recovery of the author's processing code.
        inferred=marker.mean(axis=1)-np.einsum('i,nij->nj',lc.mean(axis=0),R)
        inferred_local=np.einsum('nhi,nji->nhj',hip-inferred[:,None,:],R)
        direct=marker.mean(axis=1)[:,None,:]+np.einsum('nhi,nij->nhj',expected-lc.mean(axis=0)[None,None,:],R)
        # Forward reconstruction checks the inverse-coordinate calculation.
        inverse_residual=np.linalg.norm(inferred_local-expected,axis=-1)
        direct_residual=np.linalg.norm(direct-hip,axis=-1)
        assert np.max(np.abs(inverse_residual[valid]-direct_residual[valid]))<1e-3
        world_offset=(origin-inferred)/w[:,None]
        calibration={'hip_formula_residual_relative_width':stats((np.linalg.norm(inferred_local-expected,axis=-1)/w[:,None])[valid]),
          'stored_vs_inferred_origin_relative_width':stats((np.linalg.norm(origin-inferred,axis=-1)/w)[valid]),
          'stored_minus_inferred_world_offset_mean_relative_width':np.mean(world_offset[valid],axis=0).tolist(),
          'stored_minus_inferred_world_offset_sd_relative_width':np.std(world_offset[valid],axis=0).tolist(),
          'forward_inverse_residual_agreement_native_units':stats(np.abs(inverse_residual-direct_residual)[valid]),
          'inferred_vs_mid_asis_relative_width':stats((np.linalg.norm(inferred-marker[:,:2].mean(axis=1),axis=-1)/w)[valid])}
    matrix_checks={}
    for segment,hip_name in [('LTH','LHIP'),('RTH','RHIP')]:
        si=list(m['segmentName']).index(segment);hj=names.index(hip_name)
        translation=m['jointsTranslation_v3d'][f,si];p=m['markerLocation'][f,hj]
        good=valid&~m['jointsGaps_v3d'][f,si].astype(bool)
        row=np.einsum('ni,nij->nj',translation,R)+origin
        column=np.einsum('nij,nj->ni',R,translation)+origin
        matrix_checks[hip_name]={'row_rotation_joint_origin_difference_relative_width':stats((np.linalg.norm(row-p,axis=-1)/w)[good]),
                                'column_rotation_joint_origin_difference_relative_width':stats((np.linalg.norm(column-p,axis=-1)/w)[good])}
    rows.append({'subject':subject,'reference_sha256':digest(path),'valid_frames':int(valid.sum()),
      'stored_pelvis_type':str(m['segmentType'][root]),'subject_field_names':list(record['subject']),
      'move_field_names':list(m),'global_reference_read_is_exact':True,
      'pelvis_is_four_marker_mean_relative_width':stats((np.linalg.norm(virtual_pelvis-marker.mean(axis=1),axis=-1)/w)[valid]),
      'mhip_is_bilateral_hip_mean_relative_width':stats((np.linalg.norm(hip_mid-hip.mean(axis=1),axis=-1)/w)[valid]),
      'affine_translation_vs_translation_field':stats(np.abs(m['jointsAffine_v3d'][f,:,:3,3]-m['jointsTranslation_v3d'][f])),
      'rotation_orthogonality_residual':stats(np.abs(R@np.transpose(R,(0,2,1))-np.eye(3))),
      'stored_frame_mean_hip_coefficients':np.mean(coefficients[valid],axis=0).tolist(),
      'stored_frame_hip_coefficient_sd':np.std(coefficients[valid],axis=0).tolist(),
      'lcdata_four_pelvic_markers_usable':lc_usable,'calibration_centroid_diagnostic':calibration,
      'simple_matrix_interpretation_not_original_fk':matrix_checks})
    print('Audited exporter fields',subject,flush=True)
summary={'people':len(rows),'exact_global_reference_reads':sum(r['global_reference_read_is_exact'] for r in rows),
 'lcdata_unusable_subjects':[r['subject'] for r in rows if not r['lcdata_four_pelvic_markers_usable']],
 'calibration_centroid_formula_mean_residual_by_subject':{str(r['subject']):r['calibration_centroid_diagnostic']['hip_formula_residual_relative_width']['mean'] for r in rows if r['calibration_centroid_diagnostic']},
 'source_commit':manifest['repo_commit'],'source_text_files_inspected':len(manifest['files']),
 'no_original_visual3d_model_or_exporter_identified':True,
 'tutorial_inputs_inspected':bool(supplementary),
 'source_scope_limit':'11 small text source files plus complete input cells of the Python notebook and MATLAB Live Script XML; embedded media outputs were discarded. Current recursive tree, branch list, 59 commit messages and public issues inspected, not every historical blob.',
 'no_reference_or_model_output_changed':True}
evidence={'summary':summary,'rows':rows,'source_manifest':manifest,'supplementary_tutorial_inputs':supplementary,
 'public_sources':{'tutorial':'https://github.com/saeed1262/MoVi-Toolbox/blob/'+manifest['repo_commit']+'/MoCap/tutorial_MATLAB.m',
  'projector':'https://github.com/saeed1262/MoVi-Toolbox/blob/'+manifest['repo_commit']+'/MoCap/worldToVideo.m',
  'amass_converter':'https://github.com/saeed1262/MoVi-Toolbox/blob/'+manifest['repo_commit']+'/MoCap/utils.py',
  'author_v3d_fk_reply':'https://github.com/saeed1262/MoVi-Toolbox/issues/6',
  'paper':'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0253157'},
 'limits':['Effective width inferred from supplied hip span is a consistency check, not independent anatomical ground truth.',
           'Centroid-based translation consistency does not establish which original Visual3D frame/export step produced the difference.',
           'No exact MediaPipe-to-MoVi anatomical mapping is established.']}
assert digest(BASE/'results.json')==previous['plan']['source_results_sha256']
(OUT/'export-fields-audit.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
