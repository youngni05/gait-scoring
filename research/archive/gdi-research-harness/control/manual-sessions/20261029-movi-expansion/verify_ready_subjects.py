"""Recompute saved comparisons independently; does not certify clinical accuracy."""
import argparse
import json
import math
import time
from pathlib import Path
from run_ready_subjects import ROOT, OUT, PILOT, NAMES, MP, digest, np, loadmat, cv2


def independent_bend(points):
    result=[]
    for frame in points:
        row=[]
        for side in 'LR':
            hip,knee,ankle=frame[[NAMES.index(side+n) for n in ['HIP','KJC','AJC']]]
            a,b=hip-knee,ankle-knee
            if not np.isfinite([a,b]).all() or np.linalg.norm(a)*np.linalg.norm(b)<1e-10:
                row.append(float('nan'));continue
            cross=abs(a[0]*b[1]-a[1]*b[0]) if len(a)==2 else np.linalg.norm(np.cross(a,b))
            row.append(180-math.degrees(math.atan2(float(cross),float(np.dot(a,b)))))
        result.append(row)
    return np.array(result)


parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=OUT/'results.json')
parser.add_argument('--report',type=Path,default=OUT/'verification.json')
args=parser.parse_args()
rows=json.loads(args.results.read_text(encoding='utf-8'))['results'];checks=[]
for row in rows:
    folder=OUT/f"subject-{row['subject']:02d}"/row['variant'];name=f"{row['camera']}_{row['input_width']}"
    prediction=folder/(name+'.npz')
    assert digest(prediction)==row['prediction_sha256']
    refpath=Path(row['reference']['file']);assert digest(refpath)==row['reference']['sha256']
    move=loadmat(refpath,simplify_cells=True)[f"Subject_{row['subject']}_F"]['move']
    interval=move['flags30'][list(move['motions_list']).index('walking')].astype(int)
    with np.load(prediction) as z, np.load(folder/(name+'_comparison.npz')) as c:
        assert np.array_equal(z['video_frame_one_based'][30:],c['frames'])
        assert np.array_equal(c['frames'],np.arange(interval[0],interval[1]+1))
        virtual_ids=move['virtualMarkers'].astype(int)-1
        reference_names=[str(move['markerName'][i]) for i in virtual_ids]
        selected=[reference_names.index(n) for n in NAMES]
        mocap_indices=(c['frames']-1)*4
        expected_gaps=move['markerGaps'][mocap_indices][:,virtual_ids[selected]].astype(bool)
        expected_xyz=move['virtualMarkerLocation'][mocap_indices][:,selected].copy()
        expected_xyz[expected_gaps]=np.nan
        assert np.array_equal(c['reference_gap'],expected_gaps)
        assert np.array_equal(c['reference_3d'],expected_xyz,equal_nan=True)
        assert z['landmarks'].shape==(len(c['frames'])+30,33,5)
        assert row['evaluation_detected']==int(np.sum(z['pose_counts'][30:]>0))
        assert np.array_equal(c['predicted_xy'],z['landmarks'][30:,MP,:2]*[800,600],equal_nan=True)
        distance=np.sqrt(np.sum((c['predicted_xy']-c['reference_xy'])**2,axis=-1))
        assert np.allclose(distance,c['discrepancy_px'],equal_nan=True)
        assert abs(float(np.nanmean(distance))-row['position_discrepancy_px']['mean'])<1e-10
        assert int(np.isfinite(distance).sum())==row['position_discrepancy_px']['n']
        truth=independent_bend(c['reference_3d']);pred=independent_bend(z['world_landmarks'][30:,MP,:3])
        assert np.allclose(truth,c['reference_3d_bend_deg'],atol=1e-6,equal_nan=True)
        assert np.allclose(pred,c['predicted_3d_bend_deg'],atol=1e-6,equal_nan=True)
        assert abs(float(np.nanmean(abs(pred-truth)))-row['knee_abs_angle_difference_deg']['pred3d_vs_ref3d']['mean'])<1e-6
        truth2=independent_bend(c['reference_xy']);pred2=independent_bend(c['predicted_xy'])
        assert np.allclose(truth2,c['reference_2d_bend_deg'],atol=1e-6,equal_nan=True)
        assert np.allclose(pred2,c['predicted_2d_bend_deg'],atol=1e-6,equal_nan=True)
        for key,difference in [('pred2d_vs_ref2d',abs(pred2-truth2)),('pred2d_vs_ref3d',abs(pred2-truth)),('ref2d_vs_ref3d_projection_difference',abs(truth2-truth))]:
            assert abs(float(np.nanmean(difference))-row['knee_abs_angle_difference_deg'][key]['mean'])<1e-6
            assert int(np.isfinite(difference).sum())==row['knee_abs_angle_difference_deg'][key]['n']
        if row['subject']==1:
            previous=json.loads((PILOT/('baseline' if row['variant']=='full' else 'heavy')/'metrics.json').read_text())['results']
            previous=next(r for r in previous if r['camera']==row['camera'] and r['input_width']==row['input_width'])
            assert row['position_discrepancy_px']==previous['position_discrepancy_px']
            assert row['knee_abs_angle_difference_deg']==previous['knee_abs_angle_difference_deg']
    checks.append({'subject':row['subject'],'camera':row['camera'],'variant':row['variant'],'width':row['input_width'],'saved_arrays_verified':True})
for path in OUT.glob('subject-*/*/*_review.jpg'):assert cv2.imread(str(path)) is not None
report={'status':'saved_outputs_verified','checked_unix':time.time(),'checks':checks,
        'results_file':str(args.results),'results_sha256':digest(args.results),
        'new_inference_conditions':sum(not r['reused_pilot'] for r in rows),'reused_pilot_conditions':sum(r['reused_pilot'] for r in rows),
        'unique_people':len(set(r['subject'] for r in rows)),
        'visual_review':json.loads((OUT/'visual-review.json').read_text()) if (OUT/'visual-review.json').exists() else {'subject13_PG1':'Previously inspected; coarse alignment plausible, no manual pixel truth or turn labels.'},
        'limits':['Reference archive checksum pending where provenance says provisional','Arithmetic and saved-output verification is not clinical accuracy certification']}
args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
