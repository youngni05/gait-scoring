"""Independent arithmetic and source invariants for cause diagnostics."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
dev=STUDY.parent/'20261030-gpu-motionbert-study/cnn-multiwindow695';source=rows(dev/'received/results/predictions.csv');summary=json.loads((HERE/'development.json').read_text());binerrors=[]
for b in summary['bins']:
    selected=[r for r in source if b['lower']<=float(r['actual_gdi'])<b['upper']]
    assert len(selected)==b['side_scores']
    err=abs(sum(float(r['predicted_gdi']) for r in selected)/len(selected)-b['oof_mean']);assert err<1e-4;binerrors.append(err)
probe=np.load(DATA/'cause/perturbations.npz',allow_pickle=False);sensitivity=rows(HERE/'sensitivity_records.csv');lookup={(str(r),str(v),int(s)):i for i,(r,v,s) in enumerate(zip(probe['record_id'],probe['variant'],probe['window_start']))}
maximum=0
for r in sensitivity:
    ref='time_1.0' if r['variant'].startswith('time_') else 'baseline';deltas=[]
    for (key,variant,start),i in lookup.items():
        j=lookup.get((key,ref,start))
        if key==r['record_id'] and variant==r['variant'] and j is not None:deltas.append(float((probe['predictions'][:,i,:]-probe['predictions'][:,j,:]).mean()))
    assert len(deltas)==int(r['common_windows']);error=abs(float(np.mean(deltas))-float(r['delta']));maximum=max(maximum,error);assert error<1e-10
assert not rows(DATA/'cause/highres_prepared/excluded.csv')
with np.load(DATA/'runs/confidence_primary/prepared/prepared.npz',allow_pickle=False) as baseline,np.load(DATA/'cause/highres_prepared/prepared.npz',allow_pickle=False) as high:
    for j,k in enumerate(high['record_id']):
        i=baseline['record_id'].tolist().index(k);bm=baseline['window_record']==i;hm=high['window_record']==j
        for field in ['window_start','frame_first','frame_last']:assert np.array_equal(baseline[field][bm],high[field][hm])
        assert baseline['fps'][i]==high['fps'][j]
run=json.loads((DATA/'cause/highres_predictions/run.json').read_text());assert run['folds']==list(range(5))
assert not run['training_performed'] and not run['calibration_refitted'] and not run['resampling_performed']
highres=json.loads((HERE/'highres_summary.json').read_text());actual=rows(DATA/'cause/highres_predictions/record_predictions.csv')
for r in highres['records']:
    values=[(float(a['left_GDI'])+float(a['right_GDI']))/2 for a in actual if a['record_id']==r['record_id']];assert len(values)==5
    # All four records retain exactly the original eligible window schedule here.
    assert abs(np.mean(values)-r['highres_score'])<1e-8
q0={r['record_id']:r for r in rows(STUDY/'qa/normal_full/quality.csv')+rows(STUDY/'qa/abnormal_full/quality.csv')};q1={r['record_id']:r for r in rows(STUDY/'qa/cause_highres/quality.csv')}
quality=[]
for r in highres['records']:
    k=r['record_id'];quality.append(dict(record_id=k,video_id=r['video_id'],baseline_toe_confidence=float(q0[k]['mean_toe_confidence']),highres_toe_confidence=float(q1[k]['mean_toe_confidence']),baseline_missing_fraction=float(q0[k]['max_required_zero_fraction']),highres_missing_fraction=float(q1[k]['max_required_zero_fraction']),baseline_max_joint_jump=float(q0[k]['max_joint_jump_bbox_height']),highres_max_joint_jump=float(q1[k]['max_joint_jump_bbox_height'])))
freeze=json.loads((STUDY/'primary_prescore_freeze.json').read_text());assert all(hashlib.sha256((STUDY/f).read_bytes()).hexdigest()==h for f,h in freeze['sha256'].items())
bundle=STUDY.parents[2]/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle';manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'));assert all(hashlib.sha256((bundle/f).read_bytes()).hexdigest()==h for f,h in manifest['files'].items())
report=dict(status='passed',development_binned_predictions_vs_original_CSV_max_error=max(binerrors),sensitivity_reaggregation_max_error=maximum,highres_records=4,highres_all_five_folds_and_full_window_means_match=True,original_selection_unchanged=True,bundle_files_unchanged=len(manifest['files']),quality=quality,not_verified=['Anatomical ground-truth joints','Identical physical camera/timebase across development and GAVD','Exclusive causal contribution of model vs pose vs population shift'])
(HERE/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
