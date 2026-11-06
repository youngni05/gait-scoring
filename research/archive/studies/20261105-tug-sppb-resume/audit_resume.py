"""Verify the saved TUG restart baseline without changing prior experiments."""
from pathlib import Path
import csv,json,hashlib
import numpy as np

ROOT=Path('D:/Projects/D_P')
OUT=Path(__file__).resolve().parent
SRC=ROOT/'studies/20261104-opencap-camera-tug'
DATA=Path('D:/gait-research-data/D_P/opencap-camera-tug-20261104')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def main():
    protocol=read(SRC/'protocol.json');verification=read(SRC/'verification.json');preparation=read(SRC/'preparation.json')
    assert verification['passed']
    checks={
        'features':sha(DATA/'features.npz')==verification['features_sha256']==preparation['features_sha256'],
        'results':sha(SRC/'results.json')==verification['results_sha256'],
        'protocol':sha(SRC/'protocol.json')==verification['protocol_sha256']==preparation['protocol_sha256'],
        'source_labels':sha(protocol['labels'])==protocol['labels_sha256'],
        'person_splits':sha(SRC/'effective-splits.json')==preparation['effective_splits_sha256']}
    assert all(checks.values())
    d=np.load(DATA/'features.npz');rows=list(csv.DictReader((SRC/'person-predictions.csv').open(encoding='utf8')))
    assert len(rows)==len(d['ids'])==297
    assert len(set(r['id'] for r in rows))==297
    np.testing.assert_array_equal(d['ids'],np.array([r['id'] for r in rows]))
    y=np.array([float(r['true_TUG']) for r in rows]);np.testing.assert_array_equal(y,d['y']);assert np.isfinite(y).all() and (y>0).all()
    age=np.array([float(r['age']) for r in rows]);masks={'all':np.ones(len(y),bool),'age65plus':age>=65,'TUG20plus':y>=20}
    metrics={}
    for arm in ['speed_only','speed_dynamics','speed_plus_landmarks']:
        pr=np.array([float(r['r0_'+arm]) for r in rows]);error=pr-y
        metrics[arm]={group:{'people':int(sum(mask)),'MAE_seconds':float(abs(error[mask]).mean()),'within1_second':float(np.mean(abs(error[mask])<=1)),'within2_seconds':float(np.mean(abs(error[mask])<=2)),'bias_seconds':float(error[mask].mean())} for group,mask in masks.items()}
    for arm in metrics:assert abs(metrics[arm]['all']['MAE_seconds']-read(SRC/'results.json')['primary_repeat'][arm]['MAE'])<1e-12
    result={
        'status':'restart_baseline_audited_no_new_training',
        'user_direction':'Return from MMU JH-FRAT to actual TUG seconds and SPPB points; camera-only product scope retained.',
        'primary_next_endpoint':'TUG_seconds',
        'source':'OpenCap same-person same-visit clinical TUG, separate normal walk,297people,age65plus34',
        'input_scope':'3seconds of camera-derived 3D landmarks projected to ideal side view; calibration assumed. Not real single-phone MediaPipe validation.',
        'secondary_endpoint':'SPPB_points,retain OlderAdults44 optical-joint study separately; no pooling with sensor cohorts.',
        'comparison_plan':'Same-person-fold speed baseline / gait-feature model / learned motion model. Select inside training people only. Retain original results and failed trials.',
        'engineering_target':'TUG MAE<=1second, at least80percent within2seconds, at least20percent MAE reduction vs matched speed baseline; inspect older and slow groups separately. Not a clinical MDC.',
        'error_trigger':'An absolute error over1second triggers unit,label,visit,window,camera and model-range diagnosis; it is not itself proof of a software bug.',
        'hash_checks':checks,
        'primary_repeat0_metrics_independently_recomputed':metrics,
        'prior_followup_reference':{'path':str(ROOT/'studies/20261104-opencap-tug-tail/STATUS.txt'),'full_feature_fixed_tree_MAE_seconds':1.7528,'TUG20plus_MAE_seconds':11.5887,'aggregation':'person error averaged across5repeats; different from primary repeat0 above'},
        'sensor_reference':'FRAILPOL0.88054seconds is sensor-input development evidence,notcameraaccuracy. No sensor/phonewearing input added.',
        'MMU_record':'Preserve studies/20261105-mmu-frip-evaluation unchanged; JH-FRAT not current endpoint.',
        'no_new_models_or_external_transfers':True}
    (OUT/'resume-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'hash_checks':checks,'people':len(y),'older65':int(sum(age>=65)),'primary_repeat0':metrics},indent=2))

if __name__=='__main__':main()
