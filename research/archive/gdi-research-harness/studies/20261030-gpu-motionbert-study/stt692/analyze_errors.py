"""Descriptive held-out error audit. No model fitting, feature selection, or new exclusion."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=HERE/'error-analysis';out.mkdir(exist_ok=True)
    base=STUDY/'cnn-paper-input695'
    source=base/'bundle/data.npz';assert sha(source)==read(base/'bundle/manifest.json')['data.npz']
    with np.load(source,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    y=d['y'];ids=d['person_id'];people=np.unique(ids)
    residual=STUDY/'cnn-frozen-residual692'
    preds={'single_cnn':np.load(residual/'bundle/single_oof.npy',allow_pickle=False)}
    paths={'cpu_control':residual/'cpu-output-control/oof_predictions.npz',
        'mb20261028':residual/'analysis/residual_oof.npz',
        'mb20261029':residual/'repeats/seed20261029/analysis/residual_oof.npz',
        'mb20261030':residual/'repeats/seed20261030/analysis/residual_oof.npz'}
    for name,p in paths.items():
        with np.load(p,allow_pickle=False) as z:
            for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(z[key],d[key])
            preds[name]=z['predictions'].copy()
    missing=[];confidence=[];movement=[];raw_hashes=[]
    for i,original in enumerate(d['original_row_index']):
        path=base/'records'/('%04d.npz'%original);proof=read(path.with_suffix('.json'))
        assert sha(path)==proof['artifact_sha256'] and proof['video_id']==d['video_id'][i]
        with np.load(path,allow_pickle=False) as z:r=z['source_raw_matched'].reshape(124,25,3)
        selected=r[:,[14,13,12,19,11,10,9,22]]
        xy=selected[...,:2];bad=~np.isfinite(xy)|(xy==0)
        miss=float(bad.mean(axis=0).max());assert abs(miss-proof['missing_fraction_max_coordinate'])<1e-12
        visible=(np.isfinite(xy)&(xy!=0)).all(axis=2)&np.isfinite(selected[:,:,2])
        conf=float(selected[:,:,2][visible].mean()) if visible.any() else 0.
        hip=r[:,[12,9],0].copy();hip[(hip==0)|~np.isfinite(hip)]=np.nan
        before=hip[:10][np.isfinite(hip[:10])];after=hip[-10:][np.isfinite(hip[-10:])]
        displacement=float(np.median(after)-np.median(before)) if len(before) and len(after) else None
        missing.append(miss);confidence.append(conf);movement.append(displacement);raw_hashes.append(proof['artifact_sha256'])
    missing=np.array(missing);confidence=np.array(confidence)
    def avg(v):return np.array([v[ids==p].mean() for p in people])
    errors={k:avg(np.abs(v-y).mean(axis=1)) for k,v in preds.items()}
    bias={k:avg((v-y).mean(axis=1)) for k,v in preds.items()}
    # Mean of errors across seeds, not error of an ensemble prediction.
    errors['mb_seed_mean']=np.stack([errors['mb20261028'],errors['mb20261029'],errors['mb20261030']]).mean(0)
    bias['mb_seed_mean']=np.stack([bias['mb20261028'],bias['mb20261029'],bias['mb20261030']]).mean(0)
    true_mean=avg(y.mean(axis=1));asym=avg(np.abs(y[:,0]-y[:,1]));pmiss=avg(missing);pconf=avg(confidence)
    count=np.array([(ids==p).sum() for p in people]);fold=np.array([int(d['fold'][ids==p][0]) for p in people])
    def rows(labels):
        result=[]
        for label in dict.fromkeys(labels):
            mask=np.array(labels)==label
            result.append(dict(group=label,people=int(mask.sum()),records=int(count[mask].sum()),
                mean_true_gdi=float(true_mean[mask].mean()),mae={k:float(v[mask].mean()) for k,v in errors.items()},
                bias={k:float(v[mask].mean()) for k,v in bias.items()}))
        return result
    def bands(v,edges,names):return [names[int(np.searchsorted(edges,a,side='right'))] for a in v]
    report={
        'gdi_mean':rows(bands(true_mean,[60,70,80,90],['below60','60to70','70to80','80to90','90plus'])),
        'true_left_right_difference':rows(bands(asym,[5,10,20],['below5','5to10','10to20','20plus'])),
        'mean_record_max_missing_fraction':rows(bands(pmiss,[1e-12,.1],['zero','over0_below10pct','10pct_plus'])),
        'mean_visible_keypoint_confidence':rows(bands(pconf,[.5,.75],['below0.5','0.5to0.75','0.75plus'])),
        'fold':rows([str(f) for f in fold]),
        'record_count':rows(['one' if n==1 else 'multiple' for n in count])}
    record_direction=['unknown' if a is None else ('positive_x' if a>5 else 'negative_x' if a< -5 else 'small_displacement') for a in movement]
    person_direction=[]
    for p in people:
        labels=set(np.array(record_direction)[ids==p]);person_direction.append(next(iter(labels)) if len(labels)==1 else 'mixed')
    report['screen_hip_displacement_direction']=rows(person_direction)
    for axis,groups in report.items():
        assert sum(g['people'] for g in groups)==692
        for model in errors:
            assert abs(sum(g['people']*g['mae'][model] for g in groups)/692-errors[model].mean())<1e-10
    # Error overlap helps distinguish complementary predictions from a shared difficult cohort.
    corr=float(np.corrcoef(errors['single_cnn'],errors['mb_seed_mean'])[0,1])
    records=dict(person_id=people,y_mean=true_mean,true_asymmetry=asym,mean_missing=pmiss,mean_confidence=pconf,fold=fold,record_count=count,
        **{'error_'+k:v for k,v in errors.items()})
    np.savez_compressed(out/'person_diagnostics.npz',**records)
    result=dict(people=692,records=1061,overall_mae={k:float(v.mean()) for k,v in errors.items()},groups=report,
        person_error_correlation_cnn_vs_mb_seed_mean=corr,
        notes=['Exploratory post-hoc groups, not clinical grades or causal effects.',
        'All group metrics weight each person equally; quality is averaged over that persons records.',
        'Visible-joint confidence excludes missing joints; interpret together with missingness.',
        'Screen x displacement is not camera angle, viewpoint or true anatomical walking direction.',
        'No data exclusions, training weights or model settings were selected from these results.',
        'GDI bins describe mean ground truth per person; no clinical cutoffs were created.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,raw_records_hash_checked=1061,predictions_aligned=True,
        group_partition_totals_checked=True,weighted_group_mae_reconstructs_overall=True,raw_missingness_matches_original_audit=True,
        input_sha256=sha(source),predictions_sha256={k:sha(p) for k,p in paths.items()},actual_training_steps=0),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
