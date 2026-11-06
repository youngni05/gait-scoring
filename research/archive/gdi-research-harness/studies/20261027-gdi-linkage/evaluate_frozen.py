"""One fixed-model development transfer check, never a final-test claim."""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from audit_labels import HERE,H,ROOT,load,sha,save

DEV=H/'jobs/development-baseline/runs/20261025T044558572569Z-71adb0'
OLD=H/'control/desktop-sessions/20261026T130506Z-e876e7'
SHORT=H/'jobs/native124-cache-v1'
sys.path.insert(0,str(H))
from weighted_ridge_job import metrics

def main():
    assert not (HERE/'transfer_result.json').exists(), 'Do not repeat finished evaluation'
    prep=load(HERE/'feature_preparation.json')
    assert prep['verified'] and sha(HERE/'development_features.npz')==prep['sha256']
    sealed=load(OLD/'receipt.json')
    assert sealed['status']=='completed' and sha(OLD/'response.json')==sealed['response_sha256']
    hashes={r['path']:r['sha256'] for r in sealed['artifacts']}
    for p in (OLD/'length_models.npz',OLD/'length_comparison.json',SHORT/'development_features124.npz'):
        assert sha(p)==hashes[p.relative_to(H).as_posix()]
    assert sha(DEV/'development_packet.npz')=='6283717872ceaa8177db38ba1dac4a13581ba77ddfd2209e1ef1784c1691e3ec'
    paths=[Path(__file__),HERE/'transfer_plan.md',HERE/'development_features.npz',HERE/'development_linkage.json',
           OLD/'length_models.npz',OLD/'length_comparison.json',OLD/'receipt.json',DEV/'development_packet.npz',
           SHORT/'development_features124.npz',H/'weighted_ridge_job.py']
    before={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    save('transfer_sources.json',{'sha256':before,'new_fits':0,'encoder_fine_tuning':False})
    with np.load(OLD/'length_models.npz',allow_pickle=False) as z:
        params={k:z['short124__'+k].copy() for k in ('scaler_mean','scaler_scale','coef','intercept')}
    def predict(x):return ((x-params['scaler_mean'])/params['scaler_scale'])@params['coef'].T+params['intercept']
    with np.load(DEV/'development_packet.npz',allow_pickle=False) as z:
        old_y=z['Y'].copy();old_people=z['patient_ids'].copy();old_videos=z['video_ids'].copy();split=z['split'].copy()
    with np.load(SHORT/'development_features124.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['patient_ids'],old_people)
        np.testing.assert_array_equal(z['video_ids'],old_videos)
        np.testing.assert_array_equal(z['split'],split)
        old_x=z['X'].copy()
    train,val=split=='train',split=='validation'
    assert (int(train.sum()),int(val.sum()),len(set(old_people[train])),len(set(old_people[val])))==(203,40,175,37)
    np.testing.assert_allclose(params['scaler_mean'],old_x[train].mean(axis=0),rtol=0,atol=1e-12)
    replay=metrics(old_y[val],predict(old_x[val]),old_people[val])[0]
    assert abs(replay['person_mae']-load(OLD/'length_comparison.json')['short124']['person_mae'])<1e-9
    with np.load(HERE/'development_features.npz',allow_pickle=False) as z:
        x=z['X'].copy();y=z['Y'].copy();people=z['patient_ids'].copy();videos=z['video_ids'].copy();sequence=z['sequence'].copy()
    rows=load(HERE/'development_linkage.json')['records']
    assert x.shape==(861,512) and y.shape==(861,2) and len(set(people))==520
    assert all(str(p)==r['patient_id'] and str(v)==r['video_id'] and int(s)==r['sequence']
               for p,v,s,r in zip(people,videos,sequence,rows,strict=True))
    np.testing.assert_array_equal(y,np.asarray([r['GDI_LR'] for r in rows]))
    final_people={r['patient_id'] for r in load(HERE/'final_linkage_presence_only.json')['records']}
    assert len(final_people)==223 and not set(people)&(set(old_people)|final_people)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    baseline=np.broadcast_to(old_y[train].mean(axis=0),y.shape).copy()
    prediction=predict(x)
    mm,em=metrics(y,prediction,people);bm,eb=metrics(y,baseline,people)
    delta=eb-em
    indexes=np.random.default_rng(20261026).integers(0,len(delta),size=(10000,len(delta)))
    ci=np.percentile(delta[indexes].mean(axis=1),[2.5,97.5])
    out=HERE/'development_predictions.csv'
    with out.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['sequence','patient_id','video_id','actual_L','actual_R','frozen124_L','frozen124_R','trainmean_L','trainmean_R'])
        for i in range(len(rows)):writer.writerow([sequence[i],people[i],videos[i],*y[i],*prediction[i],*baseline[i]])
    grouped=defaultdict(lambda:defaultdict(list))
    with out.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            for method in ('frozen124','trainmean'):
                error=statistics.mean(abs(float(r['actual_'+s])-float(r[method+'_'+s])) for s in ('L','R'))
                grouped[method][r['patient_id']].append(error)
    independently={m:statistics.mean(statistics.mean(v) for v in groups.values()) for m,groups in grouped.items()}
    assert abs(independently['frozen124']-mm['person_mae'])<1e-10
    assert abs(independently['trainmean']-bm['person_mae'])<1e-10
    # A different arithmetic order replays stored coefficients without refitting.
    raw_coef=params['coef']/params['scaler_scale'][None,:]
    raw_intercept=params['intercept']-params['scaler_mean']@raw_coef.T
    alternate=x@raw_coef.T+raw_intercept
    np.testing.assert_allclose(prediction,alternate,rtol=0,atol=1e-8)
    assert all(sha(ROOT/p)==v for p,v in before.items())
    result={'verified':True,'scope':'new development520; not final evaluation','frozen124':mm,'old_train_mean_baseline':bm,
            'person_mae_reduction_vs_trainmean':float(delta.mean()),
            'relative_mae_reduction_percent_vs_trainmean':float(100*delta.mean()/bm['person_mae']),
            'people_improved':int((delta>0).sum()),'people_worsened':int((delta<0).sum()),'people_equal':int((delta==0).sum()),
            'paired_bootstrap95':[float(v) for v in ci],'bootstrap_seed':20261026,'bootstrap_resamples':10000,
            'ci_contains_zero':bool(ci[0]<=0<=ci[1]),'old_validation37_replay':replay,
            'new_fits':0,'hyperparameter_search':False,'fine_tuning':False,'new_scaler_fit':False,
            'final223_used_for_feature_extraction_or_evaluation':False,'original_evaluation38_used':False,
            'independent_final_evaluation':False,'independent_CSV_metric_check':True,
            'saved_model_alternate_equation_max_abs_difference':float(np.max(np.abs(prediction-alternate))),
            'source_hashes_unchanged':True,'predictions_sha256':sha(out),
            'limitations':['Same provider, not external or older-adult home-video validation.',
                '124-ready but outside original243-selected people; no paired243-vs124 comparison.',
                'Development data can inform future work; this result does not consume reserved final labels.',
                'Bootstrap is conditional on this frozen model and selected sample; not clinical validation.']}
    save('transfer_result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
