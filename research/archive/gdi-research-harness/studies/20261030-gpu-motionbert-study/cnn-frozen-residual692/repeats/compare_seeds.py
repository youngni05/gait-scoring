"""Compare all three completed seeds without selecting a seed or ensembling predictions."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    with np.load(SOURCE/'bundle/data.npz',allow_pickle=False) as z:
        truth={k:z[k].copy() for k in ('y','person_id','video_id','fold')}
    y=truth['y'];ids=truth['person_id'];people=np.unique(ids)
    def person_mean(v):return np.array([v[ids==q].mean() for q in people])
    def load_predictions(path):
        with np.load(path,allow_pickle=False) as z:
            for k in truth:np.testing.assert_array_equal(z[k],truth[k])
            p=z['predictions'].copy()
        assert p.shape==y.shape and np.isfinite(p).all()
        return p
    arrays={};provenance={}
    for seed in (20261028,20261029,20261030):
        root=SOURCE/'analysis' if seed==20261028 else HERE/('seed%d'%seed)/'analysis'
        proof=read(root/'verification.json');assert proof['passed'] and proof['records_reinferred']==1061
        p=root/'residual_oof.npz';arrays[str(seed)]=load_predictions(p)
        provenance[str(seed)]=dict(archive_sha256=proof['archive_sha256'],oof_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cpu_replay_max_error=proof['cpu_replay_max_error'])
    assert read(SOURCE/'cpu-output-control/audit.json')['passed']
    arrays['cpu_output_control']=load_predictions(SOURCE/'cpu-output-control/oof_predictions.npz')
    arrays['single_cnn']=np.load(SOURCE/'bundle/single_oof.npy',allow_pickle=False)
    metrics={};errors={}
    for name,p in arrays.items():
        errors[name]=person_mean(np.abs(p-y))
        bins=np.abs(np.floor(p/10)-np.floor(y/10))
        metrics[name]=dict(mae=float(errors[name].mean()),exact_10point_bin_rate=float(person_mean(bins==0).mean()),two_or_more_bins_rate=float(person_mean(bins>=2).mean()))
    seeds=['20261028','20261029','20261030'];maes=np.array([metrics[s]['mae'] for s in seeds])
    draws=np.random.default_rng(20261029).integers(0,len(people),(10000,len(people)))
    mean_errors=np.stack([errors[s] for s in seeds]).mean(axis=0)
    comparisons={}
    for name in ('single_cnn','cpu_output_control'):
        delta=errors[name]-mean_errors
        comparisons[name]=dict(mean_mae_reduction=float(delta.mean()),relative_reduction_pct=float(100*delta.mean()/errors[name].mean()),
            all_three_seeds_have_lower_mae=bool(all(metrics[s]['mae']<metrics[name]['mae'] for s in seeds)),
            exploratory95_fixed_predictions_reduction=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist())
    result=dict(people=len(people),records=len(y),metrics=metrics,
        seed_summary=dict(count=3,mean_mae=float(maes.mean()),sample_std_mae=float(maes.std(ddof=1)),minimum_mae=float(maes.min()),maximum_mae=float(maes.max()),range_mae=float(np.ptp(maes))),
        comparisons=comparisons,provenance=provenance,
        note='Average of three separately evaluated MAEs, not an ensemble. Same people and splits; seeds alter correction initialization, minibatch order and dropout only. Fixed-prediction person bootstrap excludes retraining and repeated-experiment selection uncertainty. No protected final data used.')
    (HERE/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
