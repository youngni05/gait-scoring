"""Descriptive fixed-prediction contrasts against the existing multiwindow CNN."""
from pathlib import Path
import json,hashlib,datetime,argparse
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent
OLD=S/'20261102-final223-frozen';FOLLOW=S/'20261102-final223-ranksim-followup'
BIG=Path('D:/gait-research-data/D_P');A=BIG/'final223-frozen-20261102/scored.npz';B=BIG/'final223-ranksim-followup-20261102/scored.npz'
NAMES=['CNN_bag','fusion','plus_general','plus_ranksim']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main(stage):
    if stage=='declare':
        assert not (H/'protocol.json').exists()
        protocol=dict(declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),request='Assess whether already frozen methods improve over the existing multiwindow CNN',classification='post-hoc descriptive reanalysis of the already inspected same222people377records; no new model or independent test',reference='CNN_refit: original average of five multiwindow CNN refit predictors, not single-window CNN',comparisons=NAMES,primary_metric='person-equal MAE, mean record-side absolute errors within person then mean people',bootstrap=dict(seed=20261104,draws=10000,unit='paired person',individual95=[2.5,97.5],Bonferroni_familywise95_four_contrasts=[.625,99.375]),decision='Report every contrast; negative difference favors candidate. CI crossing zero means inconclusive, not equivalence. Any nominal interval excluding zero remains post-hoc evidence, not independent confirmation.',no_new_training_selection_predictions_or_exclusions=True)
        save(H/'protocol.json',protocol)
        paths=[A,B,OLD/'results.json',OLD/'protocol.json',FOLLOW/'results.json',FOLLOW/'protocol.json']
        save(H/'freeze.json',dict(protocol_sha256=sha(H/'protocol.json'),code_sha256=sha(__file__),sources={str(p.resolve()):sha(p) for p in paths}))
        print('COMPARISONS_DECLARED');return
    f=read(H/'freeze.json');assert sha(__file__)==f['code_sha256'] and sha(H/'protocol.json')==f['protocol_sha256']
    for p,h in f['sources'].items():assert sha(p)==h
    with np.load(A) as z:y=z['y'];ids=z['person_id'];videos=z['video_id'];reference=z['CNN_refit']
    with np.load(B) as z:
        for k,v in [('y',y),('person_id',ids),('video_id',videos)]:np.testing.assert_array_equal(v,z[k])
        pred={k:z[k] for k in NAMES}
    people=np.unique(ids);assert len(people)==222 and len(y)==377
    errors={k:np.array([abs(p[ids==i]-y[ids==i]).mean() for i in people]) for k,p in {'CNN_refit':reference,**pred}.items()}
    cfg=read(H/'protocol.json')['bootstrap'];rng=np.random.default_rng(cfg['seed']);draw=rng.integers(0,len(people),(cfg['draws'],len(people)))
    contrasts={}
    for k in NAMES:
        delta=errors[k]-errors['CNN_refit'];dist=delta[draw].mean(1)
        contrasts[k]=dict(mae=float(errors[k].mean()),delta=float(delta.mean()),reduction_percent=float(-100*delta.mean()/errors['CNN_refit'].mean()),individual95=np.percentile(dist,[2.5,97.5]).tolist(),familywise95=np.percentile(dist,[.625,99.375]).tolist())
    result=dict(reference_mae=float(errors['CNN_refit'].mean()),contrasts=contrasts,people=222,records=377,post_hoc=True,protocol_sha256=f['protocol_sha256'],sources_unchanged=True)
    if stage=='compare':
        assert not (H/'results.json').exists();save(H/'results.json',result);print(json.dumps(result,indent=2));return
    assert result==read(H/'results.json')
    # Independent scalar loops: no reliance on the grouped MAE or vectorized bootstrap.
    totals={k:[] for k in ['CNN_refit']+NAMES}
    for person in sorted(set(ids)):
        rows=[i for i,v in enumerate(ids) if v==person]
        for k,p in {'CNN_refit':reference,**pred}.items():totals[k].append(sum(abs(float(p[i,j])-float(y[i,j])) for i in rows for j in [0,1])/(len(rows)*2))
    assert abs(sum(totals['CNN_refit'])/222-result['reference_mae'])<1e-12
    for k in NAMES:assert abs(sum(totals[k])/222-contrasts[k]['mae'])<1e-12
    rng=np.random.default_rng(cfg['seed']);dists={k:[] for k in NAMES}
    for _ in range(cfg['draws']):
        ix=rng.integers(0,222,222)
        for k in NAMES:dists[k].append(sum(totals[k][i]-totals['CNN_refit'][i] for i in ix)/222)
    for k in NAMES:
        np.testing.assert_allclose(np.percentile(dists[k],[2.5,97.5]),contrasts[k]['individual95'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.percentile(dists[k],[.625,99.375]),contrasts[k]['familywise95'],atol=1e-12,rtol=0)
    save(H/'verification.json',dict(passed=True,original_artifacts_unchanged=True,independent_person_mae=True,independent_bootstrap=True,results_sha256=sha(H/'results.json')))
    print('INDEPENDENT_VERIFICATION_PASSED')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['declare','compare','verify']);main(a.parse_args().stage)
