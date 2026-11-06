"""Person-equal interpretation of existing out-of-fold clinical predictions."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'feature-analysis'));import study
import numpy as np

def main():
    z=np.load(BASE/'data.npz');y=z['y'];people=z['person'];fold=z['fold'];unique=np.unique(people)
    assert len(unique)==110 and len(y)==2953
    allpred={m:[] for m in ['median','mode','rf','cnn','motion128']};records=[]
    for seed in study.SEEDS:
        pred={m:np.full(len(y),np.nan) for m in allpred}
        for f in range(6):
            p=np.load(BASE/'received/runs/main'/('fold%d'%f)/('seed%d'%seed)/'predictions.npz');i=p['indices'];assert np.array_equal(i,np.flatnonzero(fold==f))
            for m in ['median','mode','rf','cnn']:pred[m][i]=p[m]
            a=np.load(BASE/'feature-analysis/fits/motion3d'/('fold%d_seed%d.npz'%(f,seed)));assert np.array_equal(i,a['indices']);pred['motion128'][i]=a['prediction']
        for model,p in pred.items():
            assert np.isfinite(p).all();allpred[model].append(p)
            rounded=np.clip(np.floor(p+.5),0,3)
            for name,mask in [('all',np.ones(len(y),bool)),('high',y>=2),('score3',y==3)]:
                w=study.weights(people[mask]);e=p[mask]-y[mask]
                row=dict(model=model,seed=seed,group=name,MAE=float(np.average(abs(e),weights=w)),rounded_accuracy=float(np.average(rounded[mask]==y[mask],weights=w)),within_one_continuous=float(np.average(abs(e)<1,weights=w)),bias=float(np.average(e,weights=w)))
                records.append(row)
    # Average each person's errors across walks and seeds; resample people, not walks/seeds.
    error={m:np.array([np.abs(np.array(ps)[:,people==person]-y[people==person]).mean() for person in unique]) for m,ps in allpred.items()}
    rng=np.random.RandomState(20261031);boots=rng.randint(0,110,size=(10000,110));paired=[]
    for other in ['median','rf','cnn']:
        d=error['motion128']-error[other];paired.append(dict(comparison='motion128 minus '+other,delta_MAE=float(d.mean()),person_bootstrap_95CI=np.quantile(d[boots].mean(1),[.025,.975]).tolist()))
    aggregate=[]
    for model in allpred:
        for group in ['all','high','score3']:
            rows=[r for r in records if r['model']==model and r['group']==group]
            aggregate.append(dict(model=model,group=group,**{k:float(np.mean([r[k] for r in rows])) for k in ['MAE','rounded_accuracy','within_one_continuous','bias']}))
    result=dict(passed=True,people=110,walks=2953,seeds=5,aggregate=aggregate,seed_metrics=records,paired=paired,scope='Existing person-held-out CARE-PD scores, not new MoVi video clinical accuracy. Rounded ordinal classes are descriptive; no clinical decision threshold validated.')
    (HERE/'existing-prediction-interpretation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(aggregate=aggregate,paired=paired),indent=2))

if __name__=='__main__':main()
