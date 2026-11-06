"""Independent reconciliation of saved predictions and person-level paired effects."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'received/runs/main'
    protocol=json.loads((HERE/'protocol.json').read_text())
    manifest=list(csv.DictReader((HERE/'manifest.csv').open()))
    rows=list(csv.DictReader((root/'predictions.csv').open()))
    summary=json.loads((root/'summary.json').read_text())
    replays=[json.loads(p.read_text()) for p in root.glob('replay-*.json')]
    assert {p.name for p in root.glob('replay-*.json')}=={'replay-0-1-2.json','replay-3-4-5.json'}
    assert sum(r['checkpoints'] for r in replays)==30 and all(r['passed'] for r in replays)
    assert sum(r['rf_models'] for r in replays)==30
    assert max(r['max_cnn_error'] for r in replays)<=1e-6
    assert summary['complete'] and len(rows)==2953*5*5
    models=['mean','median','mode','rf','cnn']; seeds=protocol['seeds']
    keyed={}
    for r in rows:
        i=int(r['index']); seed=int(r['seed']); model=r['model']
        assert (seed,model,i) not in keyed
        assert seed in seeds and model in models
        assert r['person']==manifest[i]['person'] and int(r['fold'])==int(manifest[i]['fold'])
        assert float(r['y'])==int(manifest[i]['y'])
        assert np.isfinite(float(r['prediction'])) and 0<=float(r['prediction'])<=3
        keyed[seed,model,i]=r
    audits=[]
    for f in range(6):
        outer={i for i,r in enumerate(manifest) if int(r['fold'])!=f}
        test=set(range(2953))-outer
        for seed in seeds:
            folder=root/('fold%d'%f)/('seed%d'%seed)
            selection=json.loads((folder/'complete.json').read_text())
            assert set(selection['outer_train'])==outer and set(selection['test'])==test
            inner=set(selection['inner_train']); valid=set(selection['validation'])
            assert inner|valid==outer and not inner&valid
            assert valid=={i for i,r in enumerate(manifest) if int(r['fold'])==(f+1)%6}
            groups=[{manifest[i]['person'] for i in ids} for ids in (inner,valid,test)]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            curve=json.loads((folder/'inner/curve.json').read_text())
            epoch=min(curve,key=lambda r:r['validation_MAE'])['epoch']
            assert selection['selected_epoch']==epoch
            assert json.loads((folder/'refit/fit_complete.json').read_text())['epochs']==epoch
            assert selection['rf_leaf']==min(selection['rf_choices'],key=lambda r:r['inner_MAE'])['leaf']
            saved=np.load(folder/'predictions.npz',allow_pickle=False)
            assert set(map(int,saved['indices']))==test and len(saved['indices'])==len(test)
            for model in models:
                for index,value in zip(saved['indices'],saved[model]):
                    assert float(keyed[seed,model,int(index)]['prediction'])==float(value)
            # Reconstruct training-only baseline without importing run.py.
            byperson=defaultdict(list)
            for i in outer: byperson[manifest[i]['person']].append(int(manifest[i]['y']))
            masses=[sum(sum(y==k for y in ys)/len(ys) for ys in byperson.values())/len(byperson) for k in range(4)]
            mean=sum(k*masses[k] for k in range(4)); median=next(k for k in range(4) if sum(masses[:k+1])>=.5)
            mode=int(np.argmax(masses))
            for name,value in [('mean',mean),('median',median),('mode',mode)]:
                assert max(abs(float(keyed[seed,name,i]['prediction'])-value) for i in test)<1e-12
            audits.append(dict(fold=f,seed=seed,epoch=epoch,leaf=selection['rf_leaf']))
    independent=[]; person_errors={}
    for seed in seeds:
        for model in models:
            for group in ['all','high_2_3','score3']:
                byperson=defaultdict(list)
                for i in range(2953):
                    r=keyed[seed,model,i]; y=float(r['y']); p=float(r['prediction'])
                    if group=='high_2_3' and y<2: continue
                    if group=='score3' and y!=3: continue
                    byperson[r['person']].append(p-y)
                ma=statistics.mean(statistics.mean(abs(e) for e in es) for es in byperson.values())
                large=statistics.mean(statistics.mean(abs(e)>=2 for e in es) for es in byperson.values())
                bias=statistics.mean(statistics.mean(es) for es in byperson.values())
                rmse=statistics.mean(statistics.mean(e*e for e in es) for es in byperson.values())**.5
                orig=next(v for v in summary['metrics'] if v['seed']==seed and v['model']==model and v['group']==group)
                assert abs(ma-orig['MAE'])<1e-12 and abs(large-orig['large_error_rate'])<1e-12
                assert abs(bias-orig['bias'])<1e-12 and abs(rmse-orig['RMSE'])<1e-12
                independent.append(dict(seed=seed,model=model,group=group,MAE=ma,RMSE=rmse,large_error_rate=large,bias=bias))
                person_errors[seed,model,group]={p:statistics.mean(abs(e) for e in es) for p,es in byperson.items()}
    aggregate=[]
    for model in models:
        for group in ['all','high_2_3','score3']:
            selected=[r for r in independent if r['model']==model and r['group']==group]
            aggregate.append(dict(model=model,group=group,mae_mean=statistics.mean(r['MAE'] for r in selected),
                mae_seed_sd=statistics.stdev(r['MAE'] for r in selected),
                mae_min=min(r['MAE'] for r in selected),mae_max=max(r['MAE'] for r in selected),
                large_error_rate_mean=statistics.mean(r['large_error_rate'] for r in selected),
                bias_mean=statistics.mean(r['bias'] for r in selected)))
    paired=[]; rng=np.random.default_rng(20261031)
    for group in ['all','high_2_3','score3']:
        people=sorted(person_errors[seeds[0],'cnn',group])
        for a,b in [('rf','median'),('cnn','median'),('cnn','rf')]:
            delta=np.array([statistics.mean(person_errors[s,a,group][p]-person_errors[s,b,group][p] for s in seeds) for p in people])
            boot=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
            wins=sum(statistics.mean(person_errors[s,a,group].values())<statistics.mean(person_errors[s,b,group].values()) for s in seeds)
            paired.append(dict(group=group,comparison=a+' minus '+b,people=len(people),MAE_difference=float(delta.mean()),
                person_bootstrap_95CI=list(map(float,np.quantile(boot,[.025,.975]))),seeds_better=wins))
    result=dict(passed=True,prediction_rows=len(rows),people=110,walks=2953,fold_seed_combinations=len(audits),
        aggregate=aggregate,paired=paired,selections=audits,seed_metrics=independent,replays=replays,
        prediction_sha256=hashlib.sha256((root/'predictions.csv').read_bytes()).hexdigest(),
        limitation='Bootstrap resamples held-out people and averages seed-specific losses, not seed predictions. Training-set overlap uncertainty is not captured.')
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,aggregate=aggregate,paired=paired),indent=2))


if __name__=='__main__':main()
