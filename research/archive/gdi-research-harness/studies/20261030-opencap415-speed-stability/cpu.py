"""Paired input ablation with nested tuning and crossed split/model randomness."""
from common import *
import time
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

def fit(name, setting, seed, x, y, tr, te):
    # Preserve the original implementation: explicit uniform person weights.
    w=np.ones(len(tr),dtype=float)
    scale=StandardScaler().fit(x[tr],sample_weight=w)
    model=Ridge(alpha=setting) if name=='Ridge' else RandomForestRegressor(n_estimators=128,min_samples_leaf=setting,random_state=seed,n_jobs=2,max_features=1.)
    model.fit(scale.transform(x[tr]), y[tr],sample_weight=w)
    return model.predict(scale.transform(x[te])), (scale,model)

def main():
    started=time.time();d=load();views=features(d['x']);y=d['y'];rows=[];selections=[]
    plan=json.loads((HERE/'splits.json').read_text())
    for rep in plan:
        repeat=rep['repeat']
        for split in rep['splits']:
            tr=np.array(split['train']);te=np.array(split['test']);fold=split['fold']
            for name,value in [('Mean',float(y[tr].mean())),('Median',lower_median(y[tr]))]:
                rows.extend({'repeat':repeat,'seed':-1,'fold':fold,'view':'none','model':name,'sample':int(k),'id':d['ids'][k],'target':float(y[k]),'prediction':value} for k in te)
            for view,x in views.items():
                for name,grid,seeds in [('Ridge',[1.,100.],[-1]),('RF',[2,5],SEEDS)]:
                    for seed in seeds:
                        innerpred=[];scores=[]
                        for setting in grid:
                            pp=np.full(len(y),np.nan)
                            for item in split['inner']:
                                it=np.array(item['train']);iv=np.array(item['validation'])
                                pp[iv]=fit(name,setting,seed,x,y,it,iv)[0]
                            scores.append(float(np.mean(abs(y[tr]-pp[tr]))));innerpred.append(pp[tr].tolist())
                        best=grid[int(np.argmin(scores))]
                        pp,model=fit(name,best,seed,x,y,tr,te)
                        assert np.isfinite(pp).all()
                        stem=f'r{repeat:02d}/{view}/{name}-s{seed}-f{fold}'
                        out=HERE/'cpu-models'/f'{stem}.joblib';out.parent.mkdir(parents=True,exist_ok=True)
                        joblib.dump(model,out,compress=3)
                        selections.append({'repeat':repeat,'fold':fold,'view':view,'model':name,'seed':seed,'grid':grid,'scores':scores,'selected':best,'inner_train_order':tr.tolist(),'inner_predictions':innerpred,'model_path':out.relative_to(HERE).as_posix()})
                        rows.extend({'repeat':repeat,'seed':seed,'fold':fold,'view':view,'model':name,'sample':int(k),'id':d['ids'][k],'target':float(y[k]),'prediction':float(v)} for k,v in zip(te,pp))
            pd.DataFrame(rows).to_csv(HERE/'cpu-predictions.csv',index=False)
            save('cpu-selections.json',selections)
            print(f'CPU_REPEAT {repeat} FOLD {fold} COMPLETE',flush=True)
    save('cpu-completion.json',{'completed':True,'elapsed_seconds':time.time()-started,'models':len(selections),'input_sha256':sha(HERE/'input.npz'),'splits_sha256':sha(HERE/'splits.json'),'code_sha256':sha(HERE/'cpu.py')})
    print('CPU_STABILITY_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
