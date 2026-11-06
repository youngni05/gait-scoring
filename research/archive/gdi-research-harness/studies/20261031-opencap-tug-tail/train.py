from common import *
import time
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

def fit(model,view,weight,setting,seed,z,y,tr,te):
    scale=StandardScaler().fit(z[tr],sample_weight=np.ones(len(tr)))
    if model=='RF':
        m=RandomForestRegressor(n_estimators=128,min_samples_leaf=setting,random_state=seed,n_jobs=2,max_features=1.)
    else:m=Ridge(alpha=setting)
    w=np.where(y[tr]>=20,weight,1.).astype(float)
    m.fit(scale.transform(z[tr]),y[tr],sample_weight=w)
    p=m.predict(scale.transform(z[te]))
    if model=='Ridge':p=np.maximum(p,0)
    return p,(scale,m)

def main():
    started=time.time();d=load();views=features(d);y=d['y'];plan=json.loads((HERE/'splits.json').read_text())
    old={(r['repeat'],r['fold'],r['seed']):r for r in json.loads((HERE/'baseline-selections.json').read_text())}
    rows=[];selections=[];evaluations=[];models=0
    for rep in plan:
        for split in rep['splits']:
            tr=np.array(split['train']);te=np.array(split['test']);fold=split['fold'];repeat=rep['repeat']
            for seed in SEEDS:
                cached={}
                for view,weight,model in [('raw',1,'RF'),('raw',10,'RF'),('robust',1,'RF'),('robust',10,'RF'),('inverse',1,'Ridge')]:
                    if model=='Ridge' and seed!=SEEDS[0]:continue
                    grid=[2,5] if model=='RF' else [.1,1.,10.,100.]
                    scores=[];predictions=[]
                    for k,setting in enumerate(grid):
                        if view=='raw' and weight==1:
                            base=old[(repeat,fold,seed)];assert base['inner_train_order']==tr.tolist()
                            pp=np.array(base['inner_predictions'][k])
                        else:
                            pp_all=np.full(len(y),np.nan)
                            for inner in split['inner']:
                                it=np.array(inner['train']);iv=np.array(inner['validation'])
                                pp_all[iv]=fit(model,view,weight,setting,seed,views[view],y,it,iv)[0]
                            pp=pp_all[tr]
                        scores.append(loss(y[tr],pp));predictions.append(pp.tolist())
                    key=(view,weight,model);cached[key]=(grid,scores)
                    evaluations.append({'repeat':repeat,'fold':fold,'seed':seed if model=='RF' else -1,'view':view,'weight':weight,'model':model,'grid':grid,'train_order':tr.tolist(),'inner_predictions':predictions,'scores':scores})
                for name,(view,weight,objective,model) in METHODS.items():
                    if model=='Ridge' and seed!=SEEDS[0]:continue
                    grid,scores=cached[(view,weight,model)]
                    setting=grid[int(np.argmin([r[objective] for r in scores]))]
                    if name=='BaselineRF':
                        base=old[(repeat,fold,seed)];assert setting==base['selected']
                        pair=joblib.load(PREV/base['model_path']);pp=pair[1].predict(pair[0].transform(views[view][te]))
                        model_path=str(PREV/base['model_path']);reused=True
                    else:
                        pp,pair=fit(model,view,weight,setting,seed,views[view],y,tr,te)
                        path=HERE/'models'/f'r{repeat:02d}'/f'{name}-s{seed if model=="RF" else -1}-f{fold}.joblib'
                        path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(pair,path,compress=3)
                        model_path=path.relative_to(HERE).as_posix();reused=False;models+=1
                    assert np.isfinite(pp).all()
                    record={'repeat':repeat,'fold':fold,'seed':seed if model=='RF' else -1,'method':name,'setting':setting,'model_path':model_path,'reused_baseline':reused,'model_sha256':sha(Path(model_path) if reused else HERE/model_path)}
                    selections.append(record)
                    rows.extend({'repeat':repeat,'fold':fold,'seed':record['seed'],'method':name,'sample':int(k),'id':d['ids'][k],'target':float(y[k]),'prediction':float(v)} for k,v in zip(te,pp))
            pd.DataFrame(rows).to_csv(HERE/'predictions.csv',index=False)
            save('selections.json',selections);save('inner-evaluations.json',evaluations)
            print('TAIL_FOLD_COMPLETE',repeat,fold,flush=True)
    save('completion.json',{'completed':True,'new_models':models,'reused_models':75,'elapsed_seconds':time.time()-started,'input_sha256':sha(HERE/'input.npz'),'splits_sha256':sha(HERE/'splits.json'),'code_sha256':sha(HERE/'train.py')})
    print('TAIL_TRAINING_COMPLETE',models,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
