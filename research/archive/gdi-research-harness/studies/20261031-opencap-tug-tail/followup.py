"""Explicit second exploratory phase: moderate weights after the weight-10 tradeoff."""
from common import *
from train import fit
import time,shutil
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

def main():
    assert not (HERE/'followup-protocol.json').exists()
    assert json.loads((HERE/'verification.json').read_text())['passed']
    shutil.copy2(HERE/'summary.json',HERE/'primary-summary.json')
    protocol={'phase':'second exploratory phase, specified after inspecting primary results','reason':'weight10 reduced long-TUG MAE but increased overall and short-TUG errors; test a moderate weight without choosing it from outer test labels during fitting','methods':{name:{'view':v,'weight':w,'objective':o} for name,(v,w,o,_) in FOLLOWUP_METHODS.items()},'same_input_sha256':sha(HERE/'input.npz'),'same_splits_sha256':sha(HERE/'splits.json'),'primary_result_sha256':sha(HERE/'primary-summary.json'),'selection':'same inner macro MAE; leaf2/5; all 5 partitions x3 seeds; no new target-dependent filter'}
    save('followup-protocol.json',protocol)
    started=time.time();d=load();views=features(d);y=d['y'];rows=[];evaluations=[];selections=[]
    for rep in json.loads((HERE/'splits.json').read_text()):
        for f in rep['splits']:
            repeat=rep['repeat'];fold=f['fold'];tr=np.array(f['train']);te=np.array(f['test'])
            for seed in SEEDS:
                for name,(view,weight,objective,kind) in FOLLOWUP_METHODS.items():
                    grid=[2,5];scores=[];predictions=[]
                    for setting in grid:
                        pp=np.full(len(y),np.nan)
                        for inner in f['inner']:
                            it=np.array(inner['train']);iv=np.array(inner['validation'])
                            pp[iv]=fit(kind,view,weight,setting,seed,views[view],y,it,iv)[0]
                        scores.append(loss(y[tr],pp[tr]));predictions.append(pp[tr].tolist())
                    setting=grid[int(np.argmin([s[objective] for s in scores]))]
                    evaluations.append({'repeat':repeat,'fold':fold,'seed':seed,'view':view,'weight':weight,'model':kind,'grid':grid,'train_order':tr.tolist(),'inner_predictions':predictions,'scores':scores})
                    pp,pair=fit(kind,view,weight,setting,seed,views[view],y,tr,te)
                    path=HERE/'followup-models'/f'r{repeat:02d}'/f'{name}-s{seed}-f{fold}.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(pair,path,compress=3)
                    selections.append({'repeat':repeat,'fold':fold,'seed':seed,'method':name,'setting':setting,'model_path':path.relative_to(HERE).as_posix(),'reused_baseline':False,'model_sha256':sha(path)})
                    rows.extend({'repeat':repeat,'fold':fold,'seed':seed,'method':name,'sample':int(k),'id':d['ids'][k],'target':float(y[k]),'prediction':float(v)} for k,v in zip(te,pp))
            pd.DataFrame(rows).to_csv(HERE/'followup-predictions.csv',index=False)
            save('followup-selections.json',selections);save('followup-inner-evaluations.json',evaluations)
            print('MODERATE_WEIGHT_FOLD_COMPLETE',repeat,fold,flush=True)
    save('followup-completion.json',{'completed':True,'new_models':150,'elapsed_seconds':time.time()-started,'code_sha256':sha(HERE/'followup.py'),'protocol_sha256':sha(HERE/'followup-protocol.json')})
    print('MODERATE_WEIGHT_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
