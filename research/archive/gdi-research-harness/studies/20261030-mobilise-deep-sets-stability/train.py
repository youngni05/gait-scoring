"""Paired 120/600-epoch searches over five initialization schedules."""
import importlib.util,json,sys,time,itertools,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'20261030-mobilise-deep-sets'
spec=importlib.util.spec_from_file_location('reference_deep_sets',REFERENCE/'train.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
BASE=ref.BASE;MLP=ref.MLP;FEATURES=ref.FEATURES
SEEDS=[20261030+i*1000 for i in range(5)];BUDGETS=[120,600]
ref.EPOCHS=list(range(10,601,10))
def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def initialize():
    global DATA,ARRAYS,POOL_LIMITS
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    POOL_LIMITS=threadpool_limits(limits=1)
    DATA,ARRAYS,_=ref.load_data()
def work(task):
    target,band,k,fold,seed=task
    tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    done=HERE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();label='total_sppb_score' if target=='sppb' else 'tugtime'
    cell=DATA[DATA[target+'_ok']&DATA.band.eq(band)&DATA.k.eq(k)].sort_values(['draw','participantid','visit_number'])
    train=cell[cell.draw.eq(0)&cell.fold.ne(fold)].reset_index(drop=True);test=cell[cell.fold.eq(fold)].copy()
    assert not set(train.participantid)&set(test.participantid)
    original=json.loads((REFERENCE/'folds'/f'{target}-{band}-{k}-f{fold}.json').read_text())
    assert set(original['train_people'])==set(train.participantid)
    assert set(original['test_people'])==set(test.participantid)
    splits=[]
    for s in original['inner_people']:
        tr=train[train.participantid.isin(s['train'])];va=train[train.participantid.isin(s['valid'])]
        assert not set(tr.participantid)&set(va.participantid)
        assert set(tr.participantid)|set(va.participantid)==set(train.participantid)
        splits.append((tr,va))
    trials=[]
    for width in ref.WIDTHS:
        curves=[];sizes=[]
        for inner,(tr,va) in enumerate(splits):
            _,_,curve=ref.fit(tr,label,width,max(BUDGETS),seed+fold*10+inner,ARRAYS,k,va)
            curves.append(curve);sizes.append(va.participantid.nunique())
        scores=np.average(curves,axis=0,weights=sizes)
        trials.extend({'width':width,'epochs':epoch,'inner_person_mae':float(score)} for epoch,score in zip(ref.EPOCHS,scores))
    selected={};predictions=[]
    for budget in BUDGETS:
        best=min((t for t in trials if t['epochs']<=budget),key=lambda z:z['inner_person_mae'])
        model,bundle,_=ref.fit(train,label,best['width'],best['epochs'],seed+fold*10+9,ARRAYS,k)
        np.savez_compressed(HERE/'models'/f'{tag}-b{budget}.npz',**bundle)
        out=test[['participantid','visit_number','draw','fold',label]].rename(columns={label:'actual'}).copy()
        out['target']=target;out['band']=band;out['k']=k;out['seed']=seed;out['budget']=budget
        out['prediction']=ref.predict(model,bundle,ARRAYS[test.bag_id.to_numpy(),:k]);predictions.append(out)
        selected[str(budget)]=best
    pd.concat(predictions,ignore_index=True).to_csv(HERE/'folds'/f'{tag}.csv',index=False)
    # A completion ledger is written only after both models and predictions exist.
    save(f'folds/{tag}.json',{'target':target,'band':band,'k':k,'fold':fold,'seed':seed,'selected':selected,'trials':trials,'inner_people':original['inner_people'],'train_people':original['train_people'],'test_people':original['test_people'],'elapsed_seconds':time.time()-start})
    return tag,round(time.time()-start,1)
def run():
    started=time.time()
    protocol={'reference_study':str(REFERENCE),'source_hashes':{str(p):ref.sha(p) for p in [REFERENCE/'train.py',REFERENCE/'protocol.json',REFERENCE/'predictions.csv',REFERENCE/'verification.json',BASE/'selected-bouts.csv',BASE/'features.csv',BASE/'person-folds.csv',BASE/'predictions.csv',MLP/'predictions.csv']},'seeds':SEEDS,'budgets':BUDGETS,'epoch_candidates':ref.EPOCHS,'width_candidates':ref.WIDTHS,'data':'same people, visits, four raw bout-summary features, nested selected 1/3/5 bouts, same days and draws','outer':'same person 5-fold','inner':'reuse exact reference 3-fold person IDs; optimize overall person MAE only','seed_design':'five independent initialization schedules; seed+fold*10+inner for inner fits, seed+fold*10+9 for outer refits','paired_budget_design':'both budgets selected from the same deterministic training trajectory for each width and inner split; 600 search contains all 120 candidates','unchanged_training':'reference architecture, full-batch AdamW lr .01 weight_decay .001, person weights, training-only X/y scaling, SmoothL1 beta .5','primary_draw':0,'draws':'draw1/2 additional held-out inputs, no training or prediction averaging','groups':{'sppb_low':'actual visit score <=3','sppb_middle':'3<actual<=9','sppb_high':'actual>9','tug_long':'actual visit time >20 seconds','tug_other':'actual<=20 seconds'},'subgroup_weighting':'select visits by actual label, then equal weight per represented person; a longitudinal person can appear in multiple groups','metrics':['person MAE','signed prediction-minus-actual bias','person-weighted RMSE','weighted visit absolute-error p90'],'stability':'report all five runs, their mean/range and paired 600-120 differences; no best-seed selection, no seed ensemble','device':'CPU, four independent single-thread workers','scope':'repeat and lengthen training only; no target-dependent training weight changes','torch':torch.__version__,'numpy':np.__version__}
    if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==protocol
    save('protocol.json',protocol);(HERE/'folds').mkdir(exist_ok=True);(HERE/'models').mkdir(exist_ok=True)
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:
        futures=[pool.submit(work,t) for t in tasks]
        for n,future in enumerate(as_completed(futures),1):
            tag,elapsed=future.result()
            if n%5==0 or n==1:print(f'{n}/300 paired fold-seed jobs complete; {tag} ({elapsed}s); total {time.time()-started:.0f}s',flush=True)
    paths=sorted((HERE/'folds').glob('*.csv'));assert len(paths)==300
    p=pd.concat([pd.read_csv(path) for path in paths],ignore_index=True);p.to_csv(HERE/'predictions.csv',index=False)
    save('completion.json',{'paired_jobs':300,'outer_models':600,'inner_fits':1800,'elapsed_seconds':time.time()-started,'predictions_sha256':ref.sha(HERE/'predictions.csv')})
    print('ALL_TRAINING_FINISHED',flush=True)
if __name__=='__main__':run()
