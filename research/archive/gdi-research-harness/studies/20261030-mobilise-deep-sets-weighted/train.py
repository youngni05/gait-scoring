"""Isolate target-dependent loss weighting with frozen reference hyperparameters."""
import importlib.util,itertools,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'20261030-mobilise-deep-sets-stability'
spec=importlib.util.spec_from_file_location('stability_reference',REFERENCE/'train.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
ref=previous.ref;BASE=previous.BASE;SEEDS=previous.SEEDS;FEATURES=previous.FEATURES
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
FACTORS=[2,4]
KEY=['participantid','visit_number','draw','fold','target','band','k','seed']

def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def fit(train,label,width,epochs,seed,arrays,k,factor):
    # Keep preprocessing identical; only the loss weights change.
    w=ref.weights(train.participantid.to_numpy());x=arrays[train.bag_id.to_numpy(),:k];y=train[label].to_numpy()
    flat=x.reshape(-1,4);fw=np.repeat(w/k,k)
    xm=np.average(flat,axis=0,weights=fw);xs=np.maximum(np.sqrt(np.average((flat-xm)**2,axis=0,weights=fw)),1e-8)
    ym=float(np.average(y,weights=w));ys=max(float(np.sqrt(np.average((y-ym)**2,weights=w))),1e-8)
    tail=y<=3 if label=='total_sppb_score' else y>20
    loss_weights=w.copy()
    if factor!=1:
        loss_weights*=np.where(tail,factor,1);loss_weights/=loss_weights.sum()
    xt=torch.tensor((x-xm)/xs,dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);wt=torch.tensor(loss_weights,dtype=torch.float32)
    model=ref.make(width,seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();pred=model(xt).squeeze(-1)
        assert pred.shape==yt.shape==wt.shape
        loss=(torch.nn.functional.smooth_l1_loss(pred,yt,reduction='none',beta=.5)*wt).sum()
        assert torch.isfinite(loss)
        loss.backward();optimizer.step()
    bundle={'x_mean':xm,'x_std':xs,'y_mean':np.array(ym),'y_std':np.array(ys)}
    for i,layer in enumerate([model.layers[0],model.layers[2],model.layers[4]]):
        bundle[f'w{i}']=layer.weight.detach().numpy();bundle[f'b{i}']=layer.bias.detach().numpy()
    audit={'tail_visits':int(tail.sum()),'tail_people':int(train.loc[tail,'participantid'].nunique()),'base_tail_mass':float(w[tail].sum()),'weighted_tail_mass':float(loss_weights[tail].sum()),'loss_weight_sum':float(loss_weights.sum())}
    return model,bundle,audit

def initialize():
    global DATA,ARRAYS,LIMITS
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    LIMITS=threadpool_limits(limits=1);DATA,ARRAYS,_=ref.load_data()

def work(task):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    if (HERE/'folds'/f'{tag}.json').exists():return tag,'cached'
    started=time.time();label='total_sppb_score' if target=='sppb' else 'tugtime'
    cell=DATA[DATA[target+'_ok']&DATA.band.eq(band)&DATA.k.eq(k)].sort_values(['draw','participantid','visit_number'])
    tr=cell[cell.draw.eq(0)&cell.fold.ne(fold)].reset_index(drop=True);te=cell[cell.fold.eq(fold)].copy()
    ledger=json.loads((REFERENCE/'folds'/f'{tag}.json').read_text());choice=ledger['selected']['600']
    assert set(tr.participantid)==set(ledger['train_people'])
    assert set(te.participantid)==set(ledger['test_people'])
    assert not set(tr.participantid)&set(te.participantid)
    outputs=[];audits={};control=False
    # Exact controls demonstrate that a factor of one changes no training operation.
    if band=='long' and k==5 and fold==0 and seed==SEEDS[0]:
        _,b,_=fit(tr,label,choice['width'],choice['epochs'],seed+fold*10+9,ARRAYS,k,1)
        original=dict(np.load(REFERENCE/'models'/f'{tag}-b600.npz'))
        for name in b:np.testing.assert_array_equal(b[name],original[name])
        control=True
    for factor in FACTORS:
        model,bundle,audit=fit(tr,label,choice['width'],choice['epochs'],seed+fold*10+9,ARRAYS,k,factor)
        np.savez_compressed(HERE/'models'/f'{tag}-w{factor}.npz',**bundle)
        out=te[['participantid','visit_number','draw','fold',label]].rename(columns={label:'actual'}).copy()
        out['target']=target;out['band']=band;out['k']=k;out['seed']=seed;out['factor']=factor
        out['prediction']=ref.predict(model,bundle,ARRAYS[te.bag_id.to_numpy(),:k]);outputs.append(out);audits[str(factor)]=audit
    pd.concat(outputs,ignore_index=True).to_csv(HERE/'folds'/f'{tag}.csv',index=False)
    save(f'folds/{tag}.json',{'target':target,'band':band,'k':k,'fold':fold,'seed':seed,'selected':choice,'reference_ledger_sha256':ref.sha(REFERENCE/'folds'/f'{tag}.json'),'train_people':ledger['train_people'],'test_people':ledger['test_people'],'weight_audits':audits,'factor1_exact_control':control,'elapsed_seconds':time.time()-started})
    return tag,round(time.time()-started,1)

def run():
    started=time.time();verification=json.loads((REFERENCE/'verification.json').read_text());assert verification['passed']
    for name,digest in verification['artifacts_sha256'].items():assert ref.sha(REFERENCE/name)==digest
    for name,digest in verification['scripts_sha256'].items():assert ref.sha(REFERENCE/name)==digest
    protocol={'reference':str(REFERENCE),'source_hashes':{str(p):ref.sha(p) for p in [REFERENCE/'train.py',REFERENCE/'verification.json',REFERENCE/'predictions.csv',REFERENCE/'protocol.json',ref.HERE/'train.py',BASE/'features.csv',BASE/'selected-bouts.csv',BASE/'person-folds.csv']},'seeds':SEEDS,'loss_factors':FACTORS,'unweighted_reference':'same-seed budget600 predictions and weights','frozen_hyperparameters':'reuse each reference fold/seed width and epoch selected using only that fold training people; no weighted retuning or test-based selection','scope':'same 4..10 and 30..60 bands, 1/3/5 bouts, people, visits, folds and three draws; primary draw0','loss_weight':'base equal-person visit weight multiplied by factor iff training SPPB<=3 or TUG>20, then normalize globally; not normalized again within person','thresholds':'descriptive visit groups, not diagnostic categories','preprocessing':'unchanged base person-weighted train-only X/y scaler; no target boost in scalers','architecture_optimizer':'unchanged mean-pooled Deep Sets, full-batch AdamW lr .01 decay .001, standardized-target SmoothL1 beta .5','evaluation':'unchanged equal-person visit weighting, no tail boost in metrics; all, tail, non-tail, score bands, cohorts','comparisons':'paired same-seed/fold reference; report all five seeds, mean/range and MAE/bias/RMSE/p90; no best seed or ensemble','interpretation':'no chosen allowable global degradation margin; quantify tradeoffs, no automatic adoption or winner selection','device':'CPU four single-thread workers'}
    if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==protocol
    save('protocol.json',protocol);(HERE/'folds').mkdir(exist_ok=True);(HERE/'models').mkdir(exist_ok=True)
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:
        futures=[pool.submit(work,t) for t in tasks]
        for n,future in enumerate(as_completed(futures),1):
            tag,elapsed=future.result()
            if n%10==0 or n==1:print(f'{n}/300 paired weight jobs; {tag} ({elapsed}s); total {time.time()-started:.0f}s',flush=True)
    paths=sorted((HERE/'folds').glob('*.csv'));assert len(paths)==300
    p=pd.concat([pd.read_csv(x) for x in paths],ignore_index=True);p.to_csv(HERE/'predictions.csv',index=False)
    save('completion.json',{'jobs':300,'weighted_models':600,'factor1_controls':2,'elapsed_seconds':time.time()-started,'predictions_sha256':ref.sha(HERE/'predictions.csv')})
    print('ALL_WEIGHTED_TRAINING_FINISHED',flush=True)
if __name__=='__main__':run()
