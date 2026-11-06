"""Joint loss-weight/epoch selection under a strict inner-validation MAE constraint."""
import importlib.util,itertools,json,os,time,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
WEIGHTED=HERE.parent/'20261030-mobilise-deep-sets-weighted'
spec=importlib.util.spec_from_file_location('weighted_reference',WEIGHTED/'train.py')
weighted=importlib.util.module_from_spec(spec);spec.loader.exec_module(weighted)
ref=weighted.ref;REFERENCE=weighted.REFERENCE;BASE=weighted.BASE;SEEDS=weighted.SEEDS;KEY=weighted.KEY;FEATURES=weighted.FEATURES
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
FACTORS=[1.25,1.5,2.0];EPOCHS=list(range(10,601,10))

def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def curve_fit(tr,va,label,width,epochs,seed,arrays,k,factor,checkpoints):
    w=ref.weights(tr.participantid.to_numpy());x=arrays[tr.bag_id.to_numpy(),:k];y=tr[label].to_numpy()
    flat=x.reshape(-1,4);fw=np.repeat(w/k,k)
    xm=np.average(flat,axis=0,weights=fw);xs=np.maximum(np.sqrt(np.average((flat-xm)**2,axis=0,weights=fw)),1e-8)
    ym=float(np.average(y,weights=w));ys=max(float(np.sqrt(np.average((y-ym)**2,weights=w))),1e-8)
    tail=y<=3 if label=='total_sppb_score' else y>20
    lw=w.copy()
    if factor!=1:lw*=np.where(tail,factor,1);lw/=lw.sum()
    xt=torch.tensor((x-xm)/xs,dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);wt=torch.tensor(lw,dtype=torch.float32)
    vx=torch.tensor((arrays[va.bag_id.to_numpy(),:k]-xm)/xs,dtype=torch.float32);vy=va[label].to_numpy()
    mask=vy<=3 if label=='total_sppb_score' else vy>20
    vw=ref.weights(va.participantid.to_numpy());tw=ref.weights(va.loc[mask,'participantid'].to_numpy())
    assert mask.any()
    model=ref.make(width,seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001);curves=[]
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();pred=model(xt).squeeze(-1)
        loss=(torch.nn.functional.smooth_l1_loss(pred,yt,reduction='none',beta=.5)*wt).sum()
        assert torch.isfinite(loss)
        loss.backward();optimizer.step()
        if epoch in checkpoints:
            with torch.no_grad():vp=model(vx).squeeze(-1).numpy()*ys+ym
            error=abs(vp-vy);curves.append([float(np.dot(vw,error)),float(np.dot(tw,error[mask]))])
    return curves

def choose(trials,baseline):
    feasible=[r for r in trials if r['inner_mae']<=baseline['inner_mae']]
    assert baseline in feasible
    return min(feasible,key=lambda r:(r['inner_tail_mae'],r['inner_mae'],r['factor'],r['epochs']))

def initialize():
    global DATA,ARRAYS,LIMITS
    torch.set_num_threads(1);torch.set_num_interop_threads(1);LIMITS=threadpool_limits(limits=1)
    DATA,ARRAYS,_=ref.load_data()

def work(task):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    if (HERE/'folds'/f'{tag}.json').exists():return tag,'cached'
    started=time.time();label='total_sppb_score' if target=='sppb' else 'tugtime'
    cell=DATA[DATA[target+'_ok']&DATA.band.eq(band)&DATA.k.eq(k)].sort_values(['draw','participantid','visit_number'])
    train=cell[cell.draw.eq(0)&cell.fold.ne(fold)].reset_index(drop=True)
    prior=json.loads((REFERENCE/'folds'/f'{tag}.json').read_text());base=prior['selected']['600'];width=base['width']
    assert set(train.participantid)==set(prior['train_people'])
    splits=[];sizes=[];tail_sizes=[]
    for ids in prior['inner_people']:
        tr=train[train.participantid.isin(ids['train'])];va=train[train.participantid.isin(ids['valid'])]
        assert not set(tr.participantid)&set(va.participantid)
        assert set(tr.participantid)|set(va.participantid)==set(train.participantid)
        mask=va[label].le(3) if target=='sppb' else va[label].gt(20)
        splits.append((tr,va));sizes.append(va.participantid.nunique());tail_sizes.append(va.loc[mask,'participantid'].nunique())
    controls=[curve_fit(tr,va,label,width,base['epochs'],seed+fold*10+i,ARRAYS,k,1,[base['epochs']])[0] for i,(tr,va) in enumerate(splits)]
    baseline={'factor':1.0,'epochs':base['epochs'],'inner_mae':float(np.average(np.array(controls)[:,0],weights=sizes)),'inner_tail_mae':float(np.average(np.array(controls)[:,1],weights=tail_sizes))}
    # Same inputs, initialization and arithmetic must reproduce the reference curve exactly.
    assert baseline['inner_mae']==base['inner_person_mae']
    trials=[baseline];curves={}
    for factor in FACTORS:
        c=np.array([curve_fit(tr,va,label,width,600,seed+fold*10+i,ARRAYS,k,factor,EPOCHS) for i,(tr,va) in enumerate(splits)])
        curves[str(factor)]=c.tolist()
        overall=np.average(c[:,:,0],axis=0,weights=sizes);tail=np.average(c[:,:,1],axis=0,weights=tail_sizes)
        trials.extend({'factor':factor,'epochs':epoch,'inner_mae':float(a),'inner_tail_mae':float(b)} for epoch,a,b in zip(EPOCHS,overall,tail))
    selected=choose(trials,baseline)
    if selected['factor']==1:
        bundle=dict(np.load(REFERENCE/'models'/f'{tag}-b600.npz'));model=ref.make(width,seed)
        with torch.no_grad():
            for i,layer in enumerate([model.layers[0],model.layers[2],model.layers[4]]):
                layer.weight.copy_(torch.tensor(bundle[f'w{i}']));layer.bias.copy_(torch.tensor(bundle[f'b{i}']))
        shutil.copyfile(REFERENCE/'models'/f'{tag}-b600.npz',HERE/'models'/f'{tag}.npz')
    else:
        model,bundle,_=weighted.fit(train,label,width,selected['epochs'],seed+fold*10+9,ARRAYS,k,selected['factor'])
        np.savez_compressed(HERE/'models'/f'{tag}.npz',**bundle)
    # Held-out rows are accessed only after the selection and final fit have finished.
    test=cell[cell.fold.eq(fold)].copy()
    assert set(test.participantid)==set(prior['test_people']) and not set(test.participantid)&set(train.participantid)
    out=test[['participantid','visit_number','draw','fold',label]].rename(columns={label:'actual'}).copy()
    out['target']=target;out['band']=band;out['k']=k;out['seed']=seed
    out['prediction']=ref.predict(model,bundle,ARRAYS[test.bag_id.to_numpy(),:k]);out.to_csv(HERE/'folds'/f'{tag}.csv',index=False)
    save(f'folds/{tag}.json',{'target':target,'band':band,'k':k,'fold':fold,'seed':seed,'width':width,'baseline':baseline,'selected':selected,'trials':trials,'inner_curves':curves,'baseline_inner_metrics':controls,'inner_sizes':list(map(int,sizes)),'inner_tail_sizes':list(map(int,tail_sizes)),'inner_people':prior['inner_people'],'train_people':prior['train_people'],'test_people':prior['test_people'],'reference_ledger_sha256':ref.sha(REFERENCE/'folds'/f'{tag}.json'),'feasible_candidates':sum(t['inner_mae']<=baseline['inner_mae'] for t in trials),'elapsed_seconds':time.time()-started})
    return tag,round(time.time()-started,1)

def run():
    started=time.time();v=json.loads((REFERENCE/'verification.json').read_text());assert v['passed']
    for name,digest in v['artifacts_sha256'].items():assert ref.sha(REFERENCE/name)==digest
    for name,digest in v['scripts_sha256'].items():assert ref.sha(REFERENCE/name)==digest
    protocol={'reference':str(REFERENCE),'weighted_source':str(WEIGHTED),'source_hashes':{str(p):ref.sha(p) for p in [REFERENCE/'train.py',REFERENCE/'verification.json',REFERENCE/'predictions.csv',WEIGHTED/'train.py',WEIGHTED/'verification.json',ref.HERE/'train.py',BASE/'features.csv',BASE/'selected-bouts.csv',BASE/'person-folds.csv']},'seeds':SEEDS,'weighted_candidates':FACTORS,'epoch_candidates':EPOCHS,'baseline_candidate':'factor1 at reference selected epoch; reproduce three inner fits and keep as fallback','width':'frozen per seed/fold to reference budget600-selected width','constraint':'combined inner equal-person MAE <= reference same-seed/fold selected inner MAE; zero slack','objective':'minimum tail inner equal-person MAE among feasible candidates; ties overall MAE, lower factor, earlier epoch','tail':'SPPB<=3 or TUG>20, visit-defined descriptive groups','inner':'exact reference three person folds; overall combined by all validation people counts, tail by qualifying validation people counts','outer':'exact reference five person folds; no held-out predictions used to choose factors, epochs, seeds or fallback','input':'same 4..10 and 30..60 second bands, 1/3/5 bouts, four bout features; train draw0, assess draws0/1/2 separately','loss_scaling':'reference person weights times tail factor, globally normalized; preprocessing unchanged and based on train people only','search':'all three weighted candidates trained to600 and evaluated every10; no adaptive pruning','evaluation':'unboosted equal-person MAE/bias/RMSE/p90; paired five-seed mean errors, no ensemble; report outer violations rather than claiming guaranteed noninferiority','device':'CPU six independent single-thread workers'}
    if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==protocol
    save('protocol.json',protocol);(HERE/'folds').mkdir(exist_ok=True);(HERE/'models').mkdir(exist_ok=True)
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=6,initializer=initialize) as pool:
        futures=[pool.submit(work,t) for t in tasks]
        for n,future in enumerate(as_completed(futures),1):
            tag,elapsed=future.result()
            if n%5==0 or n==1:print(f'{n}/300 constrained searches; {tag} ({elapsed}s); total {time.time()-started:.0f}s',flush=True)
    paths=sorted((HERE/'folds').glob('*.csv'));assert len(paths)==300
    p=pd.concat([pd.read_csv(x) for x in paths],ignore_index=True);p.to_csv(HERE/'predictions.csv',index=False)
    save('completion.json',{'outer_models':300,'weighted_inner_fits':2700,'reference_inner_fits':900,'elapsed_seconds':time.time()-started,'predictions_sha256':ref.sha(HERE/'predictions.csv')})
    print('ALL_CONSTRAINED_TRAINING_FINISHED',flush=True)
if __name__=='__main__':run()
