"""Paired input x pooling ablation. All choices use training-person inner folds."""
import importlib.util,itertools,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'20261030-mobilise-deep-sets-stability'
spec=importlib.util.spec_from_file_location('stability_reference',REFERENCE/'train.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
ref=prior.ref;BASE=prior.BASE
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
SEEDS=prior.SEEDS;EPOCHS=list(range(10,601,10));FEATURES=ref.FEATURES
VARIANTS={'A':(4,'mean'),'B':(6,'mean'),'C':(4,'meanmax'),'D':(6,'meanmax')}
KEY=['target','band','k','seed','fold','draw','participantid','visit_number']

def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

class DeepSet(torch.nn.Module):
    def __init__(self,dim,width,pooling):
        super().__init__();self.pooling=pooling
        h=width//2 if pooling=='mean' else width//4
        self.layers=torch.nn.Sequential(torch.nn.Linear(dim,width),torch.nn.ReLU(),torch.nn.Linear(width if pooling=='mean' else 2*width,h),torch.nn.ReLU(),torch.nn.Linear(h,1))
    def forward(self,x):
        h=self.layers[1](self.layers[0](x));p=h.mean(dim=1)
        if self.pooling=='meanmax':p=torch.cat([p,h.amax(dim=1)],dim=1)
        return self.layers[4](self.layers[3](self.layers[2](p)))

def make(variant,width,seed):
    torch.manual_seed(seed);dim,pooling=VARIANTS[variant]
    return DeepSet(dim,width,pooling)

def load_data():
    data,core,ids=ref.load_data()
    selected=pd.read_csv(BASE/'selected-bouts.csv').sort_values(ref.BAGKEY+['rank']).reset_index(drop=True)
    audit=pd.read_csv(HERE/'audited-bouts.csv');assert audit.turn_valid.eq(True).all()
    joined=selected.merge(audit[['participantid','visit_number','wb_id','turn_rate_per_second']],on=['participantid','visit_number','wb_id'],validate='many_to_one')
    joined=joined.sort_values(ref.BAGKEY+['rank']).reset_index(drop=True)
    pd.testing.assert_frame_equal(selected,joined[selected.columns])
    np.testing.assert_array_equal(core,joined[FEATURES].to_numpy().reshape(-1,5,4))
    extra=joined[['duration','turn_rate_per_second']].to_numpy().reshape(-1,5,2)
    expanded=np.ascontiguousarray(np.concatenate([core,extra],axis=2));assert np.isfinite(expanded).all()
    return data,{'A':core,'B':expanded,'C':core,'D':expanded},ids

def fit(train,label,variant,width,epochs,seed,arrays,k,valid=None):
    w=ref.weights(train.participantid.to_numpy());x=arrays[train.bag_id.to_numpy(),:k];y=train[label].to_numpy()
    flat=x.reshape(-1,x.shape[-1]);fw=np.repeat(w/k,k)
    xm=np.average(flat,axis=0,weights=fw);xs=np.maximum(np.sqrt(np.average((flat-xm)**2,axis=0,weights=fw)),1e-8)
    ym=float(np.average(y,weights=w));ys=max(float(np.sqrt(np.average((y-ym)**2,weights=w))),1e-8)
    xt=torch.tensor((x-xm)/xs,dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
    model=make(variant,width,seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    curves=[]
    if valid is not None:
        vx=torch.tensor((arrays[valid.bag_id.to_numpy(),:k]-xm)/xs,dtype=torch.float32);vy=valid[label].to_numpy();vw=ref.weights(valid.participantid.to_numpy())
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();pred=model(xt).squeeze(-1)
        assert pred.shape==yt.shape==wt.shape
        loss=(torch.nn.functional.smooth_l1_loss(pred,yt,reduction='none',beta=.5)*wt).sum();assert torch.isfinite(loss)
        loss.backward();optimizer.step()
        if valid is not None and epoch%10==0:
            with torch.no_grad():vp=model(vx).squeeze(-1).numpy()*ys+ym
            curves.append(float(np.dot(vw,abs(vp-vy))))
    bundle={'x_mean':xm,'x_std':xs,'y_mean':np.array(ym),'y_std':np.array(ys)}
    for i,layer in enumerate([model.layers[0],model.layers[2],model.layers[4]]):
        bundle[f'w{i}']=layer.weight.detach().numpy();bundle[f'b{i}']=layer.bias.detach().numpy()
    return model,bundle,curves

def initialize():
    global DATA,ARRAYS,LIMITS
    torch.set_num_threads(1);torch.set_num_interop_threads(1);LIMITS=threadpool_limits(limits=1)
    DATA,ARRAYS,_=load_data()

def work(task):
    target,band,k,fold,seed,variant=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    name=f'{tag}-{variant}';done=HERE/'folds'/f'{name}.json'
    if done.exists():return name,'cached'
    started=time.time();label='total_sppb_score' if target=='sppb' else 'tugtime'
    cell=DATA[DATA[target+'_ok']&DATA.band.eq(band)&DATA.k.eq(k)].sort_values(['draw','participantid','visit_number'])
    train=cell[cell.draw.eq(0)&cell.fold.ne(fold)].reset_index(drop=True);test=cell[cell.fold.eq(fold)].copy()
    old=json.loads((REFERENCE/'folds'/f'{tag}.json').read_text())
    assert set(train.participantid)==set(old['train_people']) and set(test.participantid)==set(old['test_people'])
    assert not set(train.participantid)&set(test.participantid)
    width=old['selected']['600']['width'];curves=[];sizes=[]
    for inner,s in enumerate(old['inner_people']):
        tr=train[train.participantid.isin(s['train'])];va=train[train.participantid.isin(s['valid'])]
        assert set(tr.participantid)|set(va.participantid)==set(train.participantid)
        assert not set(tr.participantid)&set(va.participantid)
        _,_,curve=fit(tr,label,variant,width,600,seed+fold*10+inner,ARRAYS[variant],k,va)
        curves.append(curve);sizes.append(int(va.participantid.nunique()))
    combined=np.average(curves,axis=0,weights=sizes);chosen=int(np.argmin(combined));epochs=EPOCHS[chosen]
    model,bundle,_=fit(train,label,variant,width,epochs,seed+fold*10+9,ARRAYS[variant],k)
    np.savez_compressed(HERE/'models'/f'{name}.npz',**bundle)
    out=test[['participantid','visit_number','draw','fold',label]].rename(columns={label:'actual'}).copy()
    out['target']=target;out['band']=band;out['k']=k;out['seed']=seed
    out['prediction']=ref.predict(model,bundle,ARRAYS[variant][test.bag_id.to_numpy(),:k])
    out.to_csv(HERE/'folds'/f'{name}.csv.gz',index=False)
    save(f'folds/{name}.json',{'target':target,'band':band,'k':k,'fold':fold,'seed':seed,'variant':variant,'width':width,'epochs':epochs,'inner_person_mae':float(combined[chosen]),'inner_curves':curves,'inner_people_counts':sizes,'combined_curve':combined.tolist(),'reference_ledger_sha256':ref.sha(REFERENCE/'folds'/f'{tag}.json'),'parameters':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.time()-started})
    return name,round(time.time()-started,1)

def run():
    started=time.time();q=json.loads((HERE/'quality.json').read_text());assert q['turn_invalid_unique_bouts']==0
    for path,digest in q['source_sha256'].items():assert ref.sha(path)==digest
    for path,digest in q['artifact_sha256'].items():assert ref.sha(HERE/path)==digest
    protocol={'reference':str(REFERENCE),'source_hashes':{str(p):ref.sha(p) for p in [REFERENCE/'train.py',REFERENCE/'verification.json',REFERENCE/'predictions.csv',ref.HERE/'train.py',BASE/'features.csv',BASE/'selected-bouts.csv',BASE/'person-folds.csv',HERE/'quality.json',HERE/'audited-bouts.csv',HERE/'label-review-flags.csv',HERE/'train.py',HERE/'quality.py']},'variants':{'A':'original 4 features, mean pool (reuse verified 600-epoch-search reference)','B':'original 4 + duration seconds + turns/second, mean pool','C':'original 4, concatenate latent mean and maximum; half-sized downstream hidden layer','D':'expanded 6, concatenate latent mean and maximum; half-sized downstream hidden layer'},'feature_order':FEATURES+['duration','turn_rate_per_second'],'turn_quality':'all 205307 selected unique bouts have true original turn/duration flags and nonnegative integer turns; no clipping, imputation or bout removal','people_bouts':'exact reference people, visits, selected days/bouts, ranks <=k, folds and 3 draws; no exclusions or label corrections','seeds':SEEDS,'outer':'original 5 person folds','inner':'original 3 person folds; candidate epochs 10..600 step10; choose minimum combined overall person MAE, earlier epoch breaks ties','width':'fix per fold/seed/condition to the reference 600-search-selected encoder width (32 or64); do not choose width on test outcomes','head':'mean: Linear(width,width/2); meanmax: Linear(2width,width/4); ReLU then scalar output. Parameter counts nearly matched across architecture axis. The architecture change includes pooling AND head-width adjustment.','parameter_counts':{v:{str(w):sum(p.numel() for p in make(v,w,0).parameters()) for w in [32,64]} for v in VARIANTS},'optimization':'unchanged full-batch AdamW lr .01 weight_decay .001, person-weighted SmoothL1 beta .5; train-only person-weighted input/label scaling; no tail label weighting','seed_schedule':'inner seed+fold*10+inner; outer seed+fold*10+9','primary_draw':0,'replication':'train draw0, evaluate draw0/1/2 separately; no seed ensemble or draw averaging','tail_definitions':{'sppb':'actual<=3','tug':'actual>20 seconds'},'criteria':'paired overall MAE difference<=0 and tail MAE difference<0; separately assess tail p90/RMSE and all5 seeds. No noninferiority margin. Outer results assess hypotheses, never tune or select epochs.','uncertainty':'paired person bootstrap of mean across five seed-specific per-person errors; fixed OOF predictions, not retraining; 3 draws are sensitivity checks, not independent samples','label_sensitivity':'evaluate identical trained predictions on unflagged visits only, without replacing primary all-record results','scope':'observational ablation within reference-selected widths, not exhaustive architecture optimization or independent final validation; four input summaries plus two gait features, not videos or raw movement sequences','device':'CPU, 6 independent single-thread workers','torch':torch.__version__}
    if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==protocol
    save('protocol.json',protocol)
    for name in ['models','folds']:(HERE/name).mkdir(exist_ok=True)
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS,['B','C','D']))
    with ProcessPoolExecutor(max_workers=6,initializer=initialize) as pool:
        futures=[pool.submit(work,task) for task in tasks]
        for n,future in enumerate(as_completed(futures),1):
            name,elapsed=future.result()
            if n%6==0 or n==1:print(f'{n}/900 models complete; {name} ({elapsed}s); total {time.time()-started:.0f}s',flush=True)
    p=pd.read_csv(REFERENCE/'predictions.csv');p=p[p.budget.eq(600)].drop(columns='budget').rename(columns={'prediction':'A'}).sort_values(KEY).reset_index(drop=True)
    for variant in ['B','C','D']:
        paths=sorted((HERE/'folds').glob(f'*-{variant}.csv.gz'));assert len(paths)==300
        d=pd.concat([pd.read_csv(path) for path in paths],ignore_index=True).sort_values(KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(p[KEY],d[KEY]);np.testing.assert_allclose(p.actual,d.actual,rtol=0,atol=1e-12)
        p[variant]=d.prediction
    p.to_csv(HERE/'predictions.csv.gz',index=False)
    save('completion.json',{'new_outer_models':900,'reused_outer_reference_models':300,'new_inner_fits':2700,'predictions_rows_per_variant':len(p),'elapsed_seconds':time.time()-started,'predictions_sha256':ref.sha(HERE/'predictions.csv.gz')})
    print('ALL_TRAINING_FINISHED',flush=True)
if __name__=='__main__':run()
