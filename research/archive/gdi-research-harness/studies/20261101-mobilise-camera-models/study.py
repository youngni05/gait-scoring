"""Same camera-feasible inputs; RF leaf and Deep Sets epoch chosen by inner accuracy."""
from pathlib import Path
import importlib.util,json,time,itertools,os,argparse,datetime
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;REFERENCE=HERE.parent/'20261101-mobilise-camera-inputs'
spec=importlib.util.spec_from_file_location('camera_inputs',REFERENCE/'study.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
np,pd,joblib,old=ref.np,ref.pd,ref.joblib,ref.old
p=ref.base.p;torch=p.torch
sha,save,weights=ref.sha,ref.save,ref.weights;KEY,SEEDS=ref.KEY,ref.SEEDS
STORE=Path('D:/gait-research-data/D_P/mobilise-camera-models-20261101')
METHODS=['SpeedFixedRF','CadenceFixedRF','SpeedAccRF','CadenceAccRF','SpeedDS','CadenceDS']
EPOCHS=list(range(10,601,10));LEAVES=[5,20]

def initialize():ref.initialize()
def tasks(mode):return [('sppb',0,SEEDS[0]),('tug',0,SEEDS[0])] if mode=='pilot' else list(itertools.product(['sppb','tug'],range(5),SEEDS))
def inputs(d):
    bag=p.bag_array(d,5);rf=ref.inputs(d)
    return {'Speed':{'rf':rf['SpeedBouts'],'ds':bag[:,:,[2]]},'Cadence':{'rf':rf['SpeedCadence'],'ds':bag[:,:,[2,0]]}}

def select(probabilities,y,ids,values,prefer_larger=False):
    w=weights(ids);acc=np.array([np.average(z.argmax(1)==y,weights=w) for z in probabilities]);nll=np.array([np.average(-np.log(np.maximum(z[np.arange(len(y)),y],1e-12)),weights=w) for z in probabilities])
    best=min(range(len(values)),key=lambda i:(-round(float(acc[i]),12),round(float(nll[i]),12),-values[i] if prefer_larger else values[i]))
    return best,acc,nll

class DeepSet(torch.nn.Module):
    def __init__(self,features):
        super().__init__();self.encoder=torch.nn.Linear(features,32);self.hidden=torch.nn.Linear(32,16);self.head=torch.nn.Linear(16,4)
    def forward(self,x):return self.head(torch.relu(self.hidden(torch.relu(self.encoder(x)).mean(1))))

def ds_fit(x,y,ids,seed,epochs,valid=None):
    w=weights(ids);w=w/w.sum();flat=x.reshape(-1,x.shape[-1]);fw=np.repeat(w/x.shape[1],x.shape[1]);mean=np.average(flat,axis=0,weights=fw);std=np.maximum(np.sqrt(np.average((flat-mean)**2,axis=0,weights=fw)),1e-8)
    xt=torch.tensor((x-mean)/std,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.long);wt=torch.tensor(w,dtype=torch.float32)
    torch.manual_seed(seed);m=DeepSet(x.shape[-1]);optimizer=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.001);bundles=[];probabilities=[]
    def bundle():return {'mean':mean.copy(),'std':std.copy(),**{n:q.detach().numpy().copy() for n,q in m.state_dict().items()}}
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();loss=(torch.nn.functional.cross_entropy(m(xt),yt,reduction='none')*wt).sum();assert torch.isfinite(loss);loss.backward();optimizer.step()
        if valid is not None and epoch%10==0:
            b=bundle();bundles.append(b);probabilities.append(p.predict_ds(b,valid))
    return bundle(),bundles,probabilities

def work(task):
    target,fold,seed=task;tag=f'{target}-f{fold}-s{seed}';done=STORE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,split=old.cell_data(target,'long',5,fold);x,xt=inputs(tr),inputs(te);ids=tr.participantid.to_numpy();col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target)
    source=ref.STORE/'folds'/f'{tag}.json';sourcecsv=ref.STORE/'folds'/f'{tag}.csv.gz';sourceinner=ref.STORE/'folds'/f'{tag}-inner.npz';l0=json.loads(source.read_text());assert sha(sourcecsv)==l0['predictions_sha256'] and sha(sourceinner)==l0['inner_sha256'];ip=np.load(sourceinner);original=pd.read_csv(sourcecsv)
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    for k,v in arrays.items():np.testing.assert_array_equal(ip[k],v)
    np.testing.assert_array_equal(original[KEY],te[KEY]);np.testing.assert_allclose(original.actual,te[col],rtol=0,atol=1e-12)
    out=original[KEY+['cohort','actual','class','target','band','k','fold','seed','speed']].copy()
    for j in range(4):out[f'Prior_p{j}']=original[f'Prior_p{j}']
    ledger={'task':list(task),'code_sha256':sha(__file__),'source_files':{str(q):sha(q) for q in [source,sourcecsv,sourceinner]},'models':{},'rf_candidates':{}}
    for kind,src in [('Speed','SpeedBouts'),('Cadence','SpeedCadence')]:
        fixed=kind+'FixedRF';accname=kind+'AccRF';dsname=kind+'DS';info0=l0['models'][src];assert sha(info0['path'])==info0['sha256']
        raw0=old.probabilities(joblib.load(info0['path']),xt[kind]['rf'],'RF');oof0=ip[src+'_oof'];info=dict(info0);info.update(old.finish_prob(fixed,oof0,raw0,y,ids,inner,out,target,arrays));ledger['models'][fixed]=info
        candidates=[];oofs=[]
        for leaf in LEAVES:
            if leaf==20:info=dict(info0);oof=oof0.copy()
            else:
                saved=[];oof=np.zeros((len(tr),4))
                for i in range(3):
                    a=inner!=i;b=~a;m=old.make('RF',seed+fold*10+i);m['model'].set_params(min_samples_leaf=leaf);m=old.fit(m,x[kind]['rf'][a],y[a],ids[a],'RF');oof[b]=old.probabilities(m,x[kind]['rf'][b],'RF');saved.append(p.save_model(STORE/'models'/f'{tag}-{kind}-leaf{leaf}-inner{i}.joblib',m))
                info={'inner_models':saved}
            info['leaf']=leaf;candidates.append(info);oofs.append(oof)
        best,acc,nll=select(oofs,y,ids,LEAVES,True);leaf=LEAVES[best];oof=oofs[best];info=dict(candidates[best])
        if leaf==20:raw=raw0
        else:
            m=old.make('RF',seed+fold*10+9);m['model'].set_params(min_samples_leaf=leaf);m=old.fit(m,x[kind]['rf'],y,ids,'RF');raw=old.probabilities(m,xt[kind]['rf'],'RF');info.update(p.save_model(STORE/'models'/f'{tag}-{accname}.joblib',m))
        info.update(old.finish_prob(accname,oof,raw,y,ids,inner,out,target,arrays));info.update(selected_leaf=leaf,candidate_accuracy=acc.tolist(),candidate_nll=nll.tolist());ledger['models'][accname]=info;ledger['rf_candidates'][kind]=candidates;arrays[kind+'_rf_candidates']=np.stack(oofs)
        oofs=np.zeros((len(EPOCHS),len(tr),4));snapshots=[]
        for i in range(3):
            a=inner!=i;b=~a;_,bundles,prob=ds_fit(x[kind]['ds'][a],y[a],ids[a],seed+fold*10+i,600,x[kind]['ds'][b]);oofs[:,b,:]=np.stack(prob);snapshots.append(bundles)
        best,acc,nll=select(oofs,y,ids,EPOCHS);epoch=EPOCHS[best];oof=oofs[best];saved=[]
        for i in range(3):saved.append(p.save_model(STORE/'models'/f'{tag}-{dsname}-inner{i}.npz',snapshots[i][best]))
        bundle,_,_=ds_fit(x[kind]['ds'],y,ids,seed+fold*10+9,epoch);raw=p.predict_ds(bundle,xt[kind]['ds']);info=p.save_model(STORE/'models'/f'{tag}-{dsname}.npz',bundle)
        info.update(inner_models=saved,epochs=epoch,candidate_accuracy=acc.tolist(),candidate_nll=nll.tolist());info.update(old.finish_prob(dsname,oof,raw,y,ids,inner,out,target,arrays));ledger['models'][dsname]=info;arrays[kind+'_ds_candidates']=oofs
    out.to_csv(STORE/'folds'/f'{tag}.csv.gz',index=False);np.savez_compressed(STORE/'folds'/f'{tag}-inner.npz',**arrays)
    ledger.update(seconds=time.time()-start,predictions_sha256=sha(STORE/'folds'/f'{tag}.csv.gz'),inner_sha256=sha(STORE/'folds'/f'{tag}-inner.npz'));save(done,ledger);return tag,round(time.time()-start,1)

def prepare():
    paths=[REFERENCE/'study.py',REFERENCE/'main-verification.json',REFERENCE/'summary.json',old.BASE/'features.csv',old.BASE/'selected-bouts.csv',p.FACTORIAL/'audited-bouts.csv']
    save(HERE/'protocol.json',{'code_sha256':sha(__file__),'sources':{str(q):sha(q) for q in paths},'seeds':SEEDS,
      'hypothesis':'speed-bout preservation helps but cadence did not raise overall accuracy in fixed RF; test same 1/2features with RF/DeepSets and selection aligned to four-band accuracy.',
      'user_camera':'fixed phone,no floor marks,height once allowed; speed is anchor candidate. Sensor proxy study,not measured video scores.',
      'data':'sameSPPB2174/TUG1939people7356/5688visits,original outer5/inner3 person IDs,30-60sec5bouts,draw0,fourbands unchanged,5seeds,no metadata inputs',
      'methods':'Speed/Cadence fixed RF128 leaf20 reused; RF leaf5/20 selected by inner OOF person accuracy; DS features1/2->32 ReLU mean->16 ReLU->4,AdamWlr.01wd.001,CE person weights,epoch10..600step10 selected by same inner accuracy.',
      'selection':'highest person-equal OOF accuracy,tie smaller NLL,then larger leaf/earlier epoch; selection scores rounded12decimal for deterministic mathematical ties,not replay tolerance. Original inner3 fixed. Full calibration and warning rules remain train-only; inner selection predictions reused for calibration,not independent final performance.',
      'replication':'ProcessPool4,each worker independent torch RNG and single-thread. NumPyfloat64 validation/inference on float32 learned weights; fixed1e-12 model replay tolerance. Candidate OOFs D; selected inner/final models retained.',
      'expected':{'tasks':50,'selected_final_models':300,'new_DS_final':100,'new_RF_final_range':[0,100],'new_RF_candidate_inner':300,'new_DS_inner':300,'reused_fixed_final':100,'model_replays':1500,'selected_refits':32,'metric_rows':130,'matched_rows':60},
      'primary':'same-input RF/DS overall4band accuracy and5seed direction; report perclass/macro recall,extremeerrors,tailmiss/FPR,samecohort-speed riskorder. No changing class boundaries.',
      'comparisons':'AccRF-FixedRF within each input; DS-AccRF within each input; Cadence-Speed within each model. Conditional person bootstrap and all methods reported,not final model adoption by reused outer results.',
      'resource':'CPU4workers,estimate20-45min,rough1-3GB candidates/model outputD; pilot2tasks fold0 firstseed. Tiny sets do not require GPU.',
      'limits':'reused exploratory development people; sensor values not video restoration; no step asymmetry/variability/trunk data; .2point aggregate gains can hide severe-bin failures.',
      'storage':{'small':str(HERE),'large':str(STORE)},'versions':{'torch':torch.__version__,'numpy':np.__version__}});print('PREPARED',flush=True)

def run(mode,workers):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for q,digest in protocol['sources'].items():assert sha(q)==digest
    assert HERE.drive.upper()=='C:' and STORE.drive.upper()=='D:'
    for name in ['models','folds']:(STORE/name).mkdir(parents=True,exist_ok=True)
    for var in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[var]='1'
    start=time.time();todo=tasks(mode)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in todo]),1):print(i,len(todo),*f.result(),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(todo),'seconds':time.time()-start,'code_sha256':sha(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
