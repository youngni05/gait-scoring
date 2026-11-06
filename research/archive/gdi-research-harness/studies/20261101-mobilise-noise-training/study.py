"""Train-only speed perturbations, otherwise identical Deep Sets and source epochs."""
from pathlib import Path
import importlib.util,json,itertools,argparse,time,os,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;REFERENCE=HERE.parent/'20261101-mobilise-speed-error'
spec=importlib.util.spec_from_file_location('speed_stress_reference',REFERENCE/'study.py');stress=importlib.util.module_from_spec(spec);spec.loader.exec_module(stress)
np,pd,joblib,old,p=stress.np,stress.pd,stress.joblib,stress.old,stress.p
torch=stress.ref.torch;sha,save,KEY,SEEDS=stress.sha,stress.save,stress.KEY,stress.SEEDS
STORE=Path('D:/gait-research-data/D_P/mobilise-noise-training-20261101')
METHODS=['SpeedDS','CadenceDS','SpeedAugDS','CadenceAugDS'];SCENARIOS=stress.SCENARIOS
tasks=stress.tasks;weights=stress.ref.weights
def initialize():stress.initialize()
def fit_aug(x,y,ids,seed,epochs):
    w=weights(ids);w=w/w.sum();flat=x.reshape(-1,x.shape[-1]);fw=np.repeat(w/x.shape[1],x.shape[1]);center=np.average(flat,axis=0,weights=fw);std=np.maximum(np.sqrt(np.average((flat-center)**2,axis=0,weights=fw)),1e-8)
    xx=torch.tensor((x-center)/std,dtype=torch.float32);yy=torch.tensor(y,dtype=torch.long);ww=torch.tensor(w,dtype=torch.float32);_,inv=np.unique(ids,return_inverse=True);rng=np.random.default_rng(seed+1000000)
    torch.manual_seed(seed);model=stress.ref.DeepSet(x.shape[-1]);optim=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001);history=[]
    for epoch in range(1,epochs+1):
        shift=rng.uniform(-.1,.1,int(inv.max())+1)[inv,None];noise=rng.normal(0,.05,(len(x),5));perturbed=x.copy();perturbed[:,:,0]=np.maximum(0.,x[:,:,0]+shift+noise);xn=torch.tensor((perturbed-center)/std,dtype=torch.float32)
        optim.zero_grad();lc=(torch.nn.functional.cross_entropy(model(xx),yy,reduction='none')*ww).sum();ln=(torch.nn.functional.cross_entropy(model(xn),yy,reduction='none')*ww).sum();loss=(lc+ln)/2;assert torch.isfinite(loss);loss.backward();optim.step()
        if epoch%10==0 or epoch==epochs:history.append(dict(epoch=epoch,clean_loss=float(lc.detach()),noisy_loss=float(ln.detach())))
    return {'mean':center.copy(),'std':std.copy(),**{n:q.detach().numpy().copy() for n,q in model.state_dict().items()}},history
def work(task):
    target,fold,seed=task;tag=f'{target}-f{fold}-s{seed}';folder=STORE/'folds';done=folder/f'{tag}.json'
    if done.exists():
        q=json.loads(done.read_text());assert q['code_sha256']==sha(__file__);return tag,'cached'
    started=time.time();source=stress.STORE/'folds'/f'{tag}.json';q=json.loads(source.read_text());src=stress.STORE/'folds'/f'{tag}.npz';keys=stress.STORE/'folds'/f'{tag}-keys.csv.gz';assert sha(src)==q['predictions_sha256'] and sha(keys)==q['keys_sha256'];original=np.load(src);metadata=pd.read_csv(keys);tr,te,inner,_=old.cell_data(target,'long',5,fold);np.testing.assert_array_equal(metadata[KEY],te[KEY]);metadata.to_csv(folder/f'{tag}-keys.csv.gz',index=False)
    ids=tr.participantid.to_numpy();col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target);trainbag=p.bag_array(tr,5);ledger=dict(task=list(task),code_sha256=sha(__file__),sources={str(z):sha(z) for z in [source,src,keys]},models={},clipped=q['clipped']);arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    probabilities=np.empty((7,4,len(te),4));warnings=np.empty((7,4,len(te)),dtype=bool)
    for j,name in enumerate(METHODS[:2]):
        ix=stress.METHODS.index(name);probabilities[:,j]=original['probabilities'][:,ix];warnings[:,j]=original['warnings'][:,ix];ledger['models'][name]=q['models'][name]
    for kind in ['Speed','Cadence']:
        name=kind+'AugDS';j=METHODS.index(name);baseinfo=q['models'][kind+'DS'];epochs=baseinfo['epochs'];x=stress.model_input(trainbag,kind+'DS');oof=np.zeros((len(tr),4));inner_models=[]
        for i in range(3):
            a=inner!=i;b=~a;m,h=fit_aug(x[a],y[a],ids[a],seed+fold*10+i,epochs);oof[b]=p.predict_ds(m,x[b]);info=p.save_model(STORE/'models'/f'{tag}-{name}-inner{i}.npz',m);hp=STORE/'models'/f'{tag}-{name}-inner{i}-history.json';save(hp,h);info.update(history_path=str(hp),history_sha256=sha(hp));inner_models.append(info)
        m,h=fit_aug(x,y,ids,seed+fold*10+9,epochs);info=p.save_model(STORE/'models'/f'{tag}-{name}.npz',m);hp=STORE/'models'/f'{tag}-{name}-history.json';save(hp,h);info.update(history_path=str(hp),history_sha256=sha(hp),epochs=epochs,source_epochs=epochs,inner_models=inner_models)
        out=pd.DataFrame(index=np.arange(len(te)));clean,_=stress.changed_bag(te,'clean');raw=p.predict_ds(m,stress.model_input(clean,name));info.update(old.finish_prob(name,oof,raw,y,ids,inner,out,target,arrays));ledger['models'][name]=info
        for i,scenario in enumerate(SCENARIOS):
            bag,_=stress.changed_bag(te,scenario);raw=p.predict_ds(m,stress.model_input(bag,name));cal=old.temper(raw,info['temperature']);probabilities[i,j]=cal;warnings[i,j]=cal[:,old.lowclass(target)]>=info['thresholds']['cal']['fpr10']
    dest=folder/f'{tag}.npz';np.savez_compressed(dest,probabilities=probabilities,warnings=warnings,**arrays);ledger.update(predictions_sha256=sha(dest),keys_sha256=sha(folder/f'{tag}-keys.csv.gz'),seconds=time.time()-started);save(done,ledger);return tag,round(time.time()-started,1)
def prepare():
    files=[REFERENCE/'study.py',REFERENCE/'main-verification.json',REFERENCE/'metrics-by-seed.csv',stress.ref.REFERENCE.parent/'20261101-mobilise-camera-models/study.py',old.BASE/'features.csv',old.BASE/'selected-bouts.csv']
    save(HERE/'protocol.json',dict(code_sha256=sha(__file__),sources={str(z):sha(z) for z in files},
      hypothesis='Speed bias+.10m/s leaves low-functionmiss81/74% despite cadence. Does train-only speed augmentation reduce this sensitivity while preserving clean accuracy?',
      data='SameSPPB2174/TUG1939people7356/5688visits,30–60sec5bouts,originalpersonouter5/inner3,5seeds,fixed4bands. Same1/2features,DSarchitecture and sameinitialization asoriginal.',
      training='For each feature set,fix epochs to source DS epoch selected only from original outer-traininginnerOOF. No newepoch/grid selection. Standardize on clean train people only. SameAdamWlr.01wd.001. Per epoch use0.5cleanCE+0.5noisyCE,person-equal. Addperson-shared uniform[-.10,.10]m/s across allvisits/bouts plus independentboutGaussianSD.05 tospeed only,clip0. RNGseed=trainseed+1000000;Torchseed unchanged. Cadence untouched.',
      calibration='Augmentedmodel clean innerOOF only for temperature,cross-temperature and fpr10warning rule. No evaluationlabels/noisy-evaluation recalibration. Same old rules. Baselines frozen sourceDS probabilities for all7scenarios reused.',
      evaluation='Same7controlledscenarios as previous stress study,including unseen scale errors. Report all4models,perclass/macrorecall,Brier,extremeerrors,warningmiss/FPR,fixedsamecohort true-speedpairs,5seeds. Onefixedjitterdraw,not expected distribution.',
      primary='WithinSpeed/Cadence separately,AugDSminusbaselineDS mean6nonclean accuracy;also clean change,+.10miss,-.10FPR. No choosing winning input based on outer labels.',
      success='Exploratoryjointtarget: noncleanmeanaccuracy improves>=.5percentagepoint,cleanaccuracy drops<=.5percentagepoint,+.10miss drops>=5percentagepoints,and-.10FPR increases<=1percentagepoint. Report each/5seeds and failures. Not clinical standard.',
      expected=dict(pilot_tasks=2,main_tasks=50,outer_folds=5,seeds=SEEDS,new_final_models=100,new_inner_models=300,reused_final_models=100,model_subset_replays=1700,selected_refits=16,pilot_replays=68,pilot_refits=8,metric_rows=280,matched_rows=280),
      verification='Source/model/input/scoreSHA,exactpeople/bouts,classlinks,cleantrainingonlyscale,sourceepochidentity,calibrationandthresholdtraining-only;allnewinner/final+baseline7scenarioreplays;16inner0/final9 refits strictarrayequality;independentmetrics/pairs;replay1e-12.',
      limits='Artificialtrainingnoise,unchangedlabels may not model real camera failures. No newpeople or clinical/video-scorevalidation. Existingdevelopmentpeople reused. Cadenceexactupperbound and2secMoVi vs sensor definition gap retained.',
      resources='IndependentCPU4processes,singleTorch/BLASthreads,estimate10-25min,smallmodel<300MBtotalD,noGPU/download.',storage=dict(small=str(HERE),large=str(STORE))))
def run(mode,workers):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for z,h in protocol['sources'].items():assert sha(z)==h
    for n in ['models','folds']:(STORE/n).mkdir(parents=True,exist_ok=True)
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
    start=time.time();ts=tasks(mode)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in ts]),1):print(i,len(ts),f.result(),flush=True)
    save(HERE/f'{mode}-completion.json',dict(tasks=len(ts),new_final_models=len(ts)*2,new_inner_models=len(ts)*6,code_sha256=sha(__file__),seconds=time.time()-start))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    else:run(a.mode,a.workers)
