"""Frozen turn-only bout inputs: histogram gradient boosting versus verified RF."""
from pathlib import Path
import importlib.util,sys,os,json,time,itertools,argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'20261101-mobilise-duration-turns'
spec=importlib.util.spec_from_file_location('duration_reference',REFERENCE/'study.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
np,pd,joblib,old=base.np,base.pd,base.joblib,base.old
sha,save,weights=base.sha,base.save,base.weights
SEEDS,KEY=base.SEEDS,base.KEY
METHODS=['TurnRF','FullRF','HGB']
STORE=Path('D:/gait-research-data/D_P/mobilise-turn-hgb-20261101')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def initialize():base.initialize()
def inputs(frame,k=5):
    x=base.inputs(frame,k)
    return {'TurnRF':x['Turn5'],'FullRF':x['Bout6'],'HGB':x['Turn5']}
def make(seed,leaves):
    return Pipeline([('scale',StandardScaler()),('model',HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=leaves,min_samples_leaf=20,learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=seed))])

def work(task):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    done=STORE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,_=old.cell_data(target,band,k,fold)
    x,xt=inputs(tr,k),inputs(te,k);ids=tr.participantid.to_numpy();label='total_sppb_score' if target=='sppb' else 'tugtime'
    y=old.classes(tr[label].to_numpy(),target);out=old.outframe(te,target,band,k,fold,seed);out['speed']=te.averagestridespeed
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    prior=np.bincount(y,weights=weights(ids),minlength=4);prior/=prior.sum()
    for j in range(4):out[f'Prior_p{j}']=prior[j]
    source=base.STORE/'folds'/f'{tag}.json';sl=json.loads(source.read_text());ins=np.load(base.STORE/'folds'/f'{tag}-inner.npz')
    for key,value in arrays.items():np.testing.assert_array_equal(ins[key],value)
    ledger={'task':list(task),'source':str(source),'source_sha256':sha(source),'models':{},'code_sha256':sha(__file__)}
    for name in METHODS:
        if name!='HGB':
            original='Turn5' if name=='TurnRF' else 'Bout6';info0=sl['models'][original]
            assert sha(info0['path'])==info0['sha256'];model=joblib.load(info0['path'])
            raw=old.probabilities(model,xt[name],'RF');oof=ins[original+'_oof'].copy()
            info={'path':info0['path'],'sha256':info0['sha256'],'reused_name':original}
            if 'inner_models' in info0:info['inner_models']=info0['inner_models']
        else:
            best=None;trials=[]
            for leaves in [7,15]:
                oof=np.zeros((len(tr),4));models=[]
                for i in range(3):
                    a=inner!=i;b=~a
                    model=old.fit(make(seed+fold*10+i,leaves),x[name][a],y[a],ids[a],'RF')
                    oof[b]=old.probabilities(model,x[name][b],'RF');models.append(model)
                loss=float(np.average(-np.log(np.clip(oof[np.arange(len(y)),y],1e-12,1)),weights=weights(ids)))
                trials.append({'leaves':leaves,'inner_person_logloss':loss})
                if best is None or loss<best[0]:best=(loss,leaves,oof.copy(),models)
            loss,leaves,oof,models=best
            saved=[base.p.save_model(STORE/'models'/f'{tag}-{name}-inner{i}.joblib',model) for i,model in enumerate(models)]
            model=old.fit(make(seed+fold*10+9,leaves),x[name],y,ids,'RF');raw=old.probabilities(model,xt[name],'RF')
            info=base.p.save_model(STORE/'models'/f'{tag}-{name}.joblib',model)
            info.update(inner_models=saved,selected_leaves=leaves,trials=trials)
        info.update(old.finish_prob(name,oof,raw,y,ids,inner,out,target,arrays));ledger['models'][name]=info
    out.to_csv(STORE/'folds'/f'{tag}.csv.gz',index=False);np.savez_compressed(STORE/'folds'/f'{tag}-inner.npz',**arrays)
    ledger.update(seconds=time.time()-start,predictions_sha256=sha(STORE/'folds'/f'{tag}.csv.gz'),inner_sha256=sha(STORE/'folds'/f'{tag}-inner.npz'))
    save(done,ledger);return tag,round(ledger['seconds'],1)

def prepare():
    sources=[REFERENCE/'study.py',REFERENCE/'main-verification.json',REFERENCE/'summary.json',REFERENCE/'paired-bootstrap.json']
    protocol={'code_sha256':sha(__file__),'sources':{str(f):sha(f) for f in sources},'seeds':SEEDS,'cuts':old.CUTS,
      'hypothesis':'turn-only RF retains nearly all six-feature benefit and improves same-cohort speed matching; test whether boosting reduces RF residual miss without increasing false alarms',
      'input':'same sorted 5 bouts, original4 plus turns/sec (25 columns), duration excluded; FullRF uses prior30 columns as secondary reference',
      'scope':'fixed primary [30,60) seconds x5, draw0; same2174 SPPB /1939 TUG people and original outer5 inner3 splits; five seeds',
      'model':'HistGradientBoostingClassifier lr.05 max_iter200 min_leaf20 L2=1 early_stoppingFalse, leaves7/15 chosen only by inner person-equal OOF logloss; earlier candidate breaks ties',
      'calibration':'unchanged train-only weighted scaling, OOF temperature calibration and cross-calibrated internal fpr10/sens80 thresholds',
      'criteria':'report Brier, accuracy, tail miss/FPR, same-cohort speed ranking and each seed; strong improvement requires lower outer miss with nonworse FPR and Brier, separately report all tradeoffs; no outer-based parameter retuning',
      'resource':'CPU four independent single-thread fits; no GPU required for 25-column tabular input; estimated full30-90min, pilot measures actual time',
      'deliverables':'50 HGB final models +150 selected inner models,300 candidate inner fits;100 reused RF final models;70 global metric rows and60 matched rows; model/metric/split/scaler/calibration/threshold/refit verification',
      'limits':'input choice motivated by reused outer development results; not external validation or a clinical video score; post-selection OOF used internally for calibration, outer people never used to tune',
      'storage':{'small':str(HERE),'large':str(STORE)}}
    save(HERE/'protocol.json',protocol);print('PREPARED',flush=True)

def run(mode,workers):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for f,h in protocol['sources'].items():assert sha(f)==h
    for n in ['models','folds']:(STORE/n).mkdir(parents=True,exist_ok=True)
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
    initialize();tasks=[(t,'long',5,f,s) for t,f,s in itertools.product(['sppb','tug'],range(5),SEEDS)]
    if mode=='pilot':tasks=[(t,'long',5,0,SEEDS[0]) for t in ['sppb','tug']]
    start=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in tasks]),1):
            tag,sec=f.result();print(i,len(tasks),tag,sec,round(time.time()-start),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(tasks),'seconds':time.time()-start,'code_sha256':sha(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
