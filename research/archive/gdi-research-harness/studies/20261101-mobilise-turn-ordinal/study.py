"""Use ordered severity information with the fixed turn-only bout inputs."""
from pathlib import Path
import importlib.util,os,json,time,itertools,argparse,datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;REFERENCE=HERE.parent/'20261101-mobilise-turn-hgb';EVIDENCE=HERE.parent/'20261101-mobilise-binary-objective'
spec=importlib.util.spec_from_file_location('hgb_reference',REFERENCE/'study.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
np,pd,joblib,old=ref.np,ref.pd,ref.joblib,ref.old
sha,save,weights=ref.sha,ref.save,ref.weights;SEEDS,KEY=ref.SEEDS,ref.KEY
METHODS=['RF','OrdinalRF'];STORE=Path('D:/gait-research-data/D_P/mobilise-turn-ordinal-20261101')
def initialize():ref.initialize()
def inputs(frame,k=5):
    x=ref.inputs(frame,k)['TurnRF'];return {'RF':x,'OrdinalRF':x}
def probability(model,x,name):return old.probabilities(model,x,name)

def work(task):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}';done=STORE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,_=old.cell_data(target,band,k,fold);x,xt=inputs(tr,k),inputs(te,k);ids=tr.participantid.to_numpy()
    col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target)
    out=old.outframe(te,target,band,k,fold,seed);out['speed']=te.averagestridespeed
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    prior=np.bincount(y,weights=weights(ids),minlength=4);prior/=prior.sum()
    for j in range(4):out[f'Prior_p{j}']=prior[j]
    source=ref.STORE/'folds'/f'{tag}.json';sl=json.loads(source.read_text());ins=np.load(ref.STORE/'folds'/f'{tag}-inner.npz')
    for key,value in arrays.items():np.testing.assert_array_equal(ins[key],value)
    ledger={'task':list(task),'source':str(source),'source_sha256':sha(source),'models':{},'code_sha256':sha(__file__)}
    for name in METHODS:
        if name=='RF':
            info0=sl['models']['TurnRF'];assert sha(info0['path'])==info0['sha256'];model=joblib.load(info0['path'])
            oof=ins['TurnRF_oof'].copy();raw=probability(model,xt[name],name)
            info={'path':info0['path'],'sha256':info0['sha256'],'inner_models':info0['inner_models'],'reused_name':'TurnRF'}
        else:
            oof=np.zeros((len(tr),4));saved=[]
            for i in range(3):
                a=inner!=i;b=~a;model=old.fit(old.make(name,seed+fold*10+i),x[name][a],y[a],ids[a],name)
                oof[b]=probability(model,x[name][b],name);saved.append(ref.base.p.save_model(STORE/'models'/f'{tag}-{name}-inner{i}.joblib',model))
            model=old.fit(old.make(name,seed+fold*10+9),x[name],y,ids,name);raw=probability(model,xt[name],name)
            info=ref.base.p.save_model(STORE/'models'/f'{tag}-{name}.joblib',model);info['inner_models']=saved
        info.update(old.finish_prob(name,oof,raw,y,ids,inner,out,target,arrays));ledger['models'][name]=info
    out.to_csv(STORE/'folds'/f'{tag}.csv.gz',index=False);np.savez_compressed(STORE/'folds'/f'{tag}-inner.npz',**arrays)
    ledger.update(seconds=time.time()-start,predictions_sha256=sha(STORE/'folds'/f'{tag}.csv.gz'),inner_sha256=sha(STORE/'folds'/f'{tag}-inner.npz'));save(done,ledger)
    return tag,round(ledger['seconds'],1)

def prepare():
    paths=[REFERENCE/'study.py',REFERENCE/'main-verification.json',EVIDENCE/'main-verification.json',EVIDENCE/'summary.json',EVIDENCE/'paired-bootstrap.json']
    save(HERE/'protocol.json',{'code_sha256':sha(__file__),'sources':{str(p):sha(p) for p in paths},'seeds':SEEDS,'cuts':old.CUTS,
      'hypothesis':'direct binary objective and binary recalibration did not improve RF; test ordered four-band severity instead of discarding severity or treating bands as unrelated',
      'input':'same25 columns,5 speed-sorted30-60second bouts, original4+turns/sec, no duration or metadata',
      'data':'same2174 SPPB/1939 TUG people, original outer5/inner3 person folds, draw0,5seeds',
      'RF':'reuse verified nominal RF128trees leaf20 max_features1 gini person-equal weights',
      'OrdinalRF':'same forest hyperparameters, shared multioutput regressor to I(class>0),I(class>1),I(class>2), squared loss proportional to RPS; shared leaves ensure ordered cumulative probabilities; difference to4 probabilities',
      'selection':'no new hyperparameter search; train-only weighted scaling, unchanged OOF temperature and cross-calibrated internal fpr10/sens80 threshold choice',
      'criteria':'compare multiclass Brier/RPS/accuracy/tail miss/FPR, same-cohort speed order and5seed stability; joint gain requires lower miss with nonworse FPR and Brier; evaluate rather than tune on outer predictions',
      'resource':'CPU4single-thread models; estimated15-35min,2taskpilot first; no GPU needed for25-column input',
      'deliverables':'50 new final+150 inner ordinal models,50 reused RF final+150 inner;50tasks400model replays8selected refits50metric rows40matched rows',
      'novelty':'ordinal RF was previously tested on four medians, not the current retained25-column turn-only bout representation; this is a fixed-input objective ablation',
      'limits':'same development people repeatedly explored; no video/external validity. If no gain, prioritize measurement/input/domain diagnostics over further small objective grids.',
      'storage':{'small':str(HERE),'large':str(STORE)}})
    print('PREPARED',flush=True)

def run(mode,workers):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for p,h in protocol['sources'].items():assert sha(p)==h
    for n in ['models','folds']:(STORE/n).mkdir(parents=True,exist_ok=True)
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
    initialize();tasks=[(t,'long',5,f,s) for t,f,s in itertools.product(['sppb','tug'],range(5),SEEDS)]
    if mode=='pilot':tasks=[(t,'long',5,0,SEEDS[0]) for t in ['sppb','tug']]
    start=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in tasks]),1):
            tag,seconds=f.result();print(i,len(tasks),tag,seconds,round(time.time()-start),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(tasks),'seconds':time.time()-start,'code_sha256':sha(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
