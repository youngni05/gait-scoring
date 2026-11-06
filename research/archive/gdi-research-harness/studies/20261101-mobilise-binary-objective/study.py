"""Direct low-function objective versus multiclass risk and binary calibration."""
from pathlib import Path
import importlib.util,sys,os,json,time,itertools,argparse,datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;REFERENCE=HERE.parent/'20261101-mobilise-turn-hgb'
spec=importlib.util.spec_from_file_location('hgb_reference',REFERENCE/'study.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
np,pd,joblib,old=ref.np,ref.pd,ref.joblib,ref.old
sha,save,weights=ref.sha,ref.save,ref.weights;SEEDS,KEY=ref.SEEDS,ref.KEY
STORE=Path('D:/gait-research-data/D_P/mobilise-binary-objective-20261101')
METHODS=['OriginalRF','MultiRF','MultiHGB','BinaryRF','BinaryHGB']
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.special import logit

def initialize():ref.initialize()
def inputs(frame):return ref.inputs(frame,5)['TurnRF']
def make(name,seed,leaves=7):return old.make('RF',seed) if name=='BinaryRF' else ref.make(seed,leaves)
def prob(model,x):
    assert np.array_equal(model.classes_,[0,1])
    return model.predict_proba(x)[:,1]
def mapper(p,y,ids):
    m=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-8)
    m.fit(logit(np.clip(p,1e-6,1-1e-6))[:,None],y,sample_weight=weights(ids))
    assert m.n_iter_.max()<2000;return m
def calibrated(m,p):return m.predict_proba(logit(np.clip(p,1e-6,1-1e-6))[:,None])[:,1]
def model_save(path,m):return ref.base.p.save_model(path,m)

def work(task):
    target,fold,seed=task;tag=f'{target}-long-5-f{fold}-s{seed}';done=STORE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,sp=old.cell_data(target,'long',5,fold);x,xt=inputs(tr),inputs(te);ids=tr.participantid.to_numpy()
    label='total_sppb_score' if target=='sppb' else 'tugtime';classes=old.classes(tr[label].to_numpy(),target);y=(classes==old.lowclass(target)).astype(int)
    out=old.outframe(te,target,'long',5,fold,seed);out['is_low']=(out['class']==old.lowclass(target)).astype(int);out['speed']=te.averagestridespeed
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'class4':classes,'inner':inner}
    source=ref.STORE/'folds'/f'{tag}.json';source_ledger=json.loads(source.read_text());ins=np.load(ref.STORE/'folds'/f'{tag}-inner.npz');source_out=pd.read_csv(ref.STORE/'folds'/f'{tag}.csv.gz')
    for key,value in [('ids',ids),('inner',inner),('visits',arrays['visits']),('y',classes)]:np.testing.assert_array_equal(ins[key],value)
    np.testing.assert_array_equal(source_out[KEY],out[KEY])
    ledger={'task':list(task),'source':str(source),'source_sha256':sha(source),'source_predictions_sha256':sha(ref.STORE/'folds'/f'{tag}.csv.gz'),'code_sha256':sha(__file__),'models':{}}
    low=old.lowclass(target)
    for name in METHODS:
        info={}
        if name.startswith('Binary'):
            best=None;trials=[]
            for leaves in ([7,15] if name=='BinaryHGB' else [0]):
                oof=np.zeros(len(tr));models=[]
                for i in range(3):
                    a=inner!=i;b=~a;m=old.fit(make(name,seed+fold*10+i,leaves),x[a],y[a],ids[a],'RF')
                    oof[b]=prob(m,x[b]);models.append(m)
                loss=float(np.average(-(y*np.log(np.clip(oof,1e-12,1))+(1-y)*np.log(np.clip(1-oof,1e-12,1))),weights=weights(ids)))
                trials.append({'leaves':leaves,'inner_logloss':loss})
                if best is None or loss<best[0]:best=(loss,leaves,oof.copy(),models)
            loss,leaves,oof,models=best;info['inner_models']=[model_save(STORE/'models'/f'{tag}-{name}-inner{i}.joblib',m) for i,m in enumerate(models)]
            m=old.fit(make(name,seed+fold*10+9,leaves),x,y,ids,'RF');raw=prob(m,xt)
            info.update(model_save(STORE/'models'/f'{tag}-{name}.joblib',m));info.update(trials=trials,selected_leaves=leaves)
        else:
            original='HGB' if name=='MultiHGB' else 'TurnRF';info.update(source_model=source_ledger['models'][original],source_name=original)
            if name=='OriginalRF':
                oof=ins['TurnRF_crosscal'][:,low];raw=source_out[f'TurnRF_cal_p{low}'].to_numpy()
            else:oof=ins[original+'_oof'][:,low];raw=source_out[f'{original}_raw_p{low}'].to_numpy()
        arrays[name+'_raw_oof']=oof
        if name=='OriginalRF':cross=oof.copy();final=raw.copy();info['calibration']='unchanged multiclass temperature baseline'
        else:
            cross=np.zeros(len(tr));saved=[]
            for i in range(3):
                a=inner!=i;b=~a;m=mapper(oof[a],y[a],ids[a]);cross[b]=calibrated(m,oof[b]);saved.append(model_save(STORE/'models'/f'{tag}-{name}-cal{i}.joblib',m))
            m=mapper(oof,y,ids);final=calibrated(m,raw);info.update(calibrator=model_save(STORE/'models'/f'{tag}-{name}-cal.joblib',m),cross_calibrators=saved)
        arrays[name+'_crosscal']=cross;out[name+'_raw_probability']=raw;out[name+'_probability']=final
        thresholds=old.thresholds(cross,y,ids);info['thresholds']=thresholds
        for policy,cut in thresholds.items():out[name+'_'+policy]=(final>=cut).astype(int)
        if name=='OriginalRF':np.testing.assert_array_equal(out[name+'_fpr10'],source_out['TurnRF_cal_fpr10'])
        ledger['models'][name]=info
    out.to_csv(STORE/'folds'/f'{tag}.csv.gz',index=False);np.savez_compressed(STORE/'folds'/f'{tag}-inner.npz',**arrays)
    ledger.update(seconds=time.time()-start,predictions_sha256=sha(STORE/'folds'/f'{tag}.csv.gz'),inner_sha256=sha(STORE/'folds'/f'{tag}-inner.npz'));save(done,ledger)
    return tag,round(ledger['seconds'],1)

def prepare():
    paths=[REFERENCE/'study.py',REFERENCE/'main-verification.json',REFERENCE/'summary.json',REFERENCE/'paired-bootstrap.json']
    save(HERE/'protocol.json',{'code_sha256':sha(__file__),'sources':{str(p):sha(p) for p in paths},'seeds':SEEDS,
      'hypothesis':'four-class HGB worsened RF in all seeds; test objective mismatch by learning low function directly, separating binary calibration from classifier changes',
      'target':'SPPB<=3 or TUG>20 seconds; retain original measured scores and labels, output binary probability not a fabricated four-class distribution',
      'data':'same2174 SPPB/1939 TUG people, primary30-60sec5bouts, draw0,25columns original4+turns/sec, original outer5/inner3 person folds and5seeds',
      'methods':METHODS,'controls':'OriginalRF exact previous multiclass temperature/threshold baseline; MultiRF/MultiHGB reuse raw multiclass low probability with common binary Platt calibration; BinaryRF/BinaryHGB learn binary labels with same calibration',
      'models':'RF128 leaf20 max_features1; HGB lr.05 iterations200 minleaf20 L2=1 early_stoppingFalse leaves7/15 chosen inner person logloss. No class weighting, person-equal sample weights',
      'calibration':'LR C1 on logit risk clipped1e-6; train-inner OOF fit, cross-fit3 calibrators for internal fpr10/sens80 threshold selection; final calibrator on all inner OOF only',
      'selection_limit':'inner OOF reused for model setting and calibration; inner metrics are selection only; outer held-out people never used for parameter or threshold tuning',
      'criteria':'binary Brier/AUROC/AP/calibration, fpr10 miss and actual FPR, same-cohort nearest-speed ordering and5seed stability; call joint improvement only if miss lower without higher FPR and binary Brier; report tradeoffs and person-cluster95% intervals',
      'resource':'CPU4single-thread fits; estimate full15-45min; pilot first2tasks then same full50 tasks, no GPU needed for25columns',
      'deliverables':'50tasks,100newfinal classifiers+300selectedinner classifiers,450candidateinner fits;16calibrators per task, source multiclass models reused;50global metric rows and100match rows; replay, train-only scaler/calibration, source identity and selected refits',
      'limits':'same development evaluation people reused; binary warning experiment does not test four-band accuracy or video clinical validity',
      'storage':{'small':str(HERE),'large':str(STORE)}})
    print('PREPARED',flush=True)

def run(mode,workers):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for p,h in protocol['sources'].items():assert sha(p)==h
    for n in ['models','folds']:(STORE/n).mkdir(parents=True,exist_ok=True)
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
    initialize();tasks=list(itertools.product(['sppb','tug'],range(5),SEEDS))
    if mode=='pilot':tasks=[(t,0,SEEDS[0]) for t in ['sppb','tug']]
    start=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in tasks]),1):
            tag,sec=f.result();print(i,len(tasks),tag,sec,round(time.time()-start),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(tasks),'seconds':time.time()-start,'code_sha256':sha(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
