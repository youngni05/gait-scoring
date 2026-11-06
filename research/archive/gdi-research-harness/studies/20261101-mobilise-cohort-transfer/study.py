"""Disease-cohort holdout stress test; fixed gait inputs and RF hyperparameters."""
from pathlib import Path
import importlib.util,os,json,time,itertools,argparse,datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;REFERENCE=HERE.parent/'20261101-mobilise-duration-turns';EVIDENCE=HERE.parent/'20261101-mobilise-turn-ordinal'
spec=importlib.util.spec_from_file_location('duration_reference',REFERENCE/'study.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
np,pd,joblib,old=base.np,base.pd,base.joblib,base.old
sha,save,weights=base.sha,base.save,base.weights;SEEDS,KEY=base.SEEDS,base.KEY
COHORTS=['COPD','MS','PD','PFF'];METHODS=['MixedCoreRF','MixedTurnRF','LocoCoreRF','LocoTurnRF']
STORE=Path('D:/gait-research-data/D_P/mobilise-cohort-transfer-20261101')
from sklearn.model_selection import StratifiedKFold

def initialize():base.initialize()
def all_data(target):return old.DATA[old.DATA[target+'_ok']&old.DATA.band.eq('long')&old.DATA.k.eq(5)].sort_values(KEY).reset_index(drop=True)
def inputs(d):
    x=base.inputs(d,5);return {'Core':x['Bout4'],'Turn':x['Turn5']}
def cell(target,cohort):
    d=all_data(target);tr=d[d.cohort.ne(cohort)].reset_index(drop=True);te=d[d.cohort.eq(cohort)].reset_index(drop=True)
    people=tr[['participantid','cohort']].drop_duplicates().sort_values('participantid').reset_index(drop=True)
    assert not people.participantid.duplicated().any() and set(tr.participantid).isdisjoint(te.participantid)
    inner=np.full(len(tr),-1);splits=[]
    splitter=StratifiedKFold(3,shuffle=True,random_state=20261101+COHORTS.index(cohort))
    for i,(a,b) in enumerate(splitter.split(people,people.cohort)):
        train=people.iloc[a].participantid.to_numpy();valid=people.iloc[b].participantid.to_numpy();inner[tr.participantid.isin(valid)]=i
        splits.append({'train':train.tolist(),'valid':valid.tolist()})
    assert (inner>=0).all();return tr,te,inner,splits

def work(task):
    target,cohort,seed=task;tag=f'{target}-{cohort}-s{seed}';done=STORE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,splits=cell(target,cohort);x,xt=inputs(tr),inputs(te);ids=tr.participantid.to_numpy()
    col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target)
    assert np.array_equal(np.unique(y),np.arange(4))
    out=old.outframe(te,target,'long',5,COHORTS.index(cohort),seed);out['original_fold']=te.fold;out['speed']=te.averagestridespeed
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    prior=np.bincount(y,weights=weights(ids),minlength=4);prior/=prior.sum()
    for j in range(4):out[f'Prior_p{j}']=prior[j]
    ledger={'task':list(task),'code_sha256':sha(__file__),'inner_people':splits,'train_people':sorted(tr.participantid.unique().tolist()),'test_people':sorted(te.participantid.unique().tolist()),'models':{},'reference_files':{}}
    # Mixed-cohort OOF references use the exact same evaluation visits, not newly fit controls.
    references=[]
    for fold in range(5):
        stem=f'{target}-long-5-f{fold}-s{seed}'
        p=base.STORE/'folds'/f'{stem}.csv.gz';j=base.STORE/'folds'/f'{stem}.json'
        source=json.loads(j.read_text());assert sha(p)==source['predictions_sha256']
        references.append(pd.read_csv(p));ledger['reference_files'][str(fold)]={'predictions':str(p),'predictions_sha256':sha(p),'ledger':str(j),'ledger_sha256':sha(j)}
    source=pd.concat(references,ignore_index=True)
    selected=te[KEY].merge(source,on=KEY,validate='one_to_one',how='left');np.testing.assert_array_equal(selected.actual,te[col])
    np.testing.assert_array_equal(selected.fold,te.fold)
    for name,original in [('MixedCoreRF','Bout4'),('MixedTurnRF','Turn5')]:
        for c in source.columns:
            if c.startswith(original+'_'):out[name+c[len(original):]]=selected[c].to_numpy()
    for name,kind in [('LocoCoreRF','Core'),('LocoTurnRF','Turn')]:
        oof=np.zeros((len(tr),4));saved=[]
        for i in range(3):
            a=inner!=i;b=~a;assert np.array_equal(np.unique(y[a]),np.arange(4))
            model=old.fit(old.make('RF',seed+COHORTS.index(cohort)*10+i),x[kind][a],y[a],ids[a],'RF')
            oof[b]=old.probabilities(model,x[kind][b],'RF');saved.append(base.p.save_model(STORE/'models'/f'{tag}-{name}-inner{i}.joblib',model))
        model=old.fit(old.make('RF',seed+COHORTS.index(cohort)*10+9),x[kind],y,ids,'RF');raw=old.probabilities(model,xt[kind],'RF')
        info=base.p.save_model(STORE/'models'/f'{tag}-{name}.joblib',model);info['inner_models']=saved
        info.update(old.finish_prob(name,oof,raw,y,ids,inner,out,target,arrays));ledger['models'][name]=info
    out.to_csv(STORE/'folds'/f'{tag}.csv.gz',index=False);np.savez_compressed(STORE/'folds'/f'{tag}-inner.npz',**arrays)
    ledger.update(seconds=time.time()-start,predictions_sha256=sha(STORE/'folds'/f'{tag}.csv.gz'),inner_sha256=sha(STORE/'folds'/f'{tag}-inner.npz'));save(done,ledger)
    return tag,round(ledger['seconds'],1)

def prepare():
    initialize();counts=[]
    for target in ['sppb','tug']:
        col='total_sppb_score' if target=='sppb' else 'tugtime';d=all_data(target)
        assert set(d.cohort)==set(COHORTS)
        for cohort in COHORTS:
            tr,te,inner,splits=cell(target,cohort);y=old.classes(tr[col].to_numpy(),target)
            assert np.array_equal(np.unique(y),np.arange(4))
            for i in range(3):assert np.array_equal(np.unique(y[inner!=i]),np.arange(4))
            low=te[col].le(3) if target=='sppb' else te[col].gt(20)
            counts.append({'target':target,'cohort':cohort,'train_people':int(tr.participantid.nunique()),'test_people':int(te.participantid.nunique()),'test_visits':len(te),'low_people':int(te.loc[low,'participantid'].nunique()),'low_visits':int(low.sum())})
    paths=[REFERENCE/'study.py',REFERENCE/'main-verification.json',EVIDENCE/'main-verification.json',EVIDENCE/'summary.json',old.BASE/'features.csv',old.BASE/'selected-bouts.csv']
    save(HERE/'protocol.json',{'code_sha256':sha(__file__),'sources':{str(p):sha(p) for p in paths},'seeds':SEEDS,
      'hypothesis':'small objective changes did not lower miss; test whether the apparent gait/turn signal and low-function warning depend on disease-cohort distribution',
      'input':'same selected30-60sec5bouts, Core20 original4 or Turn25 original4+turns/sec; no age/cohort/label metadata predictors',
      'target':'unchanged4 score bands; low SPPB<=3/TUG>20s. Same2174/1939 people and7356/5688 visits across all holdouts,5seeds',
      'outer':'deliberately NEW four disease-cohort holdouts COPD/MS/PD/PFF; one whole disease cohort absent from training/calibration; this replaces original mixed-cohort5fold for the stress test',
      'inner':'3 person folds within remaining3cohorts, stratified by cohort, randomstate20261101+held-cohort-index, fixed across5seeds; all4classes supported in every inner training split',
      'controls':'MixedCore/MixedTurn reuse original person-held-out mixed-cohort OOF predictions on exact same visits; LocoCore/LocoTurn train without held cohort. Models RF128 leaf20 max_features1, person equal weights, no hyperparameter search',
      'calibration':'train-only weighted X scaling, inner OOF temperature, cross-calibrated fpr10/sens80 thresholds only from remaining cohorts; no recalibration on unseen cohort',
      'matching':'same disease and ORIGINAL person fold, different people within .05m/s; pairs shared across models/seeds. Avoid mixing differently calibrated original RF folds within a pair',
      'criteria':'diagnose per-cohort and pooled Brier/RPS/tail miss/FPR, turn benefit in mixed versus unseen cohorts, same-cohort speed ordering and5seed stability. Stress-test degradation is a result, not a reason to tune on held disease',
      'confounds':'training size differs; cohorts differ in pathology, activity environment and labels; this is not causal identification or a truly external new collection. Rare severe COPD/PD counts limit subgroup estimates',
      'resource':'CPU4single-thread fits, estimate15-40min; pilot two targets held MS first. No GPU needed',
      'deliverables':'40tasks,80newfinal+240inner models; original mixed-cohort RF references reused;720model subset/inner replays and16selected refits,90pooled metric rows40matched rows plus per-cohort metrics',
      'counts':counts,'storage':{'small':str(HERE),'large':str(STORE)}})
    print('PREPARED',counts,flush=True)

def run(mode,workers):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for p,h in protocol['sources'].items():assert sha(p)==h
    for n in ['models','folds']:(STORE/n).mkdir(parents=True,exist_ok=True)
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
    initialize();tasks=list(itertools.product(['sppb','tug'],COHORTS,SEEDS))
    if mode=='pilot':tasks=[(t,'MS',SEEDS[0]) for t in ['sppb','tug']]
    start=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in tasks]),1):
            tag,seconds=f.result();print(i,len(tasks),tag,seconds,round(time.time()-start),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(tasks),'seconds':time.time()-start,'code_sha256':sha(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
