"""Frozen RFs; calibration from held-out source cohorts rather than mixed people."""
from pathlib import Path
import importlib.util, os, json, time, itertools, argparse, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'20261101-mobilise-cohort-transfer'
spec=importlib.util.spec_from_file_location('transfer_reference',REFERENCE/'study.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
np,pd,joblib,old=ref.np,ref.pd,ref.joblib,ref.old
sha,save,weights=ref.sha,ref.save,ref.weights
SEEDS,KEY,COHORTS=ref.SEEDS,ref.KEY,ref.COHORTS
METHODS=['RawRF','PersonCal','CohortCal','CohortGuard']
STORE=Path('D:/gait-research-data/D_P/mobilise-cohort-calibration-20261101')

def initialize():ref.initialize()

def cell(target,held):
    tr,te,_,_=ref.cell(target,held)
    sources=[c for c in COHORTS if c!=held]
    inner=np.array([sources.index(c) for c in tr.cohort])
    return tr,te,inner,sources

def tasks(mode):
    return [(t,'MS',SEEDS[0]) for t in ['sppb','tug']] if mode=='pilot' else list(itertools.product(['sppb','tug'],COHORTS,SEEDS))

def work(task):
    target,held,seed=task;tag=f'{target}-{held}-s{seed}';folder=STORE/'folds'
    if (folder/f'{tag}.json').exists():return tag,'cached'
    start=time.time();tr,te,inner,sources=cell(target,held)
    ids=tr.participantid.to_numpy();col='total_sppb_score' if target=='sppb' else 'tugtime'
    y=old.classes(tr[col].to_numpy(),target);x=ref.inputs(tr)['Turn'];xt=ref.inputs(te)['Turn']
    source=ref.STORE/'folds'/f'{tag}.json';ledger0=json.loads(source.read_text())
    sourcecsv=ref.STORE/'folds'/f'{tag}.csv.gz';assert sha(sourcecsv)==ledger0['predictions_sha256']
    prior=pd.read_csv(sourcecsv);np.testing.assert_array_equal(prior[KEY],te[KEY]);np.testing.assert_allclose(prior.actual,te[col],rtol=0,atol=1e-12)
    out=prior[KEY+['cohort','actual','class','target','band','k','fold','seed','original_fold','speed']].copy()
    for j in range(4):out[f'Prior_p{j}']=prior[f'Prior_p{j}']
    final=ledger0['models']['LocoTurnRF'];assert sha(final['path'])==final['sha256']
    raw=old.probabilities(joblib.load(final['path']),xt,'RF')
    np.testing.assert_allclose(raw,prior[[f'LocoTurnRF_raw_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
    for variant,name in [('raw','RawRF'),('cal','PersonCal')]:
        for c in prior:
            prefix=f'LocoTurnRF_{variant}_'
            if c.startswith(prefix):out[name+'_'+c[len(prefix):]]=prior[c].to_numpy()
    oof=np.zeros((len(tr),4));models=[];splits=[]
    for i,cohort in enumerate(sources):
        a=inner!=i;b=~a
        assert set(tr.loc[a,'cohort'])==set(sources)-{cohort} and tr.loc[b,'cohort'].eq(cohort).all()
        assert np.array_equal(np.unique(y[a]),np.arange(4)) and set(ids[a]).isdisjoint(ids[b])
        m=old.fit(old.make('RF',seed+COHORTS.index(held)*10+i),x[a],y[a],ids[a],'RF')
        oof[b]=old.probabilities(m,x[b],'RF')
        models.append(ref.base.p.save_model(STORE/'models'/f'{tag}-source-{cohort}.joblib',m))
        splits.append({'valid_cohort':cohort,'train_people':sorted(set(ids[a].tolist())),'valid_people':sorted(set(ids[b].tolist()))})
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner,'cohorts':tr.cohort.to_numpy(dtype='U')}
    calibration=old.finish_prob('Domain',oof,raw,y,ids,inner,out,target,arrays)
    for j in range(4):
        out[f'CohortCal_p{j}']=out[f'Domain_cal_p{j}'];out[f'CohortGuard_p{j}']=out[f'Domain_cal_p{j}']
    for policy in ['p50','fpr10','sens80']:out[f'CohortCal_{policy}']=out[f'Domain_cal_{policy}']
    low=(y==old.lowclass(target)).astype(int);cross=arrays['Domain_crosscal'][:,old.lowclass(target)]
    group_cuts={c:old.thresholds(cross[inner==i],low[inner==i],ids[inner==i])['fpr10'] for i,c in enumerate(sources)}
    guard=max(group_cuts.values())
    out['CohortGuard_fpr10']=(out[f'CohortGuard_p{old.lowclass(target)}']>=guard).astype(int)
    out.to_csv(folder/f'{tag}.csv.gz',index=False);np.savez_compressed(folder/f'{tag}-inner.npz',**arrays)
    save(folder/f'{tag}.json',{'task':list(task),'code_sha256':sha(__file__),'source_ledger':str(source),'source_ledger_sha256':sha(source),'source_predictions':str(sourcecsv),'source_predictions_sha256':sha(sourcecsv),'final_model':{'path':final['path'],'sha256':final['sha256']},'inner_models':models,'splits':splits,'calibration':calibration,'guard_group_cuts':group_cuts,'guard_cut':guard,'predictions_sha256':sha(folder/f'{tag}.csv.gz'),'inner_sha256':sha(folder/f'{tag}-inner.npz'),'seconds':time.time()-start})
    return tag,round(time.time()-start,1)

def prepare():
    initialize()
    for target,held in itertools.product(['sppb','tug'],COHORTS):
        tr,te,inner,sources=cell(target,held);col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target)
        for i in range(3):
            assert np.array_equal(np.unique(y[inner!=i]),np.arange(4))
            assert set(y[inner==i]==old.lowclass(target))=={False,True}
    paths=[REFERENCE/'study.py',REFERENCE/'verify.py',REFERENCE/'main-verification.json',REFERENCE/'summary.json',old.BASE/'features.csv',old.BASE/'selected-bouts.csv',ref.base.p.FACTORIAL/'audited-bouts.csv']
    protocol={'code_sha256':sha(__file__),'sources':{str(p):sha(p) for p in paths},'seeds':SEEDS,
      'hypothesis':'Calibration OOF from unseen source disease cohorts better matches deployment to a held disease than ordinary mixed-person OOF; test calibration and a conservative source-group FPR guard without changing final RF.',
      'input':'fixed original4+turns/sec,25columns,30-60sec5bouts draw0; no disease/age metadata predictors',
      'labels':'same four SPPB/TUG bands; lowSPPB<=3/TUG>20sec;2174/1939people7356/5688visits',
      'outer':'same four disease-cohort holdouts and five seeds as cohort-transfer; exact evaluation people/visits/bouts and frozen final LocoTurnRF',
      'inner':'NEW remaining3source cohorts each held as whole; train RF on other2cohorts to create OOF. No evaluated outer disease in model/scaler/temperature/threshold selection.',
      'methods':{'RawRF':'existing final raw probabilities and raw internal warning','PersonCal':'existing mixed-person OOF temperature and warning','CohortCal':'source-cohort OOF temperature with source-cohort cross-calibration for pooled fpr10 threshold','CohortGuard':'same probabilities as CohortCal; maximum of the three source-cohort fpr10 thresholds'},
      'fixed_model':'reused40final RF128 leaf20 max_features1;120newinnerRF same parameters,person-equal weights. No hyperparameter search.',
      'selection':'temperature[.25,4], all-source person weighted OOF NLL; cross-fit temperature excluding source calibration group before warning thresholds. No choice using held-disease labels.',
      'primary_success':'CohortCal must reduce multiclass Brier and not increase pooled miss or FPR vs PersonCal in both outcomes, direction at least4/5seeds. Report all regardless. CohortGuard is a predeclared trade-off: source-group cap does not guarantee unseen-group cap; never call reducedFPR alone a gain.',
      'diagnostics':'pooled/per-disease Brier,RPS,miss,FPR,warning rate; identical same-disease/original-person-fold speed pairs. No new diagnostic cutoff search.',
      'resource':'CPU4workers, estimate5-15min training+audit; pilot2tasks heldMS firstseed; no GPU',
      'expected':{'tasks':40,'new_inner_models':120,'reused_final_models':40,'model_replays':160,'selected_refits':8,'metric_rows':50,'cohort_metric_rows':200,'matched_rows':40},
      'limits':'same development people reused after outcomes informed hypothesis; only3sourcegroups; RFs trained on2 vs3diseases for OOF/final differ; rare severe COPD/PD; no external/video validity or universal FPR guarantee',
      'storage':{'small':str(HERE),'large':str(STORE)}}
    save(HERE/'protocol.json',protocol);print('PREPARED',flush=True)

def run(mode,workers):
    p=json.loads((HERE/'protocol.json').read_text());assert p['code_sha256']==sha(__file__)
    for path,digest in p['sources'].items():assert sha(path)==digest
    assert HERE.drive.upper()=='C:' and STORE.drive.upper()=='D:'
    for folder in ['models','folds']:(STORE/folder).mkdir(parents=True,exist_ok=True)
    for var in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[var]='1'
    initialize();start=time.time();todo=tasks(mode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in todo]),1):print(i,len(todo),*f.result(),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(todo),'seconds':time.time()-start,'code_sha256':sha(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
