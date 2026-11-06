"""Do frozen gait RFs distinguish different visits of the same held-out person?"""
from pathlib import Path
import importlib.util,json,itertools,time,argparse,os,datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;REFERENCE=HERE.parent/'20261101-mobilise-cohort-transfer'
spec=importlib.util.spec_from_file_location('cohort_reference',REFERENCE/'study.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
np,pd,joblib,old=ref.np,ref.pd,ref.joblib,ref.old
sha,save,weights=ref.sha,ref.save,ref.weights;KEY,SEEDS,COHORTS=ref.KEY,ref.SEEDS,ref.COHORTS
STORE=Path('D:/gait-research-data/D_P/mobilise-within-person-20261101')
METHODS=['NoChange','Speed','CoreRF','TurnRF']

def tasks(mode):return [(t,'MS',SEEDS[0]) for t in ['sppb','tug']] if mode=='pilot' else list(itertools.product(['sppb','tug'],COHORTS,SEEDS))

def pairs(d):
    rows=[]
    for person,g in d.groupby('participantid'):
        for a,b in itertools.combinations(g.sort_values('visit_number').index,2):rows.append((int(person),int(a),int(b)))
    return rows

def make_pairs(d,target):
    index=pairs(d);out=pd.DataFrame(index,columns=['participantid','a','b']);a=out.a.to_numpy();b=out.b.to_numpy()
    for col in ['visit_number','actual','class','cohort','speed']:
        for side,ix in [('a',a),('b',b)]:out[col+'_'+side]=d[col].to_numpy()[ix]
    for name,source in [('CoreRF','LocoCoreRF_cal'),('TurnRF','LocoTurnRF_cal')]:
        prob=d[[f'{source}_p{j}' for j in range(4)]].to_numpy();expected=prob@np.arange(4);risk=prob[:,old.lowclass(target)]
        for side,ix in [('a',a),('b',b)]:
            out[f'{name}_expected_{side}']=expected[ix];out[f'{name}_risk_{side}']=risk[ix]
            out[f'{name}_class_{side}']=prob.argmax(1)[ix];out[f'{name}_warn_{side}']=d[f'{source}_fpr10'].to_numpy()[ix]
    return out

def mean(values,ids):
    d=pd.DataFrame({'id':ids,'value':values}).dropna();return float(d.groupby('id').value.mean().mean()) if len(d) else np.nan

def vectors(d,target,name):
    truth=(d.class_b-d.class_a).to_numpy();low=old.lowclass(target)
    switch=d.class_a.eq(low).to_numpy()!=d.class_b.eq(low).to_numpy();change=truth!=0
    if name=='Speed':
        delta=(d.speed_b-d.speed_a).to_numpy()*(1 if target=='sppb' else -1)
        return {'direction':np.where(change,(delta*truth>0).astype(float)+.5*(delta==0),np.nan),'large_change_direction':np.where(abs(truth)>=2,(delta*truth>0).astype(float)+.5*(delta==0),np.nan)}
    if name=='NoChange':delta=np.zeros(len(d));wa=wb=np.zeros(len(d),int);ca=cb=np.zeros(len(d),int)
    else:
        delta=(d[f'{name}_expected_b']-d[f'{name}_expected_a']).to_numpy();wa=d[f'{name}_warn_a'].to_numpy();wb=d[f'{name}_warn_b'].to_numpy();ca=d[f'{name}_class_a'].to_numpy();cb=d[f'{name}_class_b'].to_numpy()
    return {'direction':np.where(change,(delta*truth>0).astype(float)+.5*(delta==0),np.nan),
      'large_change_direction':np.where(abs(truth)>=2,(delta*truth>0).astype(float)+.5*(delta==0),np.nan),
      'class_delta_mae':abs(delta-truth),'false_class_switch':np.where(~change,ca!=cb,np.nan),
      'correct_low_switch':np.where(switch,(wa==d.class_a.eq(low).to_numpy())&(wb==d.class_b.eq(low).to_numpy()),np.nan),
      'false_low_switch':np.where(~switch,wa!=wb,np.nan)}

def work(task):
    target,cohort,seed=task;tag=f'{target}-{cohort}-s{seed}';done=STORE/f'{tag}.json'
    if done.exists():return tag,'cached'
    source=ref.STORE/'folds'/f'{tag}.csv.gz';ledger=ref.STORE/'folds'/f'{tag}.json';l=json.loads(ledger.read_text());assert sha(source)==l['predictions_sha256']
    d=pd.read_csv(source);assert not d.duplicated(KEY).any();out=make_pairs(d,target);out['target']=target;out['seed']=seed;out['held_cohort']=cohort
    out.to_csv(STORE/f'{tag}.csv.gz',index=False)
    save(done,{'task':list(task),'code_sha256':sha(__file__),'source_predictions':str(source),'source_predictions_sha256':sha(source),'source_ledger':str(ledger),'source_ledger_sha256':sha(ledger),'pairs':len(out),'people':int(out.participantid.nunique()),'predictions_sha256':sha(STORE/f'{tag}.csv.gz')})
    return tag,len(out)

def prepare():
    ref.initialize();counts=[]
    for target in ['sppb','tug']:
        d=ref.all_data(target);col='total_sppb_score' if target=='sppb' else 'tugtime';d['class']=old.classes(d[col].to_numpy(),target)
        q=pairs(d);a=np.array([r[1] for r in q]);b=np.array([r[2] for r in q]);assert (d.participantid.to_numpy()[a]==d.participantid.to_numpy()[b]).all()
        counts.append({'target':target,'all_people':int(d.participantid.nunique()),'paired_people':len(set(r[0] for r in q)),'pairs':len(q),'class_changed_pairs':int((d['class'].to_numpy()[a]!=d['class'].to_numpy()[b]).sum())})
    evidence=HERE.parent/'20261101-mobilise-cohort-calibration'
    source=[REFERENCE/'study.py',REFERENCE/'main-verification.json',REFERENCE/'summary.json',evidence/'main-verification.json',evidence/'summary.json',old.BASE/'features.csv',old.BASE/'selected-bouts.csv']
    save(HERE/'protocol.json',{'code_sha256':sha(__file__),'sources':{str(p):sha(p) for p in source},'seeds':SEEDS,'counts':counts,
      'hypothesis':'Between-person gait discrimination may reflect stable traits; test whether the SAME frozen RFs also order different-visit functional score bands within a held-out person.',
      'input_labels':'same2174/1939people,7356/5688visits,30-60sec5bouts,draw0; core20/turn25inputs and four score classes unchanged',
      'evaluation':'same disease-cohort holdout4 and5seeds as transfer study; only people with2+eligiblevisits contribute pairs; all person visits remain excluded from model/calibration',
      'pairing':'all unordered distinct visit pairs within a person; canonical lexical visit IDs, NOT an assumed chronological interval. People, not pairs, are independent units.',
      'methods':'NoChange zero expected-class difference; Speed signed speed difference only; frozen CoreRF/TurnRF expected class differences and original fpr10 warnings. Speed is an ordering control, not a calibrated score-difference model.',
      'metrics':'direction on nonzero true class changes,half credit for ties; direction for>=2classchanges; expected-class deltaMAE vs NoChange/Core; false argmax switches when true class same; exact low-warning switch and false warning switches. Person equal means and paired person bootstrap,5seedstability.',
      'success':'diagnostic, no selection/tuning. TurnRF signal beyond speed/Core requires direction better in>=4/5seeds for both targets; show uncertainty and deltaMAE/false transitions as well. Null result is retained.',
      'calibration':'reuse train-only frozen source probabilities and warning thresholds; no new calibration/model training or held-out label selection',
      'expected':{'tasks':40,'frozen_final_model_replays':80,'new_models':0,'metric_rows':40,'cohort_metric_rows':160},
      'resource':'CPU4workers; model replay80,personpair metrics and2000bootstrap replicates,estimate2-10min; pilot2tasks heldMS firstseed',
      'limits':'repeat visits are not independent people; no causal/clinical change threshold,no claim of temporal improvement or test-retest reliability; band changes include measurement noise; same development data reused',
      'storage':{'small':str(HERE),'large':str(STORE)}});print('PREPARED',counts,flush=True)

def run(mode,workers):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for f,h in protocol['sources'].items():assert sha(f)==h
    assert HERE.drive.upper()=='C:' and STORE.drive.upper()=='D:';STORE.mkdir(parents=True,exist_ok=True)
    start=time.time();todo=tasks(mode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(work,t) for t in todo]):print(*f.result(),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(todo),'seconds':time.time()-start,'code_sha256':sha(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
