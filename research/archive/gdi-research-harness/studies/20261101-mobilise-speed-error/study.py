"""Frozen score models under controlled speed-input errors; not measured video accuracy."""
from pathlib import Path
import importlib.util,json,itertools,argparse,time,datetime,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;REFERENCE=HERE.parent/'20261101-mobilise-camera-models'
spec=importlib.util.spec_from_file_location('frozen_camera_models',REFERENCE/'study.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
np,pd,joblib,old,p=ref.np,ref.pd,ref.joblib,ref.old,ref.p
sha,save,KEY,SEEDS=ref.sha,ref.save,ref.KEY,ref.SEEDS
STORE=Path('D:/gait-research-data/D_P/mobilise-speed-error-20261101')
METHODS=['SpeedFixedRF','CadenceFixedRF','SpeedDS','CadenceDS']
SCENARIOS=['clean','bias_m10','bias_p10','scale_m10','scale_p10','jitter05','jitter10']
def tasks(mode):return [('sppb',0,SEEDS[0]),('tug',0,SEEDS[0])] if mode=='pilot' else list(itertools.product(['sppb','tug'],range(5),SEEDS))
def initialize():ref.initialize()
def jitter(d):
    return np.array([np.random.default_rng(int(hashlib.sha256(f'20261101|{sid}|{visit}'.encode()).hexdigest()[:16],16)).standard_normal(5) for sid,visit in d[KEY].itertuples(index=False,name=None)])
def changed_bag(d,scenario):
    bag=p.bag_array(d,5).copy();speed=bag[:,:,2].copy()
    if scenario.startswith('bias_'):bag[:,:,2]+=-.1 if scenario=='bias_m10' else .1
    elif scenario.startswith('scale_'):bag[:,:,2]*=.9 if scenario=='scale_m10' else 1.1
    elif scenario.startswith('jitter'):bag[:,:,2]+=jitter(d)*(.05 if scenario=='jitter05' else .1)
    elif scenario!='clean':raise ValueError(scenario)
    clipped=int((bag[:,:,2]<0).sum());bag[:,:,2]=np.maximum(bag[:,:,2],0.)
    return bag,clipped
def model_input(bag,name):
    visible=bag[:,:,[2,0]] if name.startswith('Cadence') else bag[:,:,[2]]
    return visible if name.endswith('DS') else np.array([sorted(row.tolist(),key=lambda t:(t[0],*t)) for row in visible]).reshape(len(bag),-1)
def work(task):
    target,fold,seed=task;tag=f'{target}-f{fold}-s{seed}';folder=STORE/'folds';done=folder/f'{tag}.json'
    if done.exists():
        q=json.loads(done.read_text());assert q['code_sha256']==sha(__file__);return tag,'cached'
    start=time.time();_,te,_,_=old.cell_data(target,'long',5,fold);source=ref.STORE/'folds'/f'{tag}.json';q=json.loads(source.read_text());src=ref.STORE/'folds'/f'{tag}.csv.gz';assert sha(src)==q['predictions_sha256'];original=pd.read_csv(src);np.testing.assert_array_equal(original[KEY],te[KEY])
    original[KEY+['cohort','actual','class','target','fold','seed','speed']].to_csv(folder/f'{tag}-keys.csv.gz',index=False)
    values=np.empty((len(SCENARIOS),len(METHODS),len(te),4));warnings=np.empty(values.shape[:-1],dtype=bool);ledger=dict(task=list(task),code_sha256=sha(__file__),sources={str(z):sha(z) for z in [source,src]},models={},clipped={})
    models={}
    for name in METHODS:
        info=q['models'][name];assert sha(info['path'])==info['sha256'];models[name]=dict(np.load(info['path'])) if name.endswith('DS') else joblib.load(info['path']);ledger['models'][name]=info
    for i,scenario in enumerate(SCENARIOS):
        bag,clipped=changed_bag(te,scenario);ledger['clipped'][scenario]=clipped
        for j,name in enumerate(METHODS):
            x=model_input(bag,name);raw=p.predict_ds(models[name],x) if name.endswith('DS') else old.probabilities(models[name],x,'RF');info=q['models'][name];cal=old.temper(raw,info['temperature']);values[i,j]=cal;warnings[i,j]=cal[:,old.lowclass(target)]>=info['thresholds']['cal']['fpr10']
            if scenario=='clean':np.testing.assert_allclose(cal,original[[f'{name}_cal_p{k}' for k in range(4)]],rtol=0,atol=1e-12);np.testing.assert_array_equal(warnings[i,j],original[f'{name}_cal_fpr10'].astype(bool))
    dest=folder/f'{tag}.npz';np.savez_compressed(dest,probabilities=values,warnings=warnings);ledger.update(predictions_sha256=sha(dest),keys_sha256=sha(folder/f'{tag}-keys.csv.gz'),seconds=time.time()-start);save(done,ledger);return tag,round(time.time()-start,1)
def prepare():
    files=[REFERENCE/'study.py',REFERENCE/'protocol.json',REFERENCE/'main-verification.json',REFERENCE/'metrics-by-seed.csv',old.BASE/'features.csv',old.BASE/'selected-bouts.csv']
    save(HERE/'protocol.json',dict(code_sha256=sha(__file__),sources={str(z):sha(z) for z in files},
      hypothesis='Cadence had little clean-input accuracy gain. Does it protect the same frozen RF/DeepSets when speed measurement is biased or noisy?',
      data='SameSPPB2174/TUG1939people7356/5688visits,originalperson5fold/inner3fixed,30–60sec5bouts,draw0,5seeds,fixed4scorebands. No newtraining/calibration/threshold/epoch choices.',
      methods=METHODS,scenarios=SCENARIOS,
      perturbation='Speed only: additive +/-0.10m/s; common scale0.9/1.1; independent bout GaussianSD0.05/0.10m/s from fixedSHA256(participant,visit) seed,shared across all modelseeds/methods. Clip negative speed to0,report counts. Clean untouched. RF bags re-sort by perturbed speed with cadence paired; DS keeps paired bout data.',
      noise_scope='Controlled synthetic scenarios,not empirical camera-error distribution. MoVi2sec pelvis slope differs from Mobilise-D mean stride speed. Cadence left exact is an optimistic extra-input upper bound. One fixed jitter draw,not an estimated expected noise effect. Repeated noisy records are not newpeople.',
      comparison='Primary mean of +/-0.10m/s additive scenarios: cadence-minus-speed within fixed RF and DS,5seed direction; compare accuracy/macrorecall/Brier/extremeerrors/low-functionmiss/FPR and samecohort speed pairs. Also show changes from each model clean in all7conditions. Use unperturbed true speed for fixed matching; do not rematch on noisy input.',
      success='Exploratory target: cadence adds>=0.5percentagepoint accuracy in primary symmetric bias scenarios,with no low-functionmiss increase and FPR increase<=1percentagepoint. Report all3components/5seeds,not select best condition or widen scorebands. Not clinical standard.',
      expected=dict(pilot_tasks=2,main_tasks=50,outer_folds=5,seeds=SEEDS,unique_final_model_references=200,new_model_fits=0,model_scenario_replays=1400,metric_rows=280,matched_rows=280),
      verification='Code/source/modelSHA,person/bout/score links,cleanexactreplay<=1e-12,all7perturbationmodelreplays1400,independentperson-equalmetrics andfixedpairs. No refits because frozen-weight sensitivitytest.',
      resources='CPU4threads single BLASthread,no GPU,no download; estimate5-15min,total arrays<200MB D,small code/protocol/tables C.',storage=dict(small=str(HERE),large=str(STORE))))
def run(mode,workers):
    initialize();(STORE/'folds').mkdir(parents=True,exist_ok=True);start=time.time();ts=tasks(mode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in ts]),1):print(i,len(ts),f.result(),flush=True)
    save(HERE/f'{mode}-completion.json',dict(tasks=len(ts),model_scenario_replays=len(ts)*28,new_fits=0,code_sha256=sha(__file__),seconds=time.time()-start))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
