"""Separate extra speed observations from adding camera-feasible gait features."""
from pathlib import Path
import importlib.util,json,time,itertools,os,argparse,datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;REFERENCE=HERE.parent/'20261101-mobilise-duration-turns'
spec=importlib.util.spec_from_file_location('bout_reference',REFERENCE/'study.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
np,pd,joblib,old=base.np,base.pd,base.joblib,base.old
sha,save,weights=base.sha,base.save,base.weights;KEY,SEEDS=base.KEY,base.SEEDS
STORE=Path('D:/gait-research-data/D_P/mobilise-camera-inputs-20261101')
SPEED_STORE=Path('D:/gait-research-data/D_P/mobilise-input-classification-20261031')
METHODS=['SpeedMedian','SpeedBouts','SpeedCadence','Core4']

def initialize():base.initialize()
def tasks(mode):return [('sppb',0,SEEDS[0]),('tug',0,SEEDS[0])] if mode=='pilot' else list(itertools.product(['sppb','tug'],range(5),SEEDS))

def inputs(d):
    bags=base.p.bag_array(d,5);out={'SpeedMedian':d[['averagestridespeed']].to_numpy()}
    for name,cols in [('SpeedBouts',[2]),('SpeedCadence',[2,0]),('Core4',[0,1,2,3])]:
        visible=bags[:,:,cols];speed_index=2 if name=='Core4' else 0
        ordered=np.array([sorted(row.tolist(),key=lambda t:(t[speed_index],*t)) for row in visible])
        out[name]=ordered.reshape(len(d),-1)
    return out

def work(task):
    target,fold,seed=task;tag=f'{target}-f{fold}-s{seed}';done=STORE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,split=old.cell_data(target,'long',5,fold);x,xt=inputs(tr),inputs(te);ids=tr.participantid.to_numpy()
    col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target)
    out=old.outframe(te,target,'long',5,fold,seed);out['speed']=te.averagestridespeed
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    prior=np.bincount(y,weights=weights(ids),minlength=4);prior/=prior.sum()
    for j in range(4):out[f'Prior_p{j}']=prior[j]
    ledger={'task':list(task),'code_sha256':sha(__file__),'models':{},'source_files':{}}
    for name in METHODS:
        if name in ['SpeedMedian','Core4']:
            store,source_name=(SPEED_STORE,'SpeedRF') if name=='SpeedMedian' else (base.STORE,'Bout4')
            stem=f'{target}-long-5-f{fold}-s{seed}';j=store/'folds'/f'{stem}.json';ip=store/'folds'/f'{stem}-inner.npz';cp=store/'folds'/f'{stem}.csv.gz'
            l=json.loads(j.read_text());assert sha(ip)==l['inner_sha256'] and sha(cp)==l['predictions_sha256'];ins=np.load(ip)
            for key in ['ids','visits','y','inner']:np.testing.assert_array_equal(ins[key],arrays[key])
            info=dict(l['models'][source_name]);assert sha(info['path'])==info['sha256']
            raw=old.probabilities(joblib.load(info['path']),xt[name],'RF');oof=ins[source_name+'_oof'].copy()
            sd=pd.read_csv(cp);np.testing.assert_array_equal(sd[KEY],te[KEY]);np.testing.assert_allclose(sd.actual,te[col],rtol=0,atol=1e-12)
            np.testing.assert_allclose(raw,sd[[f'{source_name}_raw_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
            ledger['source_files'][name]={'ledger':str(j),'ledger_sha256':sha(j),'inner':str(ip),'inner_sha256':sha(ip),'predictions':str(cp),'predictions_sha256':sha(cp),'source_name':source_name}
        else:
            oof=np.zeros((len(tr),4));saved=[]
            for i in range(3):
                a=inner!=i;b=~a;m=old.fit(old.make('RF',seed+fold*10+i),x[name][a],y[a],ids[a],'RF');oof[b]=old.probabilities(m,x[name][b],'RF')
                saved.append(base.p.save_model(STORE/'models'/f'{tag}-{name}-inner{i}.joblib',m))
            m=old.fit(old.make('RF',seed+fold*10+9),x[name],y,ids,'RF');raw=old.probabilities(m,xt[name],'RF')
            info=base.p.save_model(STORE/'models'/f'{tag}-{name}.joblib',m);info['inner_models']=saved
        info.update(old.finish_prob(name,oof,raw,y,ids,inner,out,target,arrays));ledger['models'][name]=info
    out.to_csv(STORE/'folds'/f'{tag}.csv.gz',index=False);np.savez_compressed(STORE/'folds'/f'{tag}-inner.npz',**arrays)
    ledger.update(seconds=time.time()-start,predictions_sha256=sha(STORE/'folds'/f'{tag}.csv.gz'),inner_sha256=sha(STORE/'folds'/f'{tag}-inner.npz'));save(done,ledger)
    return tag,round(time.time()-start,1)

def prepare():
    paths=[REFERENCE/'study.py',REFERENCE/'main-verification.json',base.PREV/'study.py',base.PREV/'verification.json',old.BASE/'features.csv',old.BASE/'selected-bouts.csv',base.p.FACTORIAL/'audited-bouts.csv']
    protocol={'code_sha256':sha(__file__),'sources':{str(p):sha(p) for p in paths},'seeds':SEEDS,
      'user_constraint':'fixed single smartphone, no floor distance marks; one-time height entry allowed. Aim four-band accuracy with one anchor plus camera-feasible extras.',
      'anchor':'walking speed is the selected anchor, NOT proven globally best individual feature. Height aids metric-scale estimation but is not a validation of video speed or world translation.',
      'hypothesis':'previous median-speed vs full-bout comparison mixes information preservation with extra features. Separate five speed values from adding cadence; Core4 is a no-turn/no-duration reference.',
      'inputs':{'SpeedMedian':'one median speed','SpeedBouts':'five sorted speeds','SpeedCadence':'five speeds+corresponding cadences,10columns','Core4':'five bouts original4,20columns'},
      'data':'sameSPPB2174people7356visits,TUG1939people5688visits,draw0,30-60sec5bouts; exact original outer5/inner3 person folds and5seeds; all visits of a person grouped',
      'evaluation_scope':'restores original mixed-cohort person5fold for the original speed comparison; different question from subsequent cohort-held-out4fold diagnostics. No protected final data.',
      'model':'all RF128,minleaf20,max_features1,person equal weights. No HP search in this input ablation. Reuse SpeedMedian/Core4 verified models; new SpeedBouts/SpeedCadence.',
      'selection':'train-only scaler,temperature andwarning thresholds from original inner3 people. No selecting features or rules using outer outcomes.',
      'primary':'overall four-class accuracy,per-class recall and balanced recall; compare SpeedBouts-SpeedMedian thenSpeedCadence-SpeedBouts;5seed direction and person bootstrap. No changing class boundaries or merging classes.',
      'guardrails':'report lowclass miss/FPR and opposite-extreme errors; an aggregate accuracy gain with subgroup deterioration is not uniform improvement.',
      'camera_limits':'released sensor averages only; no pixel extraction or height-based video reconstruction tested. No within-bout step variability,left-right asymmetry,trunk sway available in this table. Bout duration andturn frequency excluded.',
      'next_stage':'compare RF/DeepSets with identical retained camera-feasible features and internal accuracy selection; validate metric video speed andcadence against reference before claiming camera-score accuracy.',
      'expected':{'tasks':50,'new_final_models':100,'new_inner_models':300,'reused_final_models':100,'reused_inner_models':300,'model_replays':800,'selected_refits':16,'metric_rows':90,'matched_rows':40},
      'resource':'CPU4single-thread workers,estimate5-20min training+audit,pilot2targets fold0 firstseed,noGPU',
      'storage':{'small':str(HERE),'large':str(STORE)}}
    save(HERE/'protocol.json',protocol);print('PREPARED',flush=True)

def run(mode,workers):
    p=json.loads((HERE/'protocol.json').read_text());assert p['code_sha256']==sha(__file__)
    for path,digest in p['sources'].items():assert sha(path)==digest
    assert HERE.drive.upper()=='C:' and STORE.drive.upper()=='D:'
    for name in ['models','folds']:(STORE/name).mkdir(parents=True,exist_ok=True)
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
    initialize();start=time.time();todo=tasks(mode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in todo]),1):print(i,len(todo),*f.result(),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(todo),'seconds':time.time()-start,'code_sha256':sha(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
