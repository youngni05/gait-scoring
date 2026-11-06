"""RF 2x2 feature-count / aggregation ablation, unchanged person splits."""
from pathlib import Path
import importlib.util,sys,os,json,time,itertools,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'20261031-mobilise-input-classification'
spec=importlib.util.spec_from_file_location('input_reference',PREV/'study.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
np,pd,joblib,old=p.np,p.pd,p.joblib,p.old
sha,save,weights=p.sha,p.save,p.weights
SEEDS,KEY,FEATURES=p.SEEDS,p.KEY,p.FEATURES
METHODS=['Median4','Median6','Bout4','Bout6']
STORE=Path('D:/gait-research-data/D_P/mobilise-input-factorial-20261101')
def initialize():
    p.initialize()

def inputs(frame,k):
    x=p.bag_array(frame,k)
    ordered=np.array([sorted(row.tolist(),key=lambda t:(t[2],*t)) for row in x])
    return {'Median4':np.median(x[:,:,:4],axis=1),'Median6':np.median(x,axis=1),
            'Bout4':ordered[:,:,:4].reshape(len(x),-1),'Bout6':ordered.reshape(len(x),-1)}

def work(task):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    done=STORE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,ref=old.cell_data(target,band,k,fold)
    x,xt=inputs(tr,k),inputs(te,k);ids=tr.participantid.to_numpy()
    label='total_sppb_score' if target=='sppb' else 'tugtime'
    y=old.classes(tr[label].to_numpy(),target)
    out=old.outframe(te,target,band,k,fold,seed);out['speed']=te.averagestridespeed
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    prior=np.bincount(y,weights=weights(ids),minlength=4);prior/=prior.sum()
    for j in range(4):out[f'Prior_p{j}']=prior[j]
    src=p.STORE/'folds'/f'{tag}.json';ledger0=json.loads(src.read_text())
    ins=np.load(p.STORE/'folds'/f'{tag}-inner.npz')
    for key in ['ids','visits','y','inner']:np.testing.assert_array_equal(ins[key],arrays[key])
    ledger={'task':list(task),'models':{},'reference_sha256':sha(src),'code_sha256':sha(__file__)}
    for name in METHODS:
        reuse={'Median4':'MedianRF','Bout6':'BoutRF'}
        if k==1:reuse.update(Median6='BoutRF',Bout4='MedianRF')
        if name in reuse:
            source_name=reuse[name];info0=ledger0['models'][source_name]
            path=Path(info0['path']);assert sha(path)==info0['sha256']
            model=joblib.load(path);raw=old.probabilities(model,xt[name],'RF')
            oof=ins[source_name+'_oof'].copy()
            info={'path':str(path),'sha256':sha(path),'reused_name':source_name,'reference':str(src)}
            if 'inner_models' in info0:info['inner_models']=info0['inner_models']
        else:
            oof=np.zeros((len(tr),4));saved=[]
            for i in range(3):
                a=inner!=i;b=~a
                model=old.fit(old.make('RF',seed+fold*10+i),x[name][a],y[a],ids[a],'RF')
                oof[b]=old.probabilities(model,x[name][b],'RF')
                saved.append(p.save_model(STORE/'models'/f'{tag}-{name}-inner{i}.joblib',model))
            model=old.fit(old.make('RF',seed+fold*10+9),x[name],y,ids,'RF')
            raw=old.probabilities(model,xt[name],'RF')
            info=p.save_model(STORE/'models'/f'{tag}-{name}.joblib',model);info['inner_models']=saved
        calibration=old.finish_prob(name,oof,raw,y,ids,inner,out,target,arrays)
        info.update(calibration);ledger['models'][name]=info
    if k==1:
        np.testing.assert_array_equal(out[[f'Median4_cal_p{j}' for j in range(4)]],out[[f'Bout4_cal_p{j}' for j in range(4)]])
        np.testing.assert_array_equal(out[[f'Median6_cal_p{j}' for j in range(4)]],out[[f'Bout6_cal_p{j}' for j in range(4)]])
    out.to_csv(STORE/'folds'/f'{tag}.csv.gz',index=False)
    np.savez_compressed(STORE/'folds'/f'{tag}-inner.npz',**arrays)
    ledger.update(seconds=time.time()-start,predictions_sha256=sha(STORE/'folds'/f'{tag}.csv.gz'),inner_sha256=sha(STORE/'folds'/f'{tag}-inner.npz'))
    save(done,ledger);return tag,round(ledger['seconds'],1)

def prepare():
    paths=[PREV/'study.py',PREV/'verification.json',PREV/'alignment-verification.json',old.BASE/'features.csv',old.BASE/'selected-bouts.csv',p.FACTORIAL/'audited-bouts.csv']
    protocol={'code_sha256':sha(__file__),'sources':{str(f):sha(f) for f in paths},'seeds':SEEDS,'cuts':old.CUTS,
      'design':'2x2 original4/expanded6 x median/bout; same 128-tree leaf20 max_features1 RF, person-equal weights, no metadata or class weighting',
      'features':FEATURES+['duration','turn_rate_per_second'],'people':{'sppb':2174,'tug':1939},
      'splits':'unchanged outer5 inner3 people, draw0, [4,10)/[30,60) seconds x1/3/5 bouts',
      'reuse':'Median4 and Bout6 from verified previous models; at k1 median/bout identical and reused; otherwise fit missing Median6/Bout4',
      'ordering':'speed then six feature values lexicographic order for both bout inputs; ties on first4 give identical core values',
      'selection':'no new hyperparameter search; previous inner OOF temperature calibration and internally selected fpr10 thresholds; outer labels used only for evaluation',
      'primary':'long5, five-seed mean not ensemble; report all6 conditions',
      'contrasts':'Median6-Median4, Bout4-Median4, Bout6-Median6, Bout6-Bout4; interaction Bout6-Bout4-Median6+Median4',
      'matching':'identical evaluation-only nearest speed pairs caliper .05m/s, different people same fold; same-cohort sensitivity; controls reusable, descriptive ranking',
      'limits':'reused development people, no video/clinical external validation; representation includes extra dimensions and ordering, not causal physiology',
      'storage':{'small':str(HERE),'large':str(STORE)}}
    save(HERE/'protocol.json',protocol);print('PREPARED',flush=True)

def run(mode,workers):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    for f,h in protocol['sources'].items():assert sha(f)==h
    assert HERE.resolve().drive.upper()=='C:' and STORE.resolve().drive.upper()=='D:'
    for n in ['folds','models']:(STORE/n).mkdir(parents=True,exist_ok=True)
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    if mode=='pilot':tasks=[('sppb','long',5,0,SEEDS[0]),('tug','long',5,0,SEEDS[0]),('sppb','short',1,0,SEEDS[0])]
    start=time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in tasks]),1):
            tag,sec=f.result()
            if mode=='pilot' or i%10==0:print(i,len(tasks),tag,sec,round(time.time()-start),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(tasks),'seconds':time.time()-start,'code_sha256':sha(__file__)})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','pilot','main']);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
