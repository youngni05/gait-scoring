"""Replay all RFs, calibrations and metrics; independent selected refits."""
from study import *
spec=importlib.util.spec_from_file_location('reference_verifier',PREV/'verify.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
v.HERE,v.STORE,v.METHODS=HERE,STORE,METHODS

def check(task,refit=False):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    ledger=json.loads((STORE/'folds'/f'{tag}.json').read_text())
    assert ledger['code_sha256']==sha(HERE/'study.py')
    src=p.STORE/'folds'/f'{tag}.json';assert ledger['reference_sha256']==sha(src)
    out=pd.read_csv(STORE/'folds'/f'{tag}.csv.gz');ins=np.load(STORE/'folds'/f'{tag}-inner.npz')
    assert ledger['predictions_sha256']==sha(STORE/'folds'/f'{tag}.csv.gz')
    assert ledger['inner_sha256']==sha(STORE/'folds'/f'{tag}-inner.npz')
    tr,te,inner,reference=old.cell_data(target,band,k,fold);x=inputs(tr,k);xt=inputs(te,k)
    ids=tr.participantid.to_numpy();label='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[label].to_numpy(),target)
    np.testing.assert_array_equal(out[KEY],te[KEY]);np.testing.assert_allclose(out.actual,te[label],rtol=0,atol=1e-12)
    for key,value in {'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}.items():np.testing.assert_array_equal(ins[key],value)
    assert set(ids).isdisjoint(te.participantid)
    replays=0;error=0;refits=0
    for name in METHODS:
        info=ledger['models'][name];path=Path(info['path']);assert sha(path)==info['sha256']
        model=joblib.load(path);v.verify_scaler(model,x[name],ids)
        assert model['model'].n_estimators==128 and model['model'].min_samples_leaf==20 and model['model'].max_features==1.
        raw=old.probabilities(model,xt[name],'RF');replays+=1
        expected=out[[f'{name}_raw_p{j}' for j in range(4)]].to_numpy()
        delta=float(np.max(abs(raw-expected)));assert delta<1e-12;error=max(error,delta)
        if 'inner_models' in info:
            for i,mi in enumerate(info['inner_models']):
                assert sha(mi['path'])==mi['sha256'];m=joblib.load(mi['path']);a=inner!=i;b=~a
                v.verify_scaler(m,x[name][a],ids[a]);pr=old.probabilities(m,x[name][b],'RF');replays+=1
                np.testing.assert_allclose(pr,ins[name+'_oof'][b],rtol=0,atol=1e-12)
        else:
            original=np.load(p.STORE/'folds'/f'{tag}-inner.npz')
            np.testing.assert_array_equal(ins[name+'_oof'],original[info['reused_name']+'_oof'])
        oof=ins[name+'_oof'];cross=ins[name+'_crosscal']
        t=old.temperature(oof,y,ids);assert abs(t-info['temperature'])<1e-12
        np.testing.assert_allclose(old.temper(raw,t),out[[f'{name}_cal_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
        for i in range(3):
            a=inner!=i;b=~a;ct=old.temperature(oof[a],y[a],ids[a])
            assert abs(ct-info['cross_temperatures'][i])<1e-12
            np.testing.assert_allclose(cross[b],old.temper(oof[b],ct),rtol=0,atol=1e-12)
        for variant,ip,op in [('raw',oof,raw),('cal',cross,old.temper(raw,t))]:
            low=old.lowclass(target)
            for policy,cut in info['thresholds'][variant].items():
                v.independently_check_threshold(ip[:,low],(y==low).astype(int),ids,cut,policy)
                np.testing.assert_array_equal(op[:,low]>=cut,out[f'{name}_{variant}_{policy}'].astype(bool))
        if refit and name in ['Length5','Turn5']:
            for i in [0,9]:
                a=inner!=i if i==0 else np.ones(len(tr),dtype=bool)
                m=old.fit(old.make('RF',seed+fold*10+i),x[name][a],y[a],ids[a],'RF')
                expected=ins[name+'_oof'][inner==0] if i==0 else raw
                test=x[name][inner==0] if i==0 else xt[name]
                np.testing.assert_array_equal(old.probabilities(m,test,'RF'),expected);refits+=1
    return {'tag':tag,'models':replays,'error':error,'refits':refits}

def run(mode):
    start=time.time();initialize();protocol=json.loads((HERE/'protocol.json').read_text())
    for f,h in protocol['sources'].items():assert sha(f)==h
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    if mode=='pilot':tasks=[('sppb','long',5,0,SEEDS[0]),('tug','long',5,0,SEEDS[0]),('sppb','short',1,0,SEEDS[0])]
    results=[]
    for i,task in enumerate(tasks,1):
        tag=f'{task[0]}-{task[1]}-{task[2]}-f{task[3]}-s{task[4]}'
        while not (STORE/'folds'/f'{tag}.json').exists():time.sleep(1)
        time.sleep(.05)
        results.append(check(task,task[1:4]==('long',5,0) and task[4] in [SEEDS[0],SEEDS[-1]]))
        if i%25==0:print('REPLAY',i,flush=True)
    if mode!='pilot':
        while not (HERE/'summary.json').exists():time.sleep(1)
        metric_error,pairchecks=v.verify_metrics()
    else:metric_error,pairchecks=None,0
    save(HERE/f'{mode}-verification.json',{'passed':True,'tasks':len(tasks),'model_replays':sum(r['models'] for r in results),
      'refits':sum(r['refits'] for r in results),'max_prediction_error':max(r['error'] for r in results),
      'independent_metric_error':metric_error,'matched_checks':pairchecks,'seconds':time.time()-start,
      'verify_sha256':sha(__file__),'analysis_sha256':sha(HERE/'analyze.py'),'tolerance':1e-12})
    print('VERIFIED',mode,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['pilot','main']);run(parser.parse_args().mode)
