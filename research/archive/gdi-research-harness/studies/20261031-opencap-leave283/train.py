from study import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from threadpoolctl import threadpool_limits
import time

def job(cohort, info, rep, fold):
    with threadpool_limits(limits=1):
        d = dict(np.load(HERE/cohort/'input.npz')); y=d['y']; ids=d['ids']
        tr=np.array(fold['train']); te=np.array(fold['test']); rows=[]; selected=[]; inner=[]
        assert REMOVED not in ids and not set(tr)&set(te)
        r=rep['repeat']; f=fold['fold']
        jobs=[('Ridge',v,-1,[.1,1.,10.,100.]) for v in info['ridge_views']]
        jobs += [('RF',v,s,[2,5]) for v in info['rf_views'] for s in SEEDS]
        for family,view,seed,grid in jobs:
            x=d[view]; scores=[]
            for setting in grid:
                p=np.full(len(y),np.nan)
                for inn in fold['inner']:
                    it=np.array(inn['train']); iv=np.array(inn['validation'])
                    assert not set(it)&set(iv) and not set(te)&(set(it)|set(iv))
                    assert set(it)|set(iv)==set(tr)
                    p[iv]=fit(x,y,it,iv,family,setting,seed)[0]
                assert np.isfinite(p[tr]).all()
                score=float(np.mean(abs(p[tr]-y[tr])));scores.append(score)
                inner.append({'repeat':r,'fold':f,'family':family,'view':view,'seed':seed,'setting':setting,'MAE':score,'train_order':tr.tolist(),'predictions':p[tr].tolist()})
            setting=grid[int(np.argmin(scores))]
            pred,pair=fit(x,y,tr,te,family,setting,seed)
            path=HERE/cohort/'models'/f'r{r}-f{f}-{family}-{view}-s{seed}.joblib'
            path.parent.mkdir(exist_ok=True); joblib.dump(pair,path,compress=3)
            selected.append({'repeat':r,'fold':f,'family':family,'view':view,'seed':seed,'setting':setting,'inner_MAE':min(scores),
                'file':path.relative_to(HERE).as_posix(),'sha256':sha(path),'train_max':float(max(y[tr]))})
            rows.extend({'repeat':r,'fold':f,'family':family,'view':view,'seed':seed,'id':str(ids[k]),'sample':int(k),
                'target':float(y[k]),'prediction':float(v),'train_max':float(max(y[tr]))} for k,v in zip(te,pred))
        return rows,selected,inner

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['fit_code_sha256']==sha(HERE/'study.py')
    assert json.loads((HERE/'input-verification.json').read_text())['passed']
    begun=time.time()
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort; assert not (out/'completion.json').exists()
        rows=[]; selected=[]; inner=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(job,cohort,info,rep,f):(rep['repeat'],f['fold']) for rep in json.loads((out/'splits.json').read_text()) for f in rep['splits']}
            for future in as_completed(futures):
                a,b,c=future.result();rows+=a;selected+=b;inner+=c
                print('FOLD',cohort,*futures[future],'MODELS',len(selected),flush=True)
        pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
        save(cohort+'/selections.json',selected);save(cohort+'/inner.json',inner)
        save(cohort+'/completion.json',{'complete':True,'models':len(selected),'elapsed_cumulative':time.time()-begun,
            'input_sha256':sha(out/'input.npz'),'protocol_sha256':sha(HERE/'protocol.json'),'driver_sha256':sha(HERE/'train.py')})
    print('TRAINING_COMPLETE',time.time()-begun,flush=True)
if __name__=='__main__':main()
