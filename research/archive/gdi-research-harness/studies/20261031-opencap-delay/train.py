from study import *
from threadpoolctl import threadpool_limits
import time

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    assert json.loads((HERE/'input-verification.json').read_text())['passed'];begun=time.time()
    with threadpool_limits(limits=1):
        for cohort,info in protocol['cohorts'].items():
            out=HERE/cohort;assert not (out/'completion.json').exists();(out/'models').mkdir(exist_ok=True)
            d=dict(np.load(out/'input.npz'));y=d['y'];rows=[];selections=[];inner=[]
            for rep in json.loads((out/'splits.json').read_text()):
                for f in rep['splits']:
                    tr=np.array(f['train']);te=np.array(f['test'])
                    for view in info['views']:
                        x=d[view];scores=[]
                        for alpha in ALPHAS:
                            pp=np.full(len(y),np.nan)
                            for v in f['inner']:
                                it=np.array(v['train']);iv=np.array(v['validation']);assert not set(it)&set(iv) and set(it)|set(iv)==set(tr) and not set(te)&set(tr)
                                pp[iv]=fit(x,y,it,iv,alpha)[0]
                            assert np.isfinite(pp[tr]).all();score=float(np.mean(abs(pp[tr]-y[tr])));scores.append(score)
                            inner.append({'repeat':rep['repeat'],'fold':f['fold'],'view':view,'alpha':alpha,'MAE':score,'train_order':tr.tolist(),'predictions':pp[tr].tolist()})
                        alpha=ALPHAS[int(np.argmin(scores))];pred,raw,pair=fit(x,y,tr,te,alpha)
                        file=out/'models'/f'r{rep["repeat"]}-f{f["fold"]}-{view}.joblib';joblib.dump(pair,file,compress=3)
                        selections.append({'repeat':rep['repeat'],'fold':f['fold'],'view':view,'alpha':alpha,'inner_MAE':min(scores),'file':file.relative_to(HERE).as_posix(),'sha256':sha(file)})
                        rows.extend({'repeat':rep['repeat'],'fold':f['fold'],'view':view,'id':str(d['ids'][k]),'sample':int(k),'target':float(y[k]),'prediction':float(p),'raw_prediction':float(q)} for k,p,q in zip(te,pred,raw))
                print(cohort,'PARTITION',rep['repeat'],'MODELS',len(selections),flush=True)
            pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False);save(cohort+'/selections.json',selections);save(cohort+'/inner.json',inner)
            save(cohort+'/completion.json',{'complete':True,'models':len(selections),'elapsed_cumulative':time.time()-begun,'protocol_sha256':sha(HERE/'protocol.json'),'driver_sha256':sha(HERE/'train.py')})
    print('COMPLETE',time.time()-begun,flush=True)
if __name__=='__main__':main()
