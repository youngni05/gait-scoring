from study import *
from threadpoolctl import threadpool_limits
import time

def main():
    assert not (HERE/'completion.json').exists()
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    assert json.loads((HERE/'input-verification.json').read_text())['passed']
    d=dict(np.load(HERE/'input.npz'));y=d['y'];ids=d['ids'];rows=[];inner=[];selections=[];start=time.time()
    (HERE/'models').mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for rep in json.loads((HERE/'splits.json').read_text()):
            for fold in rep['splits']:
                r=rep['repeat'];f=fold['fold'];tr=np.array(fold['train']);te=np.array(fold['test'])
                for view in VIEWS:
                    x=d[view];scores=[]
                    for alpha in ALPHAS:
                        p=np.full(len(y),np.nan)
                        for inn in fold['inner']:
                            it=np.array(inn['train']);iv=np.array(inn['validation'])
                            assert not set(it)&set(iv) and set(it)|set(iv)==set(tr) and not set(te)&set(tr)
                            p[iv]=fit(x,y,it,iv,alpha)[0]
                        assert np.isfinite(p[tr]).all();score=float(np.mean(abs(p[tr]-y[tr])));scores.append(score)
                        inner.append({'repeat':r,'fold':f,'view':view,'alpha':alpha,'MAE':score,'train_order':tr.tolist(),'predictions':p[tr].tolist()})
                    alpha=ALPHAS[int(np.argmin(scores))];pred,raw,pair=fit(x,y,tr,te,alpha)
                    path=HERE/'models'/f'r{r}-f{f}-{view}.joblib';joblib.dump(pair,path,compress=3)
                    selections.append({'repeat':r,'fold':f,'view':view,'alpha':alpha,'inner_MAE':min(scores),'file':path.relative_to(HERE).as_posix(),'sha256':sha(path)})
                    rows.extend({'repeat':r,'fold':f,'view':view,'id':str(ids[k]),'sample':int(k),'target':float(y[k]),'prediction':float(p),'raw_prediction':float(q)} for k,p,q in zip(te,pred,raw))
            print('PARTITION',rep['repeat'],'MODELS',len(selections),flush=True)
    pd.DataFrame(rows).to_csv(HERE/'predictions.csv',index=False);save('selections.json',selections);save('inner.json',inner)
    save('completion.json',{'complete':True,'models':len(selections),'seconds':time.time()-start,'protocol_sha256':sha(HERE/'protocol.json'),'driver_sha256':sha(HERE/'train.py')})
    print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
