from common import *
import time, joblib
from threadpoolctl import threadpool_limits

def main():
    start=time.time(); audit=json.loads((HERE/'preparation.json').read_text())
    assert audit['protocol_sha256']==sha(HERE/'protocol.json')
    assert audit['feature_code_sha256']==sha(HERE/'common.py')
    for name,meta in audit['cohorts'].items():
        if (HERE/name/'completion.json').exists():continue
        d=load_data(name); y=d['y'];rows=[];selections=[];plan=json.loads((HERE/name/'splits.json').read_text())
        assert sha(HERE/name/'input.npz')==meta['input_sha256']
        for rep in plan:
            for split in rep['splits']:
                tr=np.array(split['train']);te=np.array(split['test']); fold=split['fold'];repeat=rep['repeat']
                def add(view,family,seed,pred):
                    rows.extend({'repeat':repeat,'fold':fold,'view':view,'model':family,'seed':seed,'id':d['ids'][k],
                                 'sample':int(k),'target':float(y[k]),'prediction':float(p)} for k,p in zip(te,pred))
                add('none','Median',-1,np.repeat(np.sort(y[tr])[(len(tr)-1)//2],len(te)))
                for view in ['speed','pattern']:
                    x=d[view]
                    for family,grid,seeds in [('RF',[2,5],SEEDS),('Ridge',[1.,10.,100.],[-1])]:
                        for seed in seeds:
                            scores=[];ips=[]
                            for setting in grid:
                                pp=np.full(len(y),np.nan)
                                for inner in split['inner']:
                                    it=np.array(inner['train']);iv=np.array(inner['validation'])
                                    model=make_model(family,setting,seed).fit(x[it],y[it]);pp[iv]=model.predict(x[iv])
                                assert np.isfinite(pp[tr]).all()
                                scores.append(float(np.mean(abs(y[tr]-pp[tr]))));ips.append(pp[tr].tolist())
                            setting=grid[int(np.argmin(scores))]
                            model=make_model(family,setting,seed).fit(x[tr],y[tr]);pp=model.predict(x[te])
                            assert np.isfinite(pp).all()
                            rel=f'{name}/models/r{repeat}-{view}-{family}-{seed}-f{fold}.joblib'
                            path=HERE/rel;path.parent.mkdir(exist_ok=True);joblib.dump(model,path,compress=3)
                            selections.append({'repeat':repeat,'fold':fold,'view':view,'model':family,'seed':seed,
                                               'grid':grid,'scores':scores,'selected':setting,'inner_predictions':ips,
                                               'model_path':rel,'model_sha256':sha(path)})
                            add(view,family,seed,pp)
                pd.DataFrame(rows).to_csv(HERE/name/'predictions.csv',index=False)
                save(f'{name}/selections.json',selections)
                print(f'{name} repeat={repeat} fold={fold} done elapsed={time.time()-start:.1f}s',flush=True)
        save(f'{name}/completion.json',{'final_models':len(selections),'inner_models':sum(len(s['grid'])*3 for s in selections),
             'input_sha256':sha(HERE/name/'input.npz'),'splits_sha256':sha(HERE/name/'splits.json'),
             'predictions_sha256':sha(HERE/name/'predictions.csv'),'run_code_sha256':sha(HERE/'run.py')})
    save('execution.json',{'complete':True,'elapsed_seconds':time.time()-start,'device':'local CPU','new_packages':False})

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
