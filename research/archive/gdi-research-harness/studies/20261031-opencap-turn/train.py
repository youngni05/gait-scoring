from common import *
import joblib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
def fit(z,y,tr,te,leaf,seed):
 scale=StandardScaler().fit(z[tr],sample_weight=np.ones(len(tr)))
 model=RandomForestRegressor(n_estimators=128,min_samples_leaf=leaf,random_state=seed,n_jobs=1,max_features=1.)
 model.fit(scale.transform(z[tr]),y[tr],sample_weight=np.ones(len(tr)))
 return model.predict(scale.transform(z[te])),(scale,model)
def fold_job(cohort,rep,fold,views):
 with threadpool_limits(limits=1):
    out=HERE/cohort;d=dict(np.load(out/'input.npz'));ids=d['ids'];y=d['y']
    rows=[];selections=[];evaluations=[]
    tr=np.array(fold['train']);te=np.array(fold['test']);r=rep['repeat'];f=fold['fold']
    assert len(tr)>50 and len(te)>5 and not set(tr)&set(te)
    for seed in SEEDS:
     for name in views:
      z=d[name];candidates=[]
      for leaf in [2,5]:
       pp=np.full(len(y),np.nan)
       for inn in fold['inner']:
        it=np.array(inn['train']);iv=np.array(inn['validation']);assert not set(it)&set(iv) and not set(te)&(set(it)|set(iv))
        pp[iv]=fit(z,y,it,iv,leaf,seed)[0]
       assert np.isfinite(pp[tr]).all();score=float(np.mean(abs(y[tr]-pp[tr])));candidates.append(score)
       evaluations.append({'repeat':r,'fold':f,'seed':seed,'method':name,'leaf':leaf,'MAE':score,'train_order':tr.tolist(),'predictions':pp[tr].tolist()})
      leaf=[2,5][int(np.argmin(candidates))];prediction,pair=fit(z,y,tr,te,leaf,seed)
      modelpath=out/'models'/f'r{r}-f{f}-{name}-s{seed}.joblib';modelpath.parent.mkdir(exist_ok=True)
      joblib.dump(pair,modelpath,compress=3)
      selections.append({'repeat':r,'fold':f,'seed':seed,'method':name,'leaf':leaf,'file':modelpath.relative_to(HERE).as_posix(),'sha256':sha(modelpath)})
      rows.extend({'repeat':r,'fold':f,'seed':seed,'method':name,'sample':int(k),'id':str(ids[k]),'target':float(y[k]),'prediction':float(p)} for k,p in zip(te,prediction))
     median=float(np.sort(y[tr])[(len(tr)-1)//2])
     rows.extend({'repeat':r,'fold':f,'seed':seed,'method':'Median','sample':int(k),'id':str(ids[k]),'target':float(y[k]),'prediction':median} for k in te)
    return rows,selections,evaluations
def main():
 begun=time.time();prep=json.loads((HERE/'preparation.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text())
 assert sha(HERE/'protocol.json')==prep['protocol_sha256']
 prep['cohorts'].update(json.loads((HERE/'preparation-sensitivity.json').read_text())['cohorts'])
 protocol['event_flagged']=json.loads((HERE/'event_flagged/protocol.json').read_text())
 for cohort,info in prep['cohorts'].items():
  out=HERE/cohort
  for n,h in info['hashes'].items():assert sha(out/n)==h
  plan=json.loads((out/'splits.json').read_text());rows=[];selections=[];evaluations=[]
  with ProcessPoolExecutor(max_workers=4) as pool:
   futures={pool.submit(fold_job,cohort,rep,fold,protocol[cohort]['views']):(rep['repeat'],fold['fold']) for rep in plan for fold in rep['splits']}
   for future in as_completed(futures):
    a,b,c=future.result();rows.extend(a);selections.extend(b);evaluations.extend(c)
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False);save(cohort+'/selections.json',selections);save(cohort+'/inner-evaluations.json',evaluations)
    print('FOLD',cohort,*futures[future],'MODELS',len(selections),flush=True)
  save(cohort+'/completion.json',{'complete':True,'models':len(selections),'people':info['people'],'prediction_rows':len(rows),'elapsed_total_seconds':time.time()-begun,'input_sha256':sha(out/'input.npz'),'code_sha256':sha(HERE/'train.py'),'parallelism':'four independent outer folds; one thread per RF fit'})
 print('TRAINING_COMPLETE',time.time()-begun,flush=True)
if __name__=='__main__':
 main()
