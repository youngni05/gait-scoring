from context import *
from train import fit as fit_model
import joblib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from threadpoolctl import threadpool_limits

def job(cohort,rep,fold,views):
 with threadpool_limits(limits=1):
  out=STUDY/cohort;d=dict(np.load(out/'input.npz'));y=d['y'];ids=d['ids'];tr=np.array(fold['train']);te=np.array(fold['test']);rows=[];selections=[];inner=[];r=rep['repeat'];f=fold['fold']
  assert len(tr)>10 and len(te)>0 and not set(tr)&set(te)
  for seed in SEEDS:
   for method in views:
    z=d[method];scores=[]
    for leaf in [2,5]:
     pp=np.full(len(y),np.nan)
     for inn in fold['inner']:
      it=np.array(inn['train']);iv=np.array(inn['validation']);assert len(it)>5 and len(iv)>0 and not set(it)&set(iv) and not set(te)&(set(it)|set(iv))
      pp[iv]=fit_model(z,y,it,iv,leaf,seed)[0]
     assert np.isfinite(pp[tr]).all();score=float(np.mean(abs(y[tr]-pp[tr])));scores.append(score)
     inner.append({'repeat':r,'fold':f,'seed':seed,'method':method,'leaf':leaf,'MAE':score,'train_order':tr.tolist(),'predictions':pp[tr].tolist()})
    leaf=[2,5][int(np.argmin(scores))];pred,pair=fit_model(z,y,tr,te,leaf,seed);path=out/'models'/f'r{r}-f{f}-{method}-s{seed}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(pair,path,compress=3)
    selections.append({'repeat':r,'fold':f,'seed':seed,'method':method,'leaf':leaf,'file':path.relative_to(STUDY).as_posix(),'sha256':sha(path)})
    rows.extend({'repeat':r,'fold':f,'seed':seed,'method':method,'sample':int(k),'id':str(ids[k]),'target':float(y[k]),'prediction':float(v)} for k,v in zip(te,pred))
   median=float(np.sort(y[tr])[(len(tr)-1)//2]);rows.extend({'repeat':r,'fold':f,'seed':seed,'method':'Median','sample':int(k),'id':str(ids[k]),'target':float(y[k]),'prediction':median} for k in te)
  return rows,selections,inner

def main():
 begun=time.time();prep=json.loads((STUDY/'preparation.json').read_text());assert sha(STUDY/'protocol.json')==prep['protocol_sha256'];assert json.loads((STUDY/'input-verification.json').read_text())['passed']
 for cohort,info in prep['cohorts'].items():
  out=STUDY/cohort;assert not (out/'completion.json').exists()
  for n,h in info['hashes'].items():assert sha(out/n)==h
  plan=json.loads((out/'splits.json').read_text());rows=[];selections=[];inner=[]
  with ProcessPoolExecutor(max_workers=4) as pool:
   futures={pool.submit(job,cohort,rep,f,info['views']):(rep['repeat'],f['fold']) for rep in plan for f in rep['splits']}
   for future in as_completed(futures):
    a,b,c=future.result();rows.extend(a);selections.extend(b);inner.extend(c)
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False);save(cohort+'/selections.json',selections);save(cohort+'/inner-evaluations.json',inner)
    print('FOLD',cohort,*futures[future],'MODELS',len(selections),flush=True)
  save(cohort+'/completion.json',{'complete':True,'models':len(selections),'people':info['people'],'input_sha256':sha(out/'input.npz'),'driver_sha256':sha(STUDY/'fit_study.py'),'learner_sha256':sha(PRIOR/'train.py'),'elapsed_cumulative_seconds':time.time()-begun})
 print('TRAINING_COMPLETE',time.time()-begun,flush=True)
if __name__=='__main__':main()
