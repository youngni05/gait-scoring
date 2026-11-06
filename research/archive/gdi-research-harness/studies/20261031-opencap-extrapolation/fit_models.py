from study import *
import joblib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from threadpoolctl import threadpool_limits

def job(cohort,info,rep,fold):
 with threadpool_limits(limits=1):
  d=dict(np.load(source(info)/'input.npz'));y=d['y'];ids=d['ids'];tr=np.array(fold['train']);te=np.array(fold['test']);r=rep['repeat'];f=fold['fold'];rows=[];chosen=[];evaluations=[];family_choice=[]
  for view in info['views']:
   x=d[view];outputs={};best_scores={}
   for family,alphas in GRIDS.items():
    scores=[]
    for alpha in alphas:
     pp=np.full(len(y),np.nan);error=None
     try:
      for inn in fold['inner']:
       it=np.array(inn['train']);iv=np.array(inn['validation']);assert not set(it)&set(iv) and not set(te)&(set(it)|set(iv));pp[iv]=fit(x,y,it,iv,family,alpha)[0]
      assert np.isfinite(pp[tr]).all();score=float(np.mean(abs(pp[tr]-y[tr])))
     except (ConvergenceWarning,ValueError,FloatingPointError) as e:score=float('inf');error=str(e)
     scores.append(score);evaluations.append({'repeat':r,'fold':f,'view':view,'family':family,'alpha':alpha,'MAE':score if np.isfinite(score) else None,'error':error,'train_order':tr.tolist(),'predictions':pp[tr].tolist() if error is None else None})
    index=int(np.argmin(scores));assert np.isfinite(scores[index]),(cohort,view,family,'all candidates failed');alpha=alphas[index];pred,raw,pair=fit(x,y,tr,te,family,alpha);outputs[family]=(pred,raw);best_scores[family]=scores[index]
    path=HERE/cohort/'models'/f'r{r}-f{f}-{view}-{family}.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(pair,path,compress=3)
    chosen.append({'repeat':r,'fold':f,'view':view,'family':family,'alpha':alpha,'inner_MAE':scores[index],'file':path.relative_to(HERE).as_posix(),'sha256':sha(path),'train_target_min':float(min(y[tr])),'train_target_max':float(max(y[tr]))})
   selected=min(best_scores,key=best_scores.get);family_choice.append({'repeat':r,'fold':f,'view':view,'family':selected,'inner_MAE':best_scores[selected]});outputs['CVSelect']=outputs[selected]
   for family,(pred,raw) in outputs.items():rows.extend({'repeat':r,'fold':f,'view':view,'family':family,'seed':-1,'sample':int(k),'id':str(ids[k]),'target':float(y[k]),'prediction':float(p),'raw_prediction':float(v),'train_max':float(max(y[tr]))} for k,p,v in zip(te,pred,raw))
  return rows,chosen,evaluations,family_choice

def main():
 protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['fit_code_sha256']==sha(HERE/'study.py');assert json.loads((HERE/'input-verification.json').read_text())['passed'];begun=time.time()
 for cohort,info in protocol['cohorts'].items():
  src=source(info);out=HERE/cohort;assert not (out/'completion.json').exists()
  for n,h in info['hashes'].items():assert sha(src/n)==h
  plans=json.loads((src/'splits.json').read_text());rows=[];chosen=[];evaluations=[];families=[]
  with ProcessPoolExecutor(max_workers=4) as pool:
   futures={pool.submit(job,cohort,info,rep,f):(rep['repeat'],f['fold']) for rep in plans for f in rep['splits']}
   for future in as_completed(futures):
    a,b,c,d=future.result();rows.extend(a);chosen.extend(b);evaluations.extend(c);families.extend(d);pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False);save(cohort+'/selections.json',chosen);save(cohort+'/inner.json',evaluations);save(cohort+'/family-choice.json',families);print('FOLD',cohort,*futures[future],'MODELS',len(chosen),flush=True)
  save(cohort+'/completion.json',{'complete':True,'models':len(chosen),'people':info['people'],'elapsed_cumulative':time.time()-begun,'protocol_sha256':sha(HERE/'protocol.json'),'driver_sha256':sha(HERE/'fit_models.py'),'fit_code_sha256':sha(HERE/'study.py')})
 print('TRAINING_COMPLETE',time.time()-begun,flush=True)
if __name__=='__main__':main()
