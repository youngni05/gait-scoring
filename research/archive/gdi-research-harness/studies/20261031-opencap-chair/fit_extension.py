from context import *
from fit_study import job
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def main():
 cohort='prep_event';info=json.loads((STUDY/'preparation-extension.json').read_text())['cohorts'][cohort];out=STUDY/cohort;assert not (out/'completion.json').exists();assert cohort in json.loads((STUDY/'input-verification.json').read_text())['cohorts']
 for n,h in info['hashes'].items():assert sha(out/n)==h
 begun=time.time();rows=[];selections=[];inner=[];plan=json.loads((out/'splits.json').read_text())
 with ProcessPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(job,cohort,rep,f,info['views']):(rep['repeat'],f['fold']) for rep in plan for f in rep['splits']}
  for future in as_completed(futures):
   a,b,c=future.result();rows.extend(a);selections.extend(b);inner.extend(c);pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False);save(cohort+'/selections.json',selections);save(cohort+'/inner-evaluations.json',inner);print('PREP_FOLD',*futures[future],'MODELS',len(selections),flush=True)
 save(cohort+'/completion.json',{'complete':True,'models':len(selections),'people':info['people'],'input_sha256':sha(out/'input.npz'),'driver_sha256':sha(STUDY/'fit_study.py'),'extension_sha256':sha(STUDY/'fit_extension.py'),'learner_sha256':sha(PRIOR/'train.py'),'elapsed_seconds':time.time()-begun});print('PREPARATION_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
