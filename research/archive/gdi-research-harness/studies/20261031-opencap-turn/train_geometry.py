from train import *
def run():
 begun=time.time();out=HERE/'geometry';info=json.loads((HERE/'preparation-geometry.json').read_text())['cohorts']['geometry'];protocol=json.loads((out/'protocol.json').read_text())
 for n,h in info['hashes'].items():assert sha(out/n)==h
 plan=json.loads((out/'splits.json').read_text());rows=[];selections=[];evaluations=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(fold_job,'geometry',rep,fold,protocol['views']):(rep['repeat'],fold['fold']) for rep in plan for fold in rep['splits']}
  for future in as_completed(futures):
   a,b,c=future.result();rows.extend(a);selections.extend(b);evaluations.extend(c)
   pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False);save('geometry/selections.json',selections);save('geometry/inner-evaluations.json',evaluations)
   print('GEOMETRY_FOLD',*futures[future],'MODELS',len(selections),flush=True)
 save('geometry/completion.json',{'complete':True,'models':len(selections),'people':info['people'],'prediction_rows':len(rows),'elapsed_total_seconds':time.time()-begun,'input_sha256':sha(out/'input.npz'),'code_sha256':sha(HERE/'train.py'),'driver_sha256':sha(HERE/'train_geometry.py')})
if __name__=='__main__':run()
