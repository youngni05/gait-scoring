import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE/'deps'))
import numpy as np
from scipy.stats import rankdata
DATA=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot')
def stats(dev,speed,labels):
 ranks=rankdata(np.column_stack([dev,speed,labels]),axis=0)
 c=np.corrcoef(ranks,rowvar=False);out=[]
 for j in range(labels.shape[1]):
  k=j+2;den=(1-c[0,1]**2)*(1-c[1,k]**2)
  partial=(c[0,k]-c[0,1]*c[1,k])/np.sqrt(den) if den>1e-15 else np.nan
  out.append([c[0,k],c[1,k],c[0,1],partial])
 return np.array(out)
def loo(features,y):
 x=np.column_stack([np.ones(len(y)),features]);pred=[]
 for i in range(len(y)):
  train=np.arange(len(y))!=i;coef=np.linalg.lstsq(x[train],y[train],rcond=None)[0];pred.append(float(x[i]@coef))
 p=np.array(pred);err=np.abs(p-y)
 return {'predictions':p.tolist(),'absolute_errors':err.tolist(),'MAE':float(err.mean()),'RMSE':float(np.sqrt(np.mean((p-y)**2))),'prediction_range':[float(p.min()),float(p.max())]}
def main():
 protocol=json.loads((ROOT/'protocol.json').read_text());baseprotocol=json.loads((BASE/'protocol.json').read_text());rows=json.loads((ROOT/'participants.json').read_text())
 archive=np.load(DATA/'person-curves.npz');h=archive['healthy'][:,:,:,baseprotocol['time_samples']];s=archive['stroke'][:,:,:,baseprotocol['time_samples']]
 si=np.array([r['mat_index']-1 for r in rows]);s=s[si];speed=np.array([r['speed']['mean_mps'] for r in rows]);labels=np.array([[r[k] for k in ['POMA','FAC','TIS']] for r in rows])
 result={'n':len(rows),'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'references':{}}
 for ref_name,indices in baseprotocol['references'].items():
  hi=np.array(indices)-1;refpeople=h[hi].mean(axis=1);ref=refpeople.mean(axis=0)
  def score(subjects,reference):return np.sqrt(np.mean((subjects-reference)**2,axis=(2,3))).mean(axis=1)
  dev=score(s,ref);key='deviation_primary' if ref_name=='primary_age65plus' else 'deviation_secondary';assert np.allclose(dev,[r[key] for r in rows],atol=1e-12,rtol=0)
  observed=stats(dev,speed,labels);rng=np.random.default_rng(protocol['bootstrap']['seed']);boot=np.empty((5000,3,4))
  for b in range(5000):
   reference=refpeople[rng.integers(0,len(hi),len(hi))].mean(axis=0);sample=rng.integers(0,len(rows),len(rows));boot[b]=stats(score(s[sample],reference),speed[sample],labels[sample])
  report={}
  for j,label in enumerate(['POMA','FAC','TIS']):
   report[label]={}
   for k,stat in enumerate(['deviation_vs_outcome','speed_vs_outcome','deviation_vs_speed','partial_deviation_vs_outcome_given_speed']):
    vals=boot[:,j,k];good=vals[np.isfinite(vals)];report[label][stat]={'rho':float(observed[j,k]),'ci95':np.quantile(good,[.025,.975]).tolist(),'bootstrap_valid':len(good),'bootstrap_undefined':5000-len(good)}
  a=loo(speed[:,None],labels[:,0]);b=loo(np.column_stack([speed,dev]),labels[:,0]);delta=np.array(a['absolute_errors'])-np.array(b['absolute_errors'])
  prediction={'speed_only':a,'speed_plus_deviation':b,'MAE_reduction':a['MAE']-b['MAE'],'people_improved':int((delta>1e-12).sum()),'people_worsened':int((delta< -1e-12).sum()),'people_tied':int((np.abs(delta)<=1e-12).sum())}
  result['references'][ref_name]={'reference_n':len(hi),'associations':report,'POMA_leave_one_person_out':prediction}
  np.savez_compressed(DATA/f'speed-bootstrap-{ref_name}.npz',statistics=boot)
  print(ref_name,json.dumps({'POMA':report['POMA'],'CV':{k:v for k,v in prediction.items() if k not in ['speed_only','speed_plus_deviation']},'speed_MAE':a['MAE'],'combined_MAE':b['MAE']}),flush=True)
 (ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
