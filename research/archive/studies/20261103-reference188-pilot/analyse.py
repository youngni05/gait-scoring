"""Frozen exploratory association analysis. No fitting to clinical outcomes."""
import json,sys,hashlib,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np
from scipy.stats import spearmanr,rankdata
import scipy
DATA=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot')
def side_score(curves,reference):
 return np.sqrt(np.mean(np.square(curves-reference),axis=(-2,-1)))
def person_score(curves,reference):
 return side_score(curves,reference).mean(axis=-1)
def correlation(x,y):
 if np.ptp(x)==0 or np.ptp(y)==0:return float('nan')
 return float(spearmanr(x,y).statistic)
def run():
 protocol=json.loads((ROOT/'protocol.json').read_text());link=json.loads((ROOT/'person-linkage.json').read_text())
 archive=np.load(DATA/'person-curves.npz');ix=protocol['time_samples']
 h=archive['healthy'][:,:,:,ix];s=archive['stroke'][:,:,:,ix]
 valid_h=np.isfinite(h).all(axis=(1,2,3));valid_s=np.isfinite(s).all(axis=(1,2,3))
 stroke_indices=np.array([i-1 for i in protocol['primary_evaluation_stroke'] if valid_s[i-1]])
 assert len(stroke_indices)>=3, 'Insufficient complete stroke people for association'
 selected=s[stroke_indices];labels=np.array([[link['stroke']['rows'][i]['values'][k] for k in ['POMA','FAC','TIS']] for i in stroke_indices])
 bootstrap_n=protocol['uncertainty']['bootstrap_replicates'];seed=protocol['uncertainty']['seed']
 result={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'unit':'degrees','score_name':protocol['score_name'],'stroke_evaluation_n':len(stroke_indices),'stroke_indices_1based':(stroke_indices+1).tolist(),'analyses':{},'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
 persons={cohort:[{'id':r['supplement_candidate_ids'][0],'mat_index':r['mat_index'],'characteristics':r['values'],'usable':bool(valid[i]),'scores':{}} for i,r in enumerate(link[cohort]['rows'])] for cohort,valid in [('healthy',valid_h),('stroke',valid_s)]}
 for ref_name,indices in protocol['references'].items():
  hi=np.array([i-1 for i in indices if valid_h[i-1]]);ref_people=h[hi].mean(axis=1);reference=ref_people.mean(axis=0)
  assert len(hi)>=2, 'Insufficient healthy reference people for leave-one-out'
  scores=person_score(selected,reference);boot=np.empty((bootstrap_n,3));rng=np.random.default_rng(seed)
  for b in range(bootstrap_n):
   ref_b=ref_people[rng.integers(0,len(hi),len(hi))].mean(axis=0)
   si=rng.integers(0,len(selected),len(selected));bs=person_score(selected[si],ref_b)
   for k in range(3):boot[b,k]=correlation(bs,labels[si,k])
  outcomes={}
  for k,label in enumerate(['POMA','FAC','TIS']):
   finite=boot[:,k][np.isfinite(boot[:,k])]
   outcomes[label]={'spearman_rho':correlation(scores,labels[:,k]),'bootstrap_95_percentile_ci':np.quantile(finite,[.025,.975]).tolist(),'bootstrap_valid':len(finite),'bootstrap_discarded':int(bootstrap_n-len(finite))}
  normal_scores=[]
  for i in np.flatnonzero(valid_h):
   member=i in set(hi.tolist())
   ref_i=(ref_people.sum(axis=0)-h[i].mean(axis=0))/(len(hi)-1) if member else reference
   sides=side_score(h[i],ref_i)
   persons['healthy'][i]['scores'][ref_name]={'left':float(sides[0]),'right':float(sides[1]),'bilateral_mean':float(sides.mean()),'self_excluded':member,'reference_n':len(hi)-int(member)}
   if link['healthy']['rows'][i]['values']['Age']>=65:normal_scores.append(float(sides.mean()))
  for i in np.flatnonzero(valid_s):
   sides=side_score(s[i],reference)
   persons['stroke'][i]['scores'][ref_name]={'paretic':float(sides[0]),'nonparetic':float(sides[1]),'bilateral_mean':float(sides.mean()),'reference_n':len(hi)}
  def describe(values):
   return {'n':len(values),'median':float(np.median(values)),'q25_q75':np.quantile(values,[.25,.75]).tolist(),'min_max':[float(np.min(values)),float(np.max(values))]}
  result['analyses'][ref_name]={'reference_n':len(hi),'reference_indices_1based':(hi+1).tolist(),'clinical_associations':outcomes,'healthy65plus_leave_person_out':describe(normal_scores),'stroke65plus':describe(scores)}
  np.savez_compressed(DATA/f'bootstrap-{ref_name}.npz',correlations=boot,reference=reference)
  print(ref_name,json.dumps(result['analyses'][ref_name]),flush=True)
 (ROOT/'person-scores.json').write_text(json.dumps(persons,indent=2),encoding='utf-8')
 (ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':run()
