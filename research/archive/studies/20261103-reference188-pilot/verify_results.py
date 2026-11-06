"""Independent replay using explicit sums and rank correlation."""
import json,sys,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np
from scipy.stats import rankdata
DATA=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot')
protocol=json.loads((ROOT/'protocol.json').read_text());result=json.loads((ROOT/'results.json').read_text());persons=json.loads((ROOT/'person-scores.json').read_text());link=json.loads((ROOT/'person-linkage.json').read_text())
assert result['protocol_sha256']==hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
archive=np.load(DATA/'person-curves.npz');curves={key:archive[key] for key in ['healthy','stroke']};sel=protocol['time_samples'];maxerr=0.;checked=0;correlations={}
for name,analysis in result['analyses'].items():
 hi=[i-1 for i in analysis['reference_indices_1based']]
 normal=[(curves['healthy'][i,0][:,sel]+curves['healthy'][i,1][:,sel])/2 for i in hi]
 summed=sum(normal);ref=summed/len(hi)
 for group in ['healthy','stroke']:
  for i,row in enumerate(persons[group]):
   if not row['usable']:continue
   reference=(summed-normal[hi.index(i)])/(len(hi)-1) if group=='healthy' and i in hi else ref
   values=[]
   for side in range(2):
    arr=curves[group][i,side][:,sel]
    error=math.sqrt(math.fsum((float(arr[k,t])-float(reference[k,t]))**2 for k in range(9) for t in range(51))/(9*51))
    values.append(error)
   actual=sum(values)/2;stored=row['scores'][name]['bilateral_mean'];maxerr=max(maxerr,abs(actual-stored));assert abs(actual-stored)<1e-10
   checked+=1
 si=[i-1 for i in result['stroke_indices_1based']]
 x=np.array([persons['stroke'][i]['scores'][name]['bilateral_mean'] for i in si]);rx=rankdata(x);rx-=rx.mean()
 correlations[name]={}
 for label,record in analysis['clinical_associations'].items():
  y=np.array([link['stroke']['rows'][i]['values'][label] for i in si]);ry=rankdata(y);ry-=ry.mean()
  rho=float(np.dot(rx,ry)/np.sqrt(np.dot(rx,rx)*np.dot(ry,ry)))
  assert abs(rho-record['spearman_rho'])<1e-12
  correlations[name][label]=rho
summary={'score_records_replayed':checked,'max_abs_score_difference_degrees':maxerr,'correlations_replayed':correlations,'protocol_unchanged':True,'results_sha256':hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest(),'person_scores_sha256':hashlib.sha256((ROOT/'person-scores.json').read_bytes()).hexdigest()}
(ROOT/'independent-verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
