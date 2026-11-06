from pathlib import Path
import sys,json,hashlib,warnings
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,HuberRegressor,GammaRegressor
from sklearn.exceptions import ConvergenceWarning
GRIDS={'Ridge':[.1,1.,10.,100.],'Huber':[.01,1.,10.,100.],'Gamma':[.001,.01,.1,1.]}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def source(info):return HERE.parent/info['study']/info['folder']
def learner(family,alpha):
 if family=='Ridge':return Ridge(alpha=alpha,solver='svd')
 if family=='Huber':return HuberRegressor(alpha=alpha,epsilon=1.35,max_iter=2000,tol=1e-6)
 if family=='Gamma':return GammaRegressor(alpha=alpha,solver='lbfgs',max_iter=3000,tol=1e-7)
 raise ValueError(family)
def fit(x,y,train,test,family,alpha):
 scale=StandardScaler().fit(x[train]);model=learner(family,alpha)
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning)
  model.fit(scale.transform(x[train]),y[train])
 raw=model.predict(scale.transform(x[test]));assert np.isfinite(raw).all()
 return np.maximum(raw,0),raw,(scale,model)

def freeze():
 assert not (HERE/'protocol.json').exists()
 configs={'geometry':{'study':'20261031-opencap-turn','folder':'geometry','views':['WalkMatchedSpeed','Walk3Turn']}}
 for cohort,added in [('event_strict','ChairAll'),('fixed_strict','ChairMotion'),('fixed_flagged','ChairMotion'),('rise_strict','RiseMotion'),('prep_event','PrepAll')]:configs[cohort]={'study':'20261031-opencap-chair','folder':cohort,'views':['TurnContext',added]}
 for cohort,info in configs.items():
  src=source(info);d=dict(np.load(src/'input.npz'));assert len(set(d['ids']))==len(d['y']) and np.isfinite(d['y']).all() and min(d['y'])>0
  for view in info['views']:assert np.isfinite(d[view]).all()
  info.update({'people':len(d['y']),'long_people':int(sum(d['y']>=20)),'hashes':{n:sha(src/n) for n in ['input.npz','splits.json','manifest.json','predictions.csv','selections.json','completion.json']}})
 protocol={'cohorts':configs,'models':GRIDS,'primary_model':'CVSelect chooses among three tuned families by inner MAE only; all family results also reported','splits':'exact previous outer5partitions x5fold and inner3fold; no person or feature/window changes','determinism':'new solvers deterministic, one fit per partition/fold, no artificial3seed replication; RF comparison averages errors across its3seeds before averaging5partitions','target':'unchanged clinical tft_tug seconds; no reweighting or target transform','postprocessing':'negative linear outputs floored at0; raw stored; no upper clipping, including above training maximum','selection':'train-only scaling; alpha and family selected by inner validation MAE with floor0; convergence failures recorded as ineligible candidates; no test-driven tuning','metrics':'all/under20/20plus MAE, positive underprediction, overprediction, bias, above-training-max frequency; paired person bootstrap conditional on fits','scope':'development evidence, same-visit timed-trial and phase uncertainties persist','fit_code_sha256':sha(HERE/'study.py')}
 save('protocol.json',protocol);print(json.dumps({k:{'people':v['people'],'long':v['long_people'],'views':v['views']} for k,v in configs.items()}),flush=True)
if __name__=='__main__':freeze()
