"""Person-disjoint source-only GDI control, with independently replayed linear classifier."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files}
assert sha(BIG/'data.npz')==read(ROOT/'studies/20261104-ljmu100-mixed/comparison.json')['data_sha256']
raw=d['x'];w=np.concatenate([raw.mean(2),raw.std(2),np.ptp(raw,axis=2),np.abs(np.diff(raw,axis=2)).mean(2)],axis=2).mean(1)
x=np.zeros((len(d['y']),48));np.add.at(x,d['window_record'],w);x/=np.bincount(d['window_record'])[:,None]
y=(d['source']=='LJMU').astype(int);ids=d['person_id'];fold=d['fold'];prob=np.full(len(y),np.nan);models=[]
for f in range(5):
 tr=fold!=f;te=~tr;assert set(ids[tr]).isdisjoint(ids[te])
 weights=np.array([1/(ids[tr]==q).sum() for q in ids[tr]])
 for s in [0,1]:weights[y[tr]==s]/=weights[y[tr]==s].sum()
 weights*=tr.sum()/2
 m=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=20261105))
 m.fit(x[tr],y[tr],logisticregression__sample_weight=weights);prob[te]=m.predict_proba(x[te])[:,1]
 sc,lr=m.steps[0][1],m.steps[1][1];logit=((x[te]-sc.mean_)/sc.scale_)@lr.coef_[0]+lr.intercept_[0]
 replay=1/(1+np.exp(-logit));assert np.max(np.abs(replay-prob[te]))<1e-12
 models.append(dict(fold=f,train_people=np.unique(ids[tr]).tolist(),test_people=np.unique(ids[te]).tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0])))
def load(arm,family):
 pred=np.full_like(d['y'],np.nan,dtype=float)
 for f in range(5):
  path=BIG/'received/data/runs'/arm/family/('fold%d'%f);done=read(path/'complete.json');assert sha(path/'predictions.npz')==done['prediction_sha256']
  with np.load(path/'predictions.npz') as z:
   assert np.array_equal(z['row_index'],np.flatnonzero(fold==f));pred[z['row_index']]=z['refit']
 return pred
rows=[];comparisons=[];output={};groups={'Stanford_all':np.repeat((y==0)[:,None],2,1),'Stanford_ge90':(y==0)[:,None]&(d['y']>=90),'Stanford_ge100':(y==0)[:,None]&(d['y']>=100),'Stanford_lt70':(y==0)[:,None]&(d['y']<70),'LJMU_assumed100':np.repeat((y==1)[:,None],2,1)}
for family in ['CNN','STT']:
 base=load('baseline',family);mixed=load('mixed',family)
 preds={'baseline':base,'mixed':mixed,'hard_gate':np.where(prob[:,None]>=.5,100.,base),'soft_gate':prob[:,None]*100+(1-prob[:,None])*base}
 for group,mask in groups.items():
  people=np.unique(ids[mask.any(1)]);ind={}
  for method,pred in preds.items():
   err=pred-d['y'];v=np.array([[np.abs(err[(ids==q)[:,None]&mask]).mean(),np.maximum(-err[(ids==q)[:,None]&mask],0).mean(),err[(ids==q)[:,None]&mask].mean(),(err[(ids==q)[:,None]&mask]<0).mean(),np.maximum(err[(ids==q)[:,None]&mask],0).mean()] for q in people]);ind[method]=v
   ri,si=np.nonzero(mask);weight=np.array([1/(len(people)*(ids[ri]==q).sum()) for q in ids[ri]])
   assert abs(np.abs(err[ri,si])@weight-v[:,0].mean())<1e-10
   rows.append(dict(family=family,group=group,method=method,people=len(people),mae=float(v[:,0].mean()),shortfall=float(v[:,1].mean()),bias=float(v[:,2].mean()),under_fraction=float(v[:,3].mean()),overprediction=float(v[:,4].mean()),mean_prediction=float(np.mean([pred[(ids==q)[:,None]&mask].mean() for q in people]))))
  draws=np.random.default_rng(20261105).integers(len(people),size=(10000,len(people)))
  for method in ['mixed','hard_gate','soft_gate']:
   delta=ind[method][:,0]-ind['baseline'][:,0];ci=np.quantile(delta[draws].mean(1),[.025,.975]);comparisons.append(dict(family=family,group=group,method=method,delta_mae=float(delta.mean()),low95=float(ci[0]),high95=float(ci[1])))
 output[family]={k:v.tolist() for k,v in preds.items()}
people=np.unique(ids);pp=np.array([prob[ids==q].mean() for q in people]);truth=np.array([y[ids==q][0] for q in people])
result=dict(passed=True,protocol_sha256=sha(HERE/'protocol.json'),data_sha256=sha(BIG/'data.npz'),source_auc=float(roc_auc_score(truth,pp)),source_balanced_accuracy=float(balanced_accuracy_score(truth,pp>=.5)),rows=rows,comparisons=comparisons,models=models,record_probabilities=prob.tolist(),predictions=output,limitations='Fixed held-out development people. LJMU MAE is distance from assumption only. Source classifier replay and independent metric weighting passed.')
(HERE/'source-gate-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,source_auc=result['source_auc'],rows=[r for r in rows if r['group'] in ['Stanford_all','Stanford_ge100','LJMU_assumed100']]),indent=2))
