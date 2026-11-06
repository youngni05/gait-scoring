"""Frozen adult-LJMU models on unseen-source LIVE children; no target labels."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'studies/20261104-ljmu100-mixed'
sys.path[:0]=[str(ROOT/'studies/20261105-gdi-masked-pretrain-v3/bundle'),str(ROOT/'outputs/motionbert-smoke-20261015/deps')]
import experiment as e
import torch
BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
e.setup(20261105)
external_path=Path('D:/gait-research-data/D_P/gdi-masked-pretrain-v3-20261105/data.npz')
assert sha(external_path)==read(ROOT/'studies/20261105-gdi-masked-pretrain-v3/bundle/protocol.json')['data_sha256']
with np.load(external_path) as z:
 wr=z['window_record'];wi=np.flatnonzero(wr>=1073);x=z['x'][wi];ids=z['person_id'][1073:];records=z['record_id'][1073:] if 'record_id' in z else np.arange(12).astype(str)
assert x.shape==(12,2,124,12) and len(np.unique(ids))==9
with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files}
assert not set(ids)&set(d['person_id'])
members=read(OLD/'bundle/baseline-models.json');provenance=[];results=[];allpred={}
for family in ['CNN','STT']:
 pred={k:[] for k in ['baseline','mixed_original_calibration','mixed_Stanford_calibration']}
 for f in range(5):
  for arm in ['baseline','mixed']:
   folder=BIG/'received/data/runs'/arm/family/('fold%d'%f)
   if arm=='baseline':
    member=next(r for r in members if r['family']==family and r['fold']==f and r['tag']=='refit');path=BIG/member['checkpoint'];coef=np.array(member['calibration']);digest=member['checkpoint_sha256']
   else:
    done=read(folder/'complete.json');member=next(r for r in done['members'] if r['tag']=='refit');path=folder/'refit/latest.pt';coef=np.array(member['calibration']);digest=member['checkpoint_sha256']
   assert sha(path)==digest
   ck=torch.load(path,map_location='cpu',weights_only=True);assert not set(ids)&set(ck['identity']['train_people'])
   model=e.newmodel(family);model.load_state_dict(ck['model'],strict=True);model.eval()
   with torch.no_grad():raw=(e.forward(model,family,torch.from_numpy(x.reshape(-1,124,12)))*ck['span']+ck['lo']).reshape(-1,2).numpy()
   # Replay three original held-out records before using the frozen model externally.
   with np.load(folder/'predictions.npz') as z:
    rows=z['row_index'][:3];expected=z['refit_raw'][:3]
   original=dict(d);original['record_weight']=np.ones(len(d['y']))
   rp=e.predict(model,family,original,rows,ck['lo'],ck['span'],'cpu');error=float(np.max(np.abs(rp-expected)));assert error<.001,error
   if arm=='baseline':pred['baseline'].append(raw*coef[0]+coef[1])
   else:
    pred['mixed_original_calibration'].append(raw*coef[0]+coef[1])
    with np.load(folder/'refit/replay.npz') as z:
     tr=z['train_row'];mask=d['source'][tr]=='Stanford';sc=e.ols(z['train_raw'][mask],d['y'][tr][mask])
    pred['mixed_Stanford_calibration'].append(raw*sc[0]+sc[1])
   provenance.append(dict(family=family,fold=f,arm=arm,checkpoint=str(path),sha256=digest,heldout_three_record_replay_error=error))
  print(family,f,'replayed',flush=True)
 personpred={}
 for method,values in pred.items():
  stack=np.stack(values);mean=stack.mean(0);people=np.unique(ids);pm=np.array([mean[ids==q].mean() for q in people]);personpred[method]=pm
  results.append(dict(family=family,method=method,people=9,records=12,mean_prediction=float(pm.mean()),median=float(np.median(pm)),min=float(pm.min()),max=float(pm.max()),people_ge90=int((pm>=90).sum()),people_ge100=int((pm>=100).sum()),per_person=dict(zip(people.tolist(),pm.tolist())),fold_means=[float(np.mean([v[ids==q].mean() for q in people])) for v in stack]))
  allpred[family+'__'+method]=stack.tolist()
 for method in ['mixed_original_calibration','mixed_Stanford_calibration']:
  delta=personpred[method]-personpred['baseline'];draw=np.random.default_rng(20261105).integers(9,size=(10000,9));ci=np.quantile(delta[draw].mean(1),[.025,.975])
  next(r for r in results if r['family']==family and r['method']==method).update(delta_mean=float(delta.mean()),delta95=ci.tolist())
gate=read(HERE/'source-gate-results.json');features=np.concatenate([x.mean(2),x.std(2),np.ptp(x,axis=2),np.abs(np.diff(x,axis=2)).mean(2)],axis=2).mean(1);prob=[]
for m in gate['models']:
 logit=((features-np.array(m['mean']))/np.array(m['scale']))@np.array(m['coef'])+m['intercept'];prob.append(1/(1+np.exp(-logit)))
for family in ['CNN','STT']:
 base=np.array(allpred[family+'__baseline']);pr=np.array(prob)[:,:,None]
 for method,stack in [('hard_gate',np.where(pr>=.5,100.,base)),('soft_gate',pr*100+(1-pr)*base)]:
  mean=stack.mean(0);pm=np.array([mean[ids==q].mean() for q in np.unique(ids)])
  results.append(dict(family=family,method=method,people=9,records=12,mean_prediction=float(pm.mean()),min=float(pm.min()),max=float(pm.max()),people_ge90=int((pm>=90).sum()),people_ge100=int((pm>=100).sum()),per_person=dict(zip(np.unique(ids).tolist(),pm.tolist()))))
  allpred[family+'__'+method]=stack.tolist()
out=dict(passed=True,protocol_sha256=sha(HERE/'protocol.json'),external_data_sha256=sha(external_path),rows=results,provenance=provenance,predictions=allpred,LIVE_LJMU_source_probability=np.mean(prob,axis=0).tolist(),labels='None: score distribution and paired change only, not MAE or clinical accuracy',limitation='Five-fold model average on the same9 LIVE children; previous project exploration, never used in these checkpoints; age/source/extraction confounded.')
(HERE/'external-live-results.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(results,indent=2))
