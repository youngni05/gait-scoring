"""Independent loops verify metrics, grouping and immutable sources."""
from prepare import *
import numpy as np,itertools,math

def main():
 seal=read(H/'freeze.json');models=read(H/'models.json');res=read(H/'results.json');receipt=read(H/'analysis-receipt.json');lock=read(H/'predictions-locked.json')
 assert sha(H/'protocol.json')==seal['protocol_sha256'] and sha(H/'models.json')==seal['models_sha256']
 for n,d in seal['code'].items():assert sha(H/n)==d,n
 for p,d in models['protected'].items():assert sha(p)==d,p
 assert sha(H/'results.json')==receipt['results_sha256'] and sha(BIG/'predictions.npz')==receipt['predictions_sha256']==lock['predictions_sha256'] and sha(BIG/'case-details.npz')==receipt['case_details_sha256']
 with np.load(BIG/'predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 with np.load(BIG/'case-details.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
 ids=d['person_id'];y=c['y'];owner=d['window_record'];starts=d['window_start'];people=sorted(set(ids));checks=0
 def close(a,b):
  nonlocal checks
  assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=0),(a,b);checks+=1
 names=res['models'];membererrors=[]
 for m in models['members']:
  q=read(BIG/'members'/(m['key']+'.json'));assert q['passed'] and q['reserved_overlap']==0 and q['optimizer_steps']==0 and q['checkpoint_sha256']==m['checkpoint_sha256'] and q['prediction_sha256']==sha(BIG/'members'/(m['key']+'.npz')) and q['replay_error']<1e-4;membererrors.append(q['replay_error'])
 for n in names:
  p=d['full__'+n];w=d['window__'+n]
  for group,threshold in [('all',None),('below50',50),('below60',60),('90plus',90),('100plus',100)]:
   losses=[];biases=[];nr=0
   for person in people:
    rows=[i for i,v in enumerate(ids) if v==person];yt=sum(float(y[i,j]) for i in rows for j in [0,1])/(len(rows)*2)
    include=threshold is None or (yt<threshold if group.startswith('below') else yt>=threshold)
    if not include:continue
    losses.append(sum(abs(float(p[i,j])-float(y[i,j])) for i in rows for j in [0,1])/(len(rows)*2));biases.append(sum(float(p[i,j])-float(y[i,j]) for i in rows for j in [0,1])/(len(rows)*2));nr+=len(rows)
   z=res['full'][n][group];assert len(losses)==z['people'] and nr==z['records']
   if losses:close(sum(losses)/len(losses),z['mae']);close(sum(biases)/len(biases),z['bias'])
   else:assert z['mae'] is None and z['bias'] is None
  for i in range(len(ids)):
   ix=[k for k,v in enumerate(owner) if v==i];assert len(ix)==c['window_count'][i]
   if n!='single_CNN':
    for budget in [124,186,248,372,496]:
     chosen=[k for k in ix if int(starts[k])+124<=int(starts[ix[0]])+budget];assert chosen
     for side in [0,1]:close(sum(float(w[k,side]) for k in chosen)/len(chosen),c['prefix_%d__%s'%(budget,n)][i,side])
    np.testing.assert_allclose(c['prefix_496__'+n][i],p[i],rtol=0,atol=1e-10)
  # Independent within-record SD/range and repeat-difference loops.
  sds=[];rgs=[];reps=[]
  for person in people:
   rows=np.flatnonzero(ids==person);v=[];r=[]
   for i in rows:
    ix=np.flatnonzero(owner==i)
    if len(ix)>=3:
     v.append(sum(math.sqrt(sum((float(w[k,j])-sum(float(w[t,j]) for t in ix)/len(ix))**2 for k in ix)/(len(ix)-1)) for j in [0,1])/2);r.append(sum(max(float(w[k,j]) for k in ix)-min(float(w[k,j]) for k in ix) for j in [0,1])/2)
   if v:sds.append(sum(v)/len(v));rgs.append(sum(r)/len(r))
   pairs=list(itertools.combinations(rows,2))
   if pairs:reps.append(sum(sum(abs(float(p[b,j])-float(p[a,j])-float(y[b,j])+float(y[a,j])) for j in [0,1])/2 for a,b in pairs)/len(pairs))
  close(sum(sds)/len(sds),res['native_stability'][n]['sample_sd_min3']['mean']);close(sum(rgs)/len(rgs),res['native_stability'][n]['range_min3']['mean']);close(sum(reps)/len(reps),res['repeat_records'][n]['delta_error_MAE']['mean'])
 # Observation thresholds and side-gap errors, independently grouped with Python lists.
 for n in names:
  pred=d['full__'+n]
  for label,threshold in [('below60',60),('90plus',90),('100plus',100)]:
   vals=[];biases=[];nr=0;ns=0
   for person in people:
    pairs=[(i,j) for i in range(len(ids)) if ids[i]==person for j in [0,1] if (y[i,j]<threshold if label=='below60' else y[i,j]>=threshold)]
    if not pairs:continue
    vals.append(sum(abs(float(pred[i,j])-float(y[i,j])) for i,j in pairs)/len(pairs));biases.append(sum(float(pred[i,j])-float(y[i,j]) for i,j in pairs)/len(pairs));nr+=len(set(i for i,j in pairs));ns+=len(pairs)
   q=res['observation_groups'][n][label];assert q['people']==len(vals) and q['records']==nr and q['sides']==ns
   if vals:close(sum(vals)/len(vals),q['mae']);close(sum(biases)/len(biases),q['bias'])
   else:assert q['mae'] is None
  for threshold in [5,10]:
   vals=[]
   for person in people:
    rows=[i for i in range(len(ids)) if ids[i]==person and abs(float(y[i,0])-float(y[i,1]))>=threshold]
    if rows:vals.append(sum(abs(float(pred[i,0])-float(pred[i,1])-float(y[i,0])+float(y[i,1])) for i in rows)/len(rows))
   q=res['asymmetry'][n][str(threshold)];assert q['people']==len(vals);close(sum(vals)/len(vals),q['mean'])
 # New ensemble full predictions rebuilt from saved old fusion fold predictions and individual checkpoints.
 basepath=Path('D:/gait-research-data/D_P/final223-frozen-20261102/predictions.npz')
 with np.load(basepath,allow_pickle=False) as z:basefold=[z['fusion_%d'%f] for f in range(5)]
 for role in ['HuberOnly','HuberRank1','MAEOnly','MAERank1']:
  expected=np.zeros((len(ids),2))
  for f in range(5):
   with np.load(BIG/'members'/('%s_%d.npz'%(role,f)),allow_pickle=False) as z:wv=z['windows']
   weight=models['addition_weights'][role][f]
   for i in range(len(ids)):
    ix=np.flatnonzero(owner==i)
    for side in [0,1]:expected[i,side]+=((1-weight)*basefold[f][i,side]+weight*sum(float(wv[k,side]) for k in ix)/len(ix))/5
  np.testing.assert_allclose(expected,d['full__plus_'+role],rtol=0,atol=1e-10)
 # Existing 7 final prediction replay is also checked over every record, not only pilot examples.
 for n,e in lock['previous7_full_reproduction_max_errors'].items():assert e<1e-4
 assert len(models['members'])==94 and len(names)==17 and len(people)==222 and len(ids)==377 and receipt['optimizer_steps']==0
 save(H/'verification.json',dict(passed=True,independent_scalar_comparisons=checks,models=17,checkpoints=94,all_source_hashes_unchanged=True,all_train_eval_disjoint=True,all_94_replay_max_error=max(membererrors),previous7_full_reproduction_max_errors=lock['previous7_full_reproduction_max_errors'],results_sha256=sha(H/'results.json'),case_details_sha256=sha(BIG/'case-details.npz'),predictions_sha256=sha(BIG/'predictions.npz'),protocol_sha256=sha(H/'protocol.json'),optimizer_steps=0,new_training=False,scope='Post-hoc same-source case descriptions; no fresh final or clinical validation'))
 print(json.dumps(read(H/'verification.json')),flush=True)
if __name__=='__main__':main()
