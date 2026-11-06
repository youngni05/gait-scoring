"""Independent person/record/quality checks for the remaining-data report."""
from prepare_inputs import *
import math,csv

def main():
 seal=read(HERE/'code-seal.json');inv=read(HERE/'inventory.json');audit=read(HERE/'input-audit.json');lock=read(HERE/'predictions-locked.json');receipt=read(HERE/'evaluation-receipt.json');r=read(HERE/'results.json');models=read(CASE/'models.json');checks=0
 for n,dig in seal['code'].items():assert sha(HERE/n)==dig,n
 assert sha(HERE/'protocol.json')==seal['protocol_sha256']==r['protocol_sha256'] and sha(CASE/'models.json')==seal['models_sha256']
 for path,dig in models['protected'].items():assert sha(path)==dig,path
 for path,dig in inv['protected'].items():assert sha(path)==dig,path
 assert sha(HERE/'results.json')==receipt['results_sha256'] and sha(BIG/'scored.npz')==receipt['scored_sha256']==r['scored_sha256'] and sha(BIG/'all-record-predictions.csv')==receipt['all_record_csv_sha256'] and sha(BIG/'predictions.npz')==r['predictions_sha256']==lock['predictions_sha256']
 with np.load(BIG/'scored.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 with np.load(BIG/'features.npz',allow_pickle=False) as z:owner=z['window_record'];sourceindex=z['source_index'];start=z['window_start'];missing=z['missing'];features=z['x']
 with np.load(BIG/'predictions.npz',allow_pickle=False) as z:window={n:z['window__'+n] for n in r['models']}
 membership=read(HERE/'prediction-membership.json')['records'];assert len(membership)==lock['records']==audit['available']==4174 and len(owner)==len(features) and features.shape[1:]==(2,124,12) and np.isfinite(features).all();train=set(inv['train_validation_people']);final=set(inv['prior_final_people'])
 for i,row in enumerate(membership):
  assert row['video_id']==d['video_id'][i] and row['person_id']==d['person_id'][i] and row['index']==sourceindex[i]
  if row['cohort']=='primary_remaining':assert row['person_id'] not in train|final
  ix=np.flatnonzero(owner==i);q=audit['quality'][row['index']];assert len(ix)==q['selected_windows'] and np.all(np.diff(start[ix])>0) and start[ix].min()>=0 and start[ix].max()<=372
  if row['quality']=='standard_window_gate_only':assert (missing[ix]<=.25).all()
  else:assert (missing[ix]>.25).all()
  for n in r['models']:
   expected=window[n][ix[0]] if n=='single_CNN' else np.mean(window[n][ix],axis=0);np.testing.assert_allclose(expected,d[n][i],rtol=0,atol=1e-10,equal_nan=True)
 def close(a,b):
  nonlocal checks
  assert math.isclose(a,b,rel_tol=0,abs_tol=1e-10),(a,b);checks+=1
 ids=d['person_id'];y=d['y'];mask0=d['common_accuracy_mask'];names=r['models']
 np.testing.assert_array_equal(mask0,np.isfinite(y).all(1)&np.logical_and.reduce([np.isfinite(d[n]).all(1) for n in names]))
 for key,result in r['groups'].items():
  co,quality=key.split('__');base=mask0&(d['cohort']==co)
  if quality!='all':base&=d['quality']==quality
  for name in names:
   for band,z in result[name].items():
    vals=[];bias=[];nr=0
    for person in set(ids[base]):
     ix=[i for i in range(len(ids)) if ids[i]==person and base[i]];truth=sum(float(y[i,j]) for i in ix for j in [0,1])/(len(ix)*2)
     if band=='below60' and truth>=60 or band=='90plus' and truth<90 or band=='100plus' and truth<100:continue
     vals.append(sum(abs(float(d[name][i,j])-float(y[i,j])) for i in ix for j in [0,1])/(len(ix)*2));bias.append(sum(float(d[name][i,j])-float(y[i,j]) for i in ix for j in [0,1])/(len(ix)*2));nr+=len(ix)
    assert len(vals)==z['people'] and nr==z['records']
    if vals:close(sum(vals)/len(vals),z['mae']);close(sum(bias)/len(bias),z['bias'])
    else:assert z['mae'] is None and z['bias'] is None
 policy=read(HERE/'label-policy.json')
 if not policy['old38_numeric_labels']:assert not np.isfinite(y[d['history']=='old38_test']).any()
 assert not np.isfinite(y[d['cohort']=='unlinked']).any()
 with (BIG/'all-record-predictions.csv').open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
 assert len(csvrows)==4275 and {v['video_id'] for v in csvrows}=={v['video_id'] for v in inv['records']} and len({v['video_id'] for v in csvrows})==4275
 errors=[]
 for m in models['members']:
  q=read(BIG/'members'/(m['key']+'.json'));assert q['passed'] and q['primary_train_overlap']==0 and q['optimizer_steps']==0 and q['batch_replay_error']<1e-4 and q['prediction_sha256']==sha(BIG/'members'/(m['key']+'.npz')) and q['checkpoint_sha256']==m['checkpoint_sha256'];errors.append(q['batch_replay_error'])
 save(HERE/'verification.json',dict(passed=True,models=len(names),checkpoints=len(errors),all_source_hashes_unchanged=True,primary_train_final_person_overlap=0,identified_and_unknown_separate=True,all4275_records_retained=True,independent_metric_checks=checks,max_batch_replay_error=max(errors),old38_numeric_labels=policy['old38_numeric_labels'],unknown_identity_numeric_labels=False,results_sha256=sha(HERE/'results.json'),all_record_csv_sha256=sha(BIG/'all-record-predictions.csv'),scored_sha256=sha(BIG/'scored.npz'),optimizer_steps=0,new_training=False));print(json.dumps(read(HERE/'verification.json')),flush=True)
if __name__=='__main__':main()
