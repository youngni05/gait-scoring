"""Fixed existing94 models on all remaining compatible records; no labels."""
from prepare_inputs import *
sys.path.insert(0,str(CASE))
case=module('fixed_case_inference',CASE/'infer.py')
def guard():
 seal=read(HERE/'code-seal.json');assert sha(HERE/'protocol.json')==seal['protocol_sha256']
 for n,dig in seal['code'].items():assert sha(HERE/n)==dig,n
 manifest=case.guard();assert sha(CASE/'models.json')==seal['models_sha256'];return manifest

def batched(model,m,x,ck,src,batch):
 model.eval();parts=[]
 with torch.inference_mode():
  for begin in range(0,len(x),batch):
   v=torch.from_numpy(x[begin:begin+batch]);family=m['family']
   if family=='CNN':
    t=src.channels(v).reshape(-1,12,124)
    if m['precision']=='float64':t=t.double()
    value=model(t)[:,0]
   elif family=='STT':value=model(v[...,:8].reshape(-1,124,4,2))
   else:value=model(v[...,:8].reshape(-1,2,124,4,2))
   parts.append((value*float(ck['span'])+float(ck['lo'])).reshape(-1,2).numpy().astype(np.float64))
 return np.concatenate(parts)*m['calibration'][0]+m['calibration'][1]

def main():
 manifest=guard();locked=read(HERE/'features-locked.json');assert locked['passed'] and sha(BIG/'features.npz')==locked['features_sha256'] and sha(HERE/'inventory.json')==locked['inventory_sha256'];assert not (HERE/'predictions-locked.json').exists()
 inv=read(HERE/'inventory.json');audit=read(HERE/'input-audit.json');state=dict(pid=os.getpid(),active=True,started_at=datetime.datetime.now().astimezone().isoformat(),phase='preflight',features_sha256=locked['features_sha256'],protocol_sha256=sha(HERE/'protocol.json'),code_seal_sha256=sha(HERE/'code-seal.json'),completed_members=[],optimizer_steps=0,device='cpu',new_training=False);save(HERE/'inference-process.json',state)
 try:
  for path,dig in inv['protected'].items():assert sha(path)==dig,path
  d=case.arrays(BIG/'features.npz');x=d['x'];owner=d['window_record'];n=len(d['video_id']);assert x.shape[1:]==(2,124,12) and np.isfinite(x).all() and len(set(d['video_id']))==n;first=np.array([np.flatnonzero(owner==i)[0] for i in range(n)]);train=set(inv['train_validation_people']);final=set(inv['prior_final_people']);rows=[inv['records'][int(i)] for i in d['source_index']]
  # Derive exact historical validation membership from the old packet identity/split fields, without opening its Y.
  oldpacket=H/'jobs/development-baseline/runs/20261025T044558572569Z-71adb0/development_packet.npz'
  with np.load(oldpacket,allow_pickle=False) as z:old37=set(z['patient_ids'][z['split']=='validation'])
  old38=set(read(H/'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json')['original_test_patient_ids']);assert len(old37)==37 and len(old38)==38
  for i,r in enumerate(rows):
   assert r['video_id']==d['video_id'][i] and r['person_id']==d['person_id'][i]
   if r['cohort']=='primary_remaining':assert r['person_id'] not in train|final
   r['preparation_history_original']=r['history'];r['history']='old37_validation' if r['person_id'] in old37 else 'old38_test' if r['person_id'] in old38 else 'not_old37_38'
   r['quality']=audit['quality'][r['index']]['quality']
  save(HERE/'prediction-membership.json',dict(records=rows,old37_people=sorted(old37),old38_people=sorted(old38),old_packet_sha256=sha(oldpacket),note='Exact old37 validation membership uses split metadata. Preparation source history was the broader old-development-minus-current-training set and included one excluded old training person; original field preserved. No input/selection/cohort changed.'))
  good=[i for i,r in enumerate(rows) if r['quality']=='standard_window_gate_only'];assert len(good)>=2;probe=np.array([first[good[0]],np.flatnonzero(owner==good[-1])[-1]]);(BIG/'members').mkdir(exist_ok=True);values={};batch_choices={}
  for m in manifest['members']:
   path=BIG/'members'/(m['key']+'.npz');proofpath=path.with_suffix('.json');begin=time.monotonic();state.update(active_member=m['key'],phase='inference');save(HERE/'inference-process.json',state)
   if proofpath.exists():
    q=read(proofpath);assert q['features_sha256']==state['features_sha256'] and q['checkpoint_sha256']==m['checkpoint_sha256'] and q['prediction_sha256']==sha(path)
    with np.load(path,allow_pickle=False) as z:pred=z['windows']
   else:
    assert sha(m['checkpoint'])==m['checkpoint_sha256'] and sha(m['model_source'])==m['model_source_sha256'];ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=True);ident=ck['identity'];assert not set(r['person_id'] for r in rows if r['cohort']=='primary_remaining')&(set(ident['train_people'])|set(ident['validation_people']))
    if m['role'] in ['CNN','STT','bilateral']:model=case.prior.new_model(m['family']);src=case.prior.CNN
    else:src=case.module('remaining_model_'+m['key'],m['model_source']);model=src.GdiCNN()
    if m['precision']=='float64':model=model.double()
    model.load_state_dict(ck['model'],strict=True);model.eval();family_key=m['family']+'_'+m['precision']
    if family_key not in batch_choices:
     sample=x[np.concatenate([np.flatnonzero(owner==i) for i in good[:100]])[:1024]];baseline=case.forward(model,m,sample,ck,src);options=[16,64,128] if m['precision']=='float32' else [128,256,512];trials=[]
     for batch in options:
      t=time.monotonic();candidate=batched(model,m,sample,ck,src,batch);elapsed=time.monotonic()-t;error=float(abs(candidate-baseline).max());trials.append(dict(batch=batch,seconds=elapsed,max_difference=error,accepted=error<1e-4))
     valid=[z for z in trials if z['accepted']];assert valid;batch_choices[family_key]=min(valid,key=lambda z:z['seconds'])['batch'];save(HERE/'computational-batches.json',dict(choices=batch_choices,trials_latest_family=family_key,trials=trials,criterion='Runtime only, no labels or model selection; same <1e-4 replay criterion'))
    batch=batch_choices[family_key];pred=batched(model,m,x,ck,src,batch);small=np.vstack([case.forward(model,m,x[k:k+1],ck,src)[0] for k in probe]);assert np.isfinite(small).all() and np.isfinite(pred[probe]).all();error=float(abs(small-pred[probe]).max());assert error<1e-4,(m['key'],error)
    assert pred.shape==(len(x),2);np.savez_compressed(path,windows=pred);q=dict(key=m['key'],passed=True,features_sha256=state['features_sha256'],checkpoint_sha256=m['checkpoint_sha256'],prediction_sha256=sha(path),batch_replay_error=error,nonfinite_windows=int((~np.isfinite(pred).all(1)).sum()),seconds=time.monotonic()-begin,primary_train_overlap=0,optimizer_steps=0);save(proofpath,q)
   values[m['key']]=pred;state['completed_members'].append(m['key']);save(HERE/'inference-process.json',state);print(json.dumps(dict(member=m['key'],completed=len(values),total=len(manifest['members']),seconds=q['seconds'],nonfinite=q['nonfinite_windows'],batch_replay=q['batch_replay_error'])),flush=True)
  window=case.configurations(values,manifest);full={name:case.aggregate(pred,owner,n) for name,pred in window.items()};full['single_CNN']=window['single_CNN'][first];np.savez_compressed(BIG/'predictions.npz',**{k:d[k] for k in ['person_id','video_id','source_index','cohort','window_record','window_start','missing']},**{'full__'+name:v for name,v in full.items()},**{'window__'+name:v for name,v in window.items()})
  guard()
  for path,dig in inv['protected'].items():assert sha(path)==dig,path
  save(HERE/'predictions-locked.json',dict(passed=True,protocol_sha256=sha(HERE/'protocol.json'),models_sha256=sha(CASE/'models.json'),features_sha256=sha(BIG/'features.npz'),predictions_sha256=sha(BIG/'predictions.npz'),membership_sha256=sha(HERE/'prediction-membership.json'),members=len(values),configurations=len(full),records=n,windows=len(x),all_models_finite_records=int(np.logical_and.reduce([np.isfinite(v).all(1) for v in full.values()]).sum()),optimizer_steps=0,new_training=False,numeric_new_labels_opened=False));state['phase']='predictions_locked';print(json.dumps(read(HERE/'predictions-locked.json')),flush=True)
 except BaseException:state.update(failed=True,error=traceback.format_exc());save(HERE/'inference-failure.json',state);raise
 finally:state.update(active=False,ended_at=datetime.datetime.now().astimezone().isoformat());save(HERE/'inference-process.json',state)
if __name__=='__main__':main()
