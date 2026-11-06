"""Existing checkpoints only: CPU window traces and exact fixed combinations."""
from prepare import *
import os,sys,time,uuid,importlib.util,traceback,itertools
import numpy as np
sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
import torch
torch.set_num_threads(4);torch.manual_seed(20261105);torch.use_deterministic_algorithms(True)
F=Path('D:/gait-research-data/D_P/final223-frozen-20261102');E=Path('D:/gait-research-data/D_P/final223-extra-baselines-20261102');R=Path('D:/gait-research-data/D_P/final223-ranksim-followup-20261102')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,resolve(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
prior=module('case_prior_frozen',S/'20261101-self-video-gdi/predict.py')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def guard():
 seal=read(H/'freeze.json');assert sha(H/'protocol.json')==seal['protocol_sha256'] and sha(H/'models.json')==seal['models_sha256']
 for n,dig in seal['code'].items():assert sha(H/n)==dig,('code changed',n)
 return read(H/'models.json')
def aggregate(v,owner,n):
 sums=np.zeros((n,2));np.add.at(sums,owner,v);counts=np.bincount(owner,minlength=n);assert (counts>0).all();return sums/counts[:,None]
def combine(a,b,side,w,t):
 base=w*a+(1-w)*b;diff=(side[:,0]-side[:,1])-(base[:,0]-base[:,1]);return base+np.column_stack([t*diff/2,-t*diff/2])
def configurations(values,manifest):
 out={};refit={role:[] for role in ['CNN','STT','single','general','MSE','Rank1','HuberOnly','HuberRank1','MAEOnly','MAERank1']};fusions=[];bags=[];adds={a:[] for a in ['general','Rank1','HuberOnly','HuberRank1','MAEOnly','MAERank1']}
 for f,sf in enumerate(manifest['selections']):
  selected={family:[values[m['key']] for m in manifest['members'] if m['role']==family and m['fold']==f] for family in ['CNN','STT']}
  cnn=np.mean(selected['CNN'],axis=0);stt=np.mean(selected['STT'],axis=0);bags.append(cnn);fusion=combine(cnn,stt,values['base_bilateral_%d'%f],sf['cnn_weight'],sf['fraction']);fusions.append(fusion)
  refit['CNN'].append(values['base_CNN_%d_refit'%f]);refit['STT'].append(values['base_STT_%d_refit'%f])
  for role in refit:
   if role in ['CNN','STT']:continue
   refit[role].append(values['%s_%d'%(role,f)])
  for role in adds:
   w=manifest['addition_weights'][role][f];adds[role].append((1-w)*fusion+w*values['%s_%d'%(role,f)])
 out.update(single_CNN=np.mean(refit['single'],axis=0),multi_STT=np.mean(refit['STT'],axis=0),multi_CNN=np.mean(refit['CNN'],axis=0),CNN_bag=np.mean(bags,axis=0),fusion=np.mean(fusions,axis=0))
 for role in ['MSE','Rank1','HuberOnly','HuberRank1','MAEOnly','MAERank1']:out[role]=np.mean(refit[role],axis=0)
 for role,name in [('general','plus_general'),('Rank1','plus_ranksim'),('HuberOnly','plus_HuberOnly'),('HuberRank1','plus_HuberRank1'),('MAEOnly','plus_MAEOnly'),('MAERank1','plus_MAERank1')]:out[name]=np.mean(adds[role],axis=0)
 assert len(out)==17;return out

def forward(model,m,x,ck,src):
 model.eval();parts=[];batch=16 if m['precision']=='float32' else 128
 with torch.inference_mode():
  for start in range(0,len(x),batch):
   v=torch.from_numpy(x[start:start+batch]);family=m['family']
   if family=='CNN':
    t=src.channels(v).reshape(-1,12,124)
    if m['precision']=='float64':t=t.double()
    value=model(t)[:,0]
   elif family=='STT':value=model(v[...,:8].reshape(-1,124,4,2))
   else:value=model(v[...,:8].reshape(-1,2,124,4,2))
   p=(value*float(ck['span'])+float(ck['lo'])).reshape(-1,2).numpy();parts.append(p.astype(np.float64))
 return np.concatenate(parts)*m['calibration'][0]+m['calibration'][1]

def replay(model,m,ck,src,d,old,single_old,added,dev):
 if m['replay_kind']=='dev_multi':
  z=arrays(m['replay_source']);lookup={v:i for i,v in enumerate(dev['video_id'])};rows=[]
  for v in z['video_id'][:2]:
   i=lookup[v];p=forward(model,m,dev['x'][dev['window_record']==i],ck,src).mean(0);rows.append(p)
  error=float(abs(np.array(rows)-z['predictions'][:2,0]).max())
 elif m['replay_kind']=='old_single':error=float(abs(forward(model,m,d['canonical'][:2],ck,src)-single_old[m['replay_key']][:2]).max())
 else:
  rows=[]
  for i in range(2):rows.append(forward(model,m,d['x'][d['window_record']==i],ck,src).mean(0))
  expected=(old if m['replay_kind']=='old_final_member' else added)[m['replay_key']][:2];error=float(abs(np.array(rows)-expected).max())
 assert error<1e-4,(m['key'],error);return error

def main():
 manifest=guard();protocol=read(H/'protocol.json');BIG.mkdir(exist_ok=True);(BIG/'members').mkdir(exist_ok=True)
 state=dict(pid=os.getpid(),active=True,started_at=datetime.datetime.now().astimezone().isoformat(),script_sha256=sha(__file__),protocol_sha256=sha(H/'protocol.json'),models_sha256=sha(H/'models.json'),optimizer_steps=0,new_training=False,device='cpu',threads=4,phase='preflight',completed_members=[])
 save(H/'process-receipt.json',state)
 try:
  for p,digest in manifest['protected'].items():assert sha(p)==digest,p
  d=arrays(F/'features.npz');d['canonical']=arrays(E/'single-features.npz')['x'];old=arrays(F/'predictions.npz');single_old=arrays(E/'predictions.npz');added=arrays(R/'predictions.npz');dev=arrays(T/'cnn-record-loss692/bundle/data.npz');reserved=set(read(S/'20261102-final223-frozen/input-membership.json')['people'])
  assert d['x'].shape==(3877,2,124,12) and len(d['person_id'])==377 and len(set(d['person_id']))==222 and not reserved&set(dev['person_id'])
  for z in [old,single_old,added]:
   for k in ['person_id','video_id']:np.testing.assert_array_equal(d[k],z[k])
  # Independent grouping check on real membership, including records with one window.
  a=aggregate(np.ones((len(d['x']),2)),d['window_record'],377);np.testing.assert_array_equal(a,np.ones((377,2)))
  for i in range(377):
   ix=np.flatnonzero(d['window_record']==i);st=d['window_start'][ix];assert len(ix)>0 and np.all(np.diff(st)>0)
   for budget in protocol['case_tests']['shorter_input']['budgets_frames']:
    keep=st+124<=st[0]+budget;assert keep.any() and (st[keep]+124<=st[0]+budget).all()
   assert (st+124<=st[0]+496).all()
  # Independent algebra for the bilateral correction.
  rng=np.random.default_rng(19);aa,bb,cc=[rng.normal(size=(4,2)) for _ in range(3)];w=.35;t=.5;base=w*aa+(1-w)*bb;mean=base.mean(1);delta=(1-t)*(base[:,0]-base[:,1])+t*(cc[:,0]-cc[:,1]);np.testing.assert_allclose(combine(aa,bb,cc,w,t),np.column_stack([mean+delta/2,mean-delta/2]),rtol=0,atol=1e-12)
  save(H/'input-preflight.json',dict(passed=True,records=377,people=222,windows=3877,all_budgets_same_records=True,all_windows_native124=True,no_padding_or_resampling=True,aggregate_ones=True,independent_bilateral_formula=True,train_cohort_overlap=0))
  ordered=sorted(manifest['members'],key=lambda m:(m['key'] not in protocol['compute']['pilot_members'],manifest['members'].index(m)));values={};canonical={};timings=[]
  for m in ordered:
   guard();start=time.monotonic();path=BIG/'members'/(m['key']+'.npz');proofpath=path.with_suffix('.json')
   if proofpath.exists():
    q=read(proofpath);assert q['protocol_sha256']==state['protocol_sha256'] and q['checkpoint_sha256']==m['checkpoint_sha256'] and q['prediction_sha256']==sha(path);z=arrays(path);values[m['key']]=z['windows']
    if 'canonical' in z:canonical[m['key']]=z['canonical']
   else:
    assert sha(m['checkpoint'])==m['checkpoint_sha256'] and sha(m['model_source'])==m['model_source_sha256'];ck=torch.load(resolve(m['checkpoint']),map_location='cpu',weights_only=True);ident=ck.get('identity',{})
    assert 'train_people' in ident and 'validation_people' in ident and not reserved&(set(ident['train_people'])|set(ident['validation_people'])),m['key']
    if m['role'] in ['CNN','STT','bilateral']:model=prior.new_model(m['family']);src=prior.CNN
    else:src=module('case_model_'+m['key'],m['model_source']);model=src.GdiCNN()
    if m['precision']=='float64':model=model.double()
    model.load_state_dict(ck['model'],strict=True)
    state.update(phase='pilot' if m['key'] in protocol['compute']['pilot_members'] else 'inference',active_member=m['key']);save(H/'process-receipt.json',state)
    err=replay(model,m,ck,src,d,old,single_old,added,dev);pred=forward(model,m,d['x'],ck,src);payload={'windows':pred}
    if m['role']=='single':payload['canonical']=forward(model,m,d['canonical'],ck,src);canonical[m['key']]=payload['canonical']
    assert pred.shape==(3877,2) and np.isfinite(pred).all();np.savez_compressed(path,**payload);values[m['key']]=pred
    q=dict(passed=True,key=m['key'],checkpoint_sha256=m['checkpoint_sha256'],protocol_sha256=state['protocol_sha256'],models_sha256=state['models_sha256'],prediction_sha256=sha(path),replay_error=err,reserved_overlap=0,seconds=time.monotonic()-start,optimizer_steps=0);save(proofpath,q)
   timings.append({k:q[k] for k in ['key','seconds','replay_error']});state['completed_members'].append(m['key']);save(H/'process-receipt.json',state);print(json.dumps(dict(member=m['key'],completed=len(values),total=len(ordered),seconds=q['seconds'],replay=q['replay_error'])),flush=True)
   if len(values)==len(protocol['compute']['pilot_members']):
    assert set(values)==set(protocol['compute']['pilot_members']);save(H/'pilot-verification.json',dict(passed=True,members=timings,optimizer_steps=0,all_replays_below1e4=True,scope='Existing-model CPU pilot only; no final scores computed'))
  window=configurations(values,manifest);full={name:aggregate(pred,d['window_record'],377) for name,pred in window.items()};full['single_CNN']=np.mean(list(canonical.values()),axis=0)
  references=arrays(E/'scored.npz');errors={}
  for name in protocol['models']['core']:
   errors[name]=float(abs(full[name]-references[name]).max());assert errors[name]<1e-4,(name,errors[name])
  np.savez_compressed(BIG/'predictions.npz',person_id=d['person_id'],video_id=d['video_id'],window_record=d['window_record'],window_start=d['window_start'],**{'window__'+k:v for k,v in window.items()},**{'full__'+k:v for k,v in full.items()})
  save(H/'predictions-locked.json',dict(passed=True,protocol_sha256=state['protocol_sha256'],models_sha256=state['models_sha256'],predictions_sha256=sha(BIG/'predictions.npz'),previous7_full_reproduction_max_errors=errors,members=len(values),optimizer_steps=0,new_training=False,source_records=377,source_people=222,windows=3877))
  state.update(phase='predictions_locked',reporting_pending=True)
 except BaseException:
  state.update(failed=True,error=traceback.format_exc());save(H/'execution-failure.json',state);raise
 finally:
  state.update(active=False,ended_at=datetime.datetime.now().astimezone().isoformat());save(H/'process-receipt.json',state)
if __name__=='__main__':main()
