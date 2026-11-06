"""Remaining raw folders, including high missingness; no outcomes or training."""
from pathlib import Path
import os,sys,json,hashlib,datetime,time,importlib.util,warnings,traceback
from collections import Counter
ROOT=Path('D:/Projects/D_P');H=ROOT/'gdi-research-harness';S=H/'studies';HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/gdi-unused-audit-20261105');CASE=S/'20261105-gdi-case-audit'
sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'));sys.path.insert(0,str(S/'20261102-final223-frozen'))
import numpy as np,torch
torch.set_num_threads(4)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 v=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):v.update(b)
 return v.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
prep=module('existing_final_preparation',S/'20261102-final223-frozen/prepare.py')
# Historical helper is kept unchanged on disk; bind only this run to the actual large-data archive.
prep.arc.ARCHIVE=Path('D:/gait-research-data/D_P/outputs/gait-public-access-20261015/Stanford/video-gait-v1.zip')
def main():
 protocol_sha=sha(HERE/'protocol.json');state=dict(pid=os.getpid(),active=True,started_at=datetime.datetime.now().astimezone().isoformat(),protocol_sha256=protocol_sha,script_sha256=sha(__file__),phase='inventory',processed=0,optimizer_steps=0,new_training=False);save(HERE/'preparation-process.json',state)
 try:
  source=H/'control/desktop-sessions/20261026T150518Z-87c0f4/identity_linkage_rows.json';rows=read(source);models=read(CASE/'models.json');seal=read(CASE/'freeze.json');assert sha(CASE/'models.json')==seal['models_sha256'];train=set()
  for m in models['members']:
   assert sha(m['checkpoint'])==m['checkpoint_sha256'];identity=torch.load(m['checkpoint'],map_location='cpu',weights_only=True)['identity'];train.update(identity['train_people']);train.update(identity['validation_people'])
  assert len(train)==692
  with np.load(S/'20261030-gpu-motionbert-study/cnn-paper-input695/bundle/data.npz',allow_pickle=False) as z:devvideos=set(z['video_id'])
  with np.load(Path('D:/gait-research-data/D_P/final223-frozen-20261102/features.npz'),allow_pickle=False) as z:finalpeople=set(z['person_id']);finalvideos=set(z['video_id'])
  used=devvideos|finalvideos;rows=[dict(r) for r in rows if r['video_id'] not in used];assert len(rows)==4275 and len({r['video_id'] for r in rows})==4275
  old=read(H/'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json');old37=set(old['development_patient_ids'])-train;old38=set(old['original_test_patient_ids']);ledger=read(H/'control/desktop-sessions/20261027T065618Z-8aab63/input_readiness_ledger.json')['records'];lookup={r['video_id']:r for r in ledger}
  for i,r in enumerate(rows):
   person=r.get('patient_id','');r.update(index=i,person_id=person,cohort='additional_development' if person in train else 'additional_final' if person in finalpeople else 'primary_remaining' if person else 'unlinked',history='old37_validation' if person in old37 else 'old38_test' if person in old38 else 'not_old37_38',old_input124_status=lookup.get(r['video_id'],{}).get('input124_status','not_in_new_ledger'))
  archive=prep.arc.ARCHIVE;stat=dict(size=archive.stat().st_size,mtime_ns=archive.stat().st_mtime_ns);assert stat==read(H/'jobs/native124-cache-v1/config.json')['archive_stat']
  protected={str(p):sha(p) for p in [source,CASE/'models.json',CASE/'results.json',CASE/'verification.json',S/'20261102-final223-frozen/prepare.py',S/'20261030-gpu-motionbert-study/cnn-multiwindow695/prepare_inputs.py',S/'20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/sources/provenance.json',ROOT/'tools/stanford-sample-check/inspect_archive.py']};save(HERE/'inventory.json',dict(protocol_sha256=protocol_sha,models_sha256=sha(CASE/'models.json'),train_validation_people=sorted(train),prior_final_people=sorted(finalpeople),archive_stat=stat,protected=protected,records=rows,counts=dict(Counter(r['cohort'] for r in rows)),primary_people=len({r['person_id'] for r in rows if r['cohort']=='primary_remaining'}),unlinked_accuracy_forbidden=True))
  # Exact replay on the same three cached source records, no labels.
  devrows=read(S/'20261030-gpu-motionbert-study/cnn-multiwindow695/input_sources.json')['records'][:3];replayed=[]
  def replay(r,raw,digest,ambiguous):
   xx,q,canonical=prep.convert(raw,np.array(r['frames']))
   with np.load(S/('20261030-gpu-motionbert-study/cnn-multiwindow695/records/%04d.npz'%r['index']),allow_pickle=False) as z:
    np.testing.assert_array_equal(xx,z['multi_x']);np.testing.assert_array_equal(q['starts'],z['window_start']);np.testing.assert_array_equal(canonical.astype(np.float32),z['x'])
   replayed.append(r['index'])
  prep.scan(devrows,replay);assert len(replayed)==3;save(HERE/'feature-pilot.json',dict(passed=True,exact_development_replay=replayed,protocol_sha256=protocol_sha,optimizer_steps=0))
  state['phase']='preparing_remaining';save(HERE/'preparation-process.json',state);proofs=[];starttime=time.monotonic()
  def prepare(row,raw,digest,ambiguous):
   path=BIG/'records'/('%04d.npz'%row['index']);proofpath=path.with_suffix('.json')
   if proofpath.exists():
    q=read(proofpath);assert q['raw_sha256']==digest and q['protocol_sha256']==protocol_sha and q['video_id']==row['video_id']
    if q['available']:assert sha(path)==q['features_sha256']
   else:
    observed=np.flatnonzero(np.isfinite(raw).any(axis=1))
    if len(observed):raw=raw[:observed[-1]+1]
    q=dict(index=row['index'],video_id=row['video_id'],person_id=row['person_id'],cohort=row['cohort'],history=row['history'],old_input124_status=row['old_input124_status'],raw_sha256=digest,protocol_sha256=protocol_sha,raw_frames=len(raw),uniquely_observed_frames=int(len(observed)),ambiguous_frames=ambiguous,available=False,reason=None)
    vals=[];starts=[];miss=[]
    if len(raw)<124:q['reason']='under124_converter_extent'
    elif not len(observed):q['reason']='no_uniquely_selected_person'
    else:
     try:
      with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);processed=prep.ns['process_video_and_add_cols'](raw,swap_orientation=False)
      for k in range(0,500-124,31):
       chunk=raw[k:k+124]
       if len(chunk)!=124:continue
       xx=np.stack([processed[k:k+124,prep.LEFT],processed[k:k+124,prep.RIGHT]]).astype(np.float32)
       if not np.isfinite(xx).all():continue
       missing=float(prep.ns['max_pct_nan_or_zero_given_cols'](prep.ns['drop_confidence_cols'](chunk),[14,11,13,10,12,9,19,22]));vals.append(xx);starts.append(k);miss.append(missing)
     except (ValueError,IndexError,FloatingPointError) as e:q['processing_error']=type(e).__name__+': '+str(e)
     if vals:
      x=np.stack(vals);st=np.array(starts);mi=np.array(miss);strict=mi<=.25;selected=np.flatnonzero(strict) if strict.any() else np.arange(len(vals));quality='standard_window_gate_only' if strict.any() else 'high_missing_exploratory';q.update(available=True,reason=None,quality=quality,finite_grid_windows=len(vals),selected_windows=len(selected),selected_missing_min=float(mi[selected].min()),selected_missing_max=float(mi[selected].max()),selected_all_absent_coordinate_windows=int((mi[selected]>=1).sum()),first_selected_start=int(st[selected[0]]));np.savez_compressed(path,x=x[selected],window_start=st[selected],missing=mi[selected]);q['features_sha256']=sha(path)
     else:q['reason']='no_finite_native124_features'
    save(proofpath,q)
   proofs.append(q);state['processed']=len(proofs);state['last_video']=row['video_id'];state['seconds']=time.monotonic()-starttime;save(HERE/'preparation-process.json',state)
  prep.scan(rows,prepare);proofs.sort(key=lambda r:r['index']);save(HERE/'input-audit.json',dict(protocol_sha256=protocol_sha,records=len(rows),available=sum(r['available'] for r in proofs),quality_counts=dict(Counter(r.get('quality','unavailable') for r in proofs)),unavailable_reasons=dict(Counter(r['reason'] for r in proofs if not r['available'])),quality=proofs,development_replay_exact=True))
  parts=[];owners=[];starts=[];missing=[];eligible=[]
  for q in proofs:
   if not q['available']:continue
   r=rows[q['index']];i=len(eligible);eligible.append(r)
   with np.load(BIG/'records'/('%04d.npz'%q['index']),allow_pickle=False) as z:parts.append(z['x']);owners.extend([i]*len(z['x']));starts.extend(z['window_start']);missing.extend(z['missing'])
  assert parts
  np.savez_compressed(BIG/'features.npz',x=np.concatenate(parts),window_record=np.array(owners),window_start=np.array(starts),missing=np.array(missing),person_id=np.array([r['person_id'] for r in eligible]),video_id=np.array([r['video_id'] for r in eligible]),source_index=np.array([r['index'] for r in eligible]),cohort=np.array([r['cohort'] for r in eligible]))
  assert sha(HERE/'protocol.json')==protocol_sha
  for path,dig in protected.items():assert sha(path)==dig,path
  assert stat==dict(size=archive.stat().st_size,mtime_ns=archive.stat().st_mtime_ns)
  save(HERE/'features-locked.json',dict(passed=True,features_sha256=sha(BIG/'features.npz'),input_audit_sha256=sha(HERE/'input-audit.json'),inventory_sha256=sha(HERE/'inventory.json'),protocol_sha256=protocol_sha,records=len(eligible),windows=len(owners),optimizer_steps=0));state.update(phase='features_locked',records=len(eligible),windows=len(owners));print(json.dumps(read(HERE/'features-locked.json')),flush=True)
 except BaseException:state.update(failed=True,error=traceback.format_exc());save(HERE/'preparation-failure.json',state);raise
 finally:state.update(active=False,ended_at=datetime.datetime.now().astimezone().isoformat());save(HERE/'preparation-process.json',state)
if __name__=='__main__':main()
