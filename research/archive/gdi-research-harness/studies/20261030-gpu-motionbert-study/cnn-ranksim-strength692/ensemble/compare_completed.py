"""Use only model-held-out inner predictions to choose each convex blend."""
from pathlib import Path
import os,json,hashlib,datetime
import numpy as np
from select_weight import choose_weight
H=Path(__file__).resolve().parent;S=H.parents[1];ROOT=S.parents[1];O=H/'analysis'
CACHE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-ranksim-strength692/ensemble/cache')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def independent_mae(y,p,ids):
 return float(np.mean([np.mean(np.abs(p[ids==i]-y[ids==i])) for i in np.unique(ids)]))
cfg=read(H/'protocol.json');O.mkdir(exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
process=dict(pid=os.getpid(),active=True,started_at=datetime.datetime.now().astimezone().isoformat(),protocol_sha256=sha(H/'protocol.json'),script_sha256=sha(__file__),new_training=False)
save(O/'process-receipt.json',process)
reg=read(ROOT/'control/gpu-watch/registry.json');assert reg['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a'
sources={str(H/'protocol.json'):sha(H/'protocol.json')}
def evidence(p,digest=None):
 p=Path(p);got=sha(p)
 if digest is not None:assert got==digest,str(p)
 sources[str(p)]=got;return p
paths={}
for key,study,eid in [('rank01',H.parent,'cnn-ranksim-strength-692'),('rank03',H.parent,'cnn-ranksim-strength-692'),('mse_control',S/'cnn-ranksim692','cnn-ranksim-692'),('rank1',S/'cnn-ranksim692','cnn-ranksim-692')]:
 row=next(e for e in reg['experiments'] if e['id']==eid)
 report_path=evidence(study/'analysis/comparison.json',row.get('performance_analysis_sha256') if eid=='cnn-ranksim-strength-692' else row['reported_digest']);report=read(report_path)
 assert report['qualified_full_verification_passed'] and report['fits']==50 and report['final_cpu_replays']==10
 receipt_path=evidence(study/('checks/full-receipt.json' if eid=='cnn-ranksim-strength-692' else 'checks/closeout-receipt.json'),row['retrieval_receipt_sha256']);receipt=read(receipt_path)
 assert receipt['server_local_sha256_matched']
 paths[key]=Path(receipt['archive']).parent/'performance-review-originals'
VARIANTS={'rank01':'rank01','rank03':'rank03','mse_control':'mse','rank1':'ranksim'}
base_report=read(evidence(S/'inner-bagging692/analysis/comparison.json'))
base=arrays(evidence(base_report['predictions_path'],base_report['predictions_sha256']))
keys=['y','person_id','video_id','fold'];d={k:base[k] for k in keys};y=d['y'];ids=d['person_id'];people=np.unique(ids)
assert len(people)==cfg['people']==692 and len(y)==cfg['records']==1061
assert abs(independent_mae(y,base['bagged'],ids)-6.515816569178735)<1e-10
md=S/'mean-difference692/analysis';es=S/'cnn-stt-ensemble692/analysis'
selection=read(evidence(md/'selection.json'));proof=read(evidence(md/'verification.json'))
assert proof['passed'] and proof['selection_sha256']==sha(md/'selection.json')
assert proof['predictions_sha256']==sha(md/'predictions.npz')
old=read(evidence(es/'selection.json'));oldproof=read(evidence(es/'verification.json'))
assert oldproof['passed'] and oldproof['all_20_base_fit_person_boundaries']
for p,h in selection['source_hashes'].items():evidence(p,h)
for p,h in old['source_hashes'].items():evidence(S/p,h)
def boundary(ident,f,j):
 tr=(d['fold']!=f)&(d['fold']!=j);va=d['fold']==j;te=d['fold']==f
 for field,mask in [('train_people',tr),('validation_people',va),('outer_evaluation_people',te)]:
  assert ident[field]==np.unique(ids[mask]).tolist(),(field,f,j)
 assert not set(ident['train_people'])&set(ident['validation_people'])
 assert not (set(ident['train_people'])|set(ident['validation_people']))&set(ident['outer_evaluation_people'])
 return tr,va,te
# A leaked ID must be rejected by the same boundary checker used below.
bad=dict(train_people=['injected'],validation_people=[],outer_evaluation_people=[])
try:boundary(bad,0,1)
except AssertionError:pass
else:raise AssertionError('Injected leakage accepted')
all_pred={a:np.full_like(y,np.nan) for a in cfg['arms']};joint=np.full_like(y,np.nan);selections=[];joint_selections=[];inner_errors=[]
for f in range(5):
 sf=selection['folds'][f];assert sf['outer']==f
 v=arrays(evidence(md/('inner_outer%d.npz'%f),sf['inner_sha256']))
 ix=v['row_index'];np.testing.assert_array_equal(ix,np.flatnonzero(d['fold']!=f))
 for k in keys:np.testing.assert_array_equal(v[k],d[k][ix])
 ep=es/('inner_outer%d.npz'%f);e=arrays(evidence(ep,old['folds'][f]['inner_file_sha256']))
 np.testing.assert_array_equal(e['row_index'],ix)
 np.testing.assert_allclose(v['base'],sf['cnn_weight']*e['cnn']+(1-sf['cnn_weight'])*e['stt'],rtol=0,atol=1e-12)
 diff=(v['bilateral'][:,0]-v['bilateral'][:,1])-(v['base'][:,0]-v['base'][:,1])
 inner_base=v['base']+np.column_stack([sf['fraction']*diff/2,-sf['fraction']*diff/2])
 inner={a:np.full_like(inner_base,np.nan) for a in cfg['arms']}
 for j in range(5):
  if f==j:continue
  pos=v['fold']==j;va=d['fold']==j
  np.testing.assert_array_equal(ix[pos],np.flatnonzero(va))
  for folder in [S/'cnn-multiwindow695/received/results'/('fold%d/inner%d'%(f,j)),S/'stt692/multiwindow/analysis/unpacked/results'/('fold%d/inner%d'%(f,j)),S/'stt692/bilateral/analysis/unpacked/results'/('fold%d/inner%d'%(f,j))]:
   ident=read(evidence(folder/'identity.json'));boundary(ident,f,j)
  for a in cfg['arms']:
   variant=VARIANTS[a]
   folder=paths[a]/'runs/main'/variant/('fold%d/inner%d'%(f,j))
   ident=read(evidence(folder/'identity.json'));tr,valid,_=boundary(ident,f,j)
   assert ident['purpose']=='inner_epoch_selection' and ident['epochs']==100 and ident['outer_fold']==f and ident['inner_fold']==j
   done=read(evidence(folder/'fit_complete.json'));assert done['identity']==ident
   evidence(folder/'latest.pt',done['checkpoint_sha256']);curve=read(evidence(folder/'curve.json',done['curve_sha256']))
   row=curve[-1];assert row['epoch']==100 and len(curve)==100
   z=arrays(evidence(folder/'raw_predictions/epoch0100.npz',row['raw_predictions_sha256']))
   p=z['validation_raw']*row['calibration'][0]+row['calibration'][1]
   assert p.shape==y[valid].shape
   assert abs(independent_mae(y[valid],p,ids[valid])-row['validation_person_mae'])<1e-10
   assert hashlib.sha256(y[tr].tobytes()).hexdigest()==ident['train_y_sha']
   assert hashlib.sha256(ids[tr].tobytes()).hexdigest()==ident['train_ids_sha']
   inner[a][pos]=p
 assert not set(v['person_id'])&set(ids[d['fold']==f])
 outer_add={}
 for a in cfg['arms']:
  assert np.isfinite(inner[a]).all()
  chosen=choose_weight(v['y'],inner_base,inner[a],v['person_id'],cfg['weight_grid'])
  w=chosen['rank_weight']
  scores=[independent_mae(v['y'],(1-q)*inner_base+q*inner[a],v['person_id']) for q in cfg['weight_grid']]
  expected=next(q for q,s in zip(cfg['weight_grid'],scores) if s<=min(scores)+1e-12)
  assert expected==w
  reverse=choose_weight(v['y'][::-1],inner_base[::-1],inner[a][::-1],v['person_id'][::-1],cfg['weight_grid'])
  assert reverse['rank_weight']==w
  poisoned=y.copy();poisoned[d['fold']==f]=1e9
  assert choose_weight(poisoned[ix],inner_base,inner[a],v['person_id'],cfg['weight_grid'])['rank_weight']==w
  folder=paths[a]/'runs/main'/VARIANTS[a]/('fold%d'%f)
  done=read(evidence(folder/'complete.json'));outer=arrays(evidence(folder/'predictions.npz',done['prediction_sha256']))
  refit=read(evidence(folder/'refit/identity.json'))
  assert refit['train_people']==np.unique(ids[d['fold']!=f]).tolist() and refit['outer_evaluation_people']==np.unique(ids[d['fold']==f]).tolist() and refit['validation_people']==[]
  oi=outer['row_index'];np.testing.assert_array_equal(oi,np.flatnonzero(d['fold']==f))
  for k in keys:np.testing.assert_array_equal(outer[k],d[k][oi])
  outer_add[a]=outer['predictions'][:,0]
  all_pred[a][oi]=(1-w)*base['bagged'][oi]+w*outer_add[a]
  selections.append(dict(outer=f,arm=a,additional_weight=w,inner_scores=scores,inner_people=len(np.unique(v['person_id'])),outer_labels_used=False))
  inner_errors.append(dict(outer=f,arm=a,raw_signed_error_correlation=float(np.corrcoef((inner_base-v['y']).ravel(),(inner[a]-v['y']).ravel())[0,1])))
 candidates=[dict(arm=a,strength=cfg['strengths'][a],weight=w,score=score) for a in cfg['arms'] for w,score in zip(cfg['weight_grid'],next(q['inner_scores'] for q in selections if q['outer']==f and q['arm']==a))]
 best_score=min(q['score'] for q in candidates)
 choice=min((q for q in candidates if q['score']<=best_score+1e-12),key=lambda q:(q['weight'],q['strength']))
 te=d['fold']==f;joint[te]=(1-choice['weight'])*base['bagged'][te]+choice['weight']*outer_add[choice['arm']]
 joint_selections.append(dict(outer=f,**choice,outer_labels_used=False))
 np.savez_compressed(CACHE/('inner_outer%d.npz'%f),**{k:v[k] for k in keys+['row_index']},base=inner_base,**inner)
 evidence(CACHE/('inner_outer%d.npz'%f))
 print('ENSEMBLE_OUTER',f,flush=True)
all_pred['joint_inner_selected']=joint
old_ensemble=read(evidence(S/'ranksim-ensemble692/analysis/comparison.json'))
original=arrays(evidence(old_ensemble['predictions_path'],old_ensemble['predictions_sha256']))
for k in keys:np.testing.assert_array_equal(original[k],d[k])
np.testing.assert_array_equal(all_pred['rank1'],original['ranksim'])
pm=np.array([y[ids==p].mean() for p in people]);high=pm>=90
def perperson(v):return np.array([v[ids==i].mean() for i in people])
def metrics(p):
 e=p-y;err=perperson(abs(e));bias=perperson(e)
 out=dict(mae=float(err.mean()),bias=float(bias.mean()),high90=dict(people=int(high.sum()),mae=float(err[high].mean()),bias=float(bias[high].mean())),fold_mae=[float(np.mean([abs(e[ids==i]).mean() for i in people if d['fold'][ids==i][0]==f])) for f in range(5)],bands={})
 for w in [10,20]:out['bands'][str(w)]=float(perperson(np.searchsorted(np.arange(0,161,w),p,side='right')==np.searchsorted(np.arange(0,161,w),y,side='right')).mean())
 out['subgroups']={}
 for label,mask in [('person_mean100',pm>=100),('person_mean60below',pm<60)]:out['subgroups'][label]=dict(people=int(mask.sum()),mae=float(err[mask].mean()),bias=float(bias[mask].mean()))
 for t in [90,100]:
  maes=[];biases=[];n=0
  for i in people:
   ee=e[ids==i][y[ids==i]>=t]
   if len(ee):maes.append(abs(ee).mean());biases.append(ee.mean());n+=len(ee)
  out['subgroups']['observation_ge%d'%t]=dict(people=len(maes),side_observations=n,mae=float(np.mean(maes)),bias=float(np.mean(biases)))
 return out,err
models={};errors={}
for a,p in {'existing_best':base['bagged'],'existing_rank_ensemble':original['ranksim'],**all_pred}.items():
 assert np.isfinite(p).all();models[a],errors[a]=metrics(p)
rng=np.random.default_rng(20261101);draw=rng.integers(0,len(people),(10000,len(people)));hd=rng.integers(0,int(high.sum()),(10000,int(high.sum())))
paired={}
for a in [*cfg['arms'],'joint_inner_selected']:
 delta=errors[a]-errors['existing_best'];dh=delta[high]
 paired[a]=dict(delta=float(delta.mean()),bootstrap95=np.quantile(delta[draw].mean(1),[.025,.975]).tolist(),high90_delta=float(dh.mean()),high90_bootstrap95=np.quantile(dh[hd].mean(1),[.025,.975]).tolist(),improved_folds=int(np.sum(np.array(models[a]['fold_mae'])<np.array(models['existing_best']['fold_mae']))))
paired_rank={}
for a in [*cfg['arms'],'joint_inner_selected']:
 delta=errors[a]-errors['existing_rank_ensemble'];dh=delta[high]
 paired_rank[a]=dict(delta=float(delta.mean()),bootstrap95=np.quantile(delta[draw].mean(1),[.025,.975]).tolist(),high90_delta=float(dh.mean()),high90_bootstrap95=np.quantile(dh[hd].mean(1),[.025,.975]).tolist(),improved_folds=int(np.sum(np.array(models[a]['fold_mae'])<np.array(models['existing_rank_ensemble']['fold_mae']))))
dest=CACHE/'evaluation.npz';np.savez_compressed(dest,**d,existing_best=base['bagged'],existing_rank_ensemble=original['ranksim'],**all_pred);evidence(dest)
save(O/'selection.json',dict(protocol_sha256=sha(H/'protocol.json'),folds=selections,joint_selections=joint_selections,inner_error_correlations=inner_errors,source_hashes=sources))
report=dict(passed=True,people=692,records=1061,models=models,paired=paired,paired_vs_original_rank_ensemble=paired_rank,predictions_path=str(dest),predictions_sha256=sha(dest),selection_sha256=sha(O/'selection.json'),protocol_sha256=sha(H/'protocol.json'),outer_label_selection=False,global_OOF_meta_split=False,new_training=False,bootstrap_replicates=10000,limitations=cfg['limitations'])
save(O/'comparison.json',report)
save(O/'verification.json',dict(passed=True,independent_metric=True,independent_grid_selection=True,row_reversal=True,outer_label_poisoning=True,injected_leak_rejected=True,all_inner_train_IDs_checked=True,all_input_hashes_checked=True,baseline_reproduced=True,comparison_sha256=sha(O/'comparison.json')))
process.update(active=False,completed_at=datetime.datetime.now().astimezone().isoformat());save(O/'process-receipt.json',process)
print(json.dumps(dict(models=models,paired=paired,paired_vs_original_rank_ensemble=paired_rank,weights=selections,joint_selections=joint_selections),ensure_ascii=True),flush=True)
