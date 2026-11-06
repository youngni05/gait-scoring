"""Independent local replay and paired person analysis of the two completed CNN arms.
Run only after original remote files have been recovered and SHA256-matched.
"""
import os,sys,json,hashlib,argparse
from pathlib import Path
sys.path.insert(0,'D:/Projects/D_P/outputs/motionbert-smoke-20261015/deps')
import numpy as np
import torch
H=Path(__file__).resolve().parent;B=Path('D:/gait-research-data/D_P/cnn-stable-scale-20261103')
sys.path.insert(0,str(H/'bundle'))
from model import GdiCNN,channels
from windows import WindowSet
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def infer(model,x,lo,span):
 out=[]
 with torch.inference_mode():
  for k in range(0,len(x),64):
   a=channels(torch.from_numpy(np.array(x[k:k+64],copy=True))).reshape(-1,12,124)
   out.append((model(a)[:,0]*float(span)+float(lo)).reshape(-1,2).numpy())
 return np.concatenate(out).astype(float)
def aggregate(w,owner,n):
 out=np.zeros((n,2));np.add.at(out,owner,w);return out/np.bincount(owner,minlength=n)[:,None]
def person_values(v,ids):return np.array([v[ids==p].mean() for p in np.unique(ids)])
def describe(y,p,ids,boot):
 e=person_values(abs(p-y),ids);bias=person_values(p-y,ids);truth=person_values(y,ids)
 bands={}
 for name,mask in [('below60',truth<60),('60to75',(truth>=60)&(truth<75)),('75to90',(truth>=75)&(truth<90)),('90plus',truth>=90)]:
  bands[name]=dict(people=int(mask.sum()),mae=float(e[mask].mean()),bias=float(bias[mask].mean()),people_over10=int((e[mask]>10).sum()))
 return dict(mae=float(e.mean()),ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),bias=float(bias.mean()),p90=float(np.quantile(e,.9)),maximum=float(e.max()),people_over10=int((e>10).sum()),people_over20=int((e>20).sum()),bands=bands),e
ap=argparse.ArgumentParser();ap.add_argument('--received-root',type=Path,required=True);a=ap.parse_args();root=a.received_root
assert (root/'runs/summary_complete.json').exists(),'All ten outer predictions must complete first'
manifest=read(H/'bundle/manifest.json');assert sha(root/'manifest.json')==sha(H/'bundle/manifest.json')
for n,digest in manifest.items():
 p=B/n if n.startswith('data/') else root/n
 assert sha(p)==digest,n
marker=read(root/'runs/summary_complete.json');assert marker['expected_fits']==50 and marker['outer_predictions']==10
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
allpred={};errors={};metrics={};audits=[];inputs={}
for variant in ['baseline','stable_scale']:
 with np.load(B/'data'/f'{variant}.npz') as z:d={k:z[k] for k in z.files}
 inputs[variant]=d;ids=d['person_id'];y=d['y'];n=len(y);owner=d['window_record'];boot=np.random.default_rng(20261103).integers(0,len(np.unique(ids)),(10000,len(np.unique(ids))))
 pred=np.full((n,2),np.nan);raw=np.full_like(pred,np.nan)
 for f in range(5):
  fold=root/'runs/main'/variant/f'fold{f}';done=read(fold/'complete.json');tr=d['fold']!=f;te=~tr
  assert done['identity']['train_people']==np.unique(ids[tr]).tolist();assert done['identity']['outer_evaluation_people']==np.unique(ids[te]).tolist()
  assert not set(ids[tr])&set(ids[te])
  curves=[];weights=[]
  for j in range(5):
   if j==f:continue
   folder=fold/f'inner{j}';identity=read(folder/'identity.json');train=(d['fold']!=f)&(d['fold']!=j);valid=d['fold']==j
   assert identity['train_people']==np.unique(ids[train]).tolist();assert identity['validation_people']==np.unique(ids[valid]).tolist();assert identity['fit_seed']==20261028+100*f+j
   complete=read(folder/'fit_complete.json');assert sha(folder/'latest.pt')==complete['checkpoint_sha256'];assert sha(folder/'curve.json')==complete['curve_sha256']
   h=read(folder/'curve.json');assert len(h)==100 and complete['epochs']==100;assert all(row['validation_people']==len(np.unique(ids[valid])) for row in h)
   curves.append([row['validation_person_mae'] for row in h]);weights.append(len(np.unique(ids[valid])))
  means=np.average(curves,axis=0,weights=weights);selected=int(np.argmin(means))+1
  selection=read(fold/'selection.json');assert selected==selection['selected_epoch']==done['selected_epoch'];assert np.allclose(means,selection['inner_mean_person_mae_by_epoch'],atol=1e-12,rtol=0)
  assert sha(fold/'selection.json')==done['selection_sha256'];assert sha(fold/'predictions.npz')==done['prediction_sha256'];assert sha(fold/'refit/latest.pt')==done['checkpoint_sha256']
  ck=torch.load(fold/'refit/latest.pt',map_location='cpu',weights_only=True);assert ck['identity']==done['identity'];assert ck['identity']['fit_seed']==20261028+100*f+50
  assert len(ck['history'])==selected
  lo=float(ck['lo']);span=float(ck['span']);train_y=y[tr].astype('float32');assert lo==float(train_y.min());assert span==float(train_y.max()-train_y.min())
  m=GdiCNN();m.load_state_dict(ck['model'],strict=True);m.eval()
  w=infer(m,d['x'],lo,span);replay=aggregate(w,owner,n)
  with np.load(fold/'predictions.npz') as z:
   assert np.array_equal(z['row_index'],np.flatnonzero(te));assert np.array_equal(z['y'],y[te]);assert np.array_equal(z['person_id'],ids[te]);assert np.array_equal(z['video_id'],d['video_id'][te])
   p=z['predictions'][:,0];r=z['predictions_raw'][:,0];coef=z['calibration'];assert np.allclose(p,r*coef[0]+coef[1],atol=1e-10,rtol=0)
   difference=float(abs(replay[te]-r).max());assert difference<.001,('saved raw replay',variant,f,difference)
   independent_coef=np.linalg.lstsq(np.column_stack([replay[tr].ravel(),np.ones(2*tr.sum())]),y[tr].ravel(),rcond=None)[0]
   calibrated=replay[te]*independent_coef[0]+independent_coef[1];cdiff=float(abs(calibrated-p).max());assert cdiff<.001,('training-only calibration replay',variant,f,cdiff)
   assert np.allclose(coef,ck['history'][-1]['calibration'],atol=1e-10,rtol=0)
   pred[te]=p;raw[te]=r
  audits.append(dict(variant=variant,fold=f,selected_epoch=selected,checkpoint_sha256=done['checkpoint_sha256'],raw_replay_max_error=difference,calibrated_replay_max_error=cdiff))
  print('Verified',variant,f,'epoch',selected,'max replay difference',cdiff,flush=True)
 assert np.isfinite(pred).all();allpred[variant]=pred;metrics[variant],errors[variant]=describe(y,pred,ids,boot)
 metrics[variant]['raw_mae']=float(person_values(abs(raw-y),ids).mean());metrics[variant]['fold_mae']=[float(person_values(abs(pred[d['fold']==f]-y[d['fold']==f]),ids[d['fold']==f]).mean()) for f in range(5)]
 summary=read(root/'runs/main'/variant/'summary.json');assert summary['fits']==25;assert abs(summary['metrics'][0]['mae']-metrics[variant]['mae'])<1e-10
for k in inputs['baseline']:
 if k!='x':assert np.array_equal(inputs['baseline'][k],inputs['stable_scale'][k]),k
delta=errors['stable_scale']-errors['baseline'];ci=np.quantile(delta[boot].mean(1),[.025,.975]);result=dict(status='completed_and_independently_verified',people=692,records=1061,metrics=metrics,delta_mae=float(delta.mean()),delta_ci95=ci.tolist(),people_improved=int((delta<0).sum()),people_worsened=int((delta>0).sum()),development_improvement_supported=bool(ci[1]<0),single_seed_only=True,independent_final_validation=False,protocol_sha256=sha(H/'protocol.json'),audits=audits)
np.savez_compressed(B/'paired-predictions.npz',**allpred,y=y,person_id=ids,video_id=d['video_id'],person_error_baseline=errors['baseline'],person_error_stable_scale=errors['stable_scale'])
save(H/'results.json',result);save(H/'verification.json',dict(passed=True,results_sha256=sha(H/'results.json'),original_models_replayed=10,inner_fits_checked=40,all50_fit_hashes_checked=True,independent_train_calibration=True,person_splits_verified=True,paired_metrics_verified=True,cpu32_vs_gpu32_absolute_tolerance_gdi=.001,code_sha256=sha(__file__)))
print(json.dumps({k:result[k] for k in ['status','delta_mae','delta_ci95','people_improved','people_worsened']}),flush=True)
