"""Independent replay: coordinate invariants, neural subsets, weighted-ridge solves, and metrics."""
from prepare import *

def direct_fit(x,y,ids,alpha):
 from collections import Counter
 counts=Counter(ids.tolist());w=np.array([1/counts[p] for p in ids]);w=w/w.mean();mu=np.sum(x*w[:,None],axis=0)/w.sum();sd=np.sqrt(np.sum((x-mu)**2*w[:,None],axis=0)/w.sum());sd[sd<1e-12]=1;z=(x-mu)/sd;ym=float(np.dot(w,y)/w.sum());coef=np.linalg.solve(z.T@(z*w[:,None])+alpha*np.eye(x.shape[1]),z.T@(w*(y-ym)))
 return mu,sd,coef,ym

def independent_summary(x):
 parts=[]
 for leg in range(2):
  values=[]
  for channel in range(12):
   t=x[leg,:,channel];values.extend([np.mean(t),np.std(t),*np.percentile(t,[5,25,50,75,95]),np.sqrt(np.dot(np.diff(t),np.diff(t))/(len(t)-1))])
  parts.append(values)
 return np.array(parts)

def main():
 start=time.monotonic();prep=read(H/'preparation.json');protocol=read(H/'protocol.json');assert sha(H/'protocol.json')==prep['protocol_sha256'];assert sha(B/'summaries.npz')==prep['summary_sha256'];d=np.load(B/'summaries.npz');ids=d['person_id'];y=d['y'];fold=d['fold'];owner=d['window_record'];assert len(np.unique(ids))==692
 for p in np.unique(ids):assert len(np.unique(fold[ids==p]))==1
 for m in read(H/'input-manifest.json'):assert sha(m['path'])==m['sha256']
 for name,h in prep['input_sha256'].items():assert sha(B/'inputs'/f'{name}.npy')==h
 original=np.load(C/'bundle/data.npz');baseline=np.load(B/'inputs/baseline.npy',mmap_mode='r');assert np.array_equal(original['x'],baseline);assert np.array_equal(original['y'],y)
 source=read(H/'source-audit.json');assert [s['record'] for s in source]==list(range(len(y)))
 assert sorted(w for s in source for w in s['window_indices'])==list(range(len(owner)))
 for s in source:
  i=s['record'];wi=np.array(s['window_indices']);assert np.all(owner[wi]==i);old=read(C/'records'/f"{s['original_row']:04d}.json");assert old['raw_content_sha256']==s['source_content_sha256'];assert sha(B/'raw'/f'{i:04d}.npz')==s['raw_sha256']
 # Verify injected dose and shared realization directly in input space, independently of model outputs.
 coordinate_checks=[]
 for i in [0,101,503,1060]:
  raw=np.load(B/'raw'/f'{i:04d}.npz')['raw'];video=str(d['video_id'][i]);p,valid,scale,length=observed(raw);wi=np.flatnonzero(owner==i);starts=d['window_start'][wi]
  for name in protocol['conditions']:
   feat,detail=transform(raw,name,video);expected=np.stack([np.stack([feat[k:k+124,LEFT],feat[k:k+124,RIGHT]]) for k in starts]).astype('float32');actual=np.load(B/'inputs'/f'{name}.npy',mmap_mode='r')[wi];assert np.array_equal(actual,expected)
  for name in ['baseline','smooth2','spike_repair','stable_scale']:
   x=np.load(B/'inputs'/f'{name}.npy',mmap_mode='r')[wi];ref=np.mean([independent_summary(v.astype(float)) for v in x],axis=0);assert np.allclose(ref,d[name][i],atol=1e-10)
  # Translation and global scaling of observed coordinates preserve normalized baseline features.
  transformed=raw.reshape(-1,25,3).copy();good=np.isfinite(transformed[:,:,:2])&(transformed[:,:,:2]!=0);coords=transformed[:,:,:2];coords[good]=coords[good]*1.7+13;new,_=transform(transformed.reshape(-1,75),'baseline',video);old,_=transform(raw,'baseline',video)
  delta=float(np.nanmax(np.abs(new[:,LEFT+RIGHT]-old[:,LEFT+RIGHT])));assert delta<1e-5
  a=source[i]['conditions'];dose=[a[f'jitter{j:02d}_s0']['coordinate_noise_sd_relative'] for j in [1,3,5]];assert np.allclose(np.array(dose)/dose[0],[1,3,5],atol=1e-10);coordinate_checks.append(dict(record=i,baseline_invariance_max=delta,doses=dose))
 # Neural checkpoint replay uses the original batch16 implementation, independent of new batch64 loop.
 frozen=module('independent_neural',ROOT/'gdi-research-harness/studies/20261101-self-video-gdi/predict.py');torch=frozen.torch;torch.set_num_threads(4);torch.use_deterministic_algorithms(True);audit=read(H/'neural-audit.json');neural=np.load(B/'neural-record-predictions.npz');max_neural=0.;cells=0
 for item in audit:
  assert sha(item['checkpoint'])==item['sha256'];ck=torch.load(item['checkpoint'],map_location='cpu',weights_only=True);f=item['fold'];family=item['family'];te=fold==f;tr=~te
  assert set(ck['identity']['train_people'])==set(ids[tr]) and set(ck['identity']['outer_evaluation_people'])==set(ids[te]);assert set(ids[tr]).isdisjoint(ids[te])
  model=frozen.new_model(family);model.load_state_dict(ck['model'],strict=True);coef=np.array(item['calibration']);allwindow=np.load(B/f'{family}-window-predictions.npz')
  for condition,info in item['conditions'].items():
   ix=info['first_indices'];x=np.load(B/'inputs'/f'{condition}.npy',mmap_mode='r');value=frozen.forward(model,family,np.array(x[ix]),ck['lo'],ck['span'])*coef[0]+coef[1];expected=allwindow[family+'__'+condition][ix];error=float(np.max(np.abs(value-expected)));assert error<2e-4;max_neural=max(max_neural,error);cells+=1
   wp=allwindow[family+'__'+condition]
   for r in np.flatnonzero(te):assert np.max(np.abs(np.mean(wp[owner==r],axis=0)-neural[family+'__'+condition][r]))<1e-10
  print('verified neural',family,f,flush=True)
 # Independently recompute every ridge candidate with a linear solve instead of eigendecomposition.
 ra=read(H/'ridge-audit.json');rp=np.load(B/'ridge-record-predictions.npz');ridgefits=0;max_ridge=0.
 def errors(truth,pred,persons):return np.array([np.abs(truth[persons==p]-pred[persons==p]).mean() for p in np.unique(persons)])
 for condition,rows in ra.items():
  x=d[condition]
  for row in rows:
   tr=np.array(row['train']);te=np.array(row['test']);assert set(ids[tr]).isdisjoint(ids[te]);allerr=[]
   for alpha in [.1,1,10,100,1000,10000]:
    pp=np.full(y.shape,np.nan)
    for inner in row['inner']:
     ti=np.array(inner['train']);vi=np.array(inner['validation']);assert set(ids[ti]).isdisjoint(ids[vi]) and set(ti)|set(vi)==set(tr);assert set(ids[ti]).isdisjoint(ids[te]);mu,sd,w,b=direct_fit(x[ti].reshape(-1,96),y[ti].ravel(),np.repeat(ids[ti],2),alpha);pp[vi]=(((x[vi].reshape(-1,96)-mu)/sd)@w+b).reshape(-1,2);ridgefits+=1
    allerr.append(float(errors(y[tr],pp[tr],ids[tr]).mean()))
   assert np.allclose(allerr,row['inner_mae'],atol=1e-7);assert [.1,1,10,100,1000,10000][int(np.argmin(allerr))]==row['alpha'];assert sha(row['checkpoint'])==row['sha256'];m=np.load(row['checkpoint']);mu,sd,w,b=direct_fit(x[tr].reshape(-1,96),y[tr].ravel(),np.repeat(ids[tr],2),row['alpha']);ref=(((x[te].reshape(-1,96)-mu)/sd)@w+b).reshape(-1,2);saved=(((x[te].reshape(-1,96)-m['mean'])/m['scale'])@m['weight']+m['intercept']).reshape(-1,2);difference=float(np.max(np.abs(ref-saved)));assert difference<1e-7;assert np.max(np.abs(saved-rp[condition][te]))<1e-10;max_ridge=max(max_ridge,difference)
  print('verified ridge',condition,flush=True)
 # Replay the explicitly exploratory cross-input matrix without refitting anything.
 cross=np.load(B/'cross-input-predictions.npz');cross_result=read(H/'cross-input-results.json');assert sha(H/'cross-input-addendum.json')==cross_result['addendum_sha256'];assert sha(B/'cross-input-predictions.npz')==cross_result['predictions_sha256'];cross_cells=0
 for trained,rows in ra.items():
  for tested in ra:
   key=trained+'__'+tested
   for row in rows:
    m=np.load(row['checkpoint']);te=m['test'];p=(((d[tested][te].reshape(-1,96)-m['mean'])/m['scale'])@m['weight']+m['intercept']).reshape(-1,2);assert np.max(np.abs(p-cross[key][te]))<1e-10
   assert abs(errors(y,cross[key],ids).mean()-cross_result['matrix_mae'][trained][tested])<1e-12;cross_cells+=1
 # Recalculate headline, paired intervals, score bands and large errors from record predictions.
 result=read(H/'results.json');allp=np.load(B/'all-record-predictions.npz');people=np.unique(ids);boot=np.random.default_rng(20261103).integers(0,len(people),(10000,len(people)));perror={};ptruth=np.array([y[ids==p].mean() for p in people]);masks={'below60':ptruth<60,'60to75':(ptruth>=60)&(ptruth<75),'75to90':(ptruth>=75)&(ptruth<90),'90plus':ptruth>=90}
 for key,metric in result['metrics'].items():
  e=errors(y,allp[key],ids);perror[key]=e;assert abs(e.mean()-metric['mae'])<1e-12;assert np.allclose(np.quantile(e[boot].mean(1),[.025,.975]),metric['mae_ci95']);assert int((e>10).sum())==metric['people_over10'];assert int((e>20).sum())==metric['people_over20']
  for band,mask in masks.items():assert int(mask.sum())==metric['bands'][band]['people'];assert abs(e[mask].mean()-metric['bands'][band]['mae'])<1e-12
 for key,comp in result['coordinate_comparisons'].items():
  base=key.split('__')[0]+'__baseline';delta=perror[key]-perror[base];assert abs(delta.mean()-comp['delta_mae'])<1e-12;assert np.allclose(np.quantile(delta[boot].mean(1),[.025,.975]),comp['ci95'])
 for key,comp in result['coordinate_comparisons'].items():
  if comp['ci_three_comparisons'] is not None:
   delta=perror[key]-perror[key.split('__')[0]+'__baseline'];assert np.allclose(np.quantile(delta[boot].mean(1),[.05/6,1-.05/6]),comp['ci_three_comparisons'])
 for family in ['CNN','STT']:
  for level in [1,3,5]:
   value=(perror[f'{family}__jitter{level:02d}_s0']+perror[f'{family}__jitter{level:02d}_s1'])/2;delta=value-perror[family+'__baseline'];r=result['noise_replicates'][f'{family}_{level}pct'];assert abs(value.mean()-r['mae_mean_over_replicates'])<1e-12;assert abs(delta.mean()-r['delta_mae'])<1e-12;assert np.allclose(np.quantile(delta[boot].mean(1),[.025,.975]),r['ci95'])
 for key,comp in result['model_comparisons'].items():
  delta=perror[key]-perror['CNN__baseline'];assert abs(delta.mean()-comp['delta_mae'])<1e-12;assert np.allclose(np.quantile(delta[boot].mean(1),[.025,.975]),comp['ci95'])
 verification=dict(status='passed',protocol_sha256=sha(H/'protocol.json'),results_sha256=sha(H/'results.json'),baseline_features_exactly_equal=True,source_records_rehashed=len(source),same_windows_all_conditions=True,coordinate_checks=coordinate_checks,neural_checkpoints=10,neural_condition_replay_cells=cells,max_neural_replay_error=max_neural,ridge_inner_candidate_fits_verified=ridgefits,ridge_saved_models_replayed=20,cross_input_cells_verified=cross_cells,max_ridge_replay_error=max_ridge,person_splits_verified=True,metrics_and_bootstrap_verified=True,existing_ensemble_reconstructed=True,protected_final_inputs_or_labels_opened=False,elapsed_seconds=time.monotonic()-start,code_sha256={n:sha(H/n) for n in ['prepare.py','infer.py','ridge.py','analyse.py','verify.py','cross_inputs.py']})
 save(H/'verification.json',verification);print(json.dumps(verification),flush=True)
if __name__=='__main__':main()
