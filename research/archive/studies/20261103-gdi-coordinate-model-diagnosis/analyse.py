"""Person-weighted comparisons; no post-result model selection."""
from prepare import *
from scipy.stats import rankdata

def person_average(v,ids):return np.array([v[ids==p].mean(axis=0) for p in np.unique(ids)])
def err(y,p,ids):return person_average(np.abs(y-p).mean(1),ids)
def rho(a,b):
 a=rankdata(a);b=rankdata(b)
 return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None

def main():
 d=np.load(B/'summaries.npz');y=d['y'];ids=d['person_id'];people=np.unique(ids);n=len(people);fold=np.array([d['fold'][np.flatnonzero(ids==p)[0]] for p in people]);truth=person_average(y.mean(1),ids)
 neural=np.load(B/'neural-record-predictions.npz');ridge=np.load(B/'ridge-record-predictions.npz');pred={k:neural[k] for k in neural.files};pred.update({'ridge__'+k:ridge[k] for k in ridge.files})
 oldreport=read(S/'inner-bagging692/analysis/comparison.json');oldpath=Path(oldreport['predictions_path']);assert sha(oldpath)==oldreport['predictions_sha256'];old=np.load(oldpath)
 for key in ['y','person_id','video_id','fold']:assert np.array_equal(old[key],d[key])
 for key in ['CNN','STT']:assert np.max(np.abs(old[key]-pred[key+'__baseline']))<2e-4
 for key in ['CNN_bag','STT_bag','inner_selected_difference','bagged']:pred['existing__'+key]=old[key]
 # Reconstruct all cached bagged outputs to verify unchanged model composition.
 manifest=read(ROOT/'gdi-research-harness/studies/20261102-final223-frozen/models.json');md=np.load(S/'mean-difference692/analysis/predictions.npz');rebuilt=np.full(y.shape,np.nan)
 for f in range(5):
  te=d['fold']==f;parts={}
  for family in ['CNN','STT']:
   stack=[]
   for m in manifest['members']:
    if m['family']==family and m['fold']==f:
     assert sha(m['cache'])==m['cache_sha256']
     if m['tag']=='refit':
      z=np.load(m['cache']);assert np.array_equal(z['video_id'],d['video_id']);stack.append(z['corrected'][0][te])
     else:
      path=CACHE/'inner-bagging692/cache'/family/f"fold{f}_{m['tag']}.npz";proof=read(path.with_suffix('.json'));assert not proof['excluded'];assert proof['checkpoint_sha256']==m['checkpoint_sha256'];assert sha(path)==proof['cache_sha256'];z=np.load(path);assert np.array_equal(z['video_id'],d['video_id'][te]);stack.append(z['prediction'])
   parts[family]=np.mean(stack,axis=0)
  sf=manifest['selections'][f];base=sf['cnn_weight']*parts['CNN']+(1-sf['cnn_weight'])*parts['STT'];avg=base.mean(1);diff=(1-sf['fraction'])*(base[:,0]-base[:,1])+sf['fraction']*(md['bilateral_STT'][te,0]-md['bilateral_STT'][te,1]);rebuilt[te]=np.column_stack([avg+diff/2,avg-diff/2])
 ensemble_replay=float(np.max(np.abs(rebuilt-old['bagged'])));assert ensemble_replay<1e-4
 rng=np.random.default_rng(20261103);boot=rng.integers(0,n,(10000,n));errors={k:err(y,p,ids) for k,p in pred.items()}
 bands={'below60':truth<60,'60to75':(truth>=60)&(truth<75),'75to90':(truth>=75)&(truth<90),'90plus':truth>=90}
 def metric(key):
  p=pred[key];e=errors[key];bias=person_average((p-y).mean(1),ids)
  out=dict(mae=float(e.mean()),mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),bias=float(bias.mean()),p90_person_error=float(np.quantile(e,.9)),max_person_error=float(e.max()),people_over10=int((e>10).sum()),people_over20=int((e>20).sum()),max_record_side_error=float(np.abs(p-y).max()),fold_mae=[float(e[fold==f].mean()) for f in range(5)],bands={})
  for name,mask in bands.items():
   out['bands'][name]=dict(people=int(mask.sum()),mae=float(e[mask].mean()),bias=float(bias[mask].mean()),p90=float(np.quantile(e[mask],.9)),max=float(e[mask].max()),over10=int((e[mask]>10).sum()),over20=int((e[mask]>20).sum())) if mask.any() else dict(people=0)
  return out
 def paired(key,base,multiple=False):
  delta=errors[key]-errors[base];sample=delta[boot].mean(1);shift=person_average(np.abs(pred[key]-pred[base]).mean(1),ids)
  return dict(delta_mae=float(delta.mean()),ci95=np.quantile(sample,[.025,.975]).tolist(),ci_three_comparisons=np.quantile(sample,[.05/6,1-.05/6]).tolist() if multiple else None,improved_people=int((delta<0).sum()),worsened_people=int((delta>0).sum()),mean_prediction_shift=float(shift.mean()),p95_prediction_shift=float(np.quantile(shift,.95)),fold_deltas=[float(delta[fold==f].mean()) for f in range(5)])
 comparisons={}
 for key in pred:
  family,condition=key.split('__');base=family+'__baseline'
  if condition!='baseline' and base in pred:comparisons[key]=paired(key,base,condition in ['smooth2','spike_repair','stable_scale'])
 modelcompare={key:paired(key,'CNN__baseline') for key in ['STT__baseline','ridge__baseline','existing__CNN_bag','existing__STT_bag','existing__bagged']}
 noise={}
 for family in ['CNN','STT']:
  for level in [1,3,5]:
   e=(errors[f'{family}__jitter{level:02d}_s0']+errors[f'{family}__jitter{level:02d}_s1'])/2;delta=e-errors[family+'__baseline'];noise[f'{family}_{level}pct']=dict(mae_mean_over_replicates=float(e.mean()),delta_mae=float(delta.mean()),ci95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),replicate_deltas=[comparisons[f'{family}__jitter{level:02d}_s{s}']['delta_mae'] for s in [0,1]])
 quality=person_average(d['quality'],ids);names=read(H/'preparation.json')['quality_names'];windowcounts=np.bincount(d['window_record'],minlength=len(ids));pw=person_average(windowcounts,ids);assoc={}
 for j,name in enumerate(names):
  q=quality[:,j];v=errors['existing__bagged'];design=np.column_stack([np.ones(n),rankdata(truth),rankdata(pw)]);qr=rankdata(q)-design@np.linalg.lstsq(design,rankdata(q),rcond=None)[0];vr=rankdata(v)-design@np.linalg.lstsq(design,rankdata(v),rcond=None)[0]
  partial=float(np.corrcoef(qr,vr)[0,1]) if qr.std()>1e-10 and vr.std()>1e-10 else None
  assoc[name]=dict(spearman_with_GDI_error=rho(q,v),partial_rank_controlling_true_GDI_and_window_count=partial,median=float(np.median(q)),p90=float(np.quantile(q,.9)),nonzero_people=int((q>0).sum()))
 report=dict(protocol_sha256=sha(H/'protocol.json'),people=n,records=len(ids),windows=len(d['window_record']),metrics={k:metric(k) for k in pred},coordinate_comparisons=comparisons,model_comparisons=modelcompare,noise_replicates=noise,quality_associations=assoc,model_residual_correlations={k:float(np.corrcoef(person_average((pred[k]-y).mean(1),ids),person_average((pred['CNN__baseline']-y).mean(1),ids))[0,1]) for k in ['STT__baseline','ridge__baseline','existing__bagged']},existing_ensemble_reconstruction_max_error=ensemble_replay,source_existing_predictions_sha256=sha(oldpath),interpretation='Conditional exploratory diagnostics; no direct clean-coordinate truth and no causal error decomposition')
 save(H/'results.json',report);np.savez_compressed(B/'all-record-predictions.npz',**pred,**{k:d[k] for k in ['y','person_id','video_id','fold']});np.savez_compressed(B/'person-errors.npz',person_id=people,mean_true_GDI=truth,fold=fold,quality=quality,**errors)
 print('MODEL', {k:round(v['mae'],4) for k,v in report['metrics'].items() if k.endswith('__baseline') or k=='existing__bagged'},flush=True)
 print('REPAIRS',{k: {'delta':round(v['delta_mae'],4),'ci':v['ci95']} for k,v in comparisons.items() if any(s in k for s in ['smooth','spike','stable'])},flush=True)
 print('NOISE',noise,flush=True);print('QUALITY',assoc,flush=True)
if __name__=='__main__':main()
