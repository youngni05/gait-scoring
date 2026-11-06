"""Matched person-held-out SPPB regressions; all parameter choice stays inside train."""
from prepare import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr,rankdata
FEATURES=['speed','PCI','stride_time_CV','step_length_CV','swing_percent_CV']
ALPHAS=[.1,1.,10.,100.,1000.]
FITS=[]
def metrics(y,p):
 e=p-y;low=y<=8;pred=p<=8
 out=dict(n=len(y),MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),R2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),spearman=float(spearmanr(y,p).statistic) if np.std(p)>0 else None,within1=float((np.abs(e)<=1).mean()),errors_gt2=int((np.abs(e)>2).sum()),TP=int((low&pred).sum()),FN=int((low&~pred).sum()),TN=int((~low&~pred).sum()),FP=int((~low&pred).sum()),recall=float(pred[low].mean()) if low.any() else None,specificity=float((~pred[~low]).mean()) if (~low).any() else None,bands={})
 for name,m in [('5-8',y<=8),('9-10',(y>=9)&(y<=10)),('11-12',y>=11)]:
  out['bands'][name]=dict(n=int(m.sum()),MAE=float(np.abs(e[m]).mean()) if m.any() else None,bias=float(e[m].mean()) if m.any() else None)
 return out
def train(X,y,train_idx,test_idx,alpha,meta,ids):
 scaler=StandardScaler().fit(X[train_idx]);a=scaler.transform(X[train_idx]);model=Ridge(alpha=alpha).fit(a,y[train_idx]);p=model.predict(scaler.transform(X[test_idx]))
 FITS.append(dict(**meta,alpha=alpha,train=[ids[i] for i in train_idx],evaluation=[ids[i] for i in test_idx],mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=model.coef_.tolist(),intercept=float(model.intercept_),prediction=p.tolist()))
 return p
def run(name,people,gdi=None,variant='primary'):
 ids=[p['id'] for p in people];y=np.array([p['sppb'] for p in people]);folds=np.array([p['fold'] for p in people]);age=np.array([p['age'] for p in people])
 x=np.array([[p['features'][f] for f in FEATURES] for p in people]);mat={'A':x[:,:1],'C':x}
 if gdi is not None:
  g=np.array([gdi[pid][variant] for pid in ids])[:,None];mat.update(B=g,D=np.concatenate([x,g],axis=1))
 preds={k:np.full(len(ids),np.nan) for k in [*mat,'Mean','Median']};selection=[]
 for fold in range(5):
  tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);assert len(te)>0
  shuffled=np.array(sorted(tr,key=lambda i:ids[i]));np.random.RandomState(20261031).shuffle(shuffled);chunks=np.array_split(shuffled,3)
  preds['Mean'][te]=y[tr].mean();preds['Median'][te]=np.median(y[tr])
  for condition,X in mat.items():
   losses=[]
   for alpha in ALPHAS:
    errors=[]
    for inner,va in enumerate(chunks):
     tt=np.array([i for i in tr if i not in set(va)]);p=train(X,y,tt,va,alpha,dict(cohort=name,variant=variant,condition=condition,outer=fold,inner=inner),ids);errors.extend(np.abs(p-y[va]).tolist())
    losses.append(float(np.mean(errors)))
   chosen=ALPHAS[int(np.argmin(losses))]
   preds[condition][te]=train(X,y,tr,te,chosen,dict(cohort=name,variant=variant,condition=condition,outer=fold,inner=None),ids)
   selection.append(dict(condition=condition,outer=fold,inner_MAE=losses,alpha=chosen))
 assert all(np.isfinite(p).all() for p in preds.values())
 rng=np.random.RandomState(20261103);boot=rng.randint(0,len(ids),(10000,len(ids)));contrasts={}
 for a,b in [('C','A'),('D','C'),('D','A'),('B','A')]:
  if a not in preds:continue
  delta=np.abs(preds[a]-y)-np.abs(preds[b]-y);ci=np.quantile(delta[boot].mean(1),[.025,.975]);contrasts[a+'-'+b]=dict(MAE_difference=float(delta.mean()),CI95=ci.tolist(),interpretation='conditional fixed-OOF person bootstrap; negative favours first')
 return dict(ids=ids,y=y.tolist(),age=age.tolist(),folds=folds.tolist(),inputs={k:v.tolist() for k,v in mat.items()},predictions={k:v.tolist() for k,v in preds.items()},metrics={k:metrics(y,p) for k,p in preds.items()},age65plus={k:metrics(y[age>=65],p[age>=65]) for k,p in preds.items()},selection=selection,contrasts=contrasts)
def main():
 audit=read(HERE/'input-audit.json');assert sha(HERE/'protocol.json')==audit['protocol_sha256'];people=read(HERE/'people.json');gdi={p['id']:p['scores'] for p in read(HERE/'gdi-scores.json')};results={}
 cohorts={'strict_events':[p for p in people if p['event_eligible']], 'strict_common':[p for p in people if p['event_eligible'] and p['id'] in gdi], 'exploratory_all44':people,'exploratory_common':[p for p in people if p['id'] in gdi]}
 for name,pp in cohorts.items():
  variants=VARIANTS if 'common' in name else ['primary']
  for variant in variants:
   key=name+'/'+variant;results[key]=run(name,pp,gdi if 'common' in name else None,variant);print(key,len(pp),{k:round(m['MAE'],4) for k,m in results[key]['metrics'].items()},flush=True)
 correlations={}
 for cohort in ['strict_events','exploratory_all44']:
  pp=cohorts[cohort];y=np.array([p['sppb'] for p in pp]);speed=rankdata([p['features']['speed'] for p in pp]);design=np.column_stack([np.ones(len(pp)),speed]);yr=rankdata(y);yr-=design@np.linalg.lstsq(design,yr,rcond=None)[0];correlations[cohort]={}
  for f in FEATURES:
   x=np.array([p['features'][f] for p in pp]);rho=spearmanr(x,y);xr=rankdata(x);xr-=design@np.linalg.lstsq(design,xr,rcond=None)[0]
   correlations[cohort][f]=dict(spearman=float(rho.statistic),p=float(rho.pvalue),speed_adjusted_rank_correlation=float(np.corrcoef(xr,yr)[0,1]) if f!='speed' else None)
 save(HERE/'comparison.json',results);save(HERE/'correlations.json',correlations);save(BIG/'fits.json',FITS)
 save(HERE/'fit-receipt.json',dict(protocol_sha256=sha(HERE/'protocol.json'),operational_note_sha256=sha(HERE/'operational-note.json'),people_sha256=sha(HERE/'people.json'),gdi_scores_sha256=sha(HERE/'gdi-scores.json'),comparison_sha256=sha(HERE/'comparison.json'),fits_sha256=sha(BIG/'fits.json'),fits=len(FITS)))
if __name__=='__main__':main()
