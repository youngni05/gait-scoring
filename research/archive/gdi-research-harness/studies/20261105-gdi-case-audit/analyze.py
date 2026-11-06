"""Post-hoc fixed-case summaries; no parameter fitting or model selection."""
from prepare import *
import numpy as np,itertools,csv,math
F=Path('D:/gait-research-data/D_P/final223-frozen-20261102')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def people_values(v,ids,mask=None):
 v=np.asarray(v);mask=np.ones(len(ids),bool) if mask is None else np.asarray(mask,bool)
 names=np.unique(ids[mask]);return names,np.array([np.mean(v[(ids==p)&mask],axis=0) for p in names])
def metric(pred,y,ids,mask=None):
 mask=np.ones(len(ids),bool) if mask is None else mask
 pp,e=people_values(pred-y,ids,mask);_,a=people_values(abs(pred-y),ids,mask);_,q=people_values((pred-y)**2,ids,mask)
 return dict(people=len(pp),records=int(mask.sum()),sides=int(mask.sum()*2),mae=float(a.mean()) if len(a) else None,bias=float(e.mean()) if len(e) else None,rmse=float(np.sqrt(q.mean())) if len(q) else None)
def obsmetric(pred,y,ids,mask):
 valid=mask.any(1);names=np.unique(ids[valid]);a=[];e=[]
 for p in names:
  m=(ids==p)[:,None]&mask;a.append(abs(pred-y)[m].mean());e.append((pred-y)[m].mean())
 return dict(people=len(names),records=int(valid.sum()),sides=int(mask.sum()),mae=float(np.mean(a)) if a else None,bias=float(np.mean(e)) if e else None)
def interval(v):
 v=np.asarray(v,float);assert v.ndim==1 and len(v)>0 and np.isfinite(v).all();rng=np.random.default_rng(20261105);boot=[]
 for _ in range(100):boot.extend(v[rng.integers(0,len(v),(100,len(v)))].mean(1))
 return dict(mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),people=len(v),replicates=10000,descriptive=True)
def paired(a,b,y,ids):
 _,v=people_values((abs(a-y)-abs(b-y)).mean(1),ids);return interval(v)
def summarize_scalar(v,ids,mask):
 _,pv=people_values(v,ids,mask);return dict(people=len(pv),records=int(mask.sum()),mean=float(pv.mean()) if len(pv) else None,p90=float(np.quantile(pv,.9)) if len(pv) else None)
def correlation(a,b,w=None):
 a=np.ravel(a);b=np.ravel(b);w=np.ones_like(a)/len(a) if w is None else np.ravel(w)/np.sum(w)
 aa=a-np.sum(w*a);bb=b-np.sum(w*b);den=np.sqrt(np.sum(w*aa**2)*np.sum(w*bb**2));return float(np.sum(w*aa*bb)/den) if den>1e-20 else None
def writecsv(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def main():
 seal=read(H/'freeze.json')
 for n,d in seal['code'].items():assert sha(H/n)==d,n
 locked=read(H/'predictions-locked.json');assert locked['passed'] and sha(BIG/'predictions.npz')==locked['predictions_sha256']
 d=arrays(BIG/'predictions.npz');truth=arrays(F/'scored.npz');ids=d['person_id'];y=truth['y'];owner=d['window_record'];start=d['window_start'];N=len(ids);names=[k[6:] for k in d if k.startswith('full__')];pred={n:d['full__'+n] for n in names};win={n:d['window__'+n] for n in names}
 for k in ['person_id','video_id']:np.testing.assert_array_equal(d[k],truth[k])
 allpeople=np.unique(ids);pm={p:float(y[ids==p].mean()) for p in allpeople};meantruth=np.array([pm[p] for p in ids]);groups={'all':np.ones(N,bool),'below50':meantruth<50,'below60':meantruth<60,'90plus':meantruth>=90,'100plus':meantruth>=100};obsgroups={'below60':y<60,'90plus':y>=90,'100plus':y>=100}
 full={n:{g:metric(v,y,ids,m) for g,m in groups.items()} for n,v in pred.items()};observed={n:{g:obsmetric(v,y,ids,m) for g,m in obsgroups.items()} for n,v in pred.items()};comparisons={n:{base:paired(v,pred[base],y,ids) for base in ['fusion','plus_ranksim']} for n,v in pred.items()}
 index=[np.flatnonzero(owner==i) for i in range(N)];counts=np.array([len(ix) for ix in index]);native={};prefix={};prefixrows=[];detail={};position={};repeat={};asym={}
 for n in names:
  w=win[n];sd=np.array([np.std(w[ix],axis=0,ddof=1).mean() if len(ix)>1 else np.nan for ix in index]);ranges=np.array([np.ptp(w[ix],axis=0).mean() for ix in index]);first=np.array([w[ix[0]] for ix in index]);mid=np.array([w[ix[(len(ix)-1)//2]] for ix in index]);last=np.array([w[ix[-1]] for ix in index]);native[n]={'sample_sd_min2':summarize_scalar(sd,ids,counts>=2),'sample_sd_min3':summarize_scalar(sd,ids,counts>=3),'range_min3':summarize_scalar(ranges,ids,counts>=3),'last_first_MAD_min3':summarize_scalar(abs(last-first).mean(1),ids,counts>=3)};position[n]={k:metric(v,y,ids) for k,v in [('earliest',first),('lower_middle',mid),('latest',last)]};detail['sd__'+n]=sd;detail['range__'+n]=ranges
  if n!='single_CNN':
   prefix[n]={}
   for b in [124,186,248,372,496]:
    ixlist=[ix[start[ix]+124<=start[ix[0]]+b] for ix in index];p=np.array([w[ix].mean(0) for ix in ixlist]);nc=np.array([len(ix) for ix in ixlist]);assert (nc>0).all();z={g:metric(p,y,ids,m) for g,m in groups.items()};delta=paired(p,pred[n],y,ids);_,mad=people_values(abs(p-pred[n]).mean(1),ids);prefix[n][str(b)]=dict(metrics=z,mae_difference_from_full=delta,prediction_MAD_from_full=float(mad.mean()),minimum_windows=int(nc.min()),median_windows=float(np.median(nc)),maximum_windows=int(nc.max()));prefixrows.append(dict(model=n,budget_frames=b,mae=z['all']['mae'],bias=z['all']['bias'],high90_mae=z['90plus']['mae'],high90_bias=z['90plus']['bias'],delta_MAE=delta['mean'],delta_low=delta['ci95'][0],delta_high=delta['ci95'][1],score_MAD=float(mad.mean())));detail['prefix_%d__%s'%(b,n)]=p
    if b==496:np.testing.assert_allclose(p,pred[n],rtol=0,atol=1e-10)
  asym[n]={}
  for t in [5,10]:
   m=abs(y[:,0]-y[:,1])>=t;err=abs((pred[n][:,0]-pred[n][:,1])-(y[:,0]-y[:,1]));asym[n][str(t)]={**summarize_scalar(err,ids,m),'threshold':t,'technical_gap_error_not_clinical':True}
  perperson=[];direction=[];nside=0;npair=0;directionpeople=0
  for p in allpeople:
   ix=np.flatnonzero(ids==p)
   if len(ix)<2:continue
   errors=[];signs=[]
   for a,b in itertools.combinations(ix,2):
    dt=y[b]-y[a];dp=pred[n][b]-pred[n][a];errors.append(abs(dp-dt).mean());mask=abs(dt)>=5;signs.extend(((np.sign(dp[mask])==np.sign(dt[mask]))&(abs(dp[mask])>1e-8)).tolist());nside+=int(mask.sum());npair+=1
   perperson.append(np.mean(errors))
   if signs:direction.append(np.mean(signs));directionpeople+=1
  repeat[n]=dict(people=len(perperson),unordered_record_pairs=npair,delta_error_MAE=interval(perperson),direction_people=directionpeople,direction_side_comparisons=nside,direction_accuracy=float(np.mean(direction)) if direction else None,no_temporal_or_recapture_claim=True)
 agreement=[];reaction=[];valid=counts>=3;rcount={p:sum(valid&(ids==p)) for p in allpeople};weights=np.zeros((len(owner),2));centered={}
 for i,ix in enumerate(index):
  if valid[i]:weights[ix]=1/(sum(v>0 for v in rcount.values())*rcount[ids[i]]*len(ix)*2)
 np.testing.assert_allclose(weights.sum(),1,rtol=0,atol=1e-12)
 for n in names:
  c=win[n].copy()
  for ix in index:c[ix]-=c[ix].mean(0)
  centered[n]=c
 for a,b in itertools.combinations(names,2):
  _,mad=people_values(abs(pred[a]-pred[b]).mean(1),ids);agreement.append(dict(model_a=a,model_b=b,person_equal_MAD=float(mad.mean()),signed_error_correlation=correlation(pred[a]-y,pred[b]-y)));reaction.append(dict(model_a=a,model_b=b,centered_window_response_correlation=correlation(centered[a],centered[b],weights),centered_window_MAD=float(np.sum(weights*abs(centered[a]-centered[b]))),people=int(sum(v>0 for v in rcount.values())),records=int(valid.sum())))
 result=dict(classification=read(H/'protocol.json')['classification'],protocol_sha256=sha(H/'protocol.json'),predictions_sha256=sha(BIG/'predictions.npz'),people=len(allpeople),records=N,windows=len(owner),models=names,groups={g:dict(people=len(set(ids[m])),records=int(m.sum())) for g,m in groups.items()},full=full,observation_groups=observed,overall_MAE_contrasts=comparisons,native_stability=native,fixed_positions=position,shorter_input=prefix,asymmetry=asym,repeat_records=repeat,model_agreement=agreement,within_record_response=reaction,source_preprocessing_caveat=read(H/'protocol.json')['case_tests']['shorter_input']['limitation'],new_training=False)
 np.savez_compressed(BIG/'case-details.npz',person_id=ids,video_id=d['video_id'],y=y,window_count=counts,**detail);save(H/'results.json',result)
 writecsv(H/'model-comparison.csv',[dict(model=n,overall_MAE=full[n]['all']['mae'],bias=full[n]['all']['bias'],high90_MAE=full[n]['90plus']['mae'],high90_bias=full[n]['90plus']['bias'],low60_MAE=full[n]['below60']['mae'],window_SD=native[n]['sample_sd_min3']['mean'],window_range=native[n]['range_min3']['mean'],repeat_delta_MAE=repeat[n]['delta_error_MAE']['mean']) for n in names]);writecsv(H/'shorter-input.csv',prefixrows);writecsv(H/'model-agreement.csv',agreement);writecsv(H/'within-record-response.csv',reaction)
 save(H/'analysis-receipt.json',dict(passed=True,results_sha256=sha(H/'results.json'),case_details_sha256=sha(BIG/'case-details.npz'),predictions_sha256=sha(BIG/'predictions.npz'),optimizer_steps=0,new_training=False))
 print(json.dumps(dict(analysis_ready=True,people=len(allpeople),records=N,models=len(names),groups=result['groups'])),flush=True)
if __name__=='__main__':main()
