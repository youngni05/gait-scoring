"""Independent replay, closed-form fit checks and source feature audits."""
from prepare import S,B,read,save,sha
import numpy as np,pandas as pd,hashlib,zlib
from pathlib import Path

def main():
 results=read(S/'comparison.json');fits=read(B/'fits.json');people=read(S/'people.json');receipt=read(S/'fit-receipt.json');checks={};mx=0.;mxcoef=0.;outer=0
 assert receipt['protocol_sha256']==sha(S/'protocol.json');assert receipt['people_sha256']==sha(S/'people.json');assert receipt['fits_sha256']==sha(B/'fits.json');assert receipt['comparison_sha256']==sha(S/'comparison.json')
 for f in fits:
  key=f['run']+'/'+f['target']
  if f['run']=='provisional_gstride_to_gait2care':
   r=results['provisional_transfer/sppb'];ids=r['all_ids'];y=np.array(r['all_y']);X=np.array(r['all_inputs'][f['condition']])
  else:r=results[key];ids=r['ids'];y=np.array(r['y']);X=np.array(r['inputs'][f['condition']])
  index={p:i for i,p in enumerate(ids)};tr=np.array([index[p] for p in f['train']]);te=np.array([index[p] for p in f['evaluation']]);assert not set(tr)&set(te);assert len(set(ids))==len(ids)
  mu=X[tr].mean(0);sc=X[tr].std(0);sc[sc==0]=1
  assert np.allclose(mu,f['mean'],atol=1e-12) and np.allclose(sc,f['scale'],atol=1e-12)
  xt=(X[tr]-mu)/sc;yt=y[tr]-y[tr].mean();beta=np.linalg.solve(xt.T@xt+f['alpha']*np.eye(X.shape[1]),xt.T@yt);intercept=y[tr].mean()-xt.mean(0)@beta
  pred=((X[te]-mu)/sc)@np.array(f['coef'])+f['intercept'];ind=((X[te]-mu)/sc)@beta+intercept
  mx=max(mx,float(np.max(abs(pred-np.array(f['prediction'])))));mxcoef=max(mxcoef,float(np.max(abs(ind-pred))))
  assert np.allclose(ind,pred,atol=1e-9)
  if f['inner'] is None:
   outer+=1
   for pid,p in zip(f['evaluation'],pred):assert abs(r['predictions'][f['condition']][r['ids'].index(pid)]-p)<1e-10
  elif isinstance(f['outer'],int):
   assert all(r['folds'][index[p]]!=f['outer'] for p in f['train']+f['evaluation'])
 for key,r in results.items():
  y=np.array(r['y'])
  for c,p in r['predictions'].items():
   p=np.array(p);assert abs(np.mean(abs(y-p))-r['metrics'][c]['MAE'])<1e-12
   if r['target']=='sppb':
    m=r['metrics'][c];assert m['TP']+m['FN']==sum(y<=8);assert m['TN']+m['FP']==sum(y>8)
  for selection in r['selection']:
   alpha=[.1,1.,10.,100.,1000.][int(np.argmin(selection['inner_MAE']))];assert alpha==selection['alpha']
   losses=[]
   for a in [.1,1.,10.,100.,1000.]:
    ff=[f for f in fits if f['run']==('provisional_gstride_to_gait2care' if key.startswith('provisional') else key.split('/')[0]) and f['target']==r['target'] and f['condition']==selection['condition'] and f['alpha']==a and f['inner'] is not None and (f['outer']==selection.get('outer','all_source'))]
    yy=dict(zip(r.get('all_ids',r['ids']),r.get('all_y',r['y'])));errs=[abs(p-yy[pid]) for f in ff for pid,p in zip(f['evaluation'],f['prediction'])];assert len(ff)==3;losses.append(np.mean(errs))
   assert np.allclose(losses,selection['inner_MAE'],atol=1e-12)
  rng=np.random.RandomState(20261103);boot=rng.randint(0,len(y),(10000,len(y)))
  for k,d in r['contrasts'].items():
   a,b=k.split('-');diff=abs(np.array(r['predictions'][a])-y)-abs(np.array(r['predictions'][b])-y)
   assert np.allclose(np.quantile(diff[boot].mean(1),[.025,.975]),d['CI95'],atol=1e-12)
 # Recompute feature CV through sums of powers and group sizes, independently of extraction helper.
 max_feature_error=0.;ratio_error=0.
 for p in people:
  df=pd.read_csv(p['source_file']);df.columns=df.columns.str.strip();groups=[]
  for prefix in (['R_','L_'] if p['dataset']=='gait2care' else ['']):
   a=df[[prefix+k for k in ['CycleDuration[s]','StrideLength[m]','Per_Swing[%CD]','Stride_Velocity[m/s]','TurningAngle[degrees]']]].to_numpy(float)
   a=a[np.isfinite(a).all(1)&(a[:,0]>0)&(a[:,1]>0)&(a[:,3]>0)&(a[:,2]>0)&(a[:,2]<100)&(abs(a[:,4])<20)];groups.append(a)
   ratio_error=max(ratio_error,float(np.max(abs(a[:,1]/a[:,0]-a[:,3]))))
  allx=np.vstack(groups);assert abs(allx[:,3].mean()-p['features']['speed'])<1e-12
  for i,k in enumerate(['stride_time_CV','stride_length_CV','swing_percent_CV']):
   sse=sum(np.sum(a[:,i]**2)-np.sum(a[:,i])**2/len(a) for a in groups);cv=100*np.sqrt(sse/sum(len(a)-1 for a in groups))/np.mean(allx[:,i]);err=abs(cv-p['features'][k]);max_feature_error=max(max_feature_error,float(err));assert err<1e-8
 ca=read(B/'coordination-audit.json');pcie=0.
 for p in people:
  if p['features']['PCI'] is None:continue
  a=ca[p['id']];v=[]
  for side in ['R','L']:
   phase=np.array(a['phases'][side]);assert len(phase)>=15 and np.all((phase>0)&(phase<360));v.append(100*np.sqrt(np.sum((phase-phase.mean())**2)/(len(phase)-1))/phase.mean()+np.sum(abs(phase-180))/len(phase)/1.8)
  pcie=max(pcie,abs(np.mean(v)-p['features']['PCI']));assert pcie<1e-8
 checked=0
 for fname in ['gait2care-manifest.json','gstride-manifest.json','baseline-imu-manifest.json']:
  for m in read(S/fname):
   b=Path(m['file']).read_bytes();assert hashlib.sha256(b).hexdigest()==m['sha256']
   if 'crc32' in m:assert zlib.crc32(b)==m['crc32']
   checked+=1
 checks=dict(status='PASS_LOCAL_REPRODUCIBILITY',saved_fits=len(fits),final_models=outer,max_saved_prediction_error=mx,max_closed_form_prediction_error=mxcoef,max_CV_recalculation_error=max_feature_error,max_stride_speed_identity_error=ratio_error,max_PCI_recalculation_error=pcie,source_files_rehashed=checked,protocol_sha256=sha(S/'protocol.json'),scope='Source integrity, joins as recorded, person splits, inner selection, train-only scales, saved-fit replay, metric/bootstrap recomputation, stride feature algebra. Does not establish cross-cohort independence or IMU measurement accuracy.',cross_cohort_independence='UNRESOLVED',camera_GDI_applicability=False)
 assert ratio_error<1e-6
 save(S/'verification.json',checks);print(checks)
if __name__=='__main__':main()
