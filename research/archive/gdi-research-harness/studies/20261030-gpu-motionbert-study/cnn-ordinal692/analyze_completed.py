"""Recalculate completed ordinal OOF results; no fitting or GPU access."""
from pathlib import Path
import hashlib,json,tarfile,sys,os,subprocess
import numpy as np
ROOT=Path('D:/Projects/D_P');S=ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study'
H=S/'cnn-ordinal692';OUT=H/'analysis';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(H));os.environ['GDI_ORDINAL_DATA']=str(S/'cnn-record-loss692/bundle/data.npz')
import run as r
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
receipt=read(H/'checks/closeout-receipt.json');archive=Path(receipt['archive']);inventory=read(receipt['inventory'])
assert receipt['passed'] and receipt['all_archived_file_hashes_verified'] and sha(archive)==receipt['server_archive_sha256']
assert sha(receipt['inventory'])==receipt['inventory_sha256']
UNPACK=archive.parent/'performance-review-originals';UNPACK.mkdir(exist_ok=True)
with tarfile.open(archive) as tf:
 for member in tf.getmembers():
  p=(UNPACK/member.name).resolve();assert UNPACK.resolve() in p.parents and member.isfile() and not member.issym() and not member.islnk()
 tf.extractall(UNPACK,filter='data')
files={}
for item in inventory['files']:
 p=UNPACK/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 if item['path'].startswith('runs/main/'):files[item['path'][len('runs/main/'):]]=item['sha256']
assert sha(UNPACK/'manifest.json')==sha(H/'manifest.json')
audit_receipt=OUT/'performance-input-receipt.json'
save(audit_receipt,dict(passed=True,server_local_sha256_matched=True,manifest_sha256=sha(H/'manifest.json'),
 result_files=files,source_closeout_receipt_sha256=sha(H/'checks/closeout-receipt.json'),archive_sha256=sha(archive)))
results=UNPACK/'runs/main';summary=read(results/'summary.json');base_report=read(S/'inner-bagging692/analysis/comparison.json')
assert sha(base_report['predictions_path'])==base_report['predictions_sha256']
base=r.arrays(base_report['predictions_path']);y=base['y'];ids=base['person_id'];people,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
person_mean=np.array([y[ids==p].mean() for p in people]);weights=1/(len(people)*counts[inv])
def macro(v):return float(np.sum(weights*np.asarray(v).reshape(len(y),-1).mean(1)))
def perperson(v):return np.array([np.asarray(v)[ids==p].mean() for p in people])
def bands(v,w):return np.searchsorted(np.arange(0,161,w),v,side='right')
rng=np.random.default_rng(20261101);draws=rng.integers(0,len(people),(10000,len(people)))
def delta(v,baseline):
 d=perperson(v)-perperson(baseline);boot=d[draws].mean(1)
 return dict(delta=float(d.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),better_people=int((d<0).sum()),worse_people=int((d>0).sum()))
def points(pred):
 err=np.abs(pred-y);bias=pred-y;z=dict(mae=macro(err),bias=macro(bias),bands={})
 for w in (10,20):z['bands'][str(w)]=dict(accuracy=macro(bands(pred,w)==bands(y,w)),two_or_more=macro(abs(bands(pred,w)-bands(y,w))>=2))
 z['subgroups']={}
 for name,mask in [('person_mean90',person_mean>=90),('person_mean100',person_mean>=100),('person_mean60below',person_mean<60)]:
  z['subgroups'][name]=dict(people=int(mask.sum()),mae=float(perperson(err)[mask].mean()),bias=float(perperson(bias)[mask].mean()))
 z['paired_vs_best']=delta(err,abs(base['bagged']-y));z['fold_mae']=[float(np.mean(perperson(err)[np.array([base['fold'][ids==p][0]==f for p in people])])) for f in range(5)]
 return z
def probability(prob,w):
 cls=bands(y,w);pred=prob.argmax(-1);one=np.eye(prob.shape[-1])[cls];correct=pred==cls;conf=prob.max(-1)
 answer=dict(accuracy=macro(correct),nll=macro(-np.log(np.maximum(np.take_along_axis(prob,cls[...,None],-1)[...,0],1e-12))),
  brier=macro(((prob-one)**2).sum(-1)),rps=macro(((prob.cumsum(-1)-one.cumsum(-1))[...,:-1]**2).mean(-1)),
  mean_confidence=macro(conf),ece=0.,reliability=[])
 fullweights=np.broadcast_to(weights[:,None]/2,conf.shape)
 for k in range(10):
  mask=(conf>=k/10)&((conf<(k+1)/10) if k<9 else (conf<=1));mass=float(fullweights[mask].sum())
  if mass:
   c=float(np.sum(fullweights[mask]*conf[mask])/mass);a=float(np.sum(fullweights[mask]*correct[mask])/mass)
   answer['reliability'].append(dict(lower=k/10,upper=(k+1)/10,weight=mass,confidence=c,accuracy=a,observations=int(mask.sum())))
   answer['ece']+=mass*abs(a-c)
 answer['paired_error_vs_best']=delta(~correct,bands(base['bagged'],w)!=cls)
 answer['recall_by_true_band']=[]
 for k in np.unique(cls):
  mask=cls==k;mass=float(fullweights[mask].sum())
  answer['recall_by_true_band'].append(dict(index=int(k),lower=None if k==0 else int((k-1)*w),upper=None if k==160//w+1 else int(k*w),
    people=int(len(np.unique(np.repeat(ids,2)[mask.reshape(-1)]))),weight=mass,recall=float(np.sum(fullweights[mask]*correct[mask])/mass)))
 high=person_mean>=90
 answer['high90_person_accuracy']=float(perperson(correct)[high].mean())
 return answer
models={}
for name,key in [('existing_best','bagged'),('multiwindow_CNN','CNN'),('multiwindow_STT','STT')]:models[name]=dict(point=points(base[key]))
assert abs(models['existing_best']['point']['mae']-6.515816569178735)<1e-10
assert abs(models['existing_best']['point']['subgroups']['person_mean90']['mae']-8.782527747879264)<1e-10
for group in ['regression','ordinal10','ordinal20','joint10','joint20']:
 z=r.arrays(results/group/'oof.npz');assert sha(results/group/'oof.npz')==summary['groups'][group]['oof_sha256']
 for key in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(z[key],base[key])
 m={}
 if z['raw'].size:m['point']=points(z['raw'])
 if z['prob'].size:m['probability']=probability(z['prob'],int(group[-2:]))
 models[group]=m
report=dict(people=len(people),records=len(y),person_equal=True,models=models,raw_probabilities_uncalibrated=True,
 boundaries='0..160 with open tails; [a,b)',bootstrap_replicates=10000,bootstrap_seed=20261101,
 protected_final_opened=False,optimizer_steps=0,selection_or_probability_calibration_on_outer=False,
 source_receipt_sha256=sha(audit_receipt),full_model_audit_pending=True)
save(OUT/'comparison.json',report)
print(json.dumps({'phase':'statistics_recalculated','metrics':models},ensure_ascii=False),flush=True)
with (OUT/'full-audit.log').open('w',encoding='utf-8') as log:
 p=subprocess.run([sys.executable,'-u','-B',str(H/'review_results.py'),'--results',str(results),'--receipt',str(audit_receipt),
  '--receipt-sha256',sha(audit_receipt),'--output',str(OUT/'full-audit')],stdout=log,stderr=subprocess.STDOUT,
  env=os.environ.copy())
report['full_model_audit_pending']=False;report['full_model_audit_passed']=p.returncode==0
if p.returncode==0:
 proof=read(OUT/'full-audit/verification.json');assert proof['passed'] and proof['fits']==165 and proof['outer_replays']==25
 report['full_model_audit_sha256']=sha(OUT/'full-audit/verification.json')
report['audit_log_sha256']=sha(OUT/'full-audit.log');save(OUT/'comparison.json',report)
print(json.dumps({'phase':'finished','full_model_audit_passed':p.returncode==0,'report':str(OUT/'comparison.json')}),flush=True)
if p.returncode:raise SystemExit(p.returncode)
