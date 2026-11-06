"""Independent saved-model replay, train statistics and paired-person summaries."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np,joblib
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def metrics(y,p):
    error=p-y;high=y>=20
    return {'n':len(y),'mae':float(abs(error).mean()),'bias':float(error.mean()),'p95':float(np.quantile(abs(error),.95)),
      'high_n':int(high.sum()),'high_mae':float(abs(error[high]).mean()) if high.any() else None,'high_bias':float(error[high].mean()) if high.any() else None}
def paired(y,p,base):
    diff=abs(p-y)-abs(base-y);rng=np.random.default_rng(20261104);bs=diff[rng.integers(0,len(y),(5000,len(y)))].mean(1)
    return {'delta_mae':float(diff.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'n':len(y)}
def independent_predict(m,X):
    z=m['imputer'].transform(X);z=np.column_stack([z,1/np.maximum(z[:,0],.1)]);z=(z-m['scaler'].mean_)/m['scaler'].scale_
    return np.maximum(.1,m['model'].predict(z))
def main():
    proto=json.loads((HERE/'comparison-protocol.json').read_text());rec=json.loads((HERE/'fit-receipt.json').read_text())
    assert sha(HERE/'compare.py')==proto['code_sha256']==rec['code_sha256']
    for p,h in proto['feature_hashes'].items():assert sha(p)==h
    assert sha(BIG/'input.npz')==rec['input_sha256'];d=np.load(BIG/'input.npz');n=len(d['y']);assert len(set(d['ids']))==n
    pred={};errors=[];checked=0
    for r in rec['models']:
        assert sha(r['model'])==r['model_sha256'] and sha(r['context'])==r['context_sha256'];m=joblib.load(r['model']);c=np.load(r['context']);tr,te=c['train'],c['test']
        assert len(set(d['ids'][tr])&set(d['ids'][te]))==0 and np.all(d['fold'][tr]!=r['fold']) and np.all(d['fold'][te]==r['fold'])
        if r['mode']!='pooled':assert np.all(d['cohort'][tr]==('F' if r['mode']=='F_only' else 'G'))
        X=d[r['version']][:,:1] if r['view']=='speed' else d[r['version']]
        np.testing.assert_allclose(m['imputer'].statistics_,np.nanmedian(X[tr],axis=0),rtol=0,atol=1e-12)
        xx=np.column_stack([X[tr],1/np.maximum(X[tr,0],.1)]);np.testing.assert_allclose(m['scaler'].mean_,xx.mean(0),rtol=0,atol=1e-12)
        assert m['ntrain']==len(tr) and m['model'].n_estimators==128 and m['model'].min_samples_leaf==10
        pp=independent_predict(m,X[te]);err=float(np.max(abs(pp-c['pred'])));assert err<=1e-12;errors.append(err)
        key=(r['mode'],r['version'],r['view']);pred.setdefault(key,np.full(n,np.nan));assert np.isnan(pred[key][te]).all();pred[key][te]=pp;checked+=1
    assert checked==60
    ext=np.load(BIG/'external.npz');gt=np.flatnonzero(d['cohort']=='G')
    for ver in ['old','new']:
      for view in ['speed','core3']:
        X=d[ver][:,:1] if view=='speed' else d[ver]
        pp=[independent_predict(joblib.load(r['model']),X[gt]) for r in rec['models'] if r['mode']=='F_only' and r['version']==ver and r['view']==view]
        p=np.mean(pp,axis=0);np.testing.assert_allclose(p,ext[ver+'_'+view],rtol=0,atol=1e-12)
        pred[('external',ver,view)]=np.full(n,np.nan);pred[('external',ver,view)][gt]=p
    rows=[];deltas=[]
    for mode in ['F_only','G_only','pooled','external']:
      for cohort in (['F','G'] if mode=='pooled' else ['F' if mode=='F_only' else 'G']):
        mask=d['cohort']==cohort;y=d['y'][mask]
        for ver in ['old','new']:
          for view in ['speed','core3']:
            pp=pred[(mode,ver,view)][mask];assert np.isfinite(pp).all();rows.append(dict(mode=mode,cohort=cohort,version=ver,view=view,**metrics(y,pp)))
          deltas.append(dict(mode=mode,cohort=cohort,comparison='core3_minus_speed',version=ver,**paired(y,pred[(mode,ver,'core3')][mask],pred[(mode,ver,'speed')][mask])))
        for view in ['speed','core3']:
          deltas.append(dict(mode=mode,cohort=cohort,comparison='new_minus_old',view=view,**paired(y,pred[(mode,'new',view)][mask],pred[(mode,'old',view)][mask])))
    features=json.loads((BIG/'F-features.json').read_text());fails={}
    for r in features:
      if r['status']!='ok':fails[r['reason']]=fails.get(r['reason'],0)+1
    coverage={'F_original':len(features),'F_included':int(sum(d['cohort']=='F')),'F_excluded':len(d['F_excluded_y']),
      'included_high_TUG':int(sum(d['y'][d['cohort']=='F']>=20)),'excluded_high_TUG':int(sum(d['F_excluded_y']>=20)),
      'included_TUG_median':float(np.median(d['y'][d['cohort']=='F'])),'excluded_TUG_median':float(np.median(d['F_excluded_y'])),
      'failures':fails,'G':int(sum(d['cohort']=='G'))}
    save(HERE/'comparison-results.json',dict(coverage=coverage,metrics=rows,paired=deltas))
    np.savez_compressed(BIG/'verified-predictions.npz',ids=d['ids'],y=d['y'],**{'__'.join(k):v for k,v in pred.items()})
    save(HERE/'comparison-verification.json',dict(status='passed',model_replays=checked,external_replays=20,person_overlap=0,duplicate_eval_people=0,
      train_only_statistics_verified=True,max_prediction_difference=max(errors),input_sha256=sha(BIG/'input.npz'),results_sha256=sha(HERE/'comparison-results.json')))
    print(json.dumps({'coverage':coverage,'metrics':rows,'paired':deltas},indent=2),flush=True)
if __name__=='__main__':main()
