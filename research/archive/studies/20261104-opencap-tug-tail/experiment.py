"""Exploratory tail-error study; frozen people/features/folds from camera study."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json, hashlib, sys, warnings
import numpy as np, joblib
from scipy.ndimage import gaussian_filter1d
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, HistGradientBoostingRegressor
warnings.filterwarnings('ignore',message='X does not have valid feature names.*')
SRC=Path('D:/Projects/D_P/studies/20261104-opencap-camera-tug')
HERE=Path('D:/Projects/D_P/studies/20261104-opencap-tug-tail')
DATA=Path('D:/gait-research-data/D_P/opencap-camera-tug-20261104')
HEAVY=Path('D:/gait-research-data/D_P/opencap-tug-tail-20261104')
CORE=['speed_mean','speed_std','pelvis_bob_std','trunk_lean_mean','trunk_lean_std','L_knee_flexion_2d_std','R_knee_flexion_2d_std','ankle_separation_range','extrema_cadence','extrema_step_time_cv']
METHODS=['base','lds05','lds1','extrap','extrap_lds','predgate10','predgate12','logitgate03','logitgate05','logitgate07','treegate03','treegate05','softgate','quantilegate']
POLICIES=['overall_selected','tail_constrained','tail_balanced','two_stage_constrained']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def columns(names,view):return list(range(len(names))) if view=='all' else [names.index(n) for n in CORE]
def inputs(X,cols):return np.column_stack([X[:,cols],6/np.maximum(X[:,0],.1)])
def weights(y,power):
    bins=np.floor(y).astype(int);h=np.bincount(bins,minlength=int(max(y))+4).astype(float)
    density=gaussian_filter1d(h,2.,mode='constant')[bins]
    w=np.maximum(density,1e-8)**(-power);w/=w.mean();return np.minimum(w,10.)
def fit_bundle(X,y,ids,tr,cols):
    xx=inputs(X,cols);imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(xx[tr])
    sc=StandardScaler().fit(imp.transform(xx[tr]));z=sc.transform(imp.transform(xx[tr]));yy=y[tr]
    def et(leaf=10):return ExtraTreesRegressor(n_estimators=128,min_samples_leaf=leaf,max_features=1.,random_state=20261104,n_jobs=1)
    models={'base':et().fit(z,yy),'lds05':et().fit(z,yy,sample_weight=weights(yy,.5)),
      'lds1':et().fit(z,yy,sample_weight=weights(yy,1.)),
      'extrap':Ridge(alpha=100.).fit(z,yy),'extrap_lds':Ridge(alpha=100.).fit(z,yy,sample_weight=weights(yy,1.))}
    high=yy>=12;assert high.sum()>=5 and (~high).sum()>=5
    models['high']=et(3).fit(z[high],yy[high]);models['low']=et(3).fit(z[~high],yy[~high])
    models['logit']=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=20261104).fit(z,high)
    models['tree']=ExtraTreesClassifier(n_estimators=128,min_samples_leaf=5,class_weight='balanced',random_state=20261104,n_jobs=1).fit(z,high)
    models['q90']=HistGradientBoostingRegressor(loss='quantile',quantile=.9,max_iter=120,max_leaf_nodes=7,min_samples_leaf=10,l2_regularization=5.,early_stopping=False,random_state=20261104).fit(z,yy)
    return {'imputer':imp,'scaler':sc,'models':models,'train':tr.tolist(),'ids':ids[tr].tolist(),'cols':cols,
       'weights05':weights(yy,.5),'weights1':weights(yy,1.)}
def predict(b,X,ix):
    z=b['scaler'].transform(b['imputer'].transform(inputs(X,b['cols'])[ix]));m=b['models']
    p={n:np.maximum(.1,m[n].predict(z)) for n in ['base','lds05','lds1','extrap','extrap_lds','high','low','q90']}
    logit=m['logit'].predict_proba(z)[:,1];tree=m['tree'].predict_proba(z)[:,1]
    out={n:p[n] for n in ['base','lds05','lds1','extrap','extrap_lds']};flags={n:np.zeros(len(ix),bool) for n in out}
    for threshold in [10,12]:
        n='predgate'+str(threshold);flag=p['base']>=threshold;out[n]=np.where(flag,p['high'],p['base']);flags[n]=flag
    for code,score,threshold in [('logitgate03',logit,.3),('logitgate05',logit,.5),('logitgate07',logit,.7),('treegate03',tree,.3),('treegate05',tree,.5)]:
        flags[code]=score>=threshold;out[code]=np.where(flags[code],p['high'],p['base'])
    out['softgate']=p['low']*(1-logit)+p['high']*logit;flags['softgate']=logit>=.5
    flags['quantilegate']=logit>=.5;out['quantilegate']=np.where(flags['quantilegate'],np.maximum(p['base'],p['q90']),p['base'])
    return out,flags,{'logit':logit,'tree':tree}
def losses(y,p):
    ae=abs(y-p);groups=[y<12,(y>=12)&(y<20),y>=20]
    return {'MAE':float(ae.mean()),'low_MAE':float(ae[groups[0]].mean()),'tail_MAE':float(ae[groups[2]].mean()),
      'balanced_MAE':float(np.mean([ae[g].mean() for g in groups if g.any()]))}
def choose(scores,policy):
    if policy=='overall_selected':return min(scores,key=lambda k:scores[k]['MAE'])
    keys=list(scores)
    if policy in ['tail_constrained','two_stage_constrained']:
        base=scores['all/base'];keys=[k for k in keys if scores[k]['MAE']<=base['MAE']+.1 and scores[k]['low_MAE']<=base['low_MAE']+.1]
        if policy=='two_stage_constrained':keys=[k for k in keys if 'gate' in k or k=='all/base']
    return min(keys,key=lambda k:scores[k]['balanced_MAE'])
def freeze():
    HERE.mkdir(parents=True,exist_ok=True);HEAVY.mkdir(parents=True,exist_ok=True)
    assert not (HERE/'protocol.json').exists()
    p={'question':'Can camera-compatible short-walk features identify underpredicted high-TUG cases and reduce tail error?',
      'people':297,'source_features_sha256':sha(DATA/'features.npz'),'source_splits_sha256':sha(SRC/'effective-splits.json'),
      'source_result_sha256':sha(SRC/'results.json'),'source_verification_sha256':sha(SRC/'verification.json'),
      'code_sha256':sha(__file__),'views':{'all':'same64 features, plus deterministic inverse speed 6/v','core':CORE},
      'target':'same-visit clinical TUG seconds, no TUG video duration or other clinical scores as predictor',
      'methods':METHODS,'fit':'ET128 leaf10; high/low experts ET128 leaf3 trained on >=12/<12 training TUG; balanced logisticC0.1 and balanced tree gate; q90 HGB; Ridgealpha100; all preprocessing training-only',
      'LDS':'training-only1-second target histogram, Gaussian sigma2bins, inverse density power0.5/1.0, normalized then capped10; practical adaptation, not full DIR neural method',
      'inverse_speed':'6/max(speed,0.1), illustrative standardized walking component; not measured TUG walking duration',
      'selection':'same original25 outer folds and3 inner folds each; all thresholds/model variants selected using inner OOF predictions; never true outer group as routing input',
      'policies':{'overall_selected':'minimum inner overall MAE','tail_balanced':'minimum equally weighted mean MAE of true training TUG<12,12-20,>=20 groups',
         'tail_constrained':'minimum balanced inner MAE, requiring overall and <12 inner MAE <= fixedall-feature baseline+0.1s',
         'two_stage_constrained':'same constraints, restricted to routing methods and baseline fallback'},
      'constraint_note':'0.1s is an exploratory model-selection allowance, not a clinical acceptable-error standard; outer tradeoffs always reported',
      'evaluation':'same297 people in every arm; five person-disjoint5-fold partitions; repeated errors first averaged per person; 10000 paired person bootstraps',
      'groups':'<12,12-20,>=20 are descriptive experiment strata, not universal diagnoses or future-fall labels',
      'routing_metrics':'referral rate; recall for actualTUG>=20; fraction of actualTUG<12 referred; scores not calibrated clinical probabilities',
      'success':'report overall, low-range, high-range MAE and pairedCI together; do not claim improvement from tail alone or outer-picked best candidate',
      'scope':'further development on previously evaluated data; ideal lateral projection not actual monocular phone/MediaPipe validation;9 severe people',
      'storage_estimate':'<=2GB models and replay files on heavy data drive; no new GPU or external transfer',
      'additional_tasks':'file availability audit only unless separately acquired for linkage/pilot; no trial-duration feature'}
    save(HERE/'protocol.json',p);print('FROZEN_TAIL_STUDY',flush=True)
def train():
    prot=json.loads((HERE/'protocol.json').read_text());assert sha(__file__)==prot['code_sha256']
    assert sha(DATA/'features.npz')==prot['source_features_sha256'];assert sha(SRC/'effective-splits.json')==prot['source_splits_sha256']
    assert not (HERE/'selections.json').exists()
    d=np.load(DATA/'features.npz');X=d['X'];y=d['y'];ids=d['ids'];names=d['names'].tolist()
    splits=json.loads((SRC/'effective-splits.json').read_text());selections=[];allrows=[];bundlehash={};(HEAVY/'models').mkdir(exist_ok=True)
    for rep in splits:
      rr=rep['repeat']
      for fold in rep['splits']:
        ff=fold['fold'];tr=np.array(fold['train']);te=np.array(fold['test']);innerpred={};outerpred={};outerflags={}
        for view in ['all','core']:
          cols=columns(names,view)
          for ii,inner in enumerate(fold['inner']):
            it=np.array(inner['train']);iv=np.array(inner['validation']);assert not set(ids[it])&set(ids[iv])
            b=fit_bundle(X,y,ids,it,cols);path=HEAVY/'models'/f'r{rr}-f{ff}-{view}-inner{ii}.joblib';joblib.dump(b,path,compress=3);bundlehash[path.name]=sha(path)
            pp,flags,sc=predict(b,X,iv)
            for method,p in pp.items():
                key=view+'/'+method
                if key not in innerpred:innerpred[key]=np.full(len(y),np.nan)
                innerpred[key][iv]=p
          b=fit_bundle(X,y,ids,tr,cols);path=HEAVY/'models'/f'r{rr}-f{ff}-{view}-outer.joblib';joblib.dump(b,path,compress=3);bundlehash[path.name]=sha(path)
          pp,flags,sc=predict(b,X,te)
          for method,p in pp.items():outerpred[view+'/'+method]=p;outerflags[view+'/'+method]=flags[method]
        scores={k:losses(y[tr],pp[tr]) for k,pp in innerpred.items()};selected={p:choose(scores,p) for p in POLICIES}
        np.savez_compressed(HEAVY/'models'/f'r{rr}-f{ff}-inner-predictions.npz',**{k.replace('/','__'):p[tr] for k,p in innerpred.items()})
        selections.append({'repeat':rr,'fold':ff,'train':tr.tolist(),'test':te.tolist(),'scores':scores,'selected':selected})
        # Keep fixed candidates for transparent diagnostics; only pre-frozen policies are comparison winners.
        for name,pp in outerpred.items():
          for j,i in enumerate(te):allrows.append({'repeat':rr,'fold':ff,'name':name,'index':int(i),'id':str(ids[i]),'true':float(y[i]),'pred':float(pp[j]),'flag':bool(outerflags[name][j])})
        for policy,key in selected.items():
          for j,i in enumerate(te):allrows.append({'repeat':rr,'fold':ff,'name':policy,'index':int(i),'id':str(ids[i]),'true':float(y[i]),'pred':float(outerpred[key][j]),'flag':bool(outerflags[key][j])})
        save(HERE/'selections.json',selections);save(HEAVY/'predictions.json',allrows);save(HERE/'bundle-hashes.json',bundlehash)
        print('TAIL_FOLD_COMPLETE',rr,ff,flush=True)
    print('TAIL_TRAIN_COMPLETE',len(bundlehash),'BUNDLES',len(allrows),'ROWS',flush=True)
if __name__=='__main__':{'freeze':freeze,'train':train}[sys.argv[1]]()
