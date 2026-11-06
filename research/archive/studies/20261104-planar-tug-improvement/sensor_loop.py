"""Camera-computable quantities only: new controlled FRAILPOL development batch."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,warnings
import numpy as np,joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.svm import SVR
from sklearn.linear_model import Ridge,GammaRegressor
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import nnls
HERE=Path(__file__).resolve().parent
HEAVY=Path('D:/gait-research-data/D_P/planar-tug-improvement-20261104')
SOURCE=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104/data.npz')
AUDIT=Path('D:/Projects/D_P/studies/20261104-pose-compatible-tug/feature-audit.json')
OLD=Path('D:/gait-research-data/D_P/pose-compatible-tug-20261104')
VIEWS=['speed','core3','timing7','metric12','availability_diagnostic']
CONFIGS=[{'kind':'svr','C':c,'noise':n} for c,n in [(10,0),(100,0),(10,.05),(100,.05),(10,.10)]]+[
    {'kind':'spline','alpha':a} for a in [10,100]]+[{'kind':'trees','leaf':l} for l in [5,10]]+[
    {'kind':'median_boost'},{'kind':'weighted_boost'},{'kind':'gamma'}]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def feature_names():
    audit=json.loads(AUDIT.read_text());metric=audit['sets']['calibrated_spatiotemporal']
    return {'speed':['speed'],'core3':['speed','stride_time','swing_fraction'],
      'timing7':audit['sets']['timing'],'metric12':metric,
      'availability_diagnostic':['speed','stride_time','swing_fraction']+['missing_'+n for n in metric]}
def load():
    d=np.load(SOURCE);names=json.loads(Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn/protocol.json').read_text())['feature_names']
    sets=feature_names();X={}
    for view in VIEWS:
        if view=='availability_diagnostic':
            X[view]=np.column_stack([d['F'][:,[names.index(n) for n in sets['core3']]],
              np.isnan(d['F'][:,[names.index(n) for n in sets['metric12']]]).astype(float)])
        else:X[view]=d['F'][:,[names.index(n) for n in sets[view]]]
    return d,X,sets
def density_weights(y):
    # Three descriptive score groups, sqrt inverse frequency, capped at10.
    g=np.digitize(y,[12,20]);count=np.bincount(g,minlength=3)
    w=(len(y)/np.maximum(count[g],1))**.5;return np.minimum(w/w.mean(),10.)
def fit(X,y,cfg,names):
    imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X)
    xx=imp.transform(X);aug=xx;yy=y;weights=None
    noise=cfg.get('noise',0)
    if noise:
        rng=np.random.default_rng(20261104);copies=[xx]
        for _ in range(7):
            a=xx.copy()
            for j,name in enumerate(names):
                if name.startswith('missing_'):continue
                if name in ['speed','stride_time','stride_length']:a[:,j]*=1+rng.normal(0,noise,len(y))
                else:a[:,j]+=rng.normal(0,noise*.4,len(y))
                a[:,j]=np.maximum(a[:,j],0)
                if 'swing_fraction' in name:a[:,j]=np.minimum(a[:,j],1)
            copies.append(a)
        aug=np.concatenate(copies);yy=np.tile(y,8);weights=np.full(len(yy),1/8)
    # Inverse speed is derived from the permitted speed value, shared by every arm.
    def enrich(a):return np.column_stack([a,1/np.maximum(a[:,0],.1)])
    base=enrich(aug);sc=StandardScaler().fit(enrich(xx));zz=sc.transform(base);sp=None
    kind=cfg['kind']
    if kind=='svr':m=SVR(C=cfg['C'],epsilon=.1,gamma=1/zz.shape[1]).fit(zz,yy,sample_weight=weights)
    elif kind=='spline':
        sp=SplineTransformer(n_knots=4,degree=2,include_bias=False,extrapolation='linear').fit(zz)
        m=Ridge(alpha=cfg['alpha']).fit(sp.transform(zz),yy)
    elif kind=='trees':m=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=cfg['leaf'],random_state=20261104,n_jobs=1).fit(zz,yy)
    elif kind=='median_boost':m=HistGradientBoostingRegressor(loss='quantile',quantile=.5,max_iter=160,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=5.,early_stopping=False,random_state=20261104).fit(zz,yy)
    elif kind=='weighted_boost':m=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=5.,early_stopping=False,random_state=20261104).fit(zz,yy,sample_weight=density_weights(yy))
    else:m=GammaRegressor(alpha=1.,max_iter=500,tol=1e-8).fit(zz,yy)
    return {'imputer':imp,'scaler':sc,'spline':sp,'model':m,'config':cfg,'names':names,'ntrain':len(y)}
def predict(m,X):
    xx=m['imputer'].transform(X);z=m['scaler'].transform(np.column_stack([xx,1/np.maximum(xx[:,0],.1)]))
    if m['spline'] is not None:z=m['spline'].transform(z)
    return np.maximum(.1,m['model'].predict(z))
def stack(A,y,tail=1):
    center=A.mean(0);scale=np.maximum(A.std(0),1e-6);S=(A-center)/scale;ym=y.mean();w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.))
    coef=nnls(np.vstack([S*np.sqrt(w[:,None]),np.sqrt(10)*np.eye(S.shape[1])]),np.r_[(y-ym)*np.sqrt(w),np.zeros(S.shape[1])])[0]
    return {'center':center,'scale':scale,'ymean':ym,'coef':coef,'tail':tail}
def pred_stack(m,T):return np.maximum(.1,(T-m['center'])/m['scale']@m['coef']+m['ymean'])
def freeze():
    HERE.mkdir(exist_ok=True);HEAVY.mkdir(exist_ok=True)
    assert not (HERE/'sensor-protocol.json').exists()
    d,X,sets=load();assert len(d['y'])==len(set(d['ids']))==594
    p={'source_sha256':sha(SOURCE),'feature_audit_sha256':sha(AUDIT),'code_sha256':sha(__file__),
      'people':594,'outer_folds':'Unchanged original5 person folds','inner':'3 stratified person folds, seed20261103, labels<12/12-20/>=20',
      'views':sets,'candidates':CONFIGS,'primary':'fixed nonnegative Ridgealpha10 stack of candidate innerOOF predictions; secondary bestsingle/top3/tail4stack',
      'inputs':'Only12 predefined potentially planar-camera-computable gait quantities, or subsets. No rawIMU/CNNoutputs/age/otherclinicaltests/wholeTUGduration',
      'availability':'Missing-value mask arm is diagnostic only, never a proposed phone score. All594 retained; missing real quantities imputed from training people only',
      'noise':'Synthetic feature errors5/10percent in training, 8copies with weights1/8; hypothetical stress assumptions, not measured phone errors. No noise added to evaluation ground truth',
      'fitting':'Train-only median/scaler/spline. All choices and stacking coefficients from outer training people. Every arm/policy reported; no outer-best selection',
      'evaluation':'MAE,p95,>=20 MAE,bias;5000 paired person bootstraps; compare matched speed and prior0.98208566 model. Reused development cohort,9 severe people',
      'scope':'Sensor-derived quantities, not actual video measurements. Models do not validate observable event definitions/metric scale/phone performance',
      'estimated_extra_GB':1.5}
    save(HERE/'sensor-protocol.json',p);print('SENSOR_PROTOCOL_FROZEN',flush=True)
def train():
    p=json.loads((HERE/'sensor-protocol.json').read_text());assert sha(SOURCE)==p['source_sha256'] and sha(__file__)==p['code_sha256']
    d,X,sets=load();y=d['y'];ids=d['ids'];records=[];(HEAVY/'sensor-models').mkdir(exist_ok=True)
    for view in VIEWS:
      for f in range(5):
        tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f)
        A=np.full((len(tr),len(CONFIGS)),np.nan);T=np.empty((len(te),len(CONFIGS)));models=[]
        inner=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])))
        for k,cfg in enumerate(CONFIGS):
          for ii,(it,iv) in enumerate(inner):
            m=fit(X[view][tr[it]],y[tr[it]],cfg,sets[view]);path=HEAVY/'sensor-models'/f'{view}-f{f}-c{k}-i{ii}.joblib';joblib.dump(m,path,compress=3)
            A[iv,k]=predict(m,X[view][tr[iv]]);models.append({'file':path.name,'sha256':sha(path),'candidate':k,'inner':ii,'train':tr[it].tolist(),'test':tr[iv].tolist()})
          m=fit(X[view][tr],y[tr],cfg,sets[view]);path=HEAVY/'sensor-models'/f'{view}-f{f}-c{k}-final.joblib';joblib.dump(m,path,compress=3)
          T[:,k]=predict(m,X[view][te]);models.append({'file':path.name,'sha256':sha(path),'candidate':k,'inner':None,'train':tr.tolist(),'test':te.tolist()})
        order=np.argsort(abs(A-y[tr,None]).mean(0),kind='stable');ms={n:stack(A,y[tr],tail) for n,tail in [('stacked',1),('tail_stacked',4)]}
        pp={'selected':T[:,order[0]],'top3':T[:,order[:3]].mean(1),**{n:pred_stack(m,T) for n,m in ms.items()}}
        context=HEAVY/'sensor-models'/f'{view}-f{f}-context.npz';np.savez_compressed(context,A=A,T=T,**pp)
        meta=HEAVY/'sensor-models'/f'{view}-f{f}-meta.joblib';joblib.dump(ms,meta)
        records.append({'view':view,'fold':f,'train':tr.tolist(),'test':te.tolist(),'train_ids':ids[tr].tolist(),'test_ids':ids[te].tolist(),
          'models':models,'context':context.name,'context_sha256':sha(context),'meta':meta.name,'meta_sha256':sha(meta),'order':order.tolist()})
        save(HERE/'sensor-receipt.json',records);print('SENSOR_CASE_COMPLETE',view,f,flush=True)
    print('SENSOR_FITTING_COMPLETE',len(records),'CASES',flush=True)
if __name__=='__main__':{'freeze':freeze,'train':train}[sys.argv[1]]()
