"""Person-disjoint low-function classification; fixed models, OOF sigmoid calibration."""
from pathlib import Path
import os, sys, json, hashlib, time, itertools, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE.parent/'20261030-mobilise-bout-regression'
SETS=HERE.parent/'20261030-mobilise-deep-sets'
STABILITY=HERE.parent/'20261030-mobilise-deep-sets-stability'
MODELS=Path('D:/gait-research-data/D_P/mobilise-selective-classification-20261031/models')
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
import sklearn, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
FEATURES=['averagecadence','averagestrideduration','averagestridespeed','averagestridelength']
SEEDS=[20261030+i*1000 for i in range(5)]
KEY=['participantid','visit_number']

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def weights(ids):
    _,inv,n=np.unique(ids,return_inverse=True,return_counts=True)
    w=1/n[inv];return w/w.mean()
def label(y,target):return (y<=3 if target=='sppb' else y>20).astype(int)
def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
def mean(v,ids):return float(np.average(v,weights=weights(ids))) if len(v) else None
def quantile(v,w,q):
    ix=np.argsort(v,kind='stable');return float(v[ix][min(np.searchsorted(np.cumsum(w[ix]),q*w.sum()),len(v)-1)])
def thresholds(p,y,ids):
    # Weight each person equally within each true-label subgroup.
    vals=np.r_[0.,np.unique(p),1.00000001]
    pos=y==1;neg=~pos
    wp=weights(ids[pos]);wn=weights(ids[neg])
    def rate(a,w):
        ix=np.argsort(a);a=a[ix];w=w[ix];cum=np.r_[0.,np.cumsum(w)]
        return (cum[-1]-cum[np.searchsorted(a,vals,side='left')])/cum[-1]
    tpr=rate(p[pos],wp);fpr=rate(p[neg],wn)
    return {'p50':.5,'sens80':float(vals[np.flatnonzero(tpr>=.8-1e-12)[-1]]),
            'fpr10':float(vals[np.flatnonzero(fpr<=.1+1e-12)[0]])}
def margin(p,t):return abs(logit(p)-logit(np.array(t)))
def calibration(z,y,ids):
    m=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000,tol=1e-8))])
    w=weights(ids);m.fit(z.reshape(-1,1),y,scale__sample_weight=w,model__sample_weight=w)
    assert np.all(m['model'].n_iter_<2000)
    return m
def make(name,seed):
    m=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-8) if name=='LR' else RandomForestClassifier(n_estimators=128,min_samples_leaf=20,max_features=1.,random_state=seed,n_jobs=1)
    return Pipeline([('scale',StandardScaler()),('model',m)])
def fit(m,x,y,ids):
    w=weights(ids);m.fit(x,y,scale__sample_weight=w,model__sample_weight=w)
    if hasattr(m['model'],'n_iter_'):assert np.all(m['model'].n_iter_<2000)
    return m
def initialize():
    global DATA,LIMIT
    LIMIT=threadpool_limits(limits=1)
    DATA=pd.read_csv(BASE/'features.csv');DATA=DATA[DATA.draw.eq(0)].copy()
def work(task):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    done=HERE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();col='total_sppb_score' if target=='sppb' else 'tugtime'
    cell=DATA[DATA[target+'_ok']&DATA.band.eq(band)&DATA.k.eq(k)].sort_values(KEY)
    tr=cell[cell.fold.ne(fold)].reset_index(drop=True);te=cell[cell.fold.eq(fold)].reset_index(drop=True)
    original=json.loads((SETS/'folds'/f'{target}-{band}-{k}-f{fold}.json').read_text())
    assert sorted(tr.participantid.unique())==original['train_people']
    assert sorted(te.participantid.unique())==original['test_people']
    assert not set(tr.participantid)&set(te.participantid)
    x=tr[FEATURES].to_numpy();xt=te[FEATURES].to_numpy();ids=tr.participantid.to_numpy()
    y=label(tr[col].to_numpy(),target);yt=label(te[col].to_numpy(),target)
    inner=np.full(len(tr),-1)
    for i,sp in enumerate(original['inner_people']):
        a=tr.participantid.isin(sp['train']).to_numpy();b=tr.participantid.isin(sp['valid']).to_numpy()
        assert np.all(a^b) and not set(ids[a])&set(ids[b]);inner[b]=i
    assert (inner>=0).all()
    out=te[KEY+['fold','cohort']].copy();out['actual']=te[col];out['positive']=yt
    out['target']=target;out['band']=band;out['k']=k;out['seed']=seed
    prior=float(np.average(y,weights=weights(ids)));out['prior']=prior
    ledger={'target':target,'band':band,'k':k,'fold':fold,'seed':seed,'train_people':original['train_people'],
            'test_people':original['test_people'],'inner_people':original['inner_people'],'models':{}}
    arrays={'participantid':ids,'visit_number':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    for name in ['LR','RF']:
        oof=np.full(len(y),np.nan)
        for i in range(3):
            a=inner!=i;b=~a;m=fit(make(name,seed+fold*10+i),x[a],y[a],ids[a])
            oof[b]=m.predict_proba(x[b])[:,1]
        assert np.isfinite(oof).all()
        # Cross-fit sigmoid maps for operating-rule selection; all within outer training.
        z=logit(oof);cp=np.full(len(y),np.nan)
        for i in range(3):
            a=inner!=i;b=~a;c=calibration(z[a],y[a],ids[a]);cp[b]=c.predict_proba(z[b,None])[:,1]
        c=calibration(z,y,ids)
        final=fit(make(name,seed+fold*10+9),x,y,ids)
        raw=final.predict_proba(xt)[:,1];cal=c.predict_proba(logit(raw)[:,None])[:,1]
        out[name+'_raw']=raw;out[name+'_cal']=cal
        arrays[name+'_oof']=oof;arrays[name+'_crosscal']=cp
        rules={}
        for variant,ip,op in [('raw',oof,raw),('cal',cp,cal)]:
            ts=thresholds(ip,y,ids);rules[variant]={}
            for policy,t in ts.items():
                cuts={}
                for reject in [0.,.1,.2]:
                    cut=-1. if reject==0 else quantile(margin(ip,t),weights(ids),reject)
                    # Strict comparison leaves ties answered; actual coverage is reported.
                    dec=np.where(margin(op,t)<cut,-1,(op>=t).astype(int))
                    out[f'{name}_{variant}_{policy}_r{int(reject*100)}']=dec
                    cuts[str(reject)]=cut
                rules[variant][policy]={'threshold':t,'reject_margin':cuts}
        dest=MODELS/f'{tag}-{name}.joblib';joblib.dump({'model':final,'calibrator':c,'rules':rules},dest,compress=3)
        ledger['models'][name]={'path':str(dest),'sha256':sha(dest),'rules':rules}
    np.savez_compressed(HERE/'folds'/f'{tag}-inner.npz',**arrays)
    out.to_csv(HERE/'folds'/f'{tag}.csv',index=False)
    ledger['predictions_sha256']=sha(HERE/'folds'/f'{tag}.csv');ledger['inner_sha256']=sha(HERE/'folds'/f'{tag}-inner.npz')
    ledger['seconds']=time.time()-start
    save(done,ledger);return tag,round(ledger['seconds'],1)

def prepare():
    HERE.mkdir(exist_ok=True);(HERE/'folds').mkdir(exist_ok=True)
    files=[BASE/'features.csv',BASE/'selected-bouts.csv',BASE/'person-folds.csv',BASE/'predictions.csv',STABILITY/'predictions.csv']
    p={'seeds':SEEDS,'source_hashes':{str(f):sha(f) for f in files},'code_sha256':sha(__file__),
       'conditions':{'bands':['short','long'],'k':[1,3,5],'draw':0},'targets':{'sppb':'<=3','tug':'>20'},
       'features':FEATURES,'aggregation':'unchanged per-feature median of same selected bouts',
       'outer':'unchanged 5 person folds','inner':'exact original three person-fold ID lists',
       'models':{'LR':'C=1; weighted StandardScaler; deterministic across seed schedules',
                 'RF':'128 trees; leaf20; max_features1; seed+fold*10+inner/refit9; no class weights'},
       'selection':'fixed model hyperparameters declared before training; no outer-label selection',
       'calibration':'sigmoid logistic on inner OOF logits; cross-fitted maps for rule selection, full inner OOF map for outer test',
       'calibration_limit':'outer held-out evaluation is honest; inner cross-calibration is not a separate nested performance estimate or conformal guarantee',
       'policies':['p50','maximum threshold attaining inner positive-person sensitivity>=.8','minimum threshold with inner negative-person FPR<=.1'],
       'rejection':'inner person-weighted 0/10/20 percentiles of absolute log-odds distance to decision threshold; strict less-than; report realized rates',
       'weights':'equal total weight per person; subgroup rates reweight within true subgroup',
       'baselines':'saved original Ridge/SVR/RF and median at target cutoffs; saved DeepSets600 for each of five seeds; prior probability',
       'interpretation':'development cross-validation, not unseen external or clinical validation; abstention is boundary ambiguity, not validated OOD detection',
       'outputs':str(HERE),'large_models':str(MODELS),'download_required':False,
       'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
    if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==p
    save(HERE/'protocol.json',p)
    print('PROTOCOL_WRITTEN',flush=True)

def run(pilot=False,workers=4):
    protocol=json.loads((HERE/'protocol.json').read_text());assert sha(__file__)==protocol['code_sha256']
    MODELS.mkdir(parents=True,exist_ok=True)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    if pilot:tasks=[('sppb','long',5,0,SEEDS[0]),('tug','long',5,0,SEEDS[0])]
    start=time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        futures=[pool.submit(work,t) for t in tasks]
        for n,f in enumerate(as_completed(futures),1):
            tag,secs=f.result()
            if pilot or n%10==0:print(n,'/',len(tasks),tag,secs,'elapsed',round(time.time()-start),flush=True)
    if not pilot:
        paths=sorted((HERE/'folds').glob('*.csv'));assert len(paths)==300
        pd.concat([pd.read_csv(p) for p in paths],ignore_index=True).to_csv(HERE/'predictions.csv.gz',index=False)
        save(HERE/'completion.json',{'jobs':300,'final_models':600,'inner_base_fits':1800,'sigmoid_fits':2400,'elapsed_seconds':time.time()-start,'predictions_sha256':sha(HERE/'predictions.csv.gz')})
    print('PILOT_DONE' if pilot else 'ALL_DONE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.pilot,a.workers)
