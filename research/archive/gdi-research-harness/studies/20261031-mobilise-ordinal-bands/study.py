"""Four ordered score bands, fixed person folds, nominal versus cumulative-loss RF."""
from pathlib import Path
import sys,os,json,hashlib,itertools,time,argparse,importlib.util
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PREV=HERE.parent/'20261031-mobilise-selective-classification'
BASE=HERE.parent/'20261030-mobilise-bout-regression'
SETS=HERE.parent/'20261030-mobilise-deep-sets'
STORE=Path('D:/gait-research-data/D_P/mobilise-ordinal-bands-20261031')
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np,pandas as pd,joblib,sklearn
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from threadpoolctl import threadpool_limits
FEATURES=['averagecadence','averagestrideduration','averagestridespeed','averagestridelength']
SEEDS=[20261030+i*1000 for i in range(5)];KEY=['participantid','visit_number']
CUTS={'sppb':[3.5,6.5,9.5],'tug':[10.,15.,20.]}

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def weights(ids):
    _,inv,n=np.unique(ids,return_inverse=True,return_counts=True);w=1/n[inv];return w/w.mean()
def classes(y,target):return np.digitize(y,CUTS[target],right=True)
def lowclass(target):return 0 if target=='sppb' else 3
def cumulative(y):return (y[:,None]>np.arange(3)).astype(float)
def probabilities(model,x,name):
    if name!='OrdinalRF':p=model.predict_proba(x)
    else:
        q=model.predict(x);assert np.all(q[:,:-1]>=q[:,1:]-1e-12)
        p=np.column_stack([1-q[:,0],q[:,0]-q[:,1],q[:,1]-q[:,2],q[:,2]])
        assert p.min()>-1e-12;p=np.maximum(p,0);p/=p.sum(1,keepdims=True)
    assert p.shape==(len(x),4) and np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
    return p
def make(name,seed):
    params=dict(n_estimators=128,min_samples_leaf=20,max_features=1.,random_state=seed,n_jobs=1)
    model=RandomForestClassifier(criterion='gini',**params) if name=='RF' else RandomForestRegressor(criterion='squared_error',**params)
    return Pipeline([('scale',StandardScaler()),('model',model)])
def fit(model,x,y,ids,name):
    w=weights(ids);model.fit(x,cumulative(y) if name=='OrdinalRF' else y,scale__sample_weight=w,model__sample_weight=w);return model
def mapper(z,y,ids):
    m=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-8))])
    w=weights(ids);m.fit(z[:,None],y,scale__sample_weight=w,model__sample_weight=w)
    assert np.array_equal(m.classes_,np.arange(4)) and np.all(m['model'].n_iter_<2000)
    return m
def temper(p,t):return softmax(np.log(np.clip(p,1e-12,1))/t,axis=1)
def temperature(p,y,ids):
    w=weights(ids)
    def loss(a):return -float(np.average(np.log(temper(p,np.exp(a))[np.arange(len(y)),y]),weights=w))
    r=minimize_scalar(loss,bounds=(-np.log(4),np.log(4)),method='bounded',options={'xatol':1e-8})
    assert r.success
    return float(np.exp(r.x))
def thresholds(p,y,ids):
    # Policies matched to the preceding binary experiment; subgroup person weighting.
    values=np.r_[0.,np.unique(p),1.00000001];neg=y==0;pos=~neg
    def rates(a,w):
        ix=np.argsort(a);a=a[ix];w=w[ix];cs=np.r_[0.,np.cumsum(w)]
        return (cs[-1]-cs[np.searchsorted(a,values,side='left')])/cs[-1]
    fp=rates(p[neg],weights(ids[neg]));tp=rates(p[pos],weights(ids[pos]))
    return {'p50':.5,'fpr10':float(values[np.flatnonzero(fp<=.1+1e-12)[0]]),'sens80':float(values[np.flatnonzero(tp>=.8-1e-12)[-1]])}
def init():
    global DATA
    DATA=pd.read_csv(BASE/'features.csv');DATA=DATA[DATA.draw.eq(0)].copy()
def cell_data(target,band,k,fold):
    d=DATA[DATA[target+'_ok']&DATA.band.eq(band)&DATA.k.eq(k)].sort_values(KEY)
    train=d[d.fold.ne(fold)].reset_index(drop=True);test=d[d.fold.eq(fold)].reset_index(drop=True)
    ref=json.loads((SETS/'folds'/f'{target}-{band}-{k}-f{fold}.json').read_text())
    assert sorted(train.participantid.unique())==ref['train_people'] and sorted(test.participantid.unique())==ref['test_people']
    assert not set(train.participantid)&set(test.participantid)
    inner=np.full(len(train),-1)
    for i,sp in enumerate(ref['inner_people']):
        ix=train.participantid.isin(sp['valid']);inner[ix]=i
        assert set(train.loc[~ix,'participantid'])==set(sp['train'])
    assert (inner>=0).all();return train,test,inner,ref
def outframe(test,target,band,k,fold,seed):
    col='total_sppb_score' if target=='sppb' else 'tugtime'
    out=test[KEY+['cohort']].copy();out['actual']=test[col];out['class']=classes(test[col].to_numpy(),target)
    out['target']=target;out['band']=band;out['k']=k;out['fold']=fold;out['seed']=seed
    return out
def finish_prob(name,oof,raw,y,ids,inner,out,target,arrays):
    temps=[];cross=np.zeros_like(oof)
    for i in range(3):
        a=inner!=i;b=~a;t=temperature(oof[a],y[a],ids[a]);cross[b]=temper(oof[b],t);temps.append(t)
    t=temperature(oof,y,ids);cal=temper(raw,t)
    np.testing.assert_array_equal(raw.argmax(1),cal.argmax(1))
    arrays[name+'_oof']=oof;arrays[name+'_crosscal']=cross
    rules={};low=(y==lowclass(target)).astype(int)
    for variant,ip,op in [('raw',oof,raw),('cal',cross,cal)]:
        for j in range(4):out[f'{name}_{variant}_p{j}']=op[:,j]
        ts=thresholds(ip[:,lowclass(target)],low,ids);rules[variant]=ts
        for policy,cut in ts.items():out[f'{name}_{variant}_{policy}']=(op[:,lowclass(target)]>=cut).astype(int)
    return {'temperature':t,'cross_temperatures':temps,'thresholds':rules}
def work(task):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}';folder=STORE/'folds';done=folder/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,ref=cell_data(target,band,k,fold);col='total_sppb_score' if target=='sppb' else 'tugtime'
    x=tr[FEATURES].to_numpy();xt=te[FEATURES].to_numpy();ids=tr.participantid.to_numpy();y=classes(tr[col].to_numpy(),target)
    out=outframe(te,target,band,k,fold,seed);arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    prior=np.bincount(y,weights=weights(ids),minlength=4);prior/=prior.sum()
    for j in range(4):out[f'Prior_p{j}']=prior[j]
    ledger={'task':list(task),'reference_split_sha256':sha(SETS/'folds'/f'{target}-{band}-{k}-f{fold}.json'),'models':{}}
    for name in ['RF','OrdinalRF']:
        oof=np.zeros((len(tr),4))
        for i in range(3):
            a=inner!=i;b=~a;m=fit(make(name,seed+fold*10+i),x[a],y[a],ids[a],name);oof[b]=probabilities(m,x[b],name)
        m=fit(make(name,seed+fold*10+9),x,y,ids,name);raw=probabilities(m,xt,name)
        info=finish_prob(name,oof,raw,y,ids,inner,out,target,arrays)
        path=STORE/'models'/f'{tag}-{name}.joblib';joblib.dump(m,path,compress=3)
        info.update(path=str(path),sha256=sha(path));ledger['models'][name]=info
    np.savez_compressed(folder/f'{tag}-inner.npz',**arrays);out.to_csv(folder/f'{tag}.csv.gz',index=False)
    ledger.update(seconds=time.time()-start,predictions_sha256=sha(folder/f'{tag}.csv.gz'),inner_sha256=sha(folder/f'{tag}-inner.npz'))
    save(done,ledger);return tag,round(ledger['seconds'],1)

def regression(task):
    target,band,k,fold=task;tag=f'{target}-{band}-{k}-f{fold}';folder=STORE/'regression';done=folder/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,ref=cell_data(target,band,k,fold);col='total_sppb_score' if target=='sppb' else 'tugtime'
    ids=tr.participantid.to_numpy();y=classes(tr[col].to_numpy(),target);oldnp=PREV/'regression'/f'{tag}-inner.npz';oldcsv=PREV/'regression'/f'{tag}.csv'
    ins=np.load(oldnp);old=pd.read_csv(oldcsv);np.testing.assert_array_equal(ins['participantid'],ids);np.testing.assert_array_equal(ins['inner'],inner)
    np.testing.assert_array_equal(old[KEY].to_numpy(),te[KEY].to_numpy())
    out=outframe(te,target,band,k,fold,-1);arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    ledger={'task':list(task),'input_hashes':{str(p):sha(p) for p in [oldnp,oldcsv]},'models':{}}
    for name in ['SVRreg','RFreg']:
        z=ins[name+'_oof_prediction'];zt=old[name+'_prediction'].to_numpy();oof=np.zeros((len(tr),4));out[name+'_point']=classes(zt,target)
        for i in range(3):
            a=inner!=i;b=~a;m=mapper(z[a],y[a],ids[a]);oof[b]=m.predict_proba(z[b,None])
        m=mapper(z,y,ids);raw=m.predict_proba(zt[:,None]);arrays[name+'_scores']=z
        info=finish_prob(name,oof,raw,y,ids,inner,out,target,arrays)
        path=STORE/'models'/f'{tag}-{name}-mapper.joblib';joblib.dump(m,path,compress=3)
        info.update(path=str(path),sha256=sha(path));ledger['models'][name]=info
    np.savez_compressed(folder/f'{tag}-inner.npz',**arrays);out.to_csv(folder/f'{tag}.csv.gz',index=False)
    ledger.update(seconds=time.time()-start,predictions_sha256=sha(folder/f'{tag}.csv.gz'),inner_sha256=sha(folder/f'{tag}-inner.npz'))
    save(done,ledger);return tag,round(ledger['seconds'],1)

def prepare():
    files=[BASE/'features.csv',BASE/'selected-bouts.csv',BASE/'person-folds.csv',BASE/'predictions.csv',PREV/'verification.json',PREV/'metrics-by-seed.csv']
    p={'code_sha256':sha(__file__),'source_hashes':{str(p):sha(p) for p in files},'seeds':SEEDS,'bands':CUTS,
       'label_interpretation':{'sppb':'0-3 /4-6 /7-9 /10-12; continuous regression divided at3.5,6.5,9.5 without rounding',
                               'tug':'<=10 /(10,15] /(15,20] />20 seconds'},
       'data':'same 2174 SPPB /1939 TUG people, visits, draw0; [4,10)/[30,60) seconds x1/3/5; median four features; exact outer5 and inner3 person IDs',
       'RF':'128 trees leaf20 max_features1 gini, equal person weights, no class weights',
       'OrdinalRF':'same shared forest size; multioutput squared-error to I(class>0),I(class>1),I(class>2); objective proportional to RPS; shared leaves guarantee ordered cumulative probabilities',
       'regression_probability':'reuse verified saved SVR/RF inner scores and outer predictions; one-dimensional multinomial LR C1 maps score to4 probabilities, cross-fit map on inner groups',
       'regression_repetitions':'saved single-seed baselines repeated for pairing, not five independent runs',
       'temperature':'fit scalar T in[.25,4] by person-weighted OOF log loss; cross-fit T for internal threshold selection; raw and calibrated both reported, no choosing by outer results',
       'low':'SPPB class0 /TUG class3; argmax miss plus p50/fpr10/sens80 operating rules chosen internally',
       'metrics':['multiclass Brier sum4','RPS mean3 cumulative squared errors','log loss clipped1e-12','per-class Brier and ECE10','accuracy','balanced recall','band distance MAE','opposite extreme abs distance3','low extreme-to-best error','low miss and false positive'],
       'primary':'long5 draw0 declared before new outcomes; all6conditions and5seeds reported; no new model hyperparameter search',
       'storage':{'small':str(HERE),'large':str(STORE)},'versions':{'numpy':np.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__},
       'limits':'reused development folds; no external/video/clinical threshold validation; monotone cumulative probabilities do not guarantee accurate risk estimates'}
    if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==p
    save(HERE/'protocol.json',p);print('PROTOCOL_READY',flush=True)

def run(mode,workers):
    p=json.loads((HERE/'protocol.json').read_text());assert p['code_sha256']==sha(__file__)
    for name in ['models','folds','regression']:(STORE/name).mkdir(parents=True,exist_ok=True)
    assert HERE.resolve().drive.upper()=='C:' and STORE.resolve().drive.upper()=='D:'
    init();start=time.time()
    if mode=='regression':tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5)));fn=regression
    else:
        tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS));fn=work
        if mode=='pilot':tasks=[('sppb','long',5,0,SEEDS[0]),('tug','long',5,0,SEEDS[0])]
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fn,t) for t in tasks]
        for n,f in enumerate(as_completed(futures),1):
            tag,sec=f.result()
            if mode=='pilot' or n%10==0:print(n,'/',len(tasks),tag,sec,'elapsed',round(time.time()-start),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(tasks),'seconds':time.time()-start,'code_sha256':sha(__file__)})
    print(mode.upper()+'_DONE',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','pilot','main','regression']);a.add_argument('--workers',type=int,default=4);args=a.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
