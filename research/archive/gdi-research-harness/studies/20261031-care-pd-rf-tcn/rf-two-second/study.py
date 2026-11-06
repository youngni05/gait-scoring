"""Frozen-fold RF inspection and prespecified input ablations; no camera claim."""
import sys, os
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'deps'))
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse, csv, json, hashlib, time
from collections import Counter
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
BASE=Path(os.environ.get('CARE_FEATURE_BASE',str(HERE.parent)))
RESULTS=Path(os.environ.get('CARE_FEATURE_RESULTS',str(BASE/'received/runs/main')))
SEEDS=json.loads((BASE/'protocol.json').read_text())['seeds']
JOINTS=['left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hand','right_hand']
MAJOR=[1,2,4,5,7,8,16,17,18,19]

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')

def weights(person):
    c=Counter(person);w=np.array([1/c[p] for p in person]);return w/w.mean()

def metrics(y,p,person):
    w=weights(person);e=p-y
    return dict(people=len(set(person)),walks=len(y),MAE=float(np.average(abs(e),weights=w)),bias=float(np.average(e,weights=w)),large_error=float(np.average(abs(e)>=2,weights=w)))

def allmetrics(d,p):
    return {g:metrics(d['y'][mask],p[mask],d['person'][mask]) for g,mask in [('all',np.ones(len(p),bool)),('high',d['y']>=2),('score3',d['y']==3)]}

def stats(x):
    z=x-x.mean(0);ac=[]
    for lag in [5,10,15,20,25,30,35,40]:
        u,v=z[:-lag],z[lag:]
        ac.append(float((u*v).sum()/max(np.sqrt((u*u).sum()*(v*v).sum()),1e-12)) if len(x)>lag else 0.)
    return np.r_[x.mean(0),x.std(0),np.quantile(x,.1,axis=0),np.quantile(x,.9,axis=0),abs(np.diff(x,axis=0)).mean(0)*25,ac].astype('float32')

def load(include_x=False):
    assert hashlib.sha256((BASE/'data.npz').read_bytes()).hexdigest()==json.loads((BASE/'protocol.json').read_text())['data_sha256']
    data=np.load(BASE/'data.npz',allow_pickle=False)
    return {k:data[k] for k in data.files if include_x or k not in ('x','offsets')}

def prepare():
    d=load(True); major=[];bend=[]
    channels=np.array([6*(j-1)+k for j in MAJOR for k in range(6)])
    for i in range(len(d['y'])):
        x=d['x'][d['offsets'][i]:d['offsets'][i+1]]
        major.append(stats(x[:,channels]))
        # Local bone-axis deflection, from 3D rotations; NOT anatomical flexion or 2D detection.
        a=x.reshape(-1,23,3,2)[:,np.array(MAJOR)-1].astype('float64')
        first=a[:,:,:,0];first/=np.maximum(np.linalg.norm(first,axis=-1,keepdims=True),1e-12)
        second=a[:,:,:,1];second-=np.sum(second*first,axis=-1,keepdims=True)*first
        second/=np.maximum(np.linalg.norm(second,axis=-1,keepdims=True),1e-12)
        angles=[]
        for k,j in enumerate(MAJOR):
            cosine=second[:,k,1] if j<16 else first[:,k,0]
            angles.append(np.arccos(np.clip(cosine,-1,1)))
        bend.append(stats(np.stack(angles,axis=1)))
    major=np.stack(major);motion=major[:,np.r_[60:120,240:308]]
    np.savez_compressed(HERE/'inputs.npz',major3d=major,motion3d=motion,bend3d=np.stack(bend))
    meta=dict(major_joints=[JOINTS[j-1] for j in MAJOR],dimensions={'major3d':308,'motion3d':128,'bend3d':58},
        limitations='All derived from frozen 3D rotations. Bone-axis deflection drops twist and direction; not clinical joint flexion, synthetic image landmarks, or real monocular validation.',
        fixed_before_results=True,seeds=SEEDS,folds=6,source_sha256=hashlib.sha256((BASE/'data.npz').read_bytes()).hexdigest())
    write(HERE/'protocol.json',meta);print(json.dumps(meta),flush=True)

def inspect():
    d=load();x=d['rf'];y=d['y'];person=d['person'];cohort=d['cohort'];fold=d['fold']
    groups={}
    for name,joints in {'hips':[1,2],'knees':[4,5],'ankles_feet':[7,8,10,11],'trunk':[3,6,9],'neck_head':[12,15],'collars_shoulders':[13,14,16,17],'elbows_wrists_hands':[18,19,20,21,22,23]}.items():
        groups[name]=[s*138+6*(j-1)+k for s in range(5) for j in joints for k in range(6)]
    for s,name in enumerate(['mean','std','p10','p90','change_rate']):groups[name]=list(range(s*138,(s+1)*138))
    groups['autocorrelation']=list(range(690,698))
    mdi=[];permutation=[];leaves=[];replays=[];train_high=[]
    for f in range(6):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        for seed in SEEDS:
            folder=RESULTS/('fold%d'%f)/('seed%d'%seed)
            model=joblib.load(folder/'rf.joblib');model.n_jobs=2
            pred=model.predict(x[te]);saved=np.load(folder/'predictions.npz')
            assert np.array_equal(saved['indices'],te)
            err=float(abs(pred-saved['rf']).max());assert err<=1e-12
            replays.append(err);mdi.append(model.feature_importances_)
            base=metrics(y[te],pred,person[te])['MAE']; rng=np.random.RandomState(seed+f)
            # Shuffle jointly inside an anatomical/statistical group, on held-out walks only.
            for repeat in range(3):
                for scope in ['all','within_cohort']:
                    donor=np.arange(len(te))
                    for mask in [np.arange(len(te))] if scope=='all' else [np.flatnonzero(cohort[te]==c) for c in np.unique(cohort[te])]:
                        donor[mask]=rng.permutation(mask)
                    for name,cols in groups.items():
                        xp=x[te].copy();xp[:,cols]=xp[donor][:,cols]
                        value=metrics(y[te],model.predict(xp),person[te])['MAE']-base
                        permutation.append(dict(fold=f,seed=seed,repeat=repeat,scope=scope,group=name,delta_mae=value,people=len(set(person[te]))))
            high=te[y[te]>=2];mass=np.zeros((len(high),4));tw=weights(person[tr])
            # Recover each leaf's actual bootstrap- and person-weighted label composition.
            for tree in model.estimators_:
                boot=np.random.RandomState(tree.random_state).randint(0,len(tr),len(tr))
                wt=np.bincount(boot,minlength=len(tr))*tw
                lid=tree.apply(x[tr]);lookup=np.zeros((tree.tree_.node_count,4))
                for k in range(4):lookup[:,k]=np.bincount(lid,weights=wt*(y[tr]==k),minlength=len(lookup))
                totals=lookup.sum(1);nz=totals>0;lookup[nz]/=totals[nz,None]
                leafmass=lookup[tree.apply(x[high])]
                assert np.max(abs(leafmass@np.arange(4)-tree.predict(x[high])))<1e-10
                mass+=leafmass/len(model.estimators_)
            for j,index in enumerate(high):
                leaves.append(dict(index=int(index),person=str(person[index]),cohort=str(cohort[index]),score=float(y[index]),fold=f,seed=seed,prediction=float(mass[j]@np.arange(4)),label_mass=mass[j].tolist()))
            train3=tr[y[tr]==3]
            train_high.append(dict(fold=f,seed=seed,train_score3_people=len(set(person[train3])),train_score3_walks=len(train3),train_score3_MAE=metrics(y[train3],model.predict(x[train3]),person[train3])['MAE']))
            print('inspected',f,seed,flush=True)
    mdi=np.stack(mdi);names=[f'{stat}:{joint}:R{r}{c}' for stat in ['mean','std','p10','p90','change_rate'] for joint in JOINTS for r in range(3) for c in range(2)]+['ac_lag_'+str(k) for k in [5,10,15,20,25,30,35,40]]
    top=[dict(feature=names[i],importance=float(mdi.mean(0)[i])) for i in np.argsort(-mdi.mean(0))[:30]]
    # Sum person-weighted fold increases, then summarize seeds/repeats.
    agg=[]
    for scope in ['all','within_cohort']:
        for group,cols in groups.items():
            vals=[]
            for seed in SEEDS:
                for repeat in range(3):
                    rows=[r for r in permutation if r['scope']==scope and r['group']==group and r['seed']==seed and r['repeat']==repeat]
                    vals.append(sum(r['delta_mae']*r['people'] for r in rows)/110)
            agg.append(dict(scope=scope,group=group,delta_mae=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),mdi=float(mdi[:,cols].sum(1).mean())))
    write(HERE/'inspection.json',dict(passed=True,max_replay_error=max(replays),permutation=agg,permutation_raw=permutation,top_mdi=top,leaf_composition=leaves,training_high=train_high))

def fit_variant(name,only_fold=None):
    d=load();x=np.load(HERE/'inputs.npz')[name];y=d['y'];fold=d['fold'];person=d['person'];start=time.time()
    for f in range(6):
        if only_fold is not None and f!=only_fold:continue
        tr=np.flatnonzero((fold!=f)&(fold!=(f+1)%6));va=np.flatnonzero(fold==(f+1)%6);outer=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        assert not set(person[outer])&set(person[te])
        for seed in SEEDS:
            path=HERE/'fits'/name/('fold%d_seed%d.npz'%(f,seed))
            if path.exists() and path.with_suffix('.joblib').exists():continue
            choices=[]
            for leaf in [2,8]:
                m=RandomForestRegressor(n_estimators=200,max_features=.5,min_samples_leaf=leaf,random_state=seed,n_jobs=2)
                m.fit(x[tr],y[tr],sample_weight=weights(person[tr]));choices.append((metrics(y[va],m.predict(x[va]),person[va])['MAE'],leaf))
            leaf=min(choices,key=lambda t:t[0])[1]
            m=RandomForestRegressor(n_estimators=200,max_features=.5,min_samples_leaf=leaf,random_state=seed,n_jobs=2)
            m.fit(x[outer],y[outer],sample_weight=weights(person[outer]));pred=m.predict(x[te]);path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,indices=te,prediction=pred,selected_leaf=leaf,inner_choices=np.array(choices),train=outer,inner_train=tr,validation=va)
            joblib.dump(m,path.with_suffix('.joblib'))
            print(name,f,seed,'seconds',round(time.time()-start,1),flush=True)

def summarize():
    d=load();rows=[];allp={};original={}
    for name in ['original','major3d','motion3d','bend3d']:
        for seed in SEEDS:
            p=np.full(len(d['y']),np.nan)
            for f in range(6):
                path=RESULTS/('fold%d'%f)/('seed%d'%seed)/'predictions.npz' if name=='original' else HERE/'fits'/name/('fold%d_seed%d.npz'%(f,seed))
                z=np.load(path);idx=z['indices'];assert np.isnan(p[idx]).all();p[idx]=z['rf' if name=='original' else 'prediction']
            assert np.isfinite(p).all();allp[name,seed]=p
            for group,m in allmetrics(d,p).items():rows.append(dict(variant=name,seed=seed,group=group,**m))
    agg=[];paired=[]
    for name in ['original','major3d','motion3d','bend3d']:
        for group in ['all','high','score3']:
            r=[v for v in rows if v['variant']==name and v['group']==group]
            agg.append(dict(variant=name,group=group,MAE=float(np.mean([v['MAE'] for v in r])),bias=float(np.mean([v['bias'] for v in r])),large_error=float(np.mean([v['large_error'] for v in r])),mae_min=min(v['MAE'] for v in r),mae_max=max(v['MAE'] for v in r)))
            if name!='original':
                mask=np.ones(len(d['y']),bool) if group=='all' else d['y']>=2 if group=='high' else d['y']==3
                people=np.unique(d['person'][mask]);diff=[]
                for per in people:
                    ids=mask&(d['person']==per)
                    diff.append(np.mean([np.mean(abs(allp[name,s][ids]-d['y'][ids])-abs(allp['original',s][ids]-d['y'][ids])) for s in SEEDS]))
                rng=np.random.default_rng(20261031);diff=np.array(diff);boot=diff[rng.integers(len(diff),size=(5000,len(diff)))].mean(1)
                paired.append(dict(variant=name,group=group,delta_MAE=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    # Cohort-only mean baseline (diagnostic only, not a deployable gait model).
    cp=np.zeros(len(d['y']))
    for f in range(6):
        for c in np.unique(d['cohort']):
            tr=(d['fold']!=f)&(d['cohort']==c);te=(d['fold']==f)&(d['cohort']==c)
            cp[te]=np.average(d['y'][tr],weights=weights(d['person'][tr]))
    write(HERE/'comparison.json',dict(passed=True,seed_metrics=rows,aggregate=agg,paired=paired,cohort_only=allmetrics(d,cp)))
    print(json.dumps(agg),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('--fold',type=int);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='inspect':inspect()
    elif a.mode=='summarize':summarize()
    else:fit_variant(a.mode,a.fold)
