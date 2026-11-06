"""Compare all retrained models using fixed held-out people and MoVi clips."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;VIDEO=BASE/'video-stability/straight-followup'
sys.path.insert(0,str(BASE/'feature-analysis'));import study
import numpy as np
import joblib
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def metric(y,p,person):
    r=study.metrics(y,p,person);w=study.weights(person);mean=np.average(y,weights=w)
    r.update(score1_fraction=float(np.average((p>=.5)&(p<1.5),weights=w)),rounded_accuracy=float(np.average(np.floor(p+.5)==y,weights=w)),r2=float(1-np.average((p-y)**2,weights=w)/np.average((y-mean)**2,weights=w)) if np.ptp(y)>0 else None)
    return r
def clinical(d,p):
    out={}
    for name,mask in [('all',np.ones(len(d['y']),bool)),('high',d['y']>=2),('score3',d['y']==3)]:
        rows=[metric(d['y'][mask],v[mask],d['person'][mask]) for v in p]
        out[name]=dict(people=rows[0]['people'],records=rows[0]['walks'],**{k:float(np.mean([r[k] for r in rows])) for k in ['MAE','bias','large_error','score1_fraction','rounded_accuracy']},seeds=rows)
    return out
def view(a):
    delta=a[:,1]-a[:,0]
    return dict(mean_score=float(a.mean()),score1_fraction=float(((a>=.5)&(a<1.5)).mean()),view_abs=float(abs(delta).mean()),view_bias=float(delta.mean()),person_spread=float(a.mean(1).std(0).mean()),by_seed=[dict(seed=seed,view_abs=float(abs(delta[:,j::5]).mean()),view_bias=float(delta[:,j::5].mean()),score1_fraction=float(((a[:,:,j::5]>=.5)&(a[:,:,j::5]<1.5)).mean())) for j,seed in enumerate(study.SEEDS)])
def main():
    complete=json.loads((HERE/'received/main/complete.json').read_text());assert complete['passed'] and complete['final_models']==90
    z=np.load(HERE/'input.npz');d={k:z[k] for k in ['y','person','fold','cohort','indices']};old=np.load(VIDEO/'care-predictions.npz');movi=np.load(VIDEO/'movi-predictions.npz');clips=json.loads((VIDEO/'clips.json').read_text());subjects=movi['subjects'].tolist();index={k:i for i,k in enumerate(movi['keys'].tolist())}
    cp={'original_full':old['full'][:,d['indices']],'original_first2':old['first2'][:,d['indices']],'original_center2':old['center2'][:,d['indices']]};mp={'original':movi['predictions']};checks=[]
    for name in ['full','first2','center2']:
        c=np.full((5,len(d['y'])),np.nan);v=[]
        for f in range(6):
            for j,seed in enumerate(study.SEEDS):
                path=HERE/'received/main/fits'/name/('fold%d_seed%d'%(f,seed));a=np.load(path.with_suffix('.npz'));model=joblib.load(path.with_suffix('.joblib'));model.n_jobs=1
                assert np.isnan(c[j,a['test']]).all();c[j,a['test']]=model.predict(z[name][a['test']]);err=float(abs(c[j,a['test']]-a['prediction']).max());assert err<=1e-12;checks.append(err);v.append(model.predict(movi['features']))
        assert np.isfinite(c).all();cp['new_'+name]=c;mp['new_'+name]=np.array(v)
    for key in ['mean','median','mode','cohort_mean']:
        p=np.zeros(len(d['y']))
        for f in range(6):
            tr=d['fold']!=f;te=d['fold']==f;y=d['y'][tr];w=study.weights(d['person'][tr]);mass=np.array([w[y==k].sum() for k in range(4)])
            if key=='cohort_mean':
                for cohort in np.unique(d['cohort']):
                    train=tr&(d['cohort']==cohort);test=te&(d['cohort']==cohort);p[test]=np.average(d['y'][train],weights=study.weights(d['person'][train]))
            else:p[te]=np.average(y,weights=w) if key=='mean' else int(np.searchsorted(np.cumsum(mass),mass.sum()/2)) if key=='median' else int(mass.argmax())
        cp['baseline_'+key]=np.tile(p,(5,1))
    cm={k:clinical(d,p) for k,p in cp.items()};vm={};arrays={}
    for name,p in mp.items():
        vm[name]={}
        for condition in ['straight','turn']:
            a=np.array([[p[:,index['s%02d_%s_%s'%(sid,cam,condition)]] for cam in ['PG1','PG2']] for sid in subjects]);arrays[name+'_'+condition]=a;vm[name][condition]=view(a)
    paired=[];rng=np.random.RandomState(20261031)
    for new,oldname in [('new_first2','original_first2'),('new_center2','original_center2'),('new_first2','original_full'),('new_center2','original_full'),('new_first2','new_full'),('new_center2','new_full')]:
        for group,mask in [('all',np.ones(len(d['y']),bool)),('high',d['y']>=2),('score3',d['y']==3)]:
            difference=abs(cp[new]-d['y'][None])-abs(cp[oldname]-d['y'][None]);people=np.unique(d['person'][mask]);diff=np.array([difference[:,mask&(d['person']==s)].mean() for s in people]);boot=diff[rng.randint(0,len(diff),(10000,len(diff)))].mean(1)
            paired.append(dict(new=new,reference=oldname,group=group,delta_MAE=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),seeds_improved=sum(a['MAE']<b['MAE'] for a,b in zip(cm[new][group]['seeds'],cm[oldname][group]['seeds']))))
    vp=[]
    for name in ['new_full','new_first2','new_center2']:
        for condition in ['straight','turn']:
            a=arrays[name+'_'+condition];b=arrays['original_'+condition];diff=abs(a[:,1]-a[:,0]).mean(1)-abs(b[:,1]-b[:,0]).mean(1);boot=diff[rng.randint(0,14,(10000,14))].mean(1)
            vp.append(dict(model=name,condition=condition,delta_view_abs=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),people_improved=int((diff<0).sum()),seeds_improved=sum(a['view_abs']<b['view_abs'] for a,b in zip(vm[name][condition]['by_seed'],vm['original'][condition]['by_seed']))))
    save('analysis.json',dict(passed=True,clinical=cm,movi=vm,clinical_paired=paired,movi_paired=vp,replay_max_error=max(checks),scope='Held-out person-equal CARE error; MoVi has no score labels. CIs descriptive paired-person bootstrap, not all model-selection uncertainty.',sources={str(p):sha(p) for p in [HERE/'input.npz',HERE/'received/main/complete.json',Path(__file__),VIDEO/'movi-predictions.npz',VIDEO/'care-predictions.npz']}))
    np.savez_compressed(HERE/'predictions.npz',**cp,**{'movi_'+k:v for k,v in mp.items()},**arrays,y=d['y'],person=d['person'],fold=d['fold'],indices=d['indices'])
    print(json.dumps(dict(clinical={k:v['all']['MAE'] for k,v in cm.items()},movi={k:v['straight'] for k,v in vm.items()})),flush=True)
if __name__=='__main__':main()
