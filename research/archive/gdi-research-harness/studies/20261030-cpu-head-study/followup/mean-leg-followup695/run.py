"""CPU mean-score, feature ablation and temporal-summary experiments."""
import sys, json, csv, time, importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'side-gap-learning695'
spec=importlib.util.spec_from_file_location('side_study',PREV/'run.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
np,m=p.np,p.m
CFG=m.load_json(HERE/'protocol.json')
TASKS={}
for family in ['mb','leg','combined']:
    for kind in ['ridge','pls']:TASKS[f'mean_{family}_{kind}']=(family,'mean',kind)
for family in ['position','angle','length','no_length','leg_temporal','combined_temporal']:
    for target in ['mean','gap']:TASKS[f'{target}_{family}_ridge']=(family,target,'ridge')
METHODS=['joint_pls32','previous_leg_gap']+list(TASKS)+['joint_leg_temporal','joint_combined_temporal','inner_selected_mean']

def js(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def table(name,rows):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def temporal_features(x):
    xy=x[...,:2].astype(float);signals=[]
    for joints in [[4,5,6],[1,2,3]]:
        a=xy[:,:,joints[0]]-xy[:,:,joints[1]];b=xy[:,:,joints[2]]-xy[:,:,joints[1]]
        denom=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        assert (denom>1e-12).all()
        angle=180-np.degrees(np.arccos(np.clip((a*b).sum(-1)/denom,-1,1)))
        signals.append(np.concatenate([(xy[:,:,joints]-xy[:,:,0:1]).reshape(len(x),124,6),angle[:,:,None]],axis=2))
    stats=[]
    for s in signals:
        diff=np.diff(s,axis=1)
        stats.append(np.stack([abs(diff).mean(1),diff.std(1),abs(np.diff(diff,axis=1)).mean(1)],axis=-1).reshape(len(x),21))
    corrs=[]
    for lag in range(-15,16):
        a=signals[0][:,max(0,lag):min(124,124+lag)]
        b=signals[1][:,max(0,-lag):min(124,124-lag)]
        a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
        den=np.sqrt((a*a).sum(1)*(b*b).sum(1))
        corrs.append(np.divide((a*b).sum(1),den,out=np.zeros_like(den),where=den>1e-12))
    c=np.stack(corrs,axis=1);lags=(c.argmax(1)-15)/15
    z=np.concatenate([stats[0],stats[1],stats[0]-stats[1],c[:,15],c.max(1),lags[:,[2,4,6]]],axis=1)
    assert z.shape==(1064,80) and np.isfinite(z).all()
    return z

def features(d):
    saved=p.load(PREV/'leg_features.npz');leg=saved['X'];names=saved['names']
    for key in ['person_id','video_id']:np.testing.assert_array_equal(d[key],saved[key])
    position=np.array([any(t in n for t in ['hip_x','hip_y','knee_x','knee_y','ankle_x','ankle_y']) for n in names])
    angle=np.array(['knee_flexion' in n for n in names]);length=np.array(['length' in n for n in names])
    g=p.load(p.GPU/'expansion695/train_only.npz')
    for key in ['person_id','video_id','fold','y']:np.testing.assert_array_equal(d[key],g[key])
    temporal=temporal_features(g['x'][:,0])
    return dict(mb=d['X'],leg=leg,combined=np.concatenate([d['X'],leg],1),
        position=leg[:,position],angle=leg[:,angle],length=leg[:,length],no_length=leg[:,~length],
        leg_temporal=np.concatenate([leg,temporal],1),combined_temporal=np.concatenate([d['X'],leg,temporal],1)),temporal

def target(y,which):return y.mean(1) if which=='mean' else y[:,0]-y[:,1]

def target_mae(y,pred,ids):return float(p.person_mean(abs(y-pred),ids).mean())

def reconstruct(name,value,base,old):
    which=TASKS[name][1]
    return p.combine(value,old[:,0]-old[:,1]) if which=='mean' else p.combine(base.mean(1),value)

def main():
    assert not (HERE/'summary.json').exists(),'do not overwrite completed study'
    (HERE/'models').mkdir(exist_ok=True)
    sources={}
    for folder in [PREV,p.PARENT,p.PLS]:
        for name,digest in m.load_json(folder/'artifact_manifest.json').items():
            path=folder/name;assert m.digest(path)==digest,path
            sources[path.relative_to(m.WORKSPACE).as_posix()]=digest
    for path in [p.GPU/'expansion695/train_only.npz',HERE/'protocol.json',HERE/'run.py']:
        sources[path.relative_to(m.WORKSPACE).as_posix()]=m.digest(path)
    js('source_hashes.json',sources)
    d=p.load(p.PARENT/'development695.npz');xs,temp=features(d)
    np.savez_compressed(HERE/'temporal_features.npz',X=temp,person_id=d['person_id'],video_id=d['video_id'])
    js('feature_dimensions.json',{k:v.shape[1] for k,v in xs.items()})
    prev=p.load(PREV/'fold_predictions.npz');base=prev['predictions'][0];old=prev['predictions'][list(prev['methods']).index('leg_gap_ridge')]
    cube=np.full((len(METHODS),5,1064,2),np.nan);cube[0]=base;cube[1]=old
    choices=[];scores=[];splits=[];trainheld=[];start=time.perf_counter()
    with p.threadpool_limits(limits=2):
        for f in range(5):
            tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f)
            inn=p.load(PREV/f'models/fold{f}_inner_predictions.npz');assignment=inn['inner_assignment']
            np.testing.assert_array_equal(inn['outer_train_indices'],tr)
            ibase=inn['base_inner_predictions'];igap=inn['leg_gap_ridge__alpha_1000'];iold=p.combine(ibase.mean(1),igap)
            inner_saved=dict(train_indices=tr,inner_assignment=assignment,base_predictions=ibase,previous_predictions=iold)
            guard=[dict(task='previous_leg_gap',candidate='unchanged',score=target_mae(d['y'][tr],iold,d['person_id'][tr]))]
            for name,(family,which,kind) in TASKS.items():
                x=xs[family];y=target(d['y'],which);pool={}
                for j in range(4):
                    ti=tr[assignment!=j];vi=tr[assignment==j]
                    assert len(set(d['person_id'][ti]))==417 and len(set(d['person_id'][vi]))==139
                    assert not set(d['person_id'][ti])&set(d['person_id'][vi])
                    path=p.fit_path(x[ti],y[ti],kind)
                    if which=='mean':path.pop('zero')
                    for key,model in path.items():pool.setdefault(key,np.full(len(tr),np.nan))[assignment==j]=p.predict(x[vi],model)
                    splits.append(dict(fold=f,task=name,inner_fold=j,train_people=417,validation_people=139,overlap=0))
                local=[]
                for key,v in pool.items():
                    assert np.isfinite(v).all();inner_saved[name+'__'+key]=v
                    row=dict(fold=f,task=name,candidate=key,target_MAE=target_mae(y[tr],v,d['person_id'][tr]),
                        overall_MAE=target_mae(d['y'][tr],reconstruct(name,v,ibase,iold),d['person_id'][tr]))
                    scores.append(row);local.append(row)
                    if family in ['mb','leg','combined'] and which=='mean':guard.append(dict(task=name,candidate=key,score=row['overall_MAE']))
                best=min(local,key=lambda row:row['target_MAE']);key=best['candidate']
                full=p.fit_path(x[tr],y[tr],kind)[key]
                np.savez_compressed(HERE/f'models/fold{f}_{name}.npz',**full,train_indices=tr,evaluation_indices=te)
                cube[METHODS.index(name),f]=reconstruct(name,p.predict(x,full),base[f],old[f])
                choices.append(dict(fold=f,method=name,task=name,candidate=key,inner_target_MAE=best['target_MAE'],inner_overall_MAE=best['overall_MAE']))
            for family in ['leg_temporal','combined_temporal']:
                mean=cube[METHODS.index('mean_'+family+'_ridge'),f].mean(1)
                q=cube[METHODS.index('gap_'+family+'_ridge'),f]
                cube[METHODS.index('joint_'+family),f]=p.combine(mean,q[:,0]-q[:,1])
            winner=min(guard,key=lambda q:q['score']);name=winner['task'];key=winner['candidate']
            if name=='previous_leg_gap':cube[-1,f]=old[f]
            else:
                family,which,kind=TASKS[name];full=p.fit_path(xs[family][tr],target(d['y'][tr],which),kind)[key]
                np.savez_compressed(HERE/f'models/fold{f}_inner_selected_mean.npz',**full,train_indices=tr,evaluation_indices=te)
                cube[-1,f]=reconstruct(name,p.predict(xs[family],full),base[f],old[f])
            choices.append(dict(fold=f,method='inner_selected_mean',task=name,candidate=key,inner_target_MAE=None,inner_overall_MAE=winner['score']))
            np.savez_compressed(HERE/f'models/fold{f}_inner.npz',**inner_saved)
            for mi,name in enumerate(METHODS):
                for split,idx in [('train',tr),('held',te)]:trainheld.append(dict(fold=f,method=name,split=split,**p.metrics(d['y'][idx],cube[mi,f,idx],d['person_id'][idx])))
            print(json.dumps(dict(fold=f,selected_mean=winner,seconds=round(time.perf_counter()-start,1))),flush=True)
    assert np.isfinite(cube).all()
    oof=np.stack([c[d['fold'],np.arange(1064)] for c in cube])
    np.savez_compressed(HERE/'fold_predictions.npz',predictions=cube,methods=np.array(METHODS))
    np.savez_compressed(HERE/'oof_predictions.npz',predictions=oof,methods=np.array(METHODS),**{k:d[k] for k in ['y','person_id','video_id','fold','source_group']})
    rows=[];bands=[]
    for name,pred in zip(METHODS,oof):
        for group in ['all695','original175','additional520']:
            use=np.ones(1064,dtype=bool) if group=='all695' else d['source_group']==group
            rows.append(dict(method=name,group=group,**{**p.metrics(d['y'][use],pred[use],d['person_id'][use]),**p.parent.evaluate(d,pred,use)}))
        for band in np.unique(np.floor(d['y']/10).astype(int)):bands.append(dict(method=name,actual_bin_low=int(band*10),**p.parent.evaluate(d,pred,band=band)))
    draws=np.random.default_rng(20261029).integers(0,695,(10000,695));comparisons=[]
    for ref in ['joint_pls32','previous_leg_gap']:
        a=oof[METHODS.index(ref)]
        for name,b in zip(METHODS[2:],oof[2:]):
            diff=p.person_mean(abs(a-d['y']),d['person_id'])-p.person_mean(abs(b-d['y']),d['person_id'])
            comparisons.append(dict(reference=ref,method=name,reduction=float(diff.mean()),bootstrap95=np.percentile(diff[draws].mean(1),[2.5,97.5]).tolist()))
    for file,values in [('metrics.csv',rows),('score_bands.csv',bands),('selections.csv',choices),('inner_scores.csv',scores),('split_audit.csv',splits),('train_held.csv',trainheld)]:table(file,values)
    for path,digest in sources.items():assert m.digest(m.WORKSPACE/path)==digest,path
    js('summary.json',dict(metrics=rows,bands=bands,selections=choices,comparisons=comparisons,train_held=trainheld,seconds=time.perf_counter()-start,GPU_used=False,independent_final_evaluation=False))
    print(json.dumps([dict(method=v['method'],MAE=v['overall_MAE'],gap=v['gap_MAE'],mean=v['mean_score_MAE']) for v in rows if v['group']=='all695'],indent=2))

if __name__=='__main__':main()
