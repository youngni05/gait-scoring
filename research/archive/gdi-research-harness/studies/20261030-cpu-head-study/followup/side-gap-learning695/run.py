"""Nested CPU side-gap learning, preserving the previous mean-score prediction."""
import os,sys,csv,json,time,warnings,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE.parents[1]))
import cpu_head as m
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

PARENT=HERE.parent/'expanded695-pls-study';PLS=HERE.parent/'expanded695-pls64-study'
BLEND=HERE.parent/'pls-gpu-equal-blend695';GPU=m.ROOT/'studies/20261030-gpu-motionbert-study'
spec=importlib.util.spec_from_file_location('parent_metrics',PARENT/'run.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig')
CFG=m.load_json(HERE/'protocol.json');METHODS=CFG['methods']
FAMILIES=['mb_gap_ridge','mb_gap_pls','leg_gap_ridge','combined_gap_ridge']


def js(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def table(name,rows):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def raw_features(x):
    xy=x[...,:2].astype(float);root=xy[:,:,0:1];all_s=[];valid_f=[];names=[]
    assert (x[:,:,1:7,2]>0).all()
    for side,joints in [('L',[4,5,6]),('R',[1,2,3])]:
        hip,knee,ankle=[xy[:,:,j] for j in joints]
        a,b=hip-knee,ankle-knee
        thigh=np.linalg.norm(a,axis=-1);shank=np.linalg.norm(b,axis=-1)
        valid=(thigh*shank)>1e-12
        cosine=np.divide((a*b).sum(-1),thigh*shank,out=np.zeros_like(thigh),where=valid)
        flex=180-np.degrees(np.arccos(np.clip(cosine,-1,1)));flex[~valid]=np.nan
        assert valid.any(1).all()
        signals=np.concatenate([(xy[:,:,joints]-root).reshape(len(x),124,6),flex[:,:,None],thigh[:,:,None],shank[:,:,None]],axis=-1)
        stats=np.concatenate([np.nanpercentile(signals,[10,25,50,75,90],axis=1).transpose(1,2,0),np.nanstd(signals,axis=1)[:,:,None]],axis=-1)
        all_s.append(stats.reshape(len(x),54));valid_f.append(valid.mean(1))
        names.extend([f'{side}_{signal}_{stat}' for signal in ['hip_x','hip_y','knee_x','knee_y','ankle_x','ankle_y','knee_flexion','thigh_length','shank_length'] for stat in ['p10','p25','p50','p75','p90','sd']])
    result=np.concatenate([all_s[0],all_s[1],all_s[0]-all_s[1],np.stack(valid_f,axis=1)],axis=1)
    names+=['LminusR_'+n[2:] for n in names[:54]]+['L_knee_valid_fraction','R_knee_valid_fraction']
    assert result.shape==(1064,164) and np.isfinite(result).all()
    return result,names,valid_f


def predict(x,model):
    return ((x-model['mean'])/model['scale'])@model['coef']+model['intercept']


def fit_path(x,y,kind):
    scaler=StandardScaler().fit(x);z=scaler.transform(x)
    out={}
    for name,value in [('zero',0.),('training_record_mean',float(y.mean()))]:
        out[name]=dict(mean=scaler.mean_,scale=scaler.scale_,coef=np.zeros(x.shape[1]),intercept=np.array(value))
    if kind=='ridge':
        xm=z.mean(0);ym=y.mean();u,s,vt=np.linalg.svd(z-xm,full_matrices=False);uy=u.T@(y-ym)
        for alpha in CFG['ridge_alphas']:
            coef=vt.T@((s/(s*s+alpha))*uy)
            out[f'alpha_{alpha:g}']=dict(mean=scaler.mean_,scale=scaler.scale_,coef=coef,intercept=np.array(ym-xm@coef))
    else:
        for k in CFG['pls_components']:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model=PLSRegression(n_components=k,scale=False,max_iter=500,tol=1e-6).fit(z,y)
            if caught:raise RuntimeError([str(w.message) for w in caught])
            coef=model.coef_.reshape(-1);intercept=float(model.intercept_[0]-model._x_mean@coef)
            out[f'k_{k}']=dict(mean=scaler.mean_,scale=scaler.scale_,coef=coef,intercept=np.array(intercept))
            np.testing.assert_allclose(predict(x,out[f'k_{k}']),model.predict(z).reshape(-1),rtol=0,atol=1e-9)
    return out


def person_mean(a,ids):
    return np.array([a[ids==p].mean() for p in np.unique(ids)])


def combine(mean,gap):return np.stack([mean+gap/2,mean-gap/2],axis=-1)


def metrics(y,pred,ids):
    truthgap=y[:,0]-y[:,1];gap=pred[:,0]-pred[:,1]
    a=person_mean(truthgap,ids);b=person_mean(gap,ids)
    cor=float(np.corrcoef(a,b)[0,1]) if np.ptp(b)>1e-12 and np.ptp(a)>1e-12 else None
    direction=[]
    for pid in np.unique(ids):
        use=(ids==pid)&(abs(truthgap)>1e-12)
        if use.any():direction.append(float((np.sign(gap[use])==np.sign(truthgap[use])).mean()))
    return dict(people=len(a),overall_MAE=float(person_mean(abs(pred-y),ids).mean()),
                gap_MAE=float(person_mean(abs(gap-truthgap),ids).mean()),
                predicted_gap_magnitude=float(person_mean(abs(gap),ids).mean()),
                actual_gap_magnitude=float(person_mean(abs(truthgap),ids).mean()),
                signed_person_gap_correlation=cor,direction_accuracy=float(np.mean(direction)),
                mean_score_MAE=float(person_mean(abs(pred.mean(1)-y.mean(1)),ids).mean()))


def main():
    assert not (HERE/'summary.json').exists(),'preserve completed experiment'
    (HERE/'models').mkdir(exist_ok=True)
    source_paths=[HERE/'protocol.json',GPU/'expansion695/train_only.npz',m.ROOT/'studies/20261027-gdi-linkage/development_linkage.json',
                  m.ROOT/'studies/20261027-gdi-linkage/audit_labels.py',m.WORKSPACE/'tools/stanford-sample-check/run_ridge_experiment.py']
    for folder in [PARENT,PLS,BLEND]:
        for name,digest in m.load_json(folder/'artifact_manifest.json').items():
            path=folder/name;assert m.digest(path)==digest,path;source_paths.append(path)
    hashes={p.relative_to(m.WORKSPACE).as_posix():m.digest(p) for p in set(source_paths)};js('source_hashes.json',hashes)
    d=load(PARENT/'development695.npz');g=load(GPU/'expansion695/train_only.npz')
    for key in ['person_id','video_id','fold','y']:np.testing.assert_array_equal(d[key],g[key])
    label_rows=m.load_json(m.ROOT/'studies/20261027-gdi-linkage/development_linkage.json')['records']
    byvideo={row['video_id']:row for row in label_rows}
    matched=0
    for i in np.flatnonzero(d['source_group']=='additional520'):
        row=byvideo[str(d['video_id'][i])];assert row['patient_id']==str(d['person_id'][i])
        np.testing.assert_array_equal(d['y'][i],row['GDI_LR']);matched+=1
    feat,names,valid=raw_features(g['x'][:,0]);np.savez_compressed(HERE/'leg_features.npz',X=feat,names=np.array(names),person_id=d['person_id'],video_id=d['video_id'])
    js('input_audit.json',dict(additional_labels_checked=matched,label_order=['L','R'],
        cached_arrays_identical=True,raw_feature_dimensions=164,knee_invalid_frames=[int(((1-v)*124).round().sum()) for v in valid],
        virtual_root='midpoint of hips',anatomical_left_right_from_stored_mapping=True,
        raw_video_anatomical_side_verified=False,source_label_clinical_reliability_verified=False,
        label_audit_note='Cached new labels match recorded L/R linkage; old builder explicitly uses gdi[L],gdi[R]. Raw clinical table and video not re-audited.'))
    Xs={'mb_gap_ridge':d['X'],'mb_gap_pls':d['X'],'leg_gap_ridge':feat,'combined_gap_ridge':np.concatenate([d['X'],feat],axis=1)}
    basecube=load(PLS/'fold_predictions.npz')['predictions'];cube=np.empty((len(METHODS),5,1064,2));cube[0]=basecube
    candidates=[];selections=[];trainheld=[];audits=[];start=time.perf_counter()
    with threadpool_limits(limits=2):
        for fold in range(5):
            tr=np.flatnonzero(d['fold']!=fold);te=np.flatnonzero(d['fold']==fold)
            y=d['y'][tr];ids=d['person_id'][tr];target=y[:,0]-y[:,1]
            prev=load(PLS/f'models/fold{fold}_inner.npz');np.testing.assert_array_equal(tr,prev['outer_train_indices'])
            assignment=prev['inner_assignment'];base_inner=prev['inner_oof_predictions'][CFG['pls_components'].index(32)]
            # The source uses the same six counts; the selected joint-PLS mean is held fixed.
            innermean=base_inner.mean(1);mean=basecube[fold].mean(1)
            base_inner_mae=float(person_mean(abs(base_inner-y),ids).mean())
            cube[1,fold]=combine(mean,np.zeros(1064));cube[2,fold]=combine(mean,np.full(1064,target.mean()))
            pools={};inner_saved=dict(outer_train_indices=tr,inner_assignment=assignment,base_inner_predictions=base_inner)
            fold_candidates=[]
            for family in FAMILIES:
                pools[family]={}
                for inner in range(4):
                    ti=np.flatnonzero(assignment!=inner);vi=np.flatnonzero(assignment==inner)
                    assert len(set(ids[ti]))==417 and len(set(ids[vi]))==139 and not set(ids[ti])&set(ids[vi])
                    path=fit_path(Xs[family][tr[ti]],target[ti],'pls' if family=='mb_gap_pls' else 'ridge')
                    for key,model in path.items():
                        pools[family].setdefault(key,np.full(len(tr),np.nan))[vi]=predict(Xs[family][tr[vi]],model)
                    audits.append(dict(outer_fold=fold,family=family,inner_fold=inner,train_people=417,validation_people=139,overlap=0))
                for key,delta in pools[family].items():
                    assert np.isfinite(delta).all()
                    metric=metrics(y,combine(innermean,delta),ids)
                    row=dict(outer_fold=fold,family=family,candidate=key,inner_gap_MAE=metric['gap_MAE'],inner_overall_MAE=metric['overall_MAE'])
                    candidates.append(row);fold_candidates.append(row);inner_saved[family+'__'+key]=delta
                best=min([r for r in fold_candidates if r['family']==family],key=lambda q:q['inner_gap_MAE'])
                selected=best['candidate'];full=fit_path(Xs[family][tr],target,'pls' if family=='mb_gap_pls' else 'ridge')[selected]
                np.savez_compressed(HERE/f'models/fold{fold}_{family}.npz',**full,train_indices=tr,evaluation_indices=te)
                cube[METHODS.index(family),fold]=combine(mean,predict(Xs[family],full))
                selections.append(dict(outer_fold=fold,method=family,selected_family=family,candidate=selected,
                    inner_gap_MAE=best['inner_gap_MAE'],inner_overall_MAE=best['inner_overall_MAE'],baseline_inner_MAE=base_inner_mae))
            allowed=[dict(family='joint_pls32',candidate='unchanged',inner_gap_MAE=metrics(y,base_inner,ids)['gap_MAE'],inner_overall_MAE=base_inner_mae)]
            allowed += [q for q in fold_candidates if q['inner_overall_MAE']<=base_inner_mae+1e-12]
            guard=min(allowed,key=lambda q:q['inner_gap_MAE'])
            if guard['family']=='joint_pls32':cube[-1,fold]=basecube[fold]
            else:
                family=guard['family'];full=fit_path(Xs[family][tr],target,'pls' if family=='mb_gap_pls' else 'ridge')[guard['candidate']]
                np.savez_compressed(HERE/f'models/fold{fold}_guarded_gap.npz',**full,train_indices=tr,evaluation_indices=te)
                cube[-1,fold]=combine(mean,predict(Xs[family],full))
            selections.append(dict(outer_fold=fold,method='guarded_gap',selected_family=guard['family'],candidate=guard['candidate'],
                inner_gap_MAE=guard['inner_gap_MAE'],inner_overall_MAE=guard['inner_overall_MAE'],baseline_inner_MAE=base_inner_mae))
            np.savez_compressed(HERE/f'models/fold{fold}_inner_predictions.npz',**inner_saved)
            for mi,name in enumerate(METHODS):
                for split,idx in [('train',tr),('held',te)]:
                    trainheld.append(dict(fold=fold,method=name,split=split,**metrics(d['y'][idx],cube[mi,fold,idx],d['person_id'][idx])))
            print(json.dumps({'fold':fold,'selected':selections[-5:],'seconds':round(time.perf_counter()-start,2)}),flush=True)
    oof=np.stack([c[d['fold'],np.arange(1064)] for c in cube])
    reference=load(BLEND/'oof_predictions.npz');np.testing.assert_array_equal(oof[0],reference['predictions'][1])
    for pred in oof[1:]:np.testing.assert_allclose(pred.mean(1),oof[0].mean(1),rtol=0,atol=2e-14)
    np.savez_compressed(HERE/'fold_predictions.npz',predictions=cube,methods=np.array(METHODS))
    np.savez_compressed(HERE/'oof_predictions.npz',predictions=oof,methods=np.array(METHODS),**{k:d[k] for k in ['y','person_id','video_id','fold','source_group']})
    overall=[];bands=[];people=np.unique(d['person_id']);errors=[];gaperrors=[]
    for mi,name in enumerate(METHODS):
        pred=oof[mi];errors.append(person_mean(abs(pred-d['y']),d['person_id']))
        gaperrors.append(person_mean(abs((pred[:,0]-pred[:,1])-(d['y'][:,0]-d['y'][:,1])),d['person_id']))
        for group in ['all695','original175','additional520']:
            use=np.ones(1064,dtype=bool) if group=='all695' else d['source_group']==group
            overall.append(dict(method=name,group=group,**{**metrics(d['y'][use],pred[use],d['person_id'][use]),
                **parent.evaluate(d,pred,use)}))
        for band in np.unique(np.floor(d['y']/10).astype(int)):
            bands.append(dict(method=name,actual_bin_low=int(band*10),**parent.evaluate(d,pred,band=band)))
    draws=np.random.default_rng(20261029).integers(0,695,(10000,695));comparisons=[]
    for i,name in enumerate(METHODS[1:],1):
        for metric,arr in [('overall_MAE',errors),('gap_MAE',gaperrors)]:
            delta=arr[0]-arr[i]
            comparisons.append(dict(method=name,metric=metric,reduction=float(delta.mean()),bootstrap95=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist(),
                people_improved=int((delta>1e-12).sum()),people_worsened=int((delta < -1e-12).sum())))
    for name,rows in [('metrics.csv',overall),('score_bands.csv',bands),('train_held_metrics.csv',trainheld),
                      ('inner_candidate_scores.csv',candidates),('selections.csv',selections),('split_audit.csv',audits)]:table(name,rows)
    for path,digest in hashes.items():assert m.digest(m.WORKSPACE/path)==digest,path
    js('summary.json',dict(methods=METHODS,metrics=overall,score_bands=bands,selections=selections,comparisons=comparisons,
        train_held=trainheld,seconds=time.perf_counter()-start,source_files_preserved=len(hashes),GPU_used=False,
        unchanged_mean_prediction=True,protected_numeric_data_used=False,independent_final_evaluation=False))
    print(json.dumps({'all695':[row for row in overall if row['group']=='all695'],'seconds':time.perf_counter()-start},indent=2))


if __name__=='__main__':main()
