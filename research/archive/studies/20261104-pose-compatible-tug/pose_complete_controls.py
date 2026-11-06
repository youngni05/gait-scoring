"""Outcome-blind complete-feature controls for the pose-availability comparison."""
from pose_feature_tug import *
R=H/'complete-feature-controls';O=B/'complete-feature-controls'
SETS={'timing_core':['speed','stride_time','swing_fraction'],'metric_core':['speed','stride_time','swing_fraction','stride_length']}
def main():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);parent=read(H/'protocol.json');d=np.load(DATA);names=read(G/'protocol.json')['feature_names'];assert sha(DATA)==parent['source_data_sha256'];proto=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha256=sha(Path(__file__)),parent_protocol_sha256=sha(H/'protocol.json'),parent_code_sha256=sha(H/'pose_feature_tug.py'),source_data_sha256=sha(DATA),feature_sets=SETS,candidates={c:configs_for(len(v)) for c,v in SETS.items()},reason='Bilateralfeaturesavailable351/594,repeatedcyclefeatures198/594. These additional controls retain only original pose-computable basic quantities available594/594,so no missingness/detectorcoverage features can account for their gains. Declared from feature availability before inspecting new held-out metrics. All original594people/folds preserved.',primary=parent['primary'],secondary=parent['secondary'],verification='Sameindependentall-model/scaler/split/epoch-freeNNLS audit.280additionalbasefits(210inner70final),20metamodels. No new clinical thresholds,feature replacement,pose estimates,or model selection using outer people.')
    for vv in SETS.values():assert np.isfinite(d['F'][:,[names.index(n) for n in vv]]).all()
    if (R/'protocol.json').exists():
        old=read(R/'protocol.json');proto['created_utc']=old['created_utc'];assert old==proto
    else:save(R/'protocol.json',proto)
    for c,columns in SETS.items():
        X=d['F'][:,[names.index(n) for n in columns]];cfgs=proto['candidates'][c]
        for f in range(5):
            output=O/f'{c}-{f}.json'
            if output.exists():continue
            tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f);y=d['y'];ids=d['ids'];A=np.full((len(tr),len(cfgs)),np.nan);T=np.full((len(te),len(cfgs)),np.nan);inner=[];models=[];splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])))
            for ci,cfg in enumerate(cfgs):
                for k,(it,iv) in enumerate(splits):
                    m=fit2(X[tr[it]],y[tr[it]],cfg);path=O/f'{c}-{f}-candidate{ci}-inner{k}.joblib';joblib.dump(m,path);p=predict2(m,X[tr[iv]]);A[iv,ci]=p;inner.append(dict(candidate=ci,inner=k,path=str(path),sha256=sha(path),train_ids=ids[tr[it]].tolist(),test_ids=ids[tr[iv]].tolist(),prediction=p.tolist()))
                m=fit2(X[tr],y[tr],cfg);path=O/f'{c}-{f}-candidate{ci}-final.joblib';joblib.dump(m,path);p=predict2(m,X[te]);T[:,ci]=p;models.append(dict(candidate=ci,path=str(path),sha256=sha(path),prediction=p.tolist()));print('fitted',c,f,ci,cfg['kind'],flush=True)
            assert np.isfinite(A).all() and np.isfinite(T).all();score=np.mean(np.abs(A-y[tr,None]),axis=0);top=np.argsort(score,kind='stable')[:3];pred=dict(selected_inner_MAE=T[:,top[0]].tolist(),ensemble3_inner_MAE=T[:,top].mean(1).tolist());metas=[]
            for kind,tail in [('stacked',1),('tail_stacked',4)]:
                m=fit_meta(A,y[tr],tail);path=O/f'{c}-{f}-{kind}.joblib';joblib.dump(m,path);pred[kind]=predict_meta(m,T).tolist();metas.append(dict(kind=kind,tail=tail,path=str(path),sha256=sha(path)))
            context=O/f'{c}-{f}-context.npz';np.savez_compressed(context,Xtrain=X[tr],Xtest=X[te],ytrain=y[tr],train_ids=ids[tr],test_ids=ids[te],A=A,T=T);save(output,dict(condition=c,fold=f,features=columns,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),context=str(context),context_sha256=sha(context),inner=inner,models=models,reference=None,inner_MAE=score.tolist(),top3=top.tolist(),meta_models=metas,predictions=pred,protocol_sha256=sha(R/'protocol.json')))
    save(R/'run-receipt.json',dict(status='complete_pending_independent_review',cases=10,new_inner_models=210,new_final_models=70,meta_models=20,protocol_sha256=sha(R/'protocol.json'),hashes={str(p):sha(p) for p in O.glob('*') if p.is_file()}));print('Complete-feature controls saved; independent review required',flush=True)
if __name__=='__main__':main()
