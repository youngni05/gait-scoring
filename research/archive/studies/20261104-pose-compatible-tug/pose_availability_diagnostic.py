"""Post hoc measurement-availability diagnostic; never a proposed camera input."""
from review_pose_feature_tug import audit_case,independent_metric
from pose_feature_tug import *
R=H/'availability-diagnostic';O=B/'availability-diagnostic';C='timing_plus_availability';NAMES=['speed','stride_time','swing_fraction','bilateral_features_available','repeat_features_available']
def main():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);original=np.load(DATA);oldnames=read(G/'protocol.json')['feature_names'];X=np.column_stack([original['F'][:,[oldnames.index(n) for n in NAMES[:3]]],np.isfinite(original['F'][:,oldnames.index('bilateral_asymmetry_stride_time')]).astype(float),np.isfinite(original['F'][:,oldnames.index('within_foot_relative_std_stride_time')]).astype(float)]);assert np.isfinite(X).all() and X.shape==(594,5) and X[:,-2].sum()==351 and X[:,-1].sum()==198;d={k:original[k] for k in ['ids','y','fold']};d['F']=X;parent_v=read(H/'verification.json');assert parent_v['status']=='passed';proto=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha256=sha(Path(__file__)),parent_verification_sha256=sha(H/'verification.json'),source_data_sha256=sha(DATA),independent_audit_library_sha256=sha(H/'review_pose_feature_tug.py'),source_model_library_sha256=sha(H/'pose_feature_tug.py'),feature_sets={C:NAMES},candidates={C:configs_for(5)},feature_matrix_sha256=hashlib.sha256(X.tobytes()).hexdigest(),purpose='Posthocdiagnosticaftermainresult/availabilitysubgroups. Add onlytwo binary availabilityflagstocomplete3physicalfeatures;do notincludeactualasymmetry/variationvalues. Distinguish usefulnessofsourceIMUdetectoravailabilityfromtheproposedposequantities. Flagsareextractor-specificmetadata,notrecommendedcamerahealthfeatures. Keepprimaryfixedcomparisonunchanged.',primary='SamefixednonnegativeRidgealpha10stackoninnerOOF,tail1;reporttail4andinnerMAEselected/top3asdescriptivesecondaries. Same594/outer5/inner3. No new outer-based hyperparameter/model selection.',expected_new_base_models=140,expected_meta_models=10)
    if (R/'protocol.json').exists():
        p=read(R/'protocol.json');proto['created_utc']=p['created_utc'];assert proto==p
    else:save(R/'protocol.json',proto)
    np.savez_compressed(O/'features.npz',X=X,ids=d['ids'],y=d['y'],fold=d['fold']);y=d['y'];ids=d['ids'];cfgs=proto['candidates'][C]
    for f in range(5):
        output=O/f'{C}-{f}.json'
        if output.exists():continue
        tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f);A=np.full((len(tr),len(cfgs)),np.nan);T=np.full((len(te),len(cfgs)),np.nan);inner=[];models=[];splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])))
        for ci,cfg in enumerate(cfgs):
            for k,(it,iv) in enumerate(splits):
                m=fit2(X[tr[it]],y[tr[it]],cfg);path=O/f'{C}-{f}-candidate{ci}-inner{k}.joblib';joblib.dump(m,path);p=predict2(m,X[tr[iv]]);A[iv,ci]=p;inner.append(dict(candidate=ci,inner=k,path=str(path),sha256=sha(path),train_ids=ids[tr[it]].tolist(),test_ids=ids[tr[iv]].tolist(),prediction=p.tolist()))
            m=fit2(X[tr],y[tr],cfg);path=O/f'{C}-{f}-candidate{ci}-final.joblib';joblib.dump(m,path);p=predict2(m,X[te]);T[:,ci]=p;models.append(dict(candidate=ci,path=str(path),sha256=sha(path),prediction=p.tolist()));print('fitted availability diagnostic',f,ci,cfg['kind'],flush=True)
        score=np.mean(np.abs(A-y[tr,None]),axis=0);top=np.argsort(score,kind='stable')[:3];pred=dict(selected_inner_MAE=T[:,top[0]].tolist(),ensemble3_inner_MAE=T[:,top].mean(1).tolist());metas=[]
        for kind,tail in [('stacked',1),('tail_stacked',4)]:
            m=fit_meta(A,y[tr],tail);path=O/f'{C}-{f}-{kind}.joblib';joblib.dump(m,path);pred[kind]=predict_meta(m,T).tolist();metas.append(dict(kind=kind,tail=tail,path=str(path),sha256=sha(path)))
        ctx=O/f'{C}-{f}-context.npz';np.savez_compressed(ctx,Xtrain=X[tr],Xtest=X[te],ytrain=y[tr],train_ids=ids[tr],test_ids=ids[te],A=A,T=T);save(output,dict(condition=C,fold=f,features=NAMES,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),context=str(ctx),context_sha256=sha(ctx),inner=inner,models=models,reference=None,inner_MAE=score.tolist(),top3=top.tolist(),meta_models=metas,predictions=pred,protocol_sha256=sha(R/'protocol.json')))
    if not (R/'run-receipt.json').exists():save(R/'run-receipt.json',dict(status='complete_pending_independent_review',new_base_models=140,meta_models=10,protocol_sha256=sha(R/'protocol.json'),hashes={str(p):sha(p) for p in O.glob('*') if p.is_file()}))
    receipt=read(R/'run-receipt.json');assert receipt['protocol_sha256']==sha(R/'protocol.json')
    for path,value in receipt['hashes'].items():assert sha(path)==value
    pp={k:np.full(594,np.nan) for k in ['selected_inner_MAE','ensemble3_inner_MAE','stacked','tail_stacked']};checks=[]
    for f in range(5):
        values,check=audit_case(proto,R,O,C,f,d,NAMES);checks.append(check);te=np.flatnonzero(d['fold']==f)
        for kind,p in values.items():pp[kind][te]=p
        print('verified availability diagnostic',f,flush=True)
    assert sum(c['new_base_models_replayed'] for c in checks)==140 and sum(c['meta_models_replayed'] for c in checks)==10;out={};held=[]
    for kind,p in pp.items():
        assert np.isfinite(p).all();out[kind]=independent_metric(y,p);held.extend(dict(id=int(i),kind=kind,actual=float(a),prediction=float(v)) for i,a,v in zip(ids,y,p))
    assert parent_v['heldout_sha256']==sha(B/'heldout.json');mainrows=read(B/'heldout.json');rng=np.random.default_rng(20261104);boot=rng.integers(0,594,(5000,594));contrast={};newerror=np.abs(pp['stacked']-y)
    for c in ['speed','timing_core','timing','calibrated_spatiotemporal']:
        lookup={int(r['id']):r['prediction'] for r in mainrows if r['condition']==c and r['kind']=='stacked'};assert set(lookup)==set(ids);ref=np.array([lookup[int(i)] for i in ids]);difference=newerror-np.abs(ref-y);contrast[c]=dict(diagnostic_mae_minus_reference=float(difference.mean()),paired_ci95=np.quantile(difference[boot].mean(1),[.025,.975]).tolist())
    save(O/'heldout.json',held);save(R/'comparison.json',dict(status='posthoc_diagnostic_not_primary_or_camera_validation',diagnostic=out,contrasts=contrast,interpretation='Similarerrorsfromavailabilityflagsalonewouldmakegait-quantityinterpretationuncertain. These flags belong to sourceIMUevent detection and may not transfer to MediaPipe. No diagnostic result changes the original six-arm results,features or primarymodel. Conditionalbootstrapdoesnotadjustforposthocsearch.'));save(R/'verification.json',dict(status='passed',people=594,new_base_models_replayed=140,meta_models_replayed=10,checks=checks,comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json'),protocol_sha256=sha(R/'protocol.json'),review_code_sha256=sha(Path(__file__))));print(json.dumps(read(R/'comparison.json'),indent=2))
if __name__=='__main__':main()
