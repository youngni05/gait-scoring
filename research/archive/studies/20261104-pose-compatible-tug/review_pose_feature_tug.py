"""Independent reduced-input replay, original-reference proof and paired metrics."""
from pose_feature_tug import *
import math,statistics
from review2 import check_context_model
from review5 import check_meta
CORE_H=H/'complete-feature-controls';CORE_B=B/'complete-feature-controls'
def independent_metric(y,p):
    e=[abs(float(a)-float(b)) for a,b in zip(y,p)];s=sorted(e);q=.95*(len(s)-1);lo=math.floor(q);hi=math.ceil(q);slow=[i for i,a in enumerate(y) if a>=20];return dict(mae=statistics.fmean(e),p95=s[lo]+(s[hi]-s[lo])*(q-lo),slow_mae=statistics.fmean(e[i] for i in slow),slow_bias=statistics.fmean(float(p[i])-float(y[i]) for i in slow))
def audit_case(proto,small,large,c,f,d,names):
    rec=read(large/f'{c}-{f}.json');assert rec['protocol_sha256']==sha(small/'protocol.json');assert rec['condition']==c and rec['fold']==f;tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f);columns=proto['feature_sets'][c];assert rec['features']==columns and columns[0]=='speed';X=d['F'][:,[names.index(n) for n in columns]];ctx=np.load(rec['context']);assert sha(rec['context'])==rec['context_sha256'];assert not set(d['ids'][tr])&set(d['ids'][te]);assert rec['train_ids']==ctx['train_ids'].tolist()==d['ids'][tr].tolist() and rec['test_ids']==ctx['test_ids'].tolist()==d['ids'][te].tolist();assert np.array_equal(ctx['Xtrain'],X[tr],equal_nan=True) and np.array_equal(ctx['Xtest'],X[te],equal_nan=True) and np.array_equal(ctx['ytrain'],d['y'][tr]);cfgs=proto['candidates'][c];assert cfgs==configs_for(len(columns));count=0;delta=0.
    if c in ['speed','all_sensor']:
        A,T,proof=reference(c,f,d,X);assert proof==rec['reference'] and not rec['inner'] and not rec['models'];assert np.array_equal(ctx['A'],A) and np.array_equal(ctx['T'],T);fresh=T.copy()
    else:
        assert rec['reference'] is None;A=np.full(ctx['A'].shape,np.nan);fresh=np.full(ctx['T'].shape,np.nan);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(d['y'][tr],[12,20])));assert len(rec['inner'])==len(cfgs)*3 and len(rec['models'])==len(cfgs)
        for ci,cfg in enumerate(cfgs):
            for k,(it,iv) in enumerate(splits):
                rr=next(r for r in rec['inner'] if r['candidate']==ci and r['inner']==k);assert rr['train_ids']==d['ids'][tr[it]].tolist() and rr['test_ids']==d['ids'][tr[iv]].tolist() and sha(rr['path'])==rr['sha256'];m=joblib.load(rr['path']);check_context_model(m,X[tr[it]],d['y'][tr[it]],cfg);p=predict2(m,X[tr[iv]]);assert np.allclose(p,rr['prediction'],atol=1e-6,rtol=1e-6);delta=max(delta,float(np.max(np.abs(p-rr['prediction']))));A[iv,ci]=rr['prediction'];count+=1
            rr=next(r for r in rec['models'] if r['candidate']==ci);assert sha(rr['path'])==rr['sha256'];m=joblib.load(rr['path']);check_context_model(m,X[tr],d['y'][tr],cfg);p=predict2(m,X[te]);assert np.allclose(p,rr['prediction'],atol=1e-6,rtol=1e-6) and np.array_equal(ctx['T'][:,ci],rr['prediction']);fresh[:,ci]=p;count+=1;delta=max(delta,float(np.max(np.abs(p-rr['prediction']))))
        assert np.array_equal(A,ctx['A']) and np.isfinite(fresh).all();T=ctx['T']
    score=np.mean(np.abs(A-d['y'][tr,None]),axis=0);top=np.argsort(score,kind='stable')[:3];assert np.allclose(score,rec['inner_MAE'],atol=1e-12,rtol=1e-12) and top.tolist()==rec['top3'];assert np.array_equal(T[:,top[0]],rec['predictions']['selected_inner_MAE']) and np.array_equal(T[:,top].mean(1),rec['predictions']['ensemble3_inner_MAE']);assert np.allclose(fresh[:,top[0]],rec['predictions']['selected_inner_MAE'],atol=1e-6,rtol=1e-6) and np.allclose(fresh[:,top].mean(1),rec['predictions']['ensemble3_inner_MAE'],atol=1e-6,rtol=1e-6);meta_delta=0.
    for row in rec['meta_models']:
        assert sha(row['path'])==row['sha256'];m=joblib.load(row['path']);check_meta(m,A,d['y'][tr],row['tail']);pr=predict_meta(m,T);assert np.allclose(pr,rec['predictions'][row['kind']],atol=1e-10,rtol=1e-10);rp=predict_meta(m,fresh);bound=np.abs(fresh-T)@(np.abs(m['coef'])/m['scaler'].scale_);assert np.all(np.abs(rp-pr)<=bound+1e-10) and np.allclose(rp,pr,atol=1e-6,rtol=1e-6);meta_delta=max(meta_delta,float(np.max(np.abs(rp-pr))))
        if c in ['speed','all_sensor']:
            old_c='speed' if c=='speed' else 'bilateral';original=read(SOURCE/f'{old_c}-{f}.json');assert np.allclose(pr,original['predictions'][row['kind']],atol=1e-10,rtol=1e-10)
    return rec['predictions'],dict(condition=c,fold=f,people=len(te),features=len(columns),new_base_models_replayed=count,reference_base_model_count=0 if count else len(cfgs)*4,meta_models_replayed=2,max_base_prediction_delta=delta,max_fresh_meta_prediction_delta=meta_delta,unchanged_people_training_only_imputation_feature_selection_and_meta_KKT_verified=True)
def main():
    parent=read(H/'protocol.json');core=read(CORE_H/'protocol.json');assert parent['code_sha256']==sha(H/'pose_feature_tug.py') and core['code_sha256']==sha(H/'pose_complete_controls.py');assert core['parent_protocol_sha256']==sha(H/'protocol.json') and core['parent_code_sha256']==sha(H/'pose_feature_tug.py');assert parent['feature_audit_sha256']==sha(H/'feature-audit.json')
    for p,value in parent['library_hashes'].items():assert sha(p)==value
    assert parent['source_R5_protocol_sha256']==sha(OLD/'round05/protocol.json') and parent['source_R5_verification_sha256']==sha(OLD/'round05/verification.json') and parent['source_R5_receipt_sha256']==sha(OLD/'round05/run-receipt.json');assert read(OLD/'round05/verification.json')['status']=='passed';assert read(OLD/'round05/verification.json')['review_sha256']==sha(OLD/'review5.py');oldfactory=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');assert sha(oldfactory/'models.py')==parent['factory_sha256']
    for weight in read(oldfactory/'weights-manifest.json'):assert sha(weight['path'])==weight['sha256']
    for small in [H,CORE_H]:
        receipt=read(small/'run-receipt.json');assert receipt['protocol_sha256']==sha(small/'protocol.json')
        for path,value in receipt['hashes'].items():assert sha(path)==value
    d=np.load(DATA);names=read(G/'protocol.json')['feature_names'];assert parent['feature_sets']==feature_sets(names) and sha(DATA)==parent['source_data_sha256']==core['source_data_sha256'];assert len(d['y'])==594 and sum(d['y']>=20)==9;source=read('D:/gait-research-data/D_P/frailpol-short-walk-20261103/cohorts.json');source=[r for r in source if r['status']=='ok' and r['tug'] is not None];assert [r['id'] for r in source]==d['ids'].tolist() and np.array_equal(d['y'],[r['tug'] for r in source]) and np.array_equal(d['fold'],[r['fold'] for r in source]);checks=[];pred={};y=d['y'];ids=d['ids'];out={};held=[]
    for proto,small,large in [(parent,H,B),(core,CORE_H,CORE_B)]:
        for c,columns in proto['feature_sets'].items():
            if c.endswith('_core'):assert np.isfinite(d['F'][:,[names.index(n) for n in columns]]).all()
            pp={k:np.full(594,np.nan) for k in ['selected_inner_MAE','ensemble3_inner_MAE','stacked','tail_stacked']};seen=[]
            for f in range(5):
                values,check=audit_case(proto,small,large,c,f,d,names);checks.append(check);te=np.flatnonzero(d['fold']==f)
                for kind,p in values.items():pp[kind][te]=p
                seen.extend(ids[te]);print('verified',c,f,flush=True)
            assert len(seen)==594 and len(set(seen))==594 and set(seen)==set(ids)
            for kind,p in pp.items():
                assert np.isfinite(p).all();m=independent_metric(y,p);ref=metric(y,p)
                for k,v in m.items():assert np.isclose(v,ref[k],atol=1e-12,rtol=1e-12)
                m.update(n=594,slow_n=9,features=len(columns));out[c+'/'+kind]=m;pred[c+'/'+kind]=p;held.extend(dict(id=int(i),condition=c,kind=kind,actual=float(a),prediction=float(v)) for i,a,v in zip(ids,y,p))
    assert sum(c['new_base_models_replayed'] for c in checks)==560 and sum(c['meta_models_replayed'] for c in checks)==60 and sum(c['reference_base_model_count'] for c in checks)==400
    rng=np.random.default_rng(20261104);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))];uncertainty={}
    bilateral=np.isfinite(d['F'][:,names.index('bilateral_asymmetry_stride_time')]);repeat=np.isfinite(d['F'][:,names.index('within_foot_relative_std_stride_time')])
    for name,p in pred.items():
        kind=name.split('/')[1];base=pred['speed/'+kind];e=np.abs(p-y);diff=e-np.abs(base-y);uncertainty[name]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),matching_speed_control='speed/'+kind,mae_difference_speed=float(diff.mean()),mae_difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist(),bands={label:dict(n=int(mask.sum()),mae=float(e[mask].mean()),bias=float((p-y)[mask].mean()),max_error=float(e[mask].max())) for label,mask in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]},source_feature_availability_groups={label:dict(n=int(mask.sum()),mae=float(e[mask].mean()),mae_difference_speed=float(diff[mask].mean())) for label,mask in [('bilateral_available',bilateral),('bilateral_missing',~bilateral),('repeat_cycles_available',repeat),('repeat_cycles_missing',~repeat)]})
    save(B/'heldout.json',held);save(H/'comparison.json',out);save(H/'uncertainty.json',dict(method='5000pairedpersonbootstrap,same594people,9slowseparately. Conditionalreuseddevelopmentcohort,notexternalclinicalorsearch/multiple-comparison-adjusted. No MediaPipe extraction tested. Primaryfixedstacked;allsecondarypoliciesreported.',conditions=uncertainty));save(H/'verification.json',dict(status='passed',people=594,slow_people=9,conditions=6,policies=24,new_base_models_replayed=560,meta_models_replayed=60,unchanged_original_R5_reference_models_hashed_and_prior_full_replay_receipts_verified=400,checks=checks,comparison_sha256=sha(H/'comparison.json'),heldout_sha256=sha(B/'heldout.json'),uncertainty_sha256=sha(H/'uncertainty.json'),primary_protocol_sha256=sha(H/'protocol.json'),core_control_protocol_sha256=sha(CORE_H/'protocol.json'),review_code_sha256=sha(Path(__file__)),scope='Feature availability experiment on sensor-derived values,notactualMediaPipeaccuracy. All594includingmissingbilateral/repeatskept;core3/4featurecontrolscomplete594/594. No hidden raw waveforms,oldfullsensor latent predictions or clinical targets in reducedarms. OriginalR5 references rely on unchanged original full400modelreplay+currentfingerprints;560newbasemodels and60metamodelsnewlyreplayed. Originalpointtolerancesunchanged.'))
    print(json.dumps({k:dict(metrics=v,uncertainty=uncertainty[k]) for k,v in out.items() if k.endswith('/stacked')},indent=2))
if __name__=='__main__':main()
