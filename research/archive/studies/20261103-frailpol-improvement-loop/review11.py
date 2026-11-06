"""Independent source reconstruction, person/normalization/KKT and support-policy replay."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');G=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'round11';O=B/'round11';CONDS=['speed','speed_duration','combined'];TAILS=[1,4,8,16]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def neural_source(root,c,f,trainids,testids):
    r=read(root/c/f'fold{f}/fold.json');lookup={};epoch=r['epoch']
    for fit in r['inner']:
        ctx=read(fit['context']);assert not set(ctx['train_ids'])&set(ctx['eval_ids']) and set(ctx['train_ids'])|set(ctx['eval_ids'])==set(trainids);q=[row for row in fit['curve'] if row['epoch']==epoch];assert len(q)==1
        for pid,p in zip(ctx['eval_ids'],q[0]['prediction']):assert pid not in lookup;lookup[pid]=p
    assert set(lookup)==set(trainids) and r['test_ids']==testids.tolist();return np.array([lookup[int(i)] for i in trainids]),np.array(r['prediction'])
def main():
    p=read(R/'protocol.json');receipt=read(R/'run-receipt.json');assert p['code_sha256']==sha(H/'round11.py') and p['data_sha256']==sha(GB/'data.npz') and receipt['protocol_sha256']==sha(R/'protocol.json');assert receipt['metamodels']==60 and receipt['contexts']==15 and receipt['policy_fold_outputs']==120
    for path,value in p['source_verification_hashes'].items():assert sha(path)==value
    for path,value in receipt['hashes'].items():assert sha(path)==value
    wave_ver=read(G/'norm-aux/verification.json');assert set(['full_tail4','norm_tail4','symnorm_tail4','symnorm_tail16'])<=set(wave_ver['passed_conditions']);assert 'symnorm_aux_tail4' not in wave_ver['passed_conditions'];d=np.load(GB/'data.npz');ids=d['ids'];y=d['y'];fold=d['fold'];assert len(ids)==594 and sum(y>=20)==9;preds={};checks=[];count=0
    for c in CONDS:
      for f in range(5):
        rec=read(O/f'{c}-{f}.json');assert rec['protocol_sha256']==sha(R/'protocol.json') and rec['context_sha256']==sha(rec['context']);ctx=np.load(rec['context']);tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);assert np.array_equal(ctx['train_ids'],ids[tr]) and np.array_equal(ctx['test_ids'],ids[te]) and not set(ids[tr])&set(ids[te]) and np.array_equal(ctx['y'],y[tr]);assert np.array_equal(ctx['train_speed'],d['F'][tr,0]) and np.array_equal(ctx['test_speed'],d['F'][te,0])
        for path,value in rec['source_artifacts'].items():assert sha(path)==value
        a=np.load(B/'round07'/f'{"combined" if c=="combined" else "speed"}-{f}-context.npz');b=np.load(B/'round08'/f'{"bilateral" if c=="combined" else "speed"}-{f}-context.npz');assert np.array_equal(a['train_ids'],ids[tr]) and np.array_equal(a['test_ids'],ids[te]) and np.array_equal(b['train_ids'],ids[tr]) and np.array_equal(b['test_ids'],ids[te]);train=[a['X'][:,:-1],b['oof_matrix']];test=[a['T'][:,:-1],b['test_matrix']];names=[f'round07base{i}' for i in range(train[0].shape[1])]+[f'round08base{i}' for i in range(8)];manuals=['speed_tail4','speed_tail16','speed_aux_tail4']+(['speed_duration_tail4','speed_duration_aux_tail4'] if c!='speed' else []);fallback=None
        for m in manuals:
            aa,bb=neural_source(GB/'manual-controls',m,f,ids[tr],ids[te]);train.append(aa[:,None]);test.append(bb[:,None]);names.append('manual/'+m)
            if m==('speed_tail4' if c=='speed' else 'speed_duration_tail4'):fallback=bb
        if c=='combined':
            for w in ['full_tail4','norm_tail4','symnorm_tail4','symnorm_tail16']:
                assert wave_ver['condition_status'][w]=='passed_all65_checkpoints_and_other_gates';aa,bb=neural_source(GB/'norm-aux',w,f,ids[tr],ids[te]);train.append(aa[:,None]);test.append(bb[:,None]);names.append('wave/'+w)
                if w=='norm_tail4':fallback=bb
        X=np.column_stack(train);T=np.column_stack(test);assert np.array_equal(X,ctx['X']) and np.array_equal(T,ctx['T']) and np.array_equal(fallback,ctx['fallback']) and names==rec['feature_names'];assert X.shape[1]=={'speed':19,'speed_duration':21,'combined':34}[c];lo=float(d['F'][tr,0].min());hi=float(d['F'][tr,0].max());flags=(d['F'][te,0]<lo)|(d['F'][te,0]>hi);assert np.array_equal(flags,ctx['guard_flags']);mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1;A=(X-mu)/sd;Z=(T-mu)/sd;assert len(rec['models'])==4
        for row in rec['models']:
            assert sha(row['path'])==row['sha256'];m=np.load(row['path']);tail=row['tail'];assert tail in TAILS and np.array_equal(mu,m['mean']) and np.array_equal(sd,m['std']) and m['speed_min']==lo and m['speed_max']==hi;weights=np.where(y[tr]>=20,tail,np.where(y[tr]>=12,np.sqrt(tail),1.));assert np.array_equal(weights,m['weight']);am=np.average(A,axis=0,weights=weights);ym=np.average(y[tr],weights=weights);coef=m['coef'];assert np.min(coef)>=0;gradient=(A-am).T@(weights*((A-am)@coef-(y[tr]-ym)))+10*coef;assert np.min(gradient)>=-1e-6 and np.max(np.abs(gradient*coef))<1e-6;assert np.isclose(m['intercept'],ym-am@coef,atol=1e-12,rtol=1e-12);raw=Z@coef+m['intercept'];guarded=np.where(flags,fallback,raw);assert np.allclose(raw,row['prediction'],atol=1e-12,rtol=1e-12) and np.allclose(guarded,row['guarded_prediction'],atol=1e-12,rtol=1e-12)
            for policy,values in [('plain',row['prediction']),('guarded',row['guarded_prediction'])]:preds.setdefault(c+f'/tail{tail}/'+policy,np.full(594,np.nan))[te]=values
            count+=1
        checks.append(dict(condition=c,fold=f,source_experts=X.shape[1],models_replayed=4,policies_checked=8,guarded_people=int(flags.sum()),source_OOF_and_IDs_rebuilt=True,training_only_normalizers_weights_KKT_bounds_and_fallback_verified=True));print('verified',c,f,flush=True)
    assert count==60;rng=np.random.default_rng(20261107);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))];out={};unc={};held=[]
    for key,pred in preds.items():
        assert np.isfinite(pred).all();e=np.abs(pred-y);out[key]=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[slow].mean()),slow_bias=float((pred-y)[slow].mean()),goal_met=bool(e.mean()<=.9 and np.quantile(e,.95)<=3 and e[slow].mean()<=3));control=key.replace('combined/','speed/').replace('speed_duration/','speed/');difference=e-np.abs(preds[control]-y);unc[key]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),p95_ci95=np.quantile(np.quantile(e[boot],.95,axis=1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),matching_speed_control=control,difference_matching_speed=float(difference.mean()),difference_ci95=np.quantile(difference[boot].mean(1),[.025,.975]).tolist(),bands={label:dict(n=int(mask.sum()),mae=float(e[mask].mean()),bias=float((pred-y)[mask].mean()),max_error=float(e[mask].max())) for label,mask in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]});held.extend(dict(id=int(i),condition=key,actual=float(a),prediction=float(pr)) for i,a,pr in zip(ids,y,pred))
    save(O/'heldout.json',held);save(R/'comparison.json',out);save(R/'uncertainty.json',dict(method='5000pairedpersonbootstrap;9severeseparately;reuseddevelopmentcohortnotclinical/external/selection-adjusted. Wholepipelinecomparisonnotpureexpert-count-matchedcausality.',conditions=unc));save(R/'verification.json',dict(status='passed',contexts=15,meta_models_replayed=60,policy_fold_outputs_checked=120,checks=checks,comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json'),review_code_sha256=sha(Path(__file__))));print(out)
if __name__=='__main__':main()
