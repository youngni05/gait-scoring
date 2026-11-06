"""Audit video rotations, replay frozen RFs, and describe paired sensitivity."""
import sys, json, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent; BASE=HERE.parent; FEATURE=BASE/'feature-analysis'
sys.path.insert(0,str(FEATURE));import study
import numpy as np
import joblib

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(x):
    x=np.asarray(x);return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max()),minimum=float(x.min()))
def sequence_rotation(r):
    # Preserve CARE-PD's row-major first-two-column representation and 25Hz bins.
    six=r[...,:2].reshape(len(r),138).astype('float64')
    bins=np.floor(np.arange(len(r))*25/30+1e-8).astype(int)
    sums=np.zeros((bins.max()+1,138));np.add.at(sums,bins,six)
    return (sums/np.bincount(bins)[:,None]).astype('float32')
def feature(r):
    x=sequence_rotation(r)
    cols=[6*(j-1)+k for j in study.MAJOR for k in range(6)]
    return study.stats(x[:,cols])[np.r_[60:120,240:308]]
def angle(a,b):
    c=np.einsum('...ij,...ij->...',a,b)
    return np.degrees(np.arccos(np.clip((c-1)/2,-1,1)))

def main():
    out=HERE/'received/main';complete=json.loads((out/'complete.json').read_text())
    assert complete['passed'] and complete['frames']==10500 and len(complete['outputs'])==70
    for n in ['infer.py','protocol.json','clips.json']:assert sha(HERE/n)==complete['source_sha256'][n]
    protocol=json.loads((HERE/'protocol.json').read_text());clips=json.loads((HERE/'clips.json').read_text())
    noise_audit=json.loads((HERE/'noise-verification.json').read_text())
    assert noise_audit['passed'] and noise_audit['outputs']==70 and noise_audit['maximum_difference']==0
    assert noise_audit['source_sha256']==sha(HERE/'verify_noise.py')
    noise_rows={r['key']:r for r in noise_audit['rows']}
    keys=[];xx=[];rot={};platform_noise_differences=[]
    for row in complete['outputs']:
        p=out/(row['key']+'.npz');assert sha(p)==row['sha256']
        z=np.load(p);r=z['rotations'];assert r.shape==(150,23,3,3) and np.isfinite(r).all()
        condition=next(v for v in protocol['conditions'] if row['key'].endswith('_'+v['name']))
        expected_noise=np.clip(np.random.RandomState(condition['seed']).normal(size=(150,3)),-3,3)*condition['sigma']
        # A Windows/Linux NumPy difference in one final floating-point bit was
        # observed. Require exact source-platform replay and exact byte transfer;
        # do not relax the replay tolerance or silently round the perturbations.
        audit=noise_rows[row['key']];assert audit['exact_match'] and audit['file_sha256']==row['sha256']
        assert hashlib.sha256(z['noise'].tobytes()).hexdigest()==audit['expected_noise_sha256']
        assert int(z['fps'])==30
        platform_noise_differences.append(float(abs(z['noise']-expected_noise).max()))
        assert np.max(abs(r.transpose(0,1,3,2)@r-np.eye(3)))<=1e-5
        assert np.max(abs(np.linalg.det(r)-1))<=1e-5
        keys.append(row['key']);xx.append(feature(r));rot[row['key']]=r
    x=np.stack(xx);assert x.shape==(70,128) and np.isfinite(x).all()
    original_x=np.stack([study.stats(sequence_rotation(rot[k])) for k in keys])
    assert original_x.shape==(70,698) and np.isfinite(original_x).all()
    expected=[c['key']+'_'+v['name'] for c in clips for v in protocol['conditions']];assert keys==expected
    data=np.load(BASE/'data.npz');trainx=np.load(FEATURE/'inputs.npz')['motion3d'];fold=data['fold']
    assert sha(BASE/'data.npz')==json.loads((BASE/'protocol.json').read_text())['data_sha256']
    # Independent implementation on frozen original CARE channels, without rotations.
    cols=np.array([6*(j-1)+k for j in study.MAJOR for k in range(6)])
    flat=data['x'];off=data['offsets'];errors=[]
    for i in np.linspace(0,2952,20,dtype=int):
        a=flat[off[i]:off[i+1],cols];centered=a-a.mean(0);ac=[]
        for lag in range(5,41,5):
            u,v=centered[:-lag],centered[lag:]
            ac.append((u*v).sum()/max(np.sqrt((u*u).sum()*(v*v).sum()),1e-12) if len(a)>lag else 0.)
        value=np.r_[a.std(0),abs(np.diff(a,axis=0)).mean(0)*25,ac].astype('float32')
        errors.append(float(abs(value-trainx[i]).max()))
    assert max(errors)<=1e-6
    del flat
    predictions=[];original_predictions=[];identities=[];replay=[];ood=[];model_hashes={}
    for f in range(6):
        ref=trainx[fold!=f];lo,hi=ref.min(0),ref.max(0);qlo,qhi=np.quantile(ref,[.01,.99],axis=0)
        ood.append(dict(fold=f,outside_range=((x<lo)|(x>hi)).mean(1).tolist(),outside_q01_q99=((x<qlo)|(x>qhi)).mean(1).tolist()))
        for seed in study.SEEDS:
            path=FEATURE/'received/fits/motion3d'/('fold%d_seed%d.joblib'%(f,seed));model=joblib.load(path);model.n_jobs=1
            pred=model.predict(x);replay.append(float(abs(pred-model.predict(x)).max()));assert replay[-1]<=1e-12
            predictions.append(pred);identities.append(dict(fold=f,seed=seed));model_hashes[str(path.relative_to(BASE))]=sha(path)
            path=BASE/'received/runs/main'/('fold%d'%f)/('seed%d'%seed)/'rf.joblib'
            old=joblib.load(path);old.n_jobs=1;oldpred=old.predict(original_x)
            replay.append(float(abs(oldpred-old.predict(original_x)).max()));assert replay[-1]<=1e-12
            original_predictions.append(oldpred);model_hashes[str(path.relative_to(BASE))]=sha(path)
    predictions=np.stack(predictions);assert predictions.shape==(30,70)
    original_predictions=np.stack(original_predictions);assert original_predictions.shape==(30,70)
    idx={k:i for i,k in enumerate(keys)};subjects=[c['subject'] for c in protocol['selected']]
    def score(key):return predictions[:,idx[key]]
    view=[];jitter=[];perclip=[]
    for subject in subjects:
        a='s%02d_PG1_clean'%subject;b='s%02d_PG2_clean'%subject
        delta=abs(score(a)-score(b));angles=angle(rot[a][:,np.array(study.MAJOR)-1],rot[b][:,np.array(study.MAJOR)-1])
        view.append(dict(subject=subject,PG1=float(score(a).mean()),PG2=float(score(b).mean()),paired_abs_score_difference=describe(delta),local_rotation_difference_deg=describe(angles),per_model_delta=delta.tolist()))
    for clip in clips:
        clean=clip['key']+'_clean';clean_score=score(clean)
        perclip.append(dict(key=clip['key'],clean_score=describe(clean_score),seed_means=[float(clean_score[k::5].mean()) for k in range(5)]))
        for condition in protocol['conditions'][1:]:
            key=clip['key']+'_'+condition['name'];diff=score(key)-clean_score
            jitter.append(dict(key=key,subject=clip['subject'],sigma=condition['sigma'],seed=condition['seed'],signed_score_change=describe(diff),absolute_score_change=describe(abs(diff)),per_model_delta=diff.tolist(),local_rotation_difference_deg=describe(angle(rot[clean][:,np.array(study.MAJOR)-1],rot[key][:,np.array(study.MAJOR)-1]))))
    view_values=np.array([v['per_model_delta'] for v in view])
    summary=dict(view_abs_difference=describe(view_values),view_difference_by_rf_seed=[describe(view_values[:,k::5]) for k in range(5)],jitter={})
    for s in [.02,.05]:
        a=np.array([r['per_model_delta'] for r in jitter if r['sigma']==s]);summary['jitter'][str(s)]=dict(absolute_change=describe(abs(a)),signed_change=describe(a),by_rf_seed=[describe(abs(a[:,k::5])) for k in range(5)])
    clean_idx=[idx[c['key']+'_clean'] for c in clips]
    summary['clean_outside_training_range']=describe(np.array([v['outside_range'] for v in ood])[:,clean_idx])
    summary['clean_outside_training_q01_q99']=describe(np.array([v['outside_q01_q99'] for v in ood])[:,clean_idx])
    oldview=np.array([abs(original_predictions[:,idx['s%02d_PG1_clean'%s]]-original_predictions[:,idx['s%02d_PG2_clean'%s]]) for s in subjects])
    original_summary=dict(view_abs_difference=describe(oldview),jitter={})
    for s in [.02,.05]:
        olddelta=np.array([original_predictions[:,idx[c['key']+'_'+v['name']]]-original_predictions[:,idx[c['key']+'_clean']] for c in clips for v in protocol['conditions'] if v['sigma']==s])
        original_summary['jitter'][str(s)]=dict(absolute_change=describe(abs(olddelta)),signed_change=describe(olddelta))
    original=json.loads((BASE/'analysis.json').read_text());ablation=json.loads((FEATURE/'comparison.json').read_text())
    assert original['passed'] and ablation['passed']
    np.savez_compressed(HERE/'scores.npz',features=x,predictions=predictions,original_features=original_x,original_predictions=original_predictions,keys=np.array(keys),fold=np.array([v['fold'] for v in identities]),seed=np.array([v['seed'] for v in identities]))
    result=dict(passed=True,people=5,views=10,conditions_per_view=7,frames=10500,RF_models=30,reference_698_RF_models=30,RF_seeds=study.SEEDS,feature_reproduction_error=max(errors),RF_replay_error=max(replay),source_platform_noise_replay_error=0.,cross_platform_noise_max_difference=max(platform_noise_differences),summary=summary,original_698_video_summary=original_summary,view_pairs=view,clips=perclip,jitter=jitter,ood=ood,model_sha256=model_hashes,clinical_comparison=original['aggregate'],feature_comparison=ablation['aggregate'],limitations=protocol['scope'],prediction_sha256=sha(HERE/'scores.npz'))
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,summary=summary,view_pairs=view,clips=perclip),indent=2))

if __name__=='__main__':main()
