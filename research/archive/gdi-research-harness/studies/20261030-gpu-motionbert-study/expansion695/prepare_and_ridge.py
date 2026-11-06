"""Build authorized old175+development520 front-only data and paired Ridge controls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys, json, hashlib, time
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
H=STUDY.parents[1]
ROOT=H.parent
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
sys.dont_write_bytecode=True
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    start=time.perf_counter()
    oldpath=STUDY/'bundle-py38'
    newpath=H/'studies/20261027-gdi-linkage'
    sources={}
    def checked(p,expected):
        assert sha(p)==expected,str(p)
        sources[p.relative_to(ROOT).as_posix()]=expected
    oldmanifest=load(oldpath/'manifest.json')
    for name in ['train_only.npz','baseline_heads.npz','baseline_predictions.npz']:
        checked(oldpath/name,oldmanifest[name])
    checked(newpath/'development_features.npz',load(newpath/'feature_preparation.json')['sha256'])
    feature_sources=load(newpath/'feature_sources.json')
    checked(newpath/'development_linkage.json',feature_sources['sha256']['gdi-research-harness/studies/20261027-gdi-linkage/development_linkage.json'])
    for name in ['outputs/motionbert-smoke-20261015/assets/MB_lite.bin',
                 'outputs/motionbert-smoke-20261015/vendor/lib/model/DSTformer.py']:
        checked(ROOT/name,feature_sources['sha256'][name])
    with np.load(oldpath/'train_only.npz',allow_pickle=False) as z:
        old={k:z[k].copy() for k in z.files}
    with np.load(newpath/'development_features.npz',allow_pickle=False) as z:
        new={k:z[k].copy() for k in z.files}
    rows=load(newpath/'development_linkage.json')['records']
    assert len(rows)==861 and len(set(new['patient_ids']))==520
    assert not set(new['patient_ids'])&set(old['person_id'])
    assert not set(new['video_ids'])&set(old['video_id'])
    assert [r['patient_id'] for r in rows]==new['patient_ids'].tolist()
    assert [r['video_id'] for r in rows]==new['video_ids'].tolist()
    assert [r['sequence'] for r in rows]==new['sequence'].tolist()
    assert np.array_equal(np.array([r['GDI_LR'] for r in rows]),new['Y'])
    assert all(r['eligible'] and r['reserved_role']=='development_candidate' for r in rows)
    new_inputs=[]
    for r in rows:
        p=(H/r['artifact']).resolve()
        assert p.is_relative_to(H.resolve())
        checked(p,r['sha256'])
        with np.load(p,allow_pickle=False) as z:
            x=z['input124'].copy()
            assert x.shape==(124,17,3) and x.dtype==np.float32
            assert np.isfinite(x).all() and np.max(abs(x))<=1
            assert np.all(np.diff(z['frame_ids124'])==1)
            new_inputs.append(x)
    new_people=np.unique(new['patient_ids'])
    shuffled=np.random.default_rng(20261030).permutation(new_people)
    assignment={str(p):k for k,chunk in enumerate(np.split(shuffled,5)) for p in chunk}
    newfold=np.array([assignment[str(p)] for p in new['patient_ids']])
    d={'x':np.concatenate([old['x'][:,0],np.stack(new_inputs)])[:,None],
       'features':np.concatenate([old['features'][:,0],new['X']])[:,None],
       'y':np.concatenate([old['y'],new['Y']]),
       'person_id':np.concatenate([old['person_id'],new['patient_ids']]),
       'video_id':np.concatenate([old['video_id'],new['video_ids']]),
       'fold':np.concatenate([old['fold'],newfold]),
       'cohort':np.concatenate([np.zeros(203,dtype=np.int64),np.ones(861,dtype=np.int64)])}
    assert d['x'].shape==(1064,1,124,17,3) and d['features'].shape==(1064,1,512)
    assert len(set(d['person_id']))==695 and len(set(d['video_id']))==1064
    protocol={'status':'ready','data_scope':'original175_plus_development520_only',
        'people':695,'records':1064,'split_seed':20261030,'old_folds_preserved':True,
        'new_people_per_fold':104,'evaluation_people_per_fold':139,
        'small_training_people_per_fold':140,'expanded_training_people_per_fold':556,
        'window':'front_124_only','ridge_alpha':1.0,'fit_weight':'record_equal',
        'scaler_fit':'training_records_only','evaluation':'person_macro_MAE',
        'subgroups':['old175','new520','all695'],'primary_report':'paired results in each subgroup; no test-set selection',
        'final_reserved223_used':False,'old37_38_used':False,
        'gpu_steps':200,'gpu_batch_size':32,'gpu_head_lr':1e-4,'gpu_encoder_lr':1e-5,
        'gpu_seed':20261028,'gpu_variants':['head_only','tail','last_stage'],
        'note':'First fixed expansion comparison, no tuning. Expanded GPU has fewer data passes at same update budget; do not infer optimal training.'}
    protocol_path=HERE/'protocol.json'
    if protocol_path.exists():assert load(protocol_path)==protocol
    else:save(protocol_path,protocol)
    np.savez_compressed(HERE/'train_only.npz',**d)
    save(HERE/'source_hashes.json',sources)
    split_audit=[]
    for fold in range(5):
        tr=d['fold']!=fold;te=~tr;small=tr&(d['cohort']==0)
        assert len(set(d['person_id'][tr]))==556 and len(set(d['person_id'][te]))==139
        assert len(set(d['person_id'][small]))==140
        assert not set(d['person_id'][tr])&set(d['person_id'][te])
        split_audit.append({'fold':fold,'train_people':556,'evaluation_people':139,
            'small_train_people':140,'train_records':int(tr.sum()),'evaluation_records':int(te.sum()),
            'training_people':np.unique(d['person_id'][tr]).tolist(),
            'evaluation_people_ids':np.unique(d['person_id'][te]).tolist()})
    save(HERE/'split.json',split_audit)
    pred={n:np.full((1064,1,2),np.nan) for n in ['small_ridge','expanded_ridge']}
    heads={};fit_checks=[]
    with np.load(oldpath/'baseline_predictions.npz',allow_pickle=False) as z:old_baseline=z['predictions'][:,0].copy()
    with threadpool_limits(limits=2):
        for fold in range(5):
            te=d['fold']==fold
            for name in pred:
                tr=(~te) if name=='expanded_ridge' else ((~te)&(d['cohort']==0))
                scaler=StandardScaler().fit(d['features'][tr,0])
                x=scaler.transform(d['features'][tr,0]);y=d['y'][tr]
                model=Ridge(alpha=1.0,solver='svd').fit(x,y)
                p=model.predict(scaler.transform(d['features'][te,0]))
                pred[name][te,0]=p
                # Independent normal-equation check on the training-only standardized inputs.
                xc=x-x.mean(0);yc=y-y.mean(0)
                coef=np.linalg.solve(xc.T@xc+np.eye(512),xc.T@yc)
                q=(scaler.transform(d['features'][te,0])-x.mean(0))@coef+y.mean(0)
                error=float(abs(p-q).max());assert error<1e-6
                fit_checks.append({'fold':fold,'model':name,'independent_prediction_error':error})
                if name=='expanded_ridge':
                    prefix=f'fold{fold}__'
                    for key,value in [('scaler_mean',scaler.mean_),('scaler_scale',scaler.scale_),
                                      ('coef',model.coef_),('intercept',model.intercept_)]:heads[prefix+key]=value
    assert all(np.isfinite(p).all() for p in pred.values())
    old_replay_error=float(abs(pred['small_ridge'][:203,0]-old_baseline).max())
    assert old_replay_error<1e-6,old_replay_error
    np.savez_compressed(HERE/'baseline_heads.npz',**heads)
    np.savez_compressed(HERE/'baseline_predictions.npz',predictions=pred['expanded_ridge'])
    np.savez_compressed(HERE/'ridge_predictions.npz',**pred,person_id=d['person_id'],video_id=d['video_id'],fold=d['fold'],cohort=d['cohort'])
    metrics={};comparison={};rng=np.random.default_rng(20261030)
    for group,mask in [('old175',d['cohort']==0),('new520',d['cohort']==1),('all695',np.ones(1064,dtype=bool))]:
        people=np.unique(d['person_id'][mask]);vals={}
        for name,p in pred.items():
            e=abs(p[:,0]-d['y']);vals[name]=np.array([e[d['person_id']==pid].mean() for pid in people])
        metrics[group]={name:float(v.mean()) for name,v in vals.items()}
        delta=vals['small_ridge']-vals['expanded_ridge']
        draws=rng.integers(0,len(people),(10000,len(people)))
        comparison[group]={'people':len(people),'records':int(mask.sum()),'mae_reduction':float(delta.mean()),
            'bootstrap95':np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist(),
            'people_improved':int((delta>0).sum()),'people_worsened':int((delta<0).sum())}
    save(HERE/'ridge_summary.json',{'metrics':metrics,'comparisons':comparison,
        'independent_final_evaluation':False,'fit_count':10,'elapsed_seconds':time.perf_counter()-start})
    save(HERE/'preparation_verification.json',{'passed':True,'people':695,'records':1064,
        'source_files_verified':len(sources),'new_raw_inputs_verified':861,'old_folds_preserved':True,
        'new_labels_aligned':True,'new_old_people_overlap':0,'fold_person_disjoint':True,
        'old_baseline_replay_max_error':old_replay_error,'fit_checks':fit_checks,
        'gpu_real_training_started':False,'source_snapshot':sources,
        'outputs_sha256':{n:sha(HERE/n) for n in ['train_only.npz','baseline_heads.npz','baseline_predictions.npz','ridge_predictions.npz','protocol.json','split.json']}})
    print(json.dumps({'verified':True,'people':695,'records':1064,'metrics':metrics,'comparisons':comparison,
                     'old_replay_error':old_replay_error},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
