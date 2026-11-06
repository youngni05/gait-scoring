"""Verify expanded GPU predictions against the same-fold expanded Ridge baseline."""
from pathlib import Path
import csv, hashlib, json, re
import numpy as np

HERE=Path(__file__).resolve().parent
BUNDLE=HERE/'gpu-bundle'
REMOTE=HERE/'received/dis01-run/extracted'
RUNS=REMOTE/'runs/main'
OUT=HERE/'analysis/gpu'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_save(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

manifest=load(BUNDLE/'manifest.json')
assert sha(REMOTE/'manifest.json')==sha(BUNDLE/'manifest.json')
for name,digest in manifest.items():assert sha(BUNDLE/name)==digest,name
for name in ['run.py','batch.py']:assert sha(REMOTE/name)==manifest[name],name
cfg=load(REMOTE/'protocol.ready.json')
assert cfg==load(BUNDLE/'protocol.ready.json')
config_hash=sha(REMOTE/'protocol.ready.json');manifest_hash=sha(REMOTE/'manifest.json')
remote=load(RUNS/'summary.json')
assert remote['people']==695 and remote['records']==1064 and remote['config_sha256']==config_hash
assert cfg['seeds']==remote['seeds']==[20261028] and cfg['steps']==200
with np.load(BUNDLE/'train_only.npz',allow_pickle=False) as z:
    d={k:z[k].copy() for k in ['person_id','video_id','fold','cohort','y']}
with np.load(BUNDLE/'baseline_predictions.npz',allow_pickle=False) as z:
    predictions={'baseline':z['predictions'].copy()}
with np.load(HERE/'ridge_predictions.npz',allow_pickle=False) as z:
    assert np.array_equal(predictions['baseline'],z['expanded_ridge'])
    small=z['small_ridge'].copy()
ids=d['person_id'];people=np.unique(ids);folds=d['fold'];truth=d['y']
assert len(people)==695 and len(set(d['video_id']))==1064
assert truth.shape==(1064,2) and np.isfinite(truth).all()
checkpoint_hashes={}
for line in (REMOTE/'checkpoint-sha256.txt').read_text().splitlines():
    digest,path=line.split(maxsplit=1);path=path.lstrip('*')
    assert re.fullmatch('[0-9a-f]{64}',digest) and path not in checkpoint_hashes
    checkpoint_hashes[path]=digest
assert len(checkpoint_hashes)==15
jobs=[]
for variant in cfg['variants']:
    pred=np.full((1064,1,2),np.nan);coverage=np.zeros(1064,dtype=int);provenance=[]
    for fold in cfg['folds']:
        te=np.flatnonzero(folds==fold);tr=np.flatnonzero(folds!=fold)
        assert len(set(ids[tr]))==556 and len(set(ids[te]))==139
        assert not set(ids[tr])&set(ids[te])
        name=f'{variant}_seed20261028_fold{fold}';job=RUNS/name
        done=load(job/'complete.json');identity=load(job/'job.json')
        expected={'config':config_hash,'manifest':manifest_hash,'variant':variant,'fold':fold,
                  'seed':20261028,'torch':'2.4.0+cu121','device_type':'cuda'}
        assert done['identity']==identity==expected
        assert done['steps']==200 and done['frozen_unchanged'] is True
        assert sha(job/'predictions.npz')==done['predictions_sha256']
        with np.load(job/'predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(z['indices'],te)
            assert z['predictions'].shape==(len(te),1,2) and np.isfinite(z['predictions']).all()
            pred[te]=z['predictions'];coverage[te]+=1
        model_hash=checkpoint_hashes[f'runs/main/{name}/checkpoint.pt']
        provenance.append({'fold':fold,'model_sha256':model_hash,
            'training_people':np.unique(ids[tr]).tolist(),'evaluation_people':np.unique(ids[te]).tolist()})
        jobs.append({'variant':variant,'fold':fold,'steps':200,'train_people':556,'evaluation_people':139,
            'prediction_sha256':done['predictions_sha256'],'checkpoint_sha256_reported_by_server':model_hash})
    assert np.all(coverage==1) and np.isfinite(pred).all()
    predictions[variant]=pred
    np.savez_compressed(OUT/f'{variant}_oof.npz',predictions=pred,person_id=ids,
                        video_id=d['video_id'],fold=folds,cohort=d['cohort'],baseline=predictions['baseline'])
    save(OUT/f'{variant}_provenance.json',{'dataset_sha256':sha(BUNDLE/'train_only.npz'),
        'prediction_sha256':sha(OUT/f'{variant}_oof.npz'),'axes':['record','window','side'],
        'windows':['front'],'sides':['left_GDI','right_GDI'],'shape':[1064,1,2],
        'fold_models':provenance,'checkpoint_hash_source':'server sha256sum; checkpoint bytes not downloaded'})

groups={'old175':d['cohort']==0,'new520':d['cohort']==1,'all695':np.ones(1064,dtype=bool)}
errors={};metric_rows=[];fold_rows=[];band_rows=[];summary_error=0.0
for name,p3 in predictions.items():
    p=p3[:,0];err=abs(p-truth);bins=abs(np.floor(p/10)-np.floor(truth/10))
    errors[name]=np.array([err[ids==pid].mean() for pid in people])
    person_pred=np.array([p[ids==pid].mean() for pid in people])
    expected=remote['metrics'][name]['means']['front_mae']
    summary_error=max(summary_error,abs(errors[name].mean()-expected))
    assert abs(errors[name].mean()-expected)<1e-10
    sd_key='person_prediction_sd' if name=='baseline' else 'person_prediction_sd_per_seed'
    expected_sd=remote['metrics'][name][sd_key]
    if isinstance(expected_sd,list):expected_sd=expected_sd[0]
    assert abs(person_pred.std(ddof=1)-expected_sd)<1e-10
    for group,mask in groups.items():
        gp,counts=np.unique(ids[mask],return_counts=True);countmap=dict(zip(gp,counts))
        weights=np.array([1/(len(gp)*countmap[pid]*2) for pid in ids[mask]])[:,None]
        macro=lambda a:float(np.sum(a[mask]*weights))
        mae=macro(err);assert abs(mae-np.mean([err[ids==pid].mean() for pid in gp]))<1e-10
        metric_rows.append({'model':name,'group':group,'people':len(gp),'records':int(mask.sum()),
            'mae':mae,'bias':macro(p-truth),'exact_bin_rate':macro(bins==0),'within_one_bin_rate':macro(bins<=1),
            'two_or_more_bin_rate':macro(bins>=2),'prediction_min':float(p[mask].min()),
            'prediction_max':float(p[mask].max()),'person_prediction_sd':float(np.array([p[ids==pid].mean() for pid in gp]).std(ddof=1))})
    for fold in range(5):
        use=np.isin(people,ids[folds==fold])
        fold_rows.append({'model':name,'fold':fold,'mae':float(errors[name][use].mean())})
    for band in np.unique(np.floor(truth/10).astype(int)):
        mask=np.floor(truth/10)==band;vals=[]
        for pid in people:
            use=mask&(ids==pid)[:,None]
            if use.any():vals.append([err[use].mean(),(p-truth)[use].mean(),(bins[use]==0).mean()])
        avg=np.mean(vals,axis=0)
        band_rows.append({'model':name,'actual_bin_low':int(band*10),'people_in_band':len(vals),
            'record_side_count':int(mask.sum()),'mae':float(avg[0]),'bias':float(avg[1]),'exact_bin_rate':float(avg[2])})

draw=np.random.default_rng(20261029).integers(0,695,(10000,695));comparisons={};bootstrap_error=0.
pairs=[('baseline','head_only'),('head_only','tail'),('head_only','last_stage'),('tail','last_stage'),('baseline','tail'),('baseline','last_stage')]
for ref,name in pairs:
    delta=errors[ref]-errors[name];ci=np.percentile(delta[draw].mean(axis=1),[2.5,97.5])
    row={'mae_reduction':float(delta.mean()),'relative_reduction_percent':float(delta.mean()/errors[ref].mean()*100),
        'person_bootstrap95':ci.tolist(),'people_improved':int((delta>0).sum()),'people_worsened':int((delta<0).sum()),
        'people_equal':int((delta==0).sum()),'folds_improved':sum(
            next(r['mae'] for r in fold_rows if r['model']==ref and r['fold']==k)>
            next(r['mae'] for r in fold_rows if r['model']==name and r['fold']==k) for k in range(5))}
    key=f'{ref}_minus_{name}';comparisons[key]=row
    if key in remote['comparisons']:
        old=remote['comparisons'][key]['front_mae']
        error=max(abs(row['mae_reduction']-old['reduction']),float(abs(ci-np.array(old['person_bootstrap95'])).max()))
        bootstrap_error=max(error,bootstrap_error);assert error<1e-10
for group,mask in groups.items():
    gp=np.unique(ids[mask]);use=np.isin(people,gp)
    delta=errors['baseline'][use]-errors['last_stage'][use]
    draws=np.random.default_rng(20261029).integers(0,len(gp),(10000,len(gp)))
    comparisons[f'baseline_minus_last_stage_{group}']={'mae_reduction':float(delta.mean()),
        'person_bootstrap95':np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist(),
        'people_improved':int((delta>0).sum()),'people_worsened':int((delta<0).sum())}
small_mae=float(np.mean([abs(small[ids==pid,0]-truth[ids==pid]).mean() for pid in people]))
comparisons['small_ridge_minus_last_stage']={'small_ridge_mae':small_mae,
    'mae_reduction':float(small_mae-errors['last_stage'].mean()),
    'relative_reduction_percent':float((small_mae-errors['last_stage'].mean())/small_mae*100),
    'note':'Combined data expansion and partial tuning; not the isolated tuning effect.'}

replay=load(REMOTE/'checks/replay/replay.json');profile=load(REMOTE/'checks/profile_b32/checks.json')
assert max(r['max_score_error'] for r in replay['rows'])<1e-4 and replay['real_data_optimizer_steps']==0
assert profile['synthetic_only'] and profile['real_data_optimizer_steps']==0
assert all(r['resume_weight_max_error']==0 for r in profile['results'])
csv_save(OUT/'metrics.csv',metric_rows);csv_save(OUT/'fold_metrics.csv',fold_rows);csv_save(OUT/'score_bands.csv',band_rows)
save(OUT/'summary.json',{'metrics':metric_rows,'comparisons':comparisons,'front_only':True,
    'independent_final_evaluation':False,'one_seed_only':True,'bin_rule':'floor(GDI/10); no clipping; not clinical grades'})
save(OUT/'verification.json',{'passed':True,'completed_jobs':15,'real_data_optimizer_steps':3000,
    'people':695,'records':1064,'evaluation_people_per_fold':139,'training_people_per_fold':556,
    'all_prediction_hashes_match':True,'person_disjoint':True,'all_rows_predicted_once_per_variant':True,
    'remote_code_matches_prepared_bundle':True,'remote_config_matches':True,
    'summary_max_absolute_error':summary_error,'bootstrap_max_absolute_error':bootstrap_error,
    'archive_sha256':sha(HERE/'received/dis01-run/expanded-results-review.tar.gz'),
    'dataset_sha256':sha(BUNDLE/'train_only.npz'),'config_sha256':config_hash,
    'checkpoint_hashes_count':15,'checkpoint_bytes_downloaded':False,'model_reinference_performed':False,
    'training_and_summary_start':'2026-10-30T15:11:19+09:00','training_and_summary_end':'2026-10-30T15:31:00+09:00',
    'training_and_summary_seconds':1181,'jobs':jobs})
print(json.dumps({'verified':True,'metrics':metric_rows,'comparisons':comparisons},ensure_ascii=False,indent=2))
