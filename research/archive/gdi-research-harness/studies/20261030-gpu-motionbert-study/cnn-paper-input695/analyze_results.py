"""Independently audit returned single-window CNN results on the retained cohort."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent;B=HERE/'bundle';R=HERE/'received/results';OUT=HERE/'analysis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    assert sha(HERE/'main-review.tar.gz')=='820f9e040288f82151e0251e92d053259ff959083384873420a61c78277e6fe0'
    assert sha(HERE/'received/bundle/manifest.json')==sha(B/'manifest.json')
    manifest=load(B/'manifest.json')
    for name,digest in manifest.items():assert sha(B/name)==digest,name
    for name in ('run.py','model.py','protocol.json'):assert sha(HERE/'received/bundle'/name)==manifest[name]
    c=load(B/'protocol.json')
    with np.load(B/'data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files if k!='x'}
    idx=d['original_row_index'];y=d['y'];ids=d['person_id'];people=np.unique(ids)
    with np.load(HERE.parent/'cnn-followup695/bundle/data.npz',allow_pickle=False) as z:
        for k in ('y','person_id','video_id','fold','cohort'):np.testing.assert_array_equal(d[k],z[k][idx])
    assert len(y)==1061 and len(people)==692
    def ident(f,j,epochs,purpose):
        te=d['fold']==f;va=np.zeros(len(y),bool) if j is None else d['fold']==j;tr=~(te|va)
        groups=[np.unique(ids[m]).tolist() for m in (tr,va,te)]
        assert not(set(groups[0])&set(groups[1]) or set(groups[0])&set(groups[2]) or set(groups[1])&set(groups[2]))
        return dict(bundle_sha256=sha(B/'manifest.json'),purpose=purpose,outer_fold=f,inner_fold=j,epochs=epochs,
            fit_seed=c['seed']+f*100+(50 if j is None else j),train_people=groups[0],validation_people=groups[1],outer_evaluation_people=groups[2])
    def metric(p,mask):
        yy=y[mask];pp=p[mask];ii=ids[mask];uu=np.unique(ii)
        def macro(v):return float(np.mean([v[ii==person].mean() for person in uu]))
        bins=np.abs(np.floor(pp/10)-np.floor(yy/10))
        return dict(people=len(uu),records=int(mask.sum()),mae=macro(np.abs(pp-yy)),bias=macro(pp-yy),
            exact_10point_bin_rate=macro(bins==0),two_or_more_bins_rate=macro(bins>=2),
            prediction_min=float(pp.min()),prediction_max=float(pp.max()))
    cnn=np.full_like(y,np.nan,dtype=float);raw=np.full_like(cnn,np.nan);seen=np.zeros(len(y),int)
    diagnostics=[];steps=0;seconds=0.;curves=[]
    for f in range(5):
        dest=R/('fold%d'%f);done=load(dest/'complete.json');selection=load(dest/'selection.json')
        assert done['device']=='cuda' and done['torch']=='2.4.0+cu121'
        assert sha(dest/'selection.json')==done['selection_sha256'] and sha(dest/'predictions.npz')==done['prediction_sha256']
        histories=[];weights=[]
        for j in range(5):
            if f==j:continue
            q=dest/('inner%d'%j);complete=load(q/'fit_complete.json');hist=load(q/'curve.json')
            assert complete['identity']==load(q/'identity.json')==ident(f,j,c['max_epochs'],'inner_epoch_selection')
            assert complete['epochs']==len(hist)==c['max_epochs'] and sha(q/'curve.json')==complete['curve_sha256']
            n=int(((d['fold']!=f)&(d['fold']!=j)).sum());nv=len(set(ids[d['fold']==j]))
            expected_steps=c['max_epochs']*math.ceil(2*n/c['batch_size'])
            assert complete['optimizer_steps']==expected_steps
            steps+=expected_steps;seconds+=sum(h['seconds'] for h in hist)
            assert [h['epoch'] for h in hist]==list(range(1,c['max_epochs']+1))
            for h in hist:
                assert all(np.isfinite(h[k]) for k in ('loss','scaled_mse','train_person_mae','validation_person_mae','seconds'))
                assert h['validation_people']==nv
                assert math.isclose(h['lr'],c['learning_rate']*c['lr_drop_factor']**(h['epoch']//c['lr_drop_every']),rel_tol=1e-12)
            histories.append(hist);weights.append(nv)
        means=np.average([[h['validation_person_mae'] for h in hist] for hist in histories],axis=0,weights=weights)
        epoch=int(np.argmin(means))+1
        np.testing.assert_allclose(means,selection['inner_mean_person_mae_by_epoch'],rtol=0,atol=1e-12)
        assert epoch==selection['selected_epoch']==done['selected_epoch'] and not selection['outer_evaluation_used_for_selection']
        q=dest/'refit';fit=load(q/'fit_complete.json');hist=load(q/'curve.json')
        assert fit['identity']==load(q/'identity.json')==done['identity']==ident(f,None,epoch,'outer_refit')
        assert fit['checkpoint_sha256']==done['checkpoint_sha256'] and sha(q/'curve.json')==fit['curve_sha256']
        assert len(hist)==fit['epochs']==epoch and [h['epoch'] for h in hist]==list(range(1,epoch+1))
        assert all('validation_person_mae' not in h for h in hist)
        expected_steps=epoch*math.ceil(2*int((d['fold']!=f).sum())/c['batch_size'])
        assert fit['optimizer_steps']==expected_steps
        steps+=expected_steps;seconds+=sum(h['seconds'] for h in hist)
        with np.load(dest/'predictions.npz',allow_pickle=False) as z:
            rows=np.flatnonzero(d['fold']==f);np.testing.assert_array_equal(rows,z['row_index'])
            for k in ('y','person_id','video_id','fold','cohort'):np.testing.assert_array_equal(d[k][rows],z[k])
            assert z['predictions'].shape==z['predictions_raw'].shape==(len(rows),1,2)
            cnn[rows]=z['predictions'][:,0];raw[rows]=z['predictions_raw'][:,0];seen[rows]+=1
            np.testing.assert_allclose(z['calibration'],done['calibration'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(z['calibration'],hist[-1]['calibration'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(cnn[rows],raw[rows]*z['calibration'][0]+z['calibration'][1],rtol=0,atol=1e-10)
        assert abs(metric(cnn,d['fold']==f)['mae']-done['evaluation_person_mae'])<1e-10
        assert abs(metric(raw,d['fold']==f)['mae']-done['raw_evaluation_person_mae'])<1e-10
        diagnostics.append(dict(fold=f,selected_epoch=epoch,inner_validation_at_selected=float(means[epoch-1]),
            refit_train_mae=hist[-1]['train_person_mae'],outer_person_mae=done['evaluation_person_mae'],calibration=done['calibration']))
        curves.extend(dict(fold=f,epoch=i+1,inner_validation_mae=float(v)) for i,v in enumerate(means))
    assert (seen==1).all() and np.isfinite(cnn).all() and np.isfinite(raw).all()
    summary=load(R/'summary.json')
    assert summary['fits']==25 and summary['optimizer_steps']==steps and summary['bundle_sha256']==sha(B/'manifest.json')
    assert sha(R/'oof_predictions.npz')==summary['oof_sha256']
    with np.load(R/'oof_predictions.npz',allow_pickle=False) as z:
        for k in d:np.testing.assert_array_equal(z[k],d[k])
        np.testing.assert_array_equal(z['predictions'][:,0],cnn);np.testing.assert_array_equal(z['predictions_raw'][:,0],raw)
    with (R/'predictions.csv').open(encoding='utf-8',newline='') as stream:csvrows=list(csv.DictReader(stream))
    assert len(csvrows)==len(y)*2
    for row,(i,j) in zip(csvrows,((i,j) for i in range(len(y)) for j in range(2))):
        assert row['person_id']==ids[i] and row['video_id']==d['video_id'][i] and int(row['fold'])==d['fold'][i]
        assert row['method']==c['name'] and int(row['seed'])==c['seed'] and row['role']=='outer_evaluation'
        assert row['side']==['left','right'][j] and float(row['actual_gdi'])==y[i,j] and float(row['predicted_gdi'])==cnn[i,j]
    with np.load(B/'reference_predictions.npz',allow_pickle=False) as z:predictions={k:z[k].copy() for k in z.files}
    with np.load(HERE.parent/'cnn-followup695/bundle/reference_predictions.npz',allow_pickle=False) as z:
        for k in z.files:np.testing.assert_array_equal(z[k][idx],predictions[k])
    with np.load(HERE.parent/'cnn-followup695/analysis/cnn_oof.npz',allow_pickle=False) as z:
        for k in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(z[k][idx],d[k])
        np.testing.assert_array_equal(z['predictions'][idx,0],predictions['previous_cnn'])
    predictions.update(paper_cnn=cnn,paper_cnn_raw=raw)
    metrics=[];foldmetrics=[]
    groups=[('retained_all',np.ones(len(y),bool)),('retained_old174',d['cohort']==0),('retained_new518',d['cohort']==1)]
    for name,p in predictions.items():
        assert p.shape==y.shape and np.isfinite(p).all()
        metrics.extend(dict(model=name,group=group,**metric(p,mask)) for group,mask in groups)
        foldmetrics.extend(dict(model=name,fold=f,**metric(p,d['fold']==f)) for f in range(5))
    for rec in summary['metrics']:
        actual=next(m for m in metrics if m['model']=='paper_cnn' and m['group']==rec['group'])
        for k,v in rec.items():
            if k!='group':assert abs(actual[k]-v)<1e-10,k
    for rec in summary['fold_metrics']:
        actual=next(m for m in foldmetrics if m['model']=='paper_cnn' and m['fold']==rec['fold'])
        for k,v in rec.items():assert abs(actual[k]-v)<1e-10,k
    for k,v in summary['raw_diagnostic_metrics'].items():assert abs(metric(raw,np.ones(len(y),bool))[k]-v)<1e-10
    for name in ('ridge','pls','motionbert_last_stage','previous_cnn'):
        assert abs(metric(predictions[name],np.ones(len(y),bool))['mae']-summary['reference_metrics'][name+'_mae'])<1e-10
    errors={name:np.array([np.abs(p[ids==person]-y[ids==person]).mean() for person in people]) for name,p in predictions.items()}
    draws=np.random.default_rng(20261029).integers(0,len(people),(10000,len(people)))
    comparisons=[]
    for name in ('ridge','pls','motionbert_last_stage','previous_cnn'):
        delta=errors['paper_cnn']-errors[name]
        comparisons.append(dict(reference=name,mae_reduction=float(-delta.mean()),relative_reduction_pct=float(-100*delta.mean()/errors[name].mean()),
            people_better=int((delta<0).sum()),people_worse=int((delta>0).sum()),
            folds_better=sum(metric(cnn,d['fold']==f)['mae']<metric(predictions[name],d['fold']==f)['mae'] for f in range(5)),
            exploratory95_reduction=np.percentile(-delta[draws].mean(1),[2.5,97.5]).tolist()))
    env=[load(p) for p in R.glob('environment-*.json')]
    assert any(e['slurm_job_id']=='31374' and e['device']=='cuda' and e['hostname']=='DIS01' for e in env)
    result=dict(metrics=metrics,comparisons=comparisons,fold_metrics=foldmetrics,training_diagnostics=diagnostics,
        optimizer_steps=steps,epoch_seconds_excluding_checkpoint=seconds,environment=env,independent_final_evaluation=False,
        bootstrap_note='Exploratory person resampling of fixed OOF predictions, no retraining or correction for repeated experimentation.',
        reference_training_cohort_note='References evaluated on same 692 people; originally trained on 695. No reference refit.')
    save(OUT/'summary.json',result)
    for name,rows in [('metrics',metrics),('fold_metrics',foldmetrics),('learning_curves',curves)]:table(OUT/(name+'.csv'),rows)
    np.savez_compressed(OUT/'cnn_oof.npz',predictions=cnn[:,None,:],predictions_raw=raw[:,None,:],**d)
    proof=dict(passed=True,archive_sha256=sha(HERE/'main-review.tar.gz'),bundle_sha256=sha(B/'manifest.json'),matched_people=692,matched_records=1061,
        completed_fits=25,real_gpu_optimizer_steps=steps,all_split_and_identity_records_match=True,code_and_protocol_match=True,
        weighted_epoch_selection_recomputed=True,all_prediction_curve_hashes_match=True,calibration_applied_as_recorded=True,
        oof_csv_and_fold_predictions_match=True,metrics_independently_recomputed=True,reference_predictions_match_prior_artifacts=True,
        checkpoints_downloaded=False,independent_model_reinference=False,protected_final_people_used=False)
    save(OUT/'verification.json',proof)
    print(json.dumps(dict(metrics=[m for m in metrics if m['group']=='retained_all'],comparisons=comparisons,folds=diagnostics,verification=proof),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
