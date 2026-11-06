"""Independent returned-result audit and saved-model CPU replay; no optimization."""
from pathlib import Path
import argparse,hashlib,json,math,sys,tarfile
import numpy as np
import torch

SOURCE=Path(__file__).resolve().parent
HERE=SOURCE
B=SOURCE/'bundle'
R=HERE/'received/results'
OUT=HERE/'analysis'
sys.path.insert(0,str(B))
import run


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    global HERE,R,OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeat-seed',type=int,choices=[20261029,20261030])
    parser.add_argument('--archive-sha256')
    args=parser.parse_args()
    if args.repeat_seed:
        assert args.archive_sha256 and len(args.archive_sha256)==64
        HERE=SOURCE/'repeats'/('seed%d'%args.repeat_seed)
        R=HERE/'received/results';OUT=HERE/'analysis'
    OUT.mkdir(exist_ok=True)
    archive=HERE/'main-review.tar.gz'
    archive_sha=args.archive_sha256 if args.repeat_seed else '6ee336e680cc8e473a5c247f0913ec70ba8086812d30bf63afc9614e65b255bf'
    assert sha(archive)==archive_sha
    with tarfile.open(archive) as tf:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in tf.getnames())
        tf.extractall(HERE/'received',filter='data')
    original_manifest=read(B/'manifest.json')
    for name,digest in original_manifest.items():assert sha(B/name)==digest,name
    manifest=read(HERE/'received/bundle/manifest.json')
    bundle_sha=sha(HERE/'received/bundle/manifest.json')
    c=read(HERE/'received/bundle/protocol.json')
    original_config=read(B/'protocol.json')
    if args.repeat_seed:
        prep=read(SOURCE/'repeats/preparation.json')
        variant=next(v for v in prep['variants'] if v['seed']==args.repeat_seed)
        assert prep['source_manifest_sha256']==sha(B/'manifest.json')
        assert bundle_sha==variant['manifest_sha256']
        assert c==dict(original_config,seed=args.repeat_seed)
        assert manifest==dict(original_manifest,**{'protocol.json':sha(HERE/'received/bundle/protocol.json')})
    else:
        assert bundle_sha==sha(B/'manifest.json') and c==original_config
    for name in ('run.py','residual.py','base_model.py','protocol.json','sources.json'):
        assert sha(HERE/'received/bundle'/name)==manifest[name]
    with np.load(B/'data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    y=d['y'];ids=d['person_id'];people=np.unique(ids)
    m=np.load(B/'features.npy',mmap_mode='r',allow_pickle=False)
    assert len(y)==1061 and len(people)==692 and np.isfinite(m).all()
    def expected(f,j,epochs):
        te=d['fold']==f;va=np.zeros(len(y),bool) if j is None else d['fold']==j;tr=~(te|va)
        groups=[np.unique(ids[v]).tolist() for v in (tr,va,te)]
        assert not (set(groups[0])&set(groups[1]) or set(groups[0])&set(groups[2]) or set(groups[1])&set(groups[2]))
        path=B/'base'/('fold%d'%f)/('refit' if j is None else 'inner%d'%j)/'latest.pt'
        return dict(bundle_sha256=bundle_sha,outer_fold=f,inner_fold=j,epochs=epochs,
            seed=c['seed']+f*100+(50 if j is None else j),baseline_checkpoint_sha256=sha(path),
            train_people=groups[0],validation_people=groups[1],outer_evaluation_people=groups[2])
    def metric(p,mask=None):
        mask=np.ones(len(y),bool) if mask is None else mask
        def macro(a):return float(np.mean([a[(ids==q)&mask].mean() for q in np.unique(ids[mask])]))
        dist=np.abs(np.floor(p/10)-np.floor(y/10))
        return dict(mae=macro(np.abs(p-y)),bias=macro(p-y),exact_10point_bin_rate=macro(dist==0),two_or_more_bins_rate=macro(dist>=2))
    pred=np.full_like(y,np.nan);base=np.full_like(y,np.nan);seen=np.zeros(len(y),int)
    steps=0;folds=[];replay_max=0.0
    torch.set_num_threads(4)
    for f in range(5):
        root=R/('fold%d'%f);done=read(root/'complete.json');sel=read(root/'selection.json')
        assert done['device']=='cuda' and done['torch']=='2.4.0+cu121'
        assert sha(root/'selection.json')==done['selection_sha256'] and sha(root/'predictions.npz')==done['prediction_sha256']
        curves=[];weights=[]
        for j in range(5):
            if j==f:continue
            p=root/('inner%d'%j);info=read(p/'fit_complete.json');hist=read(p/'curve.json')
            assert info['identity']==expected(f,j,100)
            assert info['epochs']==100 and [h['epoch'] for h in hist]==list(range(101))
            assert sha(p/'curve.json')==info['curve_sha256']
            n=int(((d['fold']!=f)&(d['fold']!=j)).sum());nv=len(np.unique(ids[d['fold']==j]))
            assert info['optimizer_steps']==100*math.ceil(n/c['batch_size'])
            assert all(h['validation_people']==nv and all(np.isfinite(h[k]) for k in ('loss','train_person_mae','validation_person_mae')) for h in hist)
            curves.append([h['validation_person_mae'] for h in hist]);weights.append(nv);steps+=info['optimizer_steps']
        means=np.average(curves,axis=0,weights=weights);ep=int(np.argmin(means))
        np.testing.assert_allclose(means,sel['inner_weighted_person_mae'],rtol=0,atol=1e-12)
        assert ep==sel['selected_epoch']==done['selected_epoch'] and sel['includes_zero_correction'] and not sel['outer_evaluation_used']
        tr=d['fold']!=f;te=~tr
        info=read(root/'refit/fit_complete.json');hist=read(root/'refit/curve.json')
        assert info['identity']==done['identity']==expected(f,None,ep)
        assert info['epochs']==ep and [h['epoch'] for h in hist]==list(range(ep+1))
        assert all('validation_person_mae' not in h for h in hist)
        assert info['optimizer_steps']==ep*math.ceil(int(tr.sum())/c['batch_size'])
        assert sha(root/'refit/curve.json')==info['curve_sha256']
        steps+=info['optimizer_steps']
        rows=np.flatnonzero(te)
        with np.load(root/'predictions.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['row_index'],rows);np.testing.assert_array_equal(z['y'],y[te]);np.testing.assert_array_equal(z['person_id'],ids[te])
            assert z['predictions'].shape==z['baseline'].shape==(len(rows),2)
            pred[rows]=z['predictions'];base[rows]=z['baseline'];seen[rows]+=1
        path=HERE/'received/checkpoints'/('fold%d.pt'%f)
        assert sha(path)==info['checkpoint_sha256']==done['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=True)
        assert ck['identity']==info['identity'] and ck['history']==hist
        model=run.Correction();model.load_state_dict(ck['model'],strict=True);model.requires_grad_(False)
        v=np.asarray(m[tr],dtype=np.float64);mu=v.mean(axis=(0,1)).astype(np.float32);sd=v.std(axis=(0,1)).astype(np.float32);sd[sd<1e-6]=1
        np.testing.assert_array_equal(model.mean.numpy(),mu);np.testing.assert_array_equal(model.scale.numpy(),sd)
        del v
        cpu_base=run.baseline(d,f,None,'cpu',rows)
        cpu=run.predict(model,m[te],cpu_base,'cpu')
        error=float(np.max(np.abs(cpu-pred[te])));assert error<1e-4
        replay_max=max(error,replay_max)
        actual=metric(pred,te);bm=metric(base,te)
        assert abs(actual['mae']-done['mae'])<1e-12 and abs(bm['mae']-done['baseline_mae'])<1e-12
        folds.append(dict(fold=f,selected_epoch=ep,mae=actual['mae'],single_cnn_mae=bm['mae'],reduction=bm['mae']-actual['mae'],cpu_replay_max_error=error))
        print('VERIFIED_FOLD',f,flush=True)
    assert (seen==1).all() and np.isfinite(pred).all()
    summary=read(R/'summary.json')
    assert summary['fits']==25 and summary['optimizer_steps']==steps and summary['bundle_sha256']==bundle_sha
    if not args.repeat_seed:assert steps==47185
    assert sha(R/'oof_predictions.npz')==summary['oof_sha256']
    with np.load(R/'oof_predictions.npz',allow_pickle=False) as z:
        for k in d:
            if k!='x':np.testing.assert_array_equal(z[k],d[k])
        np.testing.assert_array_equal(z['predictions'],pred);np.testing.assert_array_equal(z['baseline'],base)
    assert abs(summary['mae']-metric(pred)['mae'])<1e-12
    assert abs(summary['baseline_mae']-metric(base)['mae'])<1e-12
    np.testing.assert_allclose(base,np.load(B/'single_oof.npy',allow_pickle=False),rtol=0,atol=1e-4)
    with np.load(SOURCE.parent/'cnn-motionbert-fusion692/bundle/reference_predictions.npz',allow_pickle=False) as z:
        predictions={k:z[k].copy() for k in ('single_cnn','pls','ridge','motionbert_last_stage')}
    with np.load(SOURCE.parent/'cnn-motionbert-fusion692/analysis/cnn_oof.npz',allow_pickle=False) as z:
        for k in ('person_id','video_id','fold','y'):np.testing.assert_array_equal(z[k],d[k])
        predictions['direct_fusion']=z['predictions'][:,0]
    np.testing.assert_array_equal(predictions['single_cnn'],np.load(B/'single_oof.npy',allow_pickle=False))
    predictions['residual']=pred
    comparison_names=['single_cnn','pls','ridge','direct_fusion']
    if args.repeat_seed:
        for name,path in [('original_residual',SOURCE/'analysis/residual_oof.npz'),('cpu_output_control',SOURCE/'cpu-output-control/oof_predictions.npz')]:
            with np.load(path,allow_pickle=False) as z:
                for k in ('person_id','video_id','fold','y'):np.testing.assert_array_equal(z[k],d[k])
                predictions[name]=z['predictions'].copy()
            comparison_names.append(name)
    metrics={k:metric(v) for k,v in predictions.items()}
    errors={k:np.array([np.abs(v[ids==q]-y[ids==q]).mean() for q in people]) for k,v in predictions.items()}
    draws=np.random.default_rng(20261029).integers(0,len(people),(10000,len(people)))
    comparisons={}
    for k in comparison_names:
        delta=errors[k]-errors['residual']
        comparisons[k]=dict(mae_reduction=float(delta.mean()),relative_reduction_pct=float(100*delta.mean()/errors[k].mean()),
            people_improved=int((delta>0).sum()),people_worsened=int((delta<0).sum()),exploratory95_reduction=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist())
    env=read(R/'environment.json');assert env['slurm_job_id']==('31755' if args.repeat_seed else '31659') and env['device']=='cuda'
    result=dict(seed=c['seed'],people=692,records=1061,metrics=metrics,comparisons=comparisons,folds=folds,optimizer_steps=steps,
        note='Fixed OOF person bootstrap only; no retraining variability or repeated-experiment correction. Older PLS/Ridge trained on 695, evaluated on common 692.')
    save(OUT/'summary.json',result)
    np.savez_compressed(OUT/'residual_oof.npz',predictions=pred,baseline=base,**{k:v for k,v in d.items() if k!='x'})
    proof=dict(passed=True,seed=c['seed'],archive_sha256=sha(archive),bundle_sha256=bundle_sha,fits=25,optimizer_steps=steps,
        all_split_identities_match=True,epoch_selection_recomputed=True,all_predictions_and_metrics_recomputed=True,
        train_only_normalization_recomputed=True,checkpoints_reinferred=5,records_reinferred=1061,
        cpu_replay_max_error=replay_max,protected_final_data_used=False)
    save(OUT/'verification.json',proof)
    print(json.dumps(result,indent=2))
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
