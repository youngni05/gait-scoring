"""CPU control: CNN's two predictions -> affine residual, without MotionBERT inputs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'bundle'
sys.path.insert(0,str(SOURCE))
import run as baseline_runner


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def train(base,y):
    """Six affine coefficients fitted only on the supplied training rows."""
    assert base.shape==y.shape and base.ndim==2 and base.shape[1]==2
    mean=base.mean(axis=0);scale=base.std(axis=0);scale[scale<1e-8]=1
    design=np.column_stack([(base-mean)/scale,np.ones(len(base))])
    coefficient=np.linalg.lstsq(design,y-base,rcond=None)[0]
    assert coefficient.shape==(3,2) and np.isfinite(coefficient).all()
    return dict(mean=mean.tolist(),scale=scale.tolist(),coefficient=coefficient.tolist())


def predict(base,model):
    design=np.column_stack([(base-np.asarray(model['mean']))/np.asarray(model['scale']),np.ones(len(base))])
    return base+design@np.asarray(model['coefficient'])


def macro(y,pred,ids):return float(np.mean([np.abs(pred[ids==p]-y[ids==p]).mean() for p in np.unique(ids)]))


def check_synthetic():
    rng=np.random.default_rng(17)
    x=rng.normal(70,10,(40,2));y=x+np.column_stack([.2*x[:,0]-10.,-.1*x[:,1]+5.])
    model=train(x[:30],y[:30])
    assert np.max(np.abs(predict(x[30:],model)-y[30:]))<1e-10
    altered=y.copy();altered[30:]+=1000
    assert train(x[:30],altered[:30])==model
    assert np.array_equal(x,x.copy())
    return dict(passed=True,known_affine_relation_recovered=True,heldout_labels_do_not_change_fit=True)


def main():
    HERE.mkdir(exist_ok=True)
    synthetic=check_synthetic()
    torch.set_num_threads(4)
    manifest=read(SOURCE/'manifest.json')
    # No MotionBERT feature array or encoder is loaded in this control.
    for name in ('data.npz','base_model.py','sources.json'):
        assert sha(SOURCE/name)==manifest[name]
    assert sha(SOURCE/'manifest.json')=='bf76bc5bc73fc9226d907d90026d9f8c5ad0944baf86e21d79448022357475f2'
    with np.load(SOURCE/'data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    ids=d['person_id'];y=d['y'];n=len(y)
    assert len(np.unique(ids))==692 and n==1061
    output=np.full_like(y,np.nan,dtype=float);base_all=np.full_like(output,np.nan);seen=np.zeros(n,int)
    folds=[]
    for f in range(5):
        dest=HERE/('fold%d'%f);dest.mkdir(exist_ok=True)
        scores=[];counts=[]
        for j in range(5):
            if f==j:continue
            tr,va,te=baseline_runner.split(d,f,j)
            rows=np.flatnonzero(tr|va)
            bp=np.full_like(y,np.nan,dtype=float)
            bp[rows]=baseline_runner.baseline(d,f,j,'cpu',rows)
            model=train(bp[tr],y[tr])
            candidate=predict(bp[va],model)
            score=[macro(y[va],bp[va],ids[va]),macro(y[va],candidate,ids[va])]
            scores.append(score);counts.append(len(np.unique(ids[va])))
            # Replay the serialized coefficients and re-check training-only statistics.
            save(dest/('inner%d.json'%j),dict(model=model,mae=score,
                train_people=np.unique(ids[tr]).tolist(),validation_people=np.unique(ids[va]).tolist(),
                outer_evaluation_people=np.unique(ids[te]).tolist(),
                baseline_checkpoint_sha256=sha(baseline_runner.baseline_path(f,j)/'latest.pt')))
            np.testing.assert_array_equal(predict(bp[va],read(dest/('inner%d.json'%j))['model']),candidate)
        means=np.average(scores,axis=0,weights=counts)
        selected=int(np.argmin(means))  # Exact tie keeps the baseline.
        tr,va,te=baseline_runner.split(d,f)
        train_rows=np.flatnonzero(tr);test_rows=np.flatnonzero(te)
        bt=baseline_runner.baseline(d,f,None,'cpu',train_rows)
        model=train(bt,y[tr])
        be=baseline_runner.baseline(d,f,None,'cpu',test_rows)
        pred=predict(be,model) if selected else be.copy()
        output[test_rows]=pred;base_all[test_rows]=be;seen[test_rows]+=1
        record=dict(fold=f,use_correction=bool(selected),inner_weighted_mae=means.tolist(),model=model,
            train_people=np.unique(ids[tr]).tolist(),outer_evaluation_people=np.unique(ids[te]).tolist(),
            mae=macro(y[te],pred,ids[te]),baseline_mae=macro(y[te],be,ids[te]),
            outer_evaluation_used_for_selection=False,baseline_checkpoint_sha256=sha(baseline_runner.baseline_path(f,None)/'latest.pt'))
        save(dest/'result.json',record)
        np.savez_compressed(dest/'predictions.npz',row_index=test_rows,y=y[te],person_id=ids[te],predictions=pred,baseline=be)
        folds.append(record)
        print('CPU_CONTROL_FOLD',f,record['mae'],selected,flush=True)
    assert (seen==1).all() and np.isfinite(output).all()
    np.testing.assert_allclose(base_all,np.load(SOURCE/'single_oof.npy',allow_pickle=False),rtol=0,atol=1e-4)
    np.savez_compressed(HERE/'oof_predictions.npz',predictions=output,baseline=base_all,**{k:v for k,v in d.items() if k!='x'})
    with np.load(HERE.parent/'analysis/residual_oof.npz',allow_pickle=False) as z:
        for k in ('person_id','video_id','fold','y'):np.testing.assert_array_equal(z[k],d[k])
        motion=z['predictions'].copy()
    predictions={'single_cnn':base_all,'cnn_output_only_control':output,'motionbert_residual_seed20261028':motion}
    metrics={}
    people=np.unique(ids)
    for name,p in predictions.items():
        dist=np.abs(np.floor(p/10)-np.floor(y/10))
        def avg(v):return float(np.mean([v[ids==q].mean() for q in people]))
        metrics[name]=dict(mae=avg(np.abs(p-y)),bias=avg(p-y),exact_10point_bin_rate=avg(dist==0),two_or_more_bins_rate=avg(dist>=2))
    errors={k:np.array([np.abs(p[ids==q]-y[ids==q]).mean() for q in people]) for k,p in predictions.items()}
    draws=np.random.default_rng(20261029).integers(0,len(people),(10000,len(people)))
    delta=errors['cnn_output_only_control']-errors['motionbert_residual_seed20261028']
    summary=dict(method='training_only_affine_residual_from_two_CNN_predictions',people=692,records=1061,
        motionbert_features_used=False,trainable_coefficients=6,selection='inner_person_weighted_MAE_apply_or_skip',metrics=metrics,
        correction_selected_by_fold=[r['use_correction'] for r in folds],folds=folds,
        motionbert_vs_control_mae_reduction=float(delta.mean()),exploratory95_motionbert_vs_control_reduction=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist(),
        note='Not capacity-matched to 9026-parameter temporal correction. Fixed OOF bootstrap; no retraining variability. All experimental models remain development comparisons.')
    save(HERE/'summary.json',summary)
    save(HERE/'verification.json',dict(passed=True,synthetic=synthetic,device='cpu',motionbert_features_loaded=False,
        baseline_checkpoints_verified=25,all_splits_disjoint=True,training_only_normalization=True,outer_selection_forbidden=True,
        all_records_evaluated_once=True,baseline_replay_max_error=float(np.max(np.abs(base_all-np.load(SOURCE/'single_oof.npy',allow_pickle=False)))),
        oof_sha256=sha(HERE/'oof_predictions.npz')))
    print(json.dumps({k:v for k,v in summary.items() if k!='folds'},indent=2))


if __name__=='__main__':main()
