"""CPU checkpoint replay and matched-person comparisons; no training or selection."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'bundle'))
import torch
import run

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    run.setup(20261028,'cpu');d=run.data();c=run.config();r=HERE/'received/results';out=HERE/'analysis'
    checks=[]
    for f in range(5):
        tr,_,te=run.split(d,f);dest=r/('fold%d'%f);done=load(dest/'complete.json')
        path=HERE/'received/checkpoints'/('fold%d.pt'%f)
        assert sha(path)==done['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=True)
        assert ck['identity']==done['identity'] and ck['history']==load(dest/'refit/curve.json')
        lo,span=run.scale_from_train(d['y'][tr])
        assert torch.equal(lo,ck['lo']) and torch.equal(span,ck['span'])
        m=run.GdiCNN(c['dropout']);m.load_state_dict(ck['model'],strict=True)
        train_raw=run.predict(m,d['x'][tr],lo,span,'cpu')
        coef=run.calibrate(train_raw,d['y'][tr]);raw=run.predict(m,d['x'][te],lo,span,'cpu')
        with np.load(dest/'predictions.npz',allow_pickle=False) as z:
            raw_error=float(np.max(np.abs(raw-z['predictions_raw'][:,0])))
            prediction_error=float(np.max(np.abs(run.corrected(raw,coef)-z['predictions'][:,0])))
        assert raw_error<1e-4 and prediction_error<1e-4,(f,raw_error,prediction_error)
        row=dict(fold=f,checkpoint_sha256=sha(path),raw_max_error=raw_error,prediction_max_error=prediction_error,
                 training_only_calibration_max_error=float(np.max(np.abs(coef-done['calibration']))),
                 train_records=int(tr.sum()),evaluation_records=int(te.sum()))
        checks.append(row);print(json.dumps(row),flush=True)
    with np.load(out/'cnn_oof.npz',allow_pickle=False) as z:multi=z['predictions'][:,0].copy()
    predictions={'multiwindow_cnn':multi}
    with np.load(HERE.parent/'cnn-motionbert-fusion692/bundle/reference_predictions.npz',allow_pickle=False) as z:
        for k in ('single_cnn','ridge','pls','motionbert_last_stage'):predictions[k]=z[k].copy()
    # Reference file is itself verified against the baseline artifact and original reference data below.
    def add(name,p):
        with np.load(p,allow_pickle=False) as z:
            for k in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(z[k],d[k])
            a=z['predictions'].copy()
            predictions[name]=a[:,0] if a.ndim==3 else a
    add('single_cnn_recheck',HERE.parent/'cnn-paper-input695/analysis/cnn_oof.npz')
    np.testing.assert_array_equal(predictions.pop('single_cnn_recheck'),predictions['single_cnn'])
    with np.load(HERE/'bundle/reference_predictions.npz',allow_pickle=False) as z:
        for k in ('ridge','pls','motionbert_last_stage'):np.testing.assert_array_equal(predictions[k],z[k])
    base=HERE.parent/'cnn-frozen-residual692'
    add('cpu_output_correction',base/'cpu-output-control/oof_predictions.npz')
    add('mb_residual_20261028',base/'analysis/residual_oof.npz')
    for seed in (20261029,20261030):add('mb_residual_%d'%seed,base/('repeats/seed%d/analysis/residual_oof.npz'%seed))
    add('direct_mb_cnn',HERE.parent/'cnn-motionbert-fusion692/analysis/cnn_oof.npz')
    ids=d['person_id'];people=np.unique(ids);y=d['y'];fold=d['fold']
    errors={k:np.array([np.abs(p[ids==i]-y[ids==i]).mean() for i in people]) for k,p in predictions.items()}
    errors['mb_residual_three_run_mean']=np.mean([errors['mb_residual_%d'%s] for s in (20261028,20261029,20261030)],axis=0)
    metrics=[dict(model=k,**run.metrics(y,p,ids)) for k,p in predictions.items()]
    draws=np.random.default_rng(20261031).integers(0,len(people),(10000,len(people)))
    comparisons=[]
    for k,e in errors.items():
        if k=='multiwindow_cnn':continue
        delta=e-errors['multiwindow_cnn']
        comparisons.append(dict(reference=k,mae=float(e.mean()),mae_reduction=float(delta.mean()),relative_reduction_pct=float(100*delta.mean()/e.mean()),
            people_better=int((delta>0).sum()),people_worse=int((delta<0).sum()),
            exploratory95_reduction=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist()))
    folds=[dict(fold=f,**{k:run.person_mae(y[fold==f],p[fold==f],ids[fold==f]) for k,p in predictions.items()}) for f in range(5)]
    mean_y=np.array([y[ids==i].mean() for i in people]);groups=[]
    for label,mask in [('below60',mean_y<60),('60to70',(mean_y>=60)&(mean_y<70)),('70to80',(mean_y>=70)&(mean_y<80)),('80to90',(mean_y>=80)&(mean_y<90)),('90plus',mean_y>=90)]:
        groups.append(dict(group=label,people=int(mask.sum()),**{k:float(e[mask].mean()) for k,e in errors.items()}))
    result=dict(metrics=metrics,comparisons=comparisons,folds=folds,actual_GDI_groups=groups,
        three_seed_mean_is_not_ensemble=True,independent_final_evaluation=False,
        bootstrap_note='Fixed person-level OOF resampling; exploratory, without retraining or multiplicity correction.',
        reference_note='Ridge, PLS and MotionBERT partial-training models originally trained on 695 people; compared on common 692. Others trained on retained 692.')
    save(out/'comparison.json',result)
    save(out/'checkpoint_replay.json',dict(passed=True,device='cpu',torch=str(torch.__version__),folds=checks,
        all1061_predictions_reinferred=True,all_training_only_scales_and_calibrations_recomputed=True,tolerance=1e-4))
    proof=load(out/'verification.json');proof.update(checkpoints_downloaded=True,independent_model_reinference=True,
        checkpoint_replay_artifact='checkpoint_replay.json',matched_comparison_artifact='comparison.json')
    save(out/'verification.json',proof)
    print(json.dumps(dict(metrics=metrics,comparisons=comparisons,groups=groups),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
