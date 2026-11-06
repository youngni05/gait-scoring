"""Replay saved models and independently recompute participant-level outcomes."""
from common import HERE, ROOT, arrays, data, load, save, sha, split
import argparse
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def independent_metrics(d,methods,reported):
    ids=sorted(set(d['person_id']))
    for name,pred in methods.items():
        assert pred.shape==(1064,2) and np.isfinite(pred).all()
        person_errors=[];bias=[];exact=[];large=[]
        for p in ids:
            rows=[i for i,v in enumerate(d['person_id']) if v==p]
            residual=pred[rows]-d['y'][rows]
            distance=np.abs(np.floor(pred[rows]/10)-np.floor(d['y'][rows]/10))
            person_errors.append(float(np.abs(residual).mean()))
            bias.append(float(residual.mean()))
            exact.append(float((distance==0).mean()));large.append(float((distance>=2).mean()))
        for key,value in [('MAE',np.mean(person_errors)),('bias',np.mean(bias)),
                          ('exact_10point_bin_rate',np.mean(exact)),('two_or_more_bins_rate',np.mean(large))]:
            assert abs(reported['metrics'][name][key]-value)<1e-10,(name,key)


def verify_rf(d):
    report=load(HERE/'rf_summary.json');protocol=load(HERE/'protocol.json')['RF']
    predictions=arrays(HERE/'rf_predictions.npz')
    reps={'rf_coordinate':d['kinematic'],'rf_motionbert':d['motionbert'],
          'rf_concatenated':np.column_stack([d['kinematic'],d['motionbert']])}
    maxdiff=0.;refits=0
    for name,x in reps.items():
        for outer in range(5):
            folder=HERE/'rf'/name/f'fold{outer}';done=load(folder/'complete.json')
            assert sha(folder/'model.joblib')==done['model_sha256']
            assert sha(folder/'inner.npz')==done['inner_sha256']
            assert sha(folder/'prediction.npz')==done['prediction_sha256']
            tr,te=split(d,outer);inner=arrays(folder/'inner.npz')
            assert set(done['train_people'])==set(d['person_id'][tr]) and set(done['evaluation_people'])==set(d['person_id'][te])
            np.testing.assert_array_equal(inner['train_indices'],tr)
            scores=[]
            for p in inner['predictions']:
                scores.append(float(np.mean([abs(p[d['person_id'][tr]==v]-d['y'][tr][d['person_id'][tr]==v]).mean() for v in sorted(set(d['person_id'][tr]))])))
            np.testing.assert_allclose(scores,done['candidate_scores'],rtol=0,atol=1e-12)
            selected=min(i for i,v in enumerate(scores) if v<=min(scores)+1e-12)
            assert selected==done['selected'] and done['parameters']==protocol['candidates'][selected]
            model=joblib.load(folder/'model.joblib')
            replay=model.predict(x[te])
            np.testing.assert_allclose(replay,predictions[name][te],rtol=0,atol=1e-10)
            maxdiff=max(maxdiff,float(abs(replay-predictions[name][te]).max()))
            # Independent selected-candidate inner refit, using training rows only.
            it,iv=split(d,outer,0)
            refit=RandomForestRegressor(n_estimators=protocol['n_estimators'],**protocol['candidates'][selected],
                random_state=protocol['seed']+outer*100,n_jobs=2).fit(x[it],d['y'][it])
            expected=inner['predictions'][selected,np.searchsorted(tr,iv)]
            np.testing.assert_allclose(refit.predict(x[iv]),expected,rtol=0,atol=1e-10)
            refits+=1
    np.testing.assert_array_equal(predictions['rf_coordinate_pls_equal'],(predictions['rf_coordinate']+d['pls'])/2)
    np.testing.assert_array_equal(predictions['rf_concatenated_pls_equal'],(predictions['rf_concatenated']+d['pls'])/2)
    independent_metrics(d,predictions,report)
    assert all(sha(HERE/name)==value for name,value in report['sha256'].items())
    proof={'verified':True,'outer_models_replayed':15,'independent_inner_refits':refits,
           'outer_prediction_max_abs_difference':maxdiff,'metrics_independently_recomputed':True,
           'source_and_split_checks':True,'protected_numeric_data_read':False,'summary_sha256':sha(HERE/'rf_summary.json')}
    save('rf_verification.json',proof)
    print(json.dumps({'RF':proof}),flush=True)


def verify_cnn(d,mode):
    import train_cnn as c
    import torch
    out=HERE/'cnn'/mode;report=load(out/'summary.json');prediction=arrays(out/'oof_predictions.npz')
    c.runner.setup(20261028,'cpu')
    x=d['x']
    if mode=='temporal':
        proof=load(HERE/'frame_features.json');assert sha(HERE/'frame_features.npy')==proof['sha256']
        x=np.load(HERE/'frame_features.npy',mmap_mode='r',allow_pickle=False)
        c.use_temporal_model()
    refits=0;maxdiff=0.;steps=0
    for outer in range(5):
        dest=out/f'fold{outer}';histories=[]
        for inner in range(4):
            it,iv=split(d,outer,inner)
            ident=load(dest/f'inner{inner}/identity.json')
            assert ident==c.identity(d,outer,inner,100,mode+'_inner_epoch_selection')
            fit=load(dest/f'inner{inner}/fit_complete.json');curve=load(dest/f'inner{inner}/curve.json')
            assert fit['epochs']==len(curve)==100 and sha(dest/f'inner{inner}/latest.pt')==fit['checkpoint_sha256']
            assert sha(dest/f'inner{inner}/curve.json')==fit['curve_sha256']
            checkpoint=torch.load(dest/f'inner{inner}/latest.pt',map_location='cpu',weights_only=True)
            lo,span=c.runner.scale_from_train(d['y'][it])
            assert torch.equal(lo,checkpoint['lo']) and torch.equal(span,checkpoint['span'])
            assert not set(ident['train_people']) & (set(ident['validation_people'])|set(ident['outer_evaluation_people']))
            histories.append(curve);steps+=fit['optimizer_steps']
        expected_epoch=int(np.argmin(np.asarray([[r['validation_person_mae'] for r in h] for h in histories]).mean(0)))+1
        selection=load(dest/'selection.json');done=load(dest/'complete.json')
        assert expected_epoch==selection['selected_epoch']==done['selected_epoch']
        assert done['identity']==c.identity(d,outer,None,expected_epoch,mode+'_outer_refit')
        assert sha(dest/'refit/latest.pt')==done['checkpoint_sha256']
        tr,te=split(d,outer)
        ck=torch.load(dest/'refit/latest.pt',map_location='cpu',weights_only=True)
        lo,span=c.runner.scale_from_train(d['y'][tr])
        assert torch.equal(lo,ck['lo']) and torch.equal(span,ck['span'])
        c.runner.setup(done['identity']['fit_seed'],'cpu')
        model=c.runner.GdiCNN(.5)
        initial={k:v.clone() for k,v in model.state_dict().items()}
        model.load_state_dict(ck['model'],strict=True)
        assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
        model.eval()
        replay=c.runner.predict(model,c.runner.channels(torch.from_numpy(np.asarray(x[te]))),lo,span,'cpu')
        np.testing.assert_allclose(replay,prediction[mode+'_cnn'][te],rtol=0,atol=1e-6)
        maxdiff=max(maxdiff,float(abs(replay-prediction[mode+'_cnn'][te]).max()))
        steps+=load(dest/'refit/fit_complete.json')['optimizer_steps'];refits+=1
    np.testing.assert_array_equal(prediction[mode+'_cnn_pls_equal'],(prediction[mode+'_cnn']+d['pls'])/2)
    if mode=='raw':
        rf=arrays(HERE/'rf_predictions.npz')
        np.testing.assert_array_equal(prediction['raw_cnn_rf_coordinate_equal'],(prediction['raw_cnn']+rf['rf_coordinate'])/2)
    independent_metrics(d,prediction,report)
    proof={'verified':True,'inner_fits_checked':20,'outer_models_replayed':refits,
           'actual_optimizer_steps':steps,'prediction_max_abs_difference':maxdiff,
           'training_target_scalers_checked':25,'nested_epoch_selection_recomputed':True,
           'encoder_finetuned':False,'protected_numeric_data_read':False,'summary_sha256':sha(out/'summary.json')}
    save(f'cnn/{mode}/verification.json',proof)
    print(json.dumps({mode:proof}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['rf','raw','temporal','all']);a=p.parse_args()
    d=data()
    for outer in range(5):
        split(d,outer)
        for inner in range(4):split(d,outer,inner)
    if a.phase in ['rf','all']:verify_rf(d)
    if a.phase in ['raw','all']:verify_cnn(d,'raw')
    if a.phase in ['temporal','all']:verify_cnn(d,'temporal')


if __name__=='__main__':main()
