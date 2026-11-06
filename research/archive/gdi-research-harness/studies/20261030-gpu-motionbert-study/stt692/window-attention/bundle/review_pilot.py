"""Recompute pilot metrics, identities, checkpoint outputs and window weights."""
from pathlib import Path
import argparse,math,numpy as np,torch
import run as r
from bags import predict_details,weight_summary
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['mean','attention'],required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r.setup(20261029,a.device);d=r.data();c=r.load(r.ROOT/'protocol.json');c.update(pooling=a.variant,name='record_loss_STT_'+a.variant)
    tr,va,te=r.split(d,0,1);s=r.load(a.input/'pilot_summary.json');ck=torch.load(a.input/'fit/latest.pt',map_location=a.device,weights_only=True)
    assert s['pilot_only'] and s['variant']==a.variant and not s['outer_evaluation_used'] and s['epochs']==3
    assert s['checkpoint_sha256']==r.sha(a.input/'fit/latest.pt')
    steps=3*math.ceil(int(tr.sum())/c['batch_records']);assert s['optimizer_steps']==steps
    ident=dict(bundle_sha256=r.sha(r.ROOT/'manifest.json'),seed=c['seed']+1,epochs=3,outer_fold=0,inner_fold=1,
        train_people=np.unique(d['person_id'][tr]).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),outer_evaluation_people=np.unique(d['person_id'][te]).tolist(),model_variant=a.variant,training_config=c)
    for prefix,mask in [('train',tr),('validation',va)]:
        for field,key in [('x','x'),('y','y'),('ids','person_id')]:ident[prefix+'_'+field+'_sha']=r.digest_array(d[key][mask])
    assert ck['identity']==ident==r.load(a.input/'fit/identity.json')
    h=r.load(a.input/'fit/curve.json');done=r.load(a.input/'fit/fit_complete.json');assert h==ck['history'] and len(h)==3
    assert done['identity']==ident and done['optimizer_steps']==steps and done['checkpoint_sha256']==s['checkpoint_sha256'] and done['curve_sha256']==r.sha(a.input/'fit/curve.json')
    assert ck['lo']==float(np.float32(d['y'][tr].min())) and ck['span']==float(np.float32(np.ptp(d['y'][tr])))
    assert all(int(v['step'])==steps for v in ck['optimizer']['state'].values())
    for row in h:
        path=a.input/'fit/raw_predictions'/('epoch%04d.npz'%row['epoch']);assert r.sha(path)==row['raw_predictions_sha256']
        with np.load(path) as z:
            coef=r.calibration(z['train_raw'],d['y'][tr]);np.testing.assert_allclose(coef,row['calibration'],atol=1e-10,rtol=0)
            for label,mask,key in [('train',tr,'train_raw'),('validation',va,'validation_raw')]:
                assert abs(r.metric(d['y'][mask],z[key],d['person_id'][mask])-row[label+'_raw_person_mae'])<1e-10
                assert abs(r.metric(d['y'][mask],r.corrected(z[key],coef),d['person_id'][mask])-row[label+'_person_mae'])<1e-10
    model=r.new_model(a.variant).to(a.device);model.load_state_dict(ck['model'],strict=True)
    train=r.predict(model,d['x'][tr],ck['lo'],ck['span'],a.device);coef=r.calibration(train,d['y'][tr])
    raw,detail=predict_details(model,d['x'][va],ck['lo'],ck['span'],a.device);pred=r.corrected(raw,coef)
    with np.load(a.input/'validation_predictions.npz') as z:
        for key in ['y','person_id']:np.testing.assert_array_equal(z[key],d[key][va])
        np.testing.assert_array_equal(z['row_index'],np.flatnonzero(va))
        raw_error=float(abs(raw-z['raw']).max());pred_error=float(abs(pred-z['predictions']).max())
        assert abs(r.metric(z['y'],z['predictions'],z['person_id'])-s['validation_person_mae'])<1e-10
    with np.load(a.input/'window_weights.npz') as z:
        for key in ['window_record','window_start','canonical_fallback']:np.testing.assert_array_equal(z[key],detail[key])
        np.testing.assert_array_equal(z['row_index'],np.flatnonzero(va))
        weight_error=float(abs(z['weights']-detail['weights']).max())
    result=dict(passed=raw_error<1e-4 and pred_error<1e-4 and weight_error<1e-6,variant=a.variant,device=a.device,raw_max_error=raw_error,corrected_max_error=pred_error,weight_max_error=weight_error,score_atol=1e-4,weight_atol=1e-6,all3_epoch_metrics_verified=True,training_only_normalization_and_calibration=True,outer_evaluation_used=False,real_optimizer_steps=0,weight_summary=weight_summary(d['x'][va],detail))
    r.save(a.output/'verification.json',result);print(result);assert result['passed'],result
if __name__=='__main__':main()
