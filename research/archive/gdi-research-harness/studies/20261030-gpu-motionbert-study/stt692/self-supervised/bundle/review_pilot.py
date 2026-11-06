"""Audit pilot identities, SSL transfer and both saved models without optimization."""
from pathlib import Path
import argparse,math
import numpy as np
import torch
import run as r
import pretrain as s
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r.setup(20261029,a.device);d=r.data();c=r.load(r.ROOT/'protocol.json');c=dict(c,ssl=dict(c['ssl'],epochs=c['ssl']['pilot_epochs']))
    tr,va,te=r.split(d,0,1);x=d['x'][tr];ids=d['person_id'][tr];seed=c['seed']+1
    summary=r.load(a.input/'pilot_summary.json');out=a.input/'fit'
    assert summary['passed'] and summary['pilot_only'] and not summary['outer_evaluation_used']
    assert summary['epochs']==summary['ssl_epochs']==3
    assert [len(set(d['person_id'][m])) for m in (tr,va,te)]==[416,138,138]
    ssl=s.audit(x,ids,c,out/'pretrain',seed)
    assert ssl['checkpoint_sha256']==summary['ssl_checkpoint_sha256']
    ident=dict(ssl_config=c['ssl'],bundle_sha256=r.sha(r.ROOT/'manifest.json'),seed=seed,epochs=3,outer_fold=0,inner_fold=1,
        train_people=np.unique(ids).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),outer_evaluation_people=np.unique(d['person_id'][te]).tolist())
    for prefix,mask in [('train',tr),('validation',va)]:
        for field,key in [('x','x'),('y','y'),('ids','person_id')]:ident[prefix+'_'+field+'_sha']=r.digest_array(d[key][mask])
    ck=torch.load(out/'latest.pt',map_location=a.device,weights_only=True)
    h=r.load(out/'curve.json');done=r.load(out/'fit_complete.json');steps=3*math.ceil(2*len(x.values)/c['batch_size'])
    assert ck['identity']==ident==done['identity']==r.load(out/'identity.json')
    assert h==ck['history'] and len(h)==3 and done['optimizer_steps']==summary['optimizer_steps']==steps
    assert done['checkpoint_sha256']==summary['checkpoint_sha256']==r.sha(out/'latest.pt') and done['curve_sha256']==r.sha(out/'curve.json')
    assert ck['lo']==float(np.float32(d['y'][tr].min())) and ck['span']==float(np.float32(np.ptp(d['y'][tr])))
    for row in h:
        path=out/'raw_predictions'/('epoch%04d.npz'%row['epoch']);assert r.sha(path)==row['raw_predictions_sha256']
        with np.load(path) as z:
            coef=r.calibration(z['train_raw'],d['y'][tr]);np.testing.assert_allclose(coef,row['calibration'],atol=1e-10,rtol=0)
            for label,mask,key in [('train',tr,'train_raw'),('validation',va,'validation_raw')]:
                assert abs(r.metric(d['y'][mask],z[key],d['person_id'][mask])-row[label+'_raw_person_mae'])<1e-10
                assert abs(r.metric(d['y'][mask],r.corrected(z[key],coef),d['person_id'][mask])-row[label+'_person_mae'])<1e-10
    m=r.new_model().to(a.device);m.load_state_dict(ck['model'],strict=True)
    train=r.predict(m,x,ck['lo'],ck['span'],a.device);coef=r.calibration(train,d['y'][tr])
    raw=r.predict(m,d['x'][va],ck['lo'],ck['span'],a.device);pred=r.corrected(raw,coef)
    with np.load(a.input/'validation_predictions.npz') as z:
        np.testing.assert_array_equal(z['row_index'],np.flatnonzero(va))
        for k in ['y','person_id']:np.testing.assert_array_equal(z[k],d[k][va])
        raw_error=float(abs(raw-z['raw']).max());pred_error=float(abs(pred-z['predictions']).max())
        assert abs(r.metric(z['y'],z['predictions'],z['person_id'])-summary['validation_person_mae'])<1e-10
    ssck=torch.load(out/'pretrain/latest.pt',map_location=a.device,weights_only=True)
    sn=s.SSL(r.new_model()).to(a.device);sn.load_state_dict(ssck['model']);sn.eval()
    transfer=r.load(out/'pretrain/transfer.json')
    assert transfer['checkpoint_sha256']==ssl['checkpoint_sha256'] and not transfer['decoder_transferred']
    assert set(transfer['transferred_keys'])=={k for k in m.state_dict() if s.encoder_key(k)}
    with torch.inference_mode():
        t=torch.from_numpy(x.values[:4].reshape(-1,124,4,2)).to(a.device)
        lc,fc,_,_=s.masked_inputs(t,42);lp=sn(lc,'local').cpu().numpy();fp=sn(fc,'future').cpu().numpy()
    probe=out/'pretrain/probe.npz'
    with np.load(probe) as z:ssl_error=max(float(abs(lp-z['local']).max()),float(abs(fp-z['future']).max()))
    result=dict(passed=raw_error<1e-4 and pred_error<1e-4 and ssl_error<1e-4,device=a.device,
        raw_max_error=raw_error,corrected_max_error=pred_error,ssl_probe_max_error=ssl_error,atol=1e-4,
        checkpoint_sha256=r.sha(out/'latest.pt'),ssl_checkpoint_sha256=ssl['checkpoint_sha256'],
        all_epoch_metrics_verified=True,train_only_pretraining=True,train_only_scale_and_calibration=True,
        train_people=416,validation_people=138,outer_people_excluded=138,real_optimizer_steps=0)
    r.save(a.output/'verification.json',result);print(result);assert result['passed'],result
if __name__=='__main__':main()
