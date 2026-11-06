"""Read-only saved-model pilot replay; CPU/GPU tolerance fixed at 1e-4 GDI."""
import argparse
from pathlib import Path
import numpy as np
import torch
import run as r
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['window_control','record_mean'],required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r.setup(7,a.device);d=r.data();c=r.config();c.update(loss_variant=a.variant,name='cnn_'+a.variant+'_packed_records')
    tr,va,te=r.split(d,0,1);folder=a.run/'pilot';done=r.load(folder/'fit_complete.json')
    assert r.sha(folder/'latest.pt')==done['checkpoint_sha256'] and r.sha(folder/'curve.json')==done['curve_sha256']
    ck=torch.load(folder/'latest.pt',map_location=a.device,weights_only=True)
    ident=r.identity_for(d,c,0,1,3,'pilot_not_performance')
    assert all(ck['identity'][k]==v for k,v in ident.items()) and ck['identity']['training_config']==c
    h=ck['history'];assert len(h)==3 and ck['identity']==done['identity']
    lo,span=r.scale_from_train(d['y'][tr]);assert torch.equal(lo,ck['lo'].cpu()) and torch.equal(span,ck['span'].cpu())
    for row in h:
        file=folder/'raw_predictions'/('epoch%04d.npz'%row['epoch']);assert r.sha(file)==row['raw_predictions_sha256']
        with np.load(file) as z:
            coef=r.calibrate(z['train_raw'],d['y'][tr]);np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)
            assert abs(r.person_mae(d['y'][va],r.corrected(z['validation_raw'],coef),d['person_id'][va])-row['validation_person_mae'])<1e-10
    m=r.GdiCNN(c['dropout']).to(a.device);m.load_state_dict(ck['model'])
    train=r.predict(m,d['x'][tr],lo,span,a.device);raw=r.predict(m,d['x'][va],lo,span,a.device)
    coef=r.calibrate(train,d['y'][tr]);last=folder/'raw_predictions/epoch0003.npz'
    with np.load(last) as z:
        error=float(np.max(abs(r.corrected(raw,coef)-r.corrected(z['validation_raw'],h[-1]['calibration']))))
    assert error<1e-4
    r.save(a.output,dict(passed=True,variant=a.variant,device=a.device,max_corrected_prediction_error=error,tolerance=1e-4,train_people=len(np.unique(d['person_id'][tr])),validation_people=len(np.unique(d['person_id'][va])),outer_evaluation_used=False,checkpoint_sha256=r.sha(folder/'latest.pt')))
if __name__=='__main__':main()
