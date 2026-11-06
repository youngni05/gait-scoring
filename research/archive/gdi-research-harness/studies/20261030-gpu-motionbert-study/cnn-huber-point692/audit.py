"""Verify every epoch using training-only calibration before accepting a full run."""
import hashlib
import numpy as np
import torch

def audit_fit(r,d,c,f,j,folder,epochs,purpose):
    tr,va,_=r.split(d,f,j);x=d['x'][tr];y=d['y'][tr];ids=d['person_id'][tr]
    ident=r.identity_for(d,c,f,j,epochs,purpose)
    ident.update(training_config=c,train_x_sha=hashlib.sha256(x.values.tobytes()+x.owner.tobytes()).hexdigest(),
        train_y_sha=hashlib.sha256(y.tobytes()).hexdigest(),train_ids_sha=hashlib.sha256(ids.tobytes()).hexdigest())
    done=r.load(folder/'fit_complete.json');h=r.load(folder/'curve.json')
    r.require(done['identity']==ident==r.load(folder/'identity.json'),'Fit provenance mismatch')
    r.require(r.sha(folder/'latest.pt')==done['checkpoint_sha256'] and r.sha(folder/'curve.json')==done['curve_sha256'],'Fit hash mismatch')
    ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
    r.require(ck['identity']==ident and ck['history']==h,'Checkpoint history mismatch')
    lo,span=r.scale_from_train(y)
    r.require(torch.equal(lo,ck['lo']) and torch.equal(span,ck['span']),'Scaling used wrong training labels')
    r.require(done['epochs']==epochs and len(h)==epochs and done['optimizer_steps']==epochs*r.steps_per_epoch(x,ids,c['batch_size']),'Epoch/step count mismatch')
    for k,row in enumerate(h,1):
        r.require(row['epoch']==k,'Nonsequential curve')
        order=r.sample_epoch(x,ids,ident['fit_seed']+k-1)
        r.require(row['sample_sha256']==r.sample_digest(order) and row['sample_count']==len(order), 'Wrong person-balanced samples')
        r.require(row['samples_per_person']==len(order)//len(np.unique(ids)), 'Wrong sample mass')
        path=folder/'raw_predictions'/('epoch%04d.npz'%k)
        r.require(r.sha(path)==row['raw_predictions_sha256'],'Raw prediction changed')
        with np.load(path) as z:
            raw=z['train_raw'];valid=z['validation_raw']
            r.require(raw.shape==y.shape,'Wrong train predictions')
            coef=r.calibrate(raw,y)
            np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)
            for key,pred in [('train_raw_person_mae',raw),('train_person_mae',r.corrected(raw,coef))]:
                r.require(abs(r.person_mae(y,pred,ids)-row[key])<1e-10,'Wrong train metric')
            if j is None:
                r.require(valid.shape==(0,2) and 'validation_person_mae' not in row,'Refit read validation')
            else:
                for key,pred in [('validation_raw_person_mae',valid),('validation_person_mae',r.corrected(valid,coef))]:
                    r.require(abs(r.person_mae(d['y'][va],pred,d['person_id'][va])-row[key])<1e-10,'Wrong validation metric')
    return h
