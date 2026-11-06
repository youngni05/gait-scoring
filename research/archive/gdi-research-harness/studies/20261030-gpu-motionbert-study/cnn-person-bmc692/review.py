"""All-epoch audit and frozen checkpoint replay; no training or tolerance changes."""
from pathlib import Path
import argparse,sys
import numpy as np
import torch
import run as r
from audit import audit_fit

def replay(d,c,f,j,epochs,folder,device,purpose):
    h=audit_fit(r,d,c,f,j,folder,epochs,purpose)
    tr,va,_=r.split(d,f,j)
    ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
    model=r.GdiCNN(c['dropout']);model.load_state_dict(ck['model']);model.to(device)
    if device=='cpu':model.double()
    model.eval()
    def predict(mask):
        x=d['x'][mask];parts=[]
        with torch.no_grad():
            for start in range(0,len(x.values),128):
                t=r.channels(torch.from_numpy(x.values[start:start+128])).reshape(-1,12,124).to(device)
                if device=='cpu':t=t.double()
                p=model(t)[:,0]*ck['span'].to(device)+ck['lo'].to(device)
                parts.append(p.reshape(-1,2).cpu().numpy())
        return x.aggregate(np.concatenate(parts))
    coef=np.asarray(h[-1]['calibration']);raw_error=0.;corrected_error=0.
    with np.load(folder/'raw_predictions'/('epoch%04d.npz'%epochs)) as z:
        for mask,key in [(tr,'train_raw')]+([] if j is None else [(va,'validation_raw')]):
            got=predict(mask);ref=z[key]
            raw_error=max(raw_error,float(np.max(abs(got-ref))))
            corrected_error=max(corrected_error,float(np.max(abs(r.corrected(got,coef)-r.corrected(ref,coef)))))
    assert max(raw_error,corrected_error)<1e-4,(raw_error,corrected_error)
    return dict(passed=True,device=device,reference_dtype='float64' if device=='cpu' else 'float32',
        max_raw_error=raw_error,max_corrected_error=corrected_error,tolerance=1e-4,
        checkpoint_sha256=r.sha(folder/'latest.pt'),epochs=epochs,outer_evaluation_used=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['mse','bmc'],required=True)
    p.add_argument('--device',choices=['cuda','cpu'],required=True);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r.setup(7,a.device);d=r.data();c=r.config();c.update(loss_variant=a.variant,name='cnn_'+a.variant+'_person_balanced')
    q=replay(d,c,0,1,3,a.folder,a.device,'pilot_not_performance');r.save(a.output,q);print(q,flush=True)
if __name__=='__main__':main()
