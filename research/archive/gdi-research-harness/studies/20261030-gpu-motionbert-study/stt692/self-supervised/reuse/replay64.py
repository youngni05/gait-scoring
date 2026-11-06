"""Read-only float64 evaluation of unchanged float32 pilot weights."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import run as r
import pretrain as ssl

def main():
    r.setup(20261029,'cpu');d=r.data();tr,va,_=r.split(d,0,1);out=ROOT/'runs/pilot'
    c=r.load(ROOT/'protocol.json');c=dict(c,ssl=dict(c['ssl'],epochs=c['ssl']['pilot_epochs']))
    ssl.audit(d['x'][tr],d['person_id'][tr],c,out/'fit/pretrain',c['seed']+1)
    old=r.load(ROOT/'checks/replay-cpu/verification.json')
    gpu=r.load(ROOT/'checks/replay-reuse-gpu/verification.json')
    assert gpu['passed'] and old['checkpoint_sha256']==gpu['checkpoint_sha256']==r.sha(out/'fit/latest.pt')
    ck=torch.load(out/'fit/latest.pt',map_location='cpu',weights_only=True)
    m=r.new_model().double();m.load_state_dict(ck['model']);m.eval()
    def predict(x):
        result=[]
        with torch.inference_mode():
            for start in range(0,len(x.values),64):
                t=torch.from_numpy(x.values[start:start+64].reshape(-1,124,4,2)).double()
                result.append((m(t)*ck['span']+ck['lo']).reshape(-1,2).numpy())
        return x.aggregate(np.concatenate(result))
    train=predict(d['x'][tr]);coef=r.calibration(train,d['y'][tr]);raw=predict(d['x'][va]);pred=r.corrected(raw,coef)
    with np.load(out/'validation_predictions.npz') as z:
        er=float(abs(raw-z['raw']).max());ep=float(abs(pred-z['predictions']).max())
    with np.load(out/'fit/raw_predictions/epoch0003.npz') as z:et=float(abs(train-z['train_raw']).max())
    sk=torch.load(out/'fit/pretrain/latest.pt',map_location='cpu',weights_only=True)
    sm=ssl.SSL(r.new_model()).double();sm.load_state_dict(sk['model']);sm.eval()
    with torch.inference_mode():
        t=torch.from_numpy(d['x'][tr].values[:4].reshape(-1,124,4,2)).double();lc,fc,_,_=ssl.masked_inputs(t,42)
        lp=sm(lc,'local').numpy();fp=sm(fc,'future').numpy()
    with np.load(out/'fit/pretrain/probe.npz') as z:es=max(float(abs(lp-z['local']).max()),float(abs(fp-z['future']).max()))
    result=dict(passed=max(er,ep,et,es)<1e-4,device='cpu',arithmetic='float64_exact_values_of_original_float32_weights',
        raw_max_error=er,corrected_max_error=ep,train_raw_max_error=et,ssl_probe_max_error=es,atol=1e-4,
        checkpoint_sha256=r.sha(out/'fit/latest.pt'),ssl_checkpoint_sha256=r.sha(out/'fit/pretrain/latest.pt'),
        float32_failure_preserved=old,original_gpu_verified=gpu,training_only_calibration=True,real_optimizer_steps=0)
    r.save(ROOT/'checks/replay-reuse-cpu64/verification.json',result);print(result,flush=True)
    assert result['passed'],result
if __name__=='__main__':main()
