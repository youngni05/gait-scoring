"""CPU checks for train-only calibration, frozen weights, idempotence and batch invariance."""
import json
from pathlib import Path
import numpy as np
import torch
from model import TransferCNN
from bn import calibrate,reset_running_statistics
from train import partition
import manage


def main():
    torch.set_num_threads(2)
    with np.load(manage.BUNDLE/'target.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    tr,va,te=partition(d,0,0)
    rows=[]
    for arm,initial in [('scratch',None),('gdi',manage.BUNDLE/'gdi.pt')]:
        torch.manual_seed(20261101)
        m=TransferCNN(1)
        m.initialize(torch.load(initial,map_location='cpu',weights_only=True) if initial else {},arm)
        reset_running_statistics(m)
        c1=calibrate(m,d,tr,'cpu',1,manage.sha(manage.BUNDLE/'target.npz'))
        saved={n:v.clone() for n,v in m.state_dict().items()}
        changed={k:v.copy() for k,v in d.items()}
        heldout=~tr
        changed['x'][heldout[d['owner']]]+=123.
        changed['y'][heldout]=-999.
        c2=calibrate(m,changed,tr,'cpu',1,manage.sha(manage.BUNDLE/'target.npz'))
        assert c1==c2
        assert all(torch.equal(v,saved[n]) for n,v in m.state_dict().items())
        m.eval();probe=torch.from_numpy(d['x'][va[d['owner']]][:7])
        with torch.no_grad():
            p=m(probe);q=torch.cat([m(x[None]) for x in probe])
        error=float((p-q).abs().max());assert error<1e-5
        rows.append(dict(arm=arm,heldout_mutation_no_effect=True,repeated_calibration_identical=True,batch_size_max_error=error))
    for fold in range(5):
        for inner in range(3):partition(d,fold,inner)
    result=dict(passed=True,target_sha256=manage.sha(manage.BUNDLE/'target.npz'),checks=rows,people=44,outer_folds=5,inner_folds=3)
    manage.save(manage.HERE/'selfcheck.json',result);print(json.dumps(result))


if __name__=='__main__':main()
