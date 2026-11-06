"""Window ownership, record-balanced loss and inference aggregation checks."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
import torch
import run
from windows import WindowSet,weighted_mse
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run.setup(21,'cpu')
    owner=np.array([0,1,1,1,2,2]);values=np.zeros((6,2,124,4,2),np.float32)
    w=WindowSet(values,owner,3,np.arange(6));w.validate()
    sub=w[np.array([2,0])]
    np.testing.assert_array_equal(sub.owner,[1,0,0]);np.testing.assert_array_equal(sub.starts,[0,4,5])
    targets=torch.arange(12,dtype=torch.float32)
    actual=weighted_mse(torch.zeros(12),targets,torch.from_numpy(w.side_weights()))
    expected=np.mean([(targets.numpy().reshape(6,2)[owner==i]**2).mean() for i in range(3)])
    np.testing.assert_allclose(actual.item(),expected,rtol=1e-7,atol=1e-7)
    try:weighted_mse(torch.zeros(2,1),torch.zeros(2),torch.ones(2))
    except ValueError:pass
    else:raise AssertionError('Broadcast accepted')
    raw=np.arange(12).reshape(6,2).astype(np.float32)
    np.testing.assert_array_equal(w.aggregate(raw),np.stack([raw[owner==i].mean(0) for i in range(3)]))
    # Real inference order: each side belongs to the correct window; average only
    # after restoring GDI scale. A deterministic fake model isolates this logic.
    w.values[:,0,0,0,0]=np.arange(6);w.values[:,1,0,0,0]=10+np.arange(6)
    class Probe(torch.nn.Module):
        def forward(self,x):return x[:,:,0,0,0].reshape(-1)
    got=run.predict(Probe(),w,10.,2.,'cpu')
    reference=w.aggregate(w.values[:,:,0,0,0]*2.+10.)
    np.testing.assert_array_equal(got,reference)
    d=run.data()
    assert d['x'].values.shape==(11261,2,124,4,2)
    checks=[]
    for f in range(5):
        for j in [None]+[k for k in range(5) if k!=f]:
            tr,va,te=run.split(d,f,j);x=d['x'][tr]
            weights=x.side_weights().reshape(-1,2)
            np.testing.assert_allclose(weights.mean(),1,rtol=0,atol=2e-7)
            per_record=np.zeros(len(x));np.add.at(per_record,x.owner,weights.sum(1))
            np.testing.assert_allclose(per_record,2*len(x.values)/len(x),rtol=2e-7,atol=2e-7)
            assert set(d['person_id'][tr]).isdisjoint(d['person_id'][te])
            checks.append(dict(outer=f,inner=j,train_records=int(tr.sum()),train_windows=len(x.values)))
    run.save(a.output/'verification.json',dict(passed=True,real_GDI_optimizer_steps=0,window_count=11261,
        all25_training_subset_weights_checked=True,subset_reordering_verified=True,
        loss_and_prediction_aggregation_verified=True,fold_counts=checks))
    print('WINDOW_VERIFICATION_PASSED',flush=True)
if __name__=='__main__':main()
