"""Actual-input plan, architecture compatibility and calibration isolation checks."""
import importlib.util
import numpy as np
import torch
from common import data,make_plan,check_plan,save,mask
from model import TransferCNN
from bn import calibrate
import manage
def main():
    torch.set_num_threads(2);d=data(manage.BUNDLE/'target.npz');plan=make_plan(d);check_plan(plan,d)
    save(manage.HERE/'plan.json',plan);save(manage.BUNDLE/'plan.json',plan)
    spec=importlib.util.spec_from_file_location('prior_cnn',manage.HERE.parent/'20261101-healthgait-bn-consistent/model.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    torch.manual_seed(20261101);old=prior.TransferCNN(1,.5)
    torch.manual_seed(20261101);new=TransferCNN(1,.5,32)
    assert all(torch.equal(new.state_dict()[k],v) for k,v in old.state_dict().items())
    old.eval();new.eval();probe=torch.from_numpy(d['x'][:5])
    with torch.no_grad():np.testing.assert_array_equal(old(probe).numpy(),new(probe).numpy())
    case=plan['cases'][0];tr=mask(d,case['inner'][0]['train']);checks=[]
    for channels in (32,16,8):
        torch.manual_seed(20261101);m=TransferCNN(1,.5,channels);m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
        xx=torch.from_numpy(d['x'][tr[d['owner']]]);opt.zero_grad();loss=m(xx).square().mean();loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
        a=calibrate(m,d,tr,'cpu',1,'selfcheck');state={k:v.clone() for k,v in m.state_dict().items()}
        altered={k:v.copy() for k,v in d.items()};altered['x'][~tr[d['owner']]]+=123;altered['y'][~tr]=-999
        b=calibrate(m,altered,tr,'cpu',1,'selfcheck');assert a==b and all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
        m.eval()
        with torch.no_grad():np.testing.assert_allclose(m(probe).numpy(),torch.cat([m(x[None]) for x in probe]).numpy(),atol=1e-5,rtol=0)
        checks.append(dict(channels=channels,parameters=sum(p.numel() for p in m.parameters()),optimizer_step_finite=True,heldout_mutation_no_effect=True,batch_invariant=True))
    save(manage.HERE/'selfcheck.json',dict(passed=True,cases=50,people=44,architecture32_exact_prior_match=True,checks=checks))
    print(checks)
if __name__=='__main__':main()
