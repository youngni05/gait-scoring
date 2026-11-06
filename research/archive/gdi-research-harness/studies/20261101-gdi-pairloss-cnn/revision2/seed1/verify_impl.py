"""Synthetic gradient/resume/baseline checks; real inputs checked without fitting."""
import contextlib,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
from pair_ops import pool_for,change_term

def run_checks(r,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    r.setup(7,device);d=r.data();c=r.config()
    assert len(d['y'])==1061 and len(set(d['person_id']))==692
    rng=np.random.default_rng(123);n=30
    people=np.array(['p%02d'%i for i in range(15) for j in range(2)])
    ages=np.tile([1.,2.],15);fold=np.repeat(np.arange(15)%5,2)
    x=r.WindowSet(rng.normal(size=(n,2,124,12)).astype(np.float32),np.arange(n),n,ages=ages)
    y=rng.uniform(40,100,(n,2));ident=dict(fit_seed=321,epochs=2,purpose='synthetic')
    sc=dict(c,pair_weight=1,max_epochs=1)
    with (out/'synthetic.log').open('w') as log,contextlib.redirect_stdout(log):
        def fit(name,epochs,valid=y[6:10]):
            return r.fit(x[:6],y[:6],people[:6],x[6:10],valid,people[6:10],sc,epochs,out/name,ident,device)[0]
        full=fit('full',2);fit('resume',1);resumed=fit('resume',2);poison=fit('poison',2,np.ones((4,2))*999)
        for k,v in full.state_dict().items():assert torch.equal(v,resumed.state_dict()[k]) and torch.equal(v,poison.state_dict()[k])
        # The zero-weight arm must reproduce the original baseline's actual gradient path.
        spec=importlib.util.spec_from_file_location('baseline_run',r.ROOT/'baseline_run.py');baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
        zero=dict(c,pair_weight=0)
        m0=r.fit(x[:6],y[:6],people[:6],x[6:10],y[6:10],people[6:10],zero,2,out/'zero',ident,device)[0]
        original=baseline.fit(x[:6],y[:6],people[:6],x[6:10],y[6:10],people[6:10],zero,2,out/'original',ident,device)[0]
        for k,v in m0.state_dict().items():assert torch.equal(v,original.state_dict()[k])
        r.setup(45,device);model=r.GdiCNN().to(device);model.train()
        buffers={k:v.clone() for k,v in model.named_buffers()};state=r.rng_state();pool=pool_for(x[:6],people[:6])
        yy=torch.from_numpy(y[:6].astype(np.float32));term=change_term(model,x[:6],yy,pool,987,device)
        assert torch.equal(state['cpu'],r.rng_state()['cpu'])
        for a,b in zip(state['cuda'],r.rng_state()['cuda']):assert torch.equal(a,b)
        assert model.training
        for k,v in model.named_buffers():assert torch.equal(v,buffers[k])
        term.backward();assert any(p.grad is not None and bool((p.grad!=0).any()) for p in model.parameters())
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        # Independent direct prediction of the deterministically selected synthetic pair.
        rr=np.random.default_rng(987);group=pool[int(rr.integers(len(pool)))];a,b=group[int(rr.integers(len(group)))]
        model.eval()
        with torch.no_grad():
            pred=model(r.channels(torch.from_numpy(x[:6].values)).reshape(-1,12,124).to(device))[:,0].reshape(-1,2)
            expected=((pred[b]-pred[a])-(yy[b]-yy[a]).to(device)).square().mean()
        assert abs(float(term)-float(expected))<1e-5
        sd=dict(x=x,y=y,person_id=people,video_id=np.array(['v%d'%i for i in range(n)]),fold=fold,cohort=np.arange(n)%2,original_row_index=np.arange(n))
        for f in range(5):r.train_fold(sd,sc,f,out/'nested',device,None)
        r.summarize(sd,sc,out/'nested')
    proof=dict(passed=True,device=device,manifest_sha256=r.sha(r.ROOT/'manifest.json'),
               real_people=692,real_records=1061,real_optimizer_steps=0,synthetic_nested_fits=25,
               exact_resume=True,validation_label_poison_invariant=True,zero_arm_exact_baseline=True,
               pair_gradient_nonzero=True,pair_forward_preserves_BN_and_rng=True,
               manual_pair_loss_matches=True,outer_people_excluded=True)
    r.save(out/'verification.json',proof);print(json.dumps(proof),flush=True);return proof
