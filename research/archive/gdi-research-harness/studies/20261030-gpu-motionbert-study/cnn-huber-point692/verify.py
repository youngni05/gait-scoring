"""Bounded synthetic optimization; real data checks only, never real-data CPU fits."""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import numpy as np
import torch
import run as r
from rank_aux import feature_loss
from balanced import objective,sample_epoch,batches,steps_per_epoch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    env=r.setup(42,a.device);c=r.config();d=r.data()
    from source_contract import verify
    verify()
    checked=0
    for f in range(5):
        for j in range(5):
            if f==j:continue
            tr,va,te=r.split(d,f,j);x=d['x'][tr];people=d['person_id'][tr]
            order=sample_epoch(x,people,99);again=sample_epoch(x,people,99)
            np.testing.assert_array_equal(order,again)
            _,cnt=np.unique(people[x.owner[order[:,0]]],return_counts=True)
            assert cnt.min()==cnt.max() and len(cnt)==len(np.unique(people))
            assert len(batches(len(order),32))==steps_per_epoch(x,people,32)
            # Fit API receives only train/inner-valid inputs; outer outcomes cannot alter these.
            altered=d['y'].copy();altered[te]+=12345
            np.testing.assert_array_equal(altered[tr],d['y'][tr]);np.testing.assert_array_equal(altered[va],d['y'][va])
            checked+=1
    # Independent new Huber formula/gradient, including boundaries and large tails.
    qh=torch.tensor([-.5,-.1,-.02,0.,.03,.1,.8],device=a.device,dtype=torch.float64,requires_grad=True)
    lh=2*torch.nn.functional.huber_loss(qh,torch.zeros_like(qh),delta=.1,reduction='none')
    refh=torch.where(qh.abs()<.1,qh.square(),.2*qh.abs()-.01)
    torch.testing.assert_close(lh,refh,rtol=0,atol=1e-14)
    gh=torch.autograd.grad(lh.sum(),qh)[0];expected=2*qh.detach().clamp(-.1,.1)
    torch.testing.assert_close(gh,expected,rtol=0,atol=1e-14)
    assert bool(torch.all(lh<=qh.square()+1e-14))
    # Independent log-sum-exp expression and gradients, including repeated labels.
    p=torch.tensor([.1,.5,.8,.9],dtype=torch.float64,requires_grad=True,device=a.device)
    y=torch.tensor([.2,.2,.7,.9],dtype=torch.float64,device=a.device);sigma=.12
    v=objective(p,y,'bmc',sigma);logits=-(p[:,None]-y[None,:])**2/(2*sigma**2)
    ref=(-torch.diag(logits)+torch.logsumexp(logits,dim=1)).mean()*(2*sigma**2)
    torch.testing.assert_close(v,ref,rtol=0,atol=1e-12)
    g=torch.autograd.grad(v,p,retain_graph=True)[0];gr=torch.autograd.grad(ref,p)[0]
    torch.testing.assert_close(g,gr,rtol=0,atol=1e-12)
    q=torch.tensor([.1,.4],dtype=torch.float64,requires_grad=True,device=a.device)
    z=objective(q,torch.ones_like(q),'bmc',.1);gz=torch.autograd.grad(z,q)[0]
    assert torch.isfinite(z) and torch.isfinite(gz).all()
    assert all(len(b)>=2 for b in batches(65,32))
    features=torch.randn(8,10,device=a.device,requires_grad=True)
    target=torch.tensor([1.,1.,2.,3.,4.,5.,6.,7.],device=a.device)
    aux=feature_loss(features,target,2.0);aux.backward()
    assert torch.isfinite(aux) and torch.isfinite(features.grad).all()
    # Real CNN on synthetic windows, same initial weights/draws and exact resumed run.
    rng=np.random.default_rng(4);owner=np.repeat(np.arange(10),np.arange(1,11)%3+1)
    x=r.WindowSet(rng.normal(size=(len(owner),2,124,12)).astype('float32'),owner,10)
    people=np.array(['a','a','a','b','c','c','d','e','e','f'])
    labels=rng.uniform(40,110,(10,2));checks={}
    c=dict(c,max_epochs=2,kernel_l2=c['kernel_l2'])
    for variant in ['mse','ranksim']:
        cc=dict(c,loss_variant=variant,name='synthetic_'+variant)
        ident=dict(fit_seed=123,synthetic=True)
        full=a.output/variant/'full';part=a.output/variant/'resume'
        m,lo,span,h=r.fit(x,labels,people,None,None,None,cc,2,full,ident,a.device)
        r.fit(x,labels,people,None,None,None,cc,1,part,ident,a.device)
        mr,_,_,hr=r.fit(x,labels,people,None,None,None,cc,2,part,ident,a.device)
        for k,vv in m.state_dict().items():assert torch.equal(vv,mr.state_dict()[k]),k
        r.setup(123,a.device);initial=r.GdiCNN(cc['dropout']).to(a.device)
        changes=sum(not torch.equal(vv,initial.state_dict()[k]) for k,vv in m.state_dict().items())
        assert changes>0 and h[0]['sample_sha256']==hr[0]['sample_sha256']
        checks[variant]=dict(changed_tensors=changes,resume_max_error=0,steps=2*steps_per_epoch(x,people,32))
    # Exercise the full nested runner, epoch selection, all-epoch audits and OOF assembly.
    ids=np.repeat(np.array(['p%02d'%i for i in range(10)]),2)
    xx=r.WindowSet(rng.normal(size=(40,2,124,12)).astype('float32'),np.repeat(np.arange(20),2),20)
    sd=dict(x=xx,y=rng.uniform(35,115,(20,2)),person_id=ids,video_id=np.array(['v%d'%i for i in range(20)]),
        fold=np.repeat(np.arange(10)%5,2),cohort=np.repeat(np.arange(10)%2,2),original_row_index=np.arange(20))
    for variant in ['mse','ranksim']:
        cc=dict(c,max_epochs=1,loss_variant=variant,name='synthetic_nested_'+variant)
        out=a.output/'nested'/variant
        for f in range(5):r.train_fold(sd,cc,f,out,a.device,None)
        r.summarize(sd,cc,out)
        assert r.load(out/'summary.json')['fits']==25
    r.save(a.output/'verification.json',dict(passed=True,environment=env,inner_splits=checked,
        real_data_optimizer_steps=0,synthetic=checks,manifest_sha256=r.sha(r.ROOT/'manifest.json'),
        bmc_reference_and_gradient=True,huber_piecewise_and_gradient=True,exact_person_counts=True,protected_final_used=False,synthetic_nested_fits=50,synthetic_outer_folds=10))
    print('VERIFICATION_PASSED',flush=True)
if __name__=='__main__':main()
