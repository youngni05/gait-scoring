"""Synthetic symmetry, paired batching and masked window aggregation properties."""
import argparse,json,torch
from pathlib import Path
import run
from mil_pool import WindowAttentionPool
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run.setup(21,a.device);m=run.new_model().to(a.device);x=torch.randn(3,2,124,4,2,device=a.device)
    y=m(x).reshape(3,2);sw=m(x.flip(1)).reshape(3,2).flip(1)
    torch.testing.assert_close(y,sw,rtol=1e-5,atol=1e-6)
    equal=x[:,0:1].repeat(1,2,1,1,1);p0=m(equal).reshape(3,2)
    torch.testing.assert_close(p0[:,0],p0[:,1],rtol=0,atol=0)
    y.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    target=torch.randn(3,2,device=a.device);err=y-target
    torch.testing.assert_close(err.square().mean(),(err.mean(1).square()+((err[:,0]-err[:,1])/2).square()).mean())
    # CPU-only independent second branch: padded windows never enter learned pooling.
    pool=WindowAttentionPool(8);v=torch.randn(2,4,8);mask=torch.tensor([[True,True,False,False],[True,True,True,False]])
    out,w=pool(v,mask);order=torch.tensor([2,0,3,1]);permuted,_=pool(v[:,order],mask[:,order]);torch.testing.assert_close(out,permuted)
    changed=v.clone();changed[~mask]=float('nan');same,_=pool(changed,mask);torch.testing.assert_close(out,same)
    assert (w[~mask]==0).all();torch.testing.assert_close(w.sum(1),torch.ones(2))
    try:pool(v,torch.zeros_like(mask))
    except ValueError:pass
    else:raise AssertionError('Empty record accepted')
    out.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pool.parameters())
    result=dict(passed=True,device=a.device,synthetic_only=True,real_GDI_optimizer_steps=0,swap_equivariance=True,equal_legs_equal_outputs=True,paired_loss_identity=True,finite_gradients=True,mil_permutation_invariance=True,mil_padding_excluded=True,mil_empty_record_rejected=True,parameters=sum(p.numel() for p in m.parameters()))
    run.save(a.output/'verification.json',result);print(json.dumps(result))
if __name__=='__main__':main()
