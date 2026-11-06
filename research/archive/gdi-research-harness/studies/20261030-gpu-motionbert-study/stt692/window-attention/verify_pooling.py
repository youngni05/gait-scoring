"""Synthetic pooling properties plus real input ownership; no real GDI optimization."""
import argparse,json,numpy as np,torch
from pathlib import Path
import run as r
from bags import pack,predict_details

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r.setup(31,a.device);rng=np.random.default_rng(31)
    x=r.WindowSet(rng.normal(size=(6,2,124,4,2)).astype(np.float32),np.array([0,1,1,2,2,2]),3,np.arange(6))
    values,mask,mapping=pack(x,np.arange(3),a.device)
    r.setup(31,a.device);mean=r.new_model('mean').to(a.device)
    r.setup(31,a.device);attn=r.new_model('attention').to(a.device)
    for k,v in mean.encoder.state_dict().items():assert torch.equal(v,attn.encoder.state_dict()[k])
    pred,w,s=attn(values,mask,True);uniform=mean(values,mask)
    torch.testing.assert_close(pred,uniform,rtol=0,atol=0)
    original=mean.encoder(values[mask].reshape(-1,124,4,2)).reshape(-1,2)
    reference=x.aggregate(original.detach().cpu().numpy())
    np.testing.assert_allclose(uniform.detach().cpu().numpy(),reference,rtol=1e-5,atol=1e-6)
    assert bool((w[~mask]==0).all());torch.testing.assert_close(w.sum(1),torch.ones_like(pred),rtol=0,atol=2e-7)
    target=torch.randn_like(pred);opt=torch.optim.Adam(attn.parameters(),lr=0.001)
    opt.zero_grad();(pred-target).square().mean().backward()
    assert attn.gate.weight.grad.abs().max().item()>0;opt.step();assert attn.gate.weight.abs().max().item()>0
    # Property checks after the weights actually learned a non-uniform rule.
    attn.eval();base,weights,scores=attn(values,mask,True)
    assert float((weights[1,:2]-0.5).abs().max().detach())>1e-7
    order=torch.tensor([2,0,1],device=a.device)
    torch.testing.assert_close(base,attn(values[:,order],mask[:,order]),rtol=1e-5,atol=1e-6)
    corrupt=values.detach().clone();corrupt[~mask]=float('nan')
    torch.testing.assert_close(base,attn(corrupt,mask),rtol=0,atol=0)
    # Padding has neither a forward influence nor a training gradient.
    grad_input=values.detach().clone().requires_grad_();attn(grad_input,mask).sum().backward();assert bool((grad_input.grad[~mask]==0).all())
    swapped=attn(values.flip(2),mask).flip(1);torch.testing.assert_close(base,swapped,rtol=1e-5,atol=1e-6)
    single=attn(values[:1,:1],mask[:1,:1],True);torch.testing.assert_close(single[1],torch.ones_like(single[1]),rtol=0,atol=0)
    for invalid in [torch.zeros_like(mask)]:
        try:attn(values,invalid)
        except ValueError:pass
        else:raise AssertionError('Empty bag accepted')
    altered=values.clone();altered[2]+=100
    torch.testing.assert_close(base[:2],attn(altered,mask)[:2],rtol=0,atol=0)
    got,detail=predict_details(attn,x,20,80,a.device,batch_records=2)
    summed=np.zeros_like(got);np.add.at(summed,x.owner,detail['weights']*detail['window_scores'])
    np.testing.assert_allclose(got,summed,rtol=0,atol=1e-4)
    real=r.data();xreal=real['x'];counts=[]
    for f in range(5):
        for j in [None]+[v for v in range(5) if v!=f]:
            tr,va,te=r.split(real,f,j)
            for selected in [tr,va,te]:
                if selected.any():
                    subset=xreal[selected];assert len(subset.values)==int(np.isin(xreal.owner,np.flatnonzero(selected)).sum())
                    assert subset.counts.sum()==len(subset.values)
            counts.append({'outer':f,'inner':j,'train_records':int(tr.sum()),'train_people':len(set(real['person_id'][tr]))})
    all_indices=[]
    for start in range(0,len(xreal),8):
        _,mask0,where=pack(xreal,np.arange(start,min(start+8,len(xreal))),'cpu')
        assert np.array_equal(mask0.numpy(),where>=0);all_indices.extend(where[where>=0].tolist())
    assert sorted(all_indices)==list(range(11261))
    result=dict(passed=True,device=a.device,real_GDI_optimizer_steps=0,people=692,records=1061,windows=11261,max_windows_per_record=int(xreal.counts.max()),
        identical_initial_encoder=True,zero_gate_equals_mean_exact=True,original_window_mean_agreement=True,gate_gradient_and_update_nonzero=True,
        nonuniform_weights_verified=True,permutation_invariance=True,padding_forward_and_gradient_excluded=True,empty_bag_rejected=True,
        single_window_weight_one=True,record_independence=True,side_order_verified=True,all_real_windows_assigned_exactly_once=True,
        all25_record_splits_preserved=True,weight_reconstruction_atol=1e-4,parameter_counts={k:sum(p.numel() for p in r.new_model(k).parameters()) for k in ['mean','attention']},splits=counts)
    r.save(a.output/'verification.json',result);print(json.dumps(result))
if __name__=='__main__':main()
