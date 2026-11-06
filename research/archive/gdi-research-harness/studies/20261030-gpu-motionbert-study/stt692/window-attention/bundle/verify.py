"""Synthetic training checks; never optimize on real GDI labels."""
from pathlib import Path
import argparse,ast,json,tempfile
import numpy as np
import torch
import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['mean','attention'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=run.load(run.ROOT/'protocol.json');c.update(pooling=a.variant,name='record_loss_STT_'+a.variant)
    run.setup(17,a.device)
    rng=np.random.default_rng(17);x=rng.normal(size=(8,2,124,4,2)).astype(np.float32)
    x=run.WindowSet(x,np.arange(8),8)
    y=rng.normal(70,10,(8,2));ids=np.array(['t%d'%i for i in range(6)]);vids=np.array(['v0','v1'])
    ident=dict(seed=17,epochs=3,synthetic_only=True)
    def reject(fn):
        try:fn()
        except ValueError:return True
        raise AssertionError('Invalid case accepted')
    reject(lambda:run.aligned_mse(torch.zeros(2,1),torch.zeros(2)))
    np.testing.assert_allclose(run.aligned_mse(torch.tensor([1.,2.]),torch.tensor([0.,0.])).item(),2.5)
    # Reproduce the upstream weight-broadcast behavior with the actual function body.
    tree=ast.parse((run.ROOT/'sources/transformer_model.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='weighted_loss')
    ns={'nn':torch.nn};exec(compile(ast.Module(body=[fn],type_ignores=[]),'upstream_weighted_loss','exec'),ns)
    upstream=float(ns['weighted_loss'](torch.tensor([1.,2.]),torch.zeros(2),torch.tensor([[1.],[3.]]),2.))
    expected=(1.*1+4.*3)/4
    assert upstream==2.5 and expected==3.25
    model=run.new_model(c['pooling']).to(a.device).eval()
    values,valid,_=run.pack(x,np.array([0,1]),a.device)
    with torch.inference_mode():
        grouped=model(values,valid)
        separate=torch.cat([model(values[i:i+1],valid[i:i+1]) for i in range(2)])
    torch.testing.assert_close(grouped,separate,rtol=1e-5,atol=1e-6)
    with tempfile.TemporaryDirectory(dir=a.output,prefix='synthetic-') as temp:
        root=Path(temp)
        run.setup(ident['seed'],a.device)
        initial={k:v.detach().cpu().clone() for k,v in run.new_model(c['pooling']).state_dict().items()}
        training=run.WindowSet(np.concatenate([x.values[:6],x.values[:2]]),np.array([0,1,2,3,4,5,0,1]),6)
        args=(training,y[:6],ids,x[6:],y[6:],vids,c,3)
        full=run.fit(*args,root/'full',ident,a.device)
        run.fit(*args,root/'resume',ident,a.device,stop_after=1)
        resumed=run.fit(*args,root/'resume',ident,a.device)
        fullck=torch.load(root/'full/latest.pt',map_location='cpu',weights_only=True)
        resumeck=torch.load(root/'resume/latest.pt',map_location='cpu',weights_only=True)
        def same(a,b):
            if torch.is_tensor(a):assert torch.equal(a,b)
            elif isinstance(a,dict):
                assert set(a)==set(b)
                for k in a:same(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for l,r in zip(a,b):same(l,r)
            else:assert a==b
        for key in ('model','optimizer','scheduler'):same(fullck[key],resumeck[key])
        altered=run.fit(training,y[:6],ids,run.WindowSet(x[6:].values+100,x[6:].owner,2),y[6:]+1000,vids,c,3,root/'altered',ident,a.device)
        alteredck=torch.load(root/'altered/latest.pt',map_location='cpu',weights_only=True)
        for key in ('model','optimizer','scheduler','lo','span'):same(fullck[key],alteredck[key])
        for h1,h2 in zip(fullck['history'],alteredck['history']):same(h1['calibration'],h2['calibration'])
        assert fullck['lo']==float(np.float32(y[:6].min()))
        reject(lambda:run.fit(*args,root/'resume',dict(ident,seed=18),a.device))
        reject(lambda:run.fit(training,y[:6],ids,x[6:],y[6:],np.array(['t0','v1']),c,3,root/'bad',ident,a.device))
        replay=run.new_model(c['pooling']).to(a.device);replay.load_state_dict(fullck['model'])
        before=run.predict(full[0],x[6:],full[1],full[2],a.device)
        after=run.predict(replay,x[6:],fullck['lo'],fullck['span'],a.device)
        np.testing.assert_array_equal(before,after)
        changed=sum(not torch.equal(v.cpu(),initial[k]) for k,v in fullck['model'].items())
        assert changed>0
    result=dict(passed=True,device=a.device,synthetic_only=True,real_GDI_optimizer_steps=0,
        exact_model_optimizer_scheduler_resume=True,validation_input_and_label_changes_do_not_change_fit=True,
        train_only_target_scale_and_calibration=True,person_overlap_rejected=True,checkpoint_identity_change_rejected=True,
        batch_side_layout_verified=True,loss_broadcast_rejected=True,checkpoint_prediction_replay_exact=True,
        official_weighted_loss_counterexample=dict(upstream=upstream,aligned_weighted=expected),
        changed_parameter_tensors=changed,parameters=sum(p.numel() for p in model.parameters()))
    run.save(a.output/'verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
