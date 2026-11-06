"""Synthetic optimizer/resume/leakage checks. Never train on real GDI labels."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import torch
import run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    c=dict(learning_rate=0.001,weight_decay=0.0001,batch_size=4)
    rng=np.random.default_rng(71)
    x=rng.normal(size=(9,124,512)).astype(np.float32)
    vx=rng.normal(size=(4,124,512)).astype(np.float32)
    ids=np.array(['p%d'%i for i in range(9)])
    vi=np.array(['v%d'%i for i in range(4)])
    base=rng.normal(70,4,(9,2));y=base+rng.normal(2,1,(9,2))
    vb=rng.normal(70,4,(4,2));vy=vb+rng.normal(2,1,(4,2))
    ident=dict(seed=20261028,epochs=3,test_fixture=True)
    a.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='residual-check-',dir=str(a.output)) as tmp:
        root=Path(tmp)
        full,h=run.fit(x,base,y,ids,(vx,vb,vy,vi),c,3,root/'full',ident,a.device)
        run.fit(x,base,y,ids,(vx,vb,vy,vi),c,3,root/'resume',ident,a.device,stop_after=1)
        resumed,_=run.fit(x,base,y,ids,(vx,vb,vy,vi),c,3,root/'resume',ident,a.device)
        assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in full.state_dict().items())
        assert not torch.equal(full.output.weight,torch.zeros_like(full.output.weight))
        altered,_=run.fit(x,base,y,ids,(vx*7+3,vb+100,vy-100,vi),c,3,root/'altered',ident,a.device)
        assert all(torch.equal(v,altered.state_dict()[k]) for k,v in full.state_dict().items())
        rejected=False
        try:
            run.fit(x,base,y,ids,None,c,3,root/'resume',dict(ident,seed=7),a.device)
        except ValueError:
            rejected=True
        assert rejected
        zero,h0=run.fit(x,base,y,ids,None,c,0,root/'zero',dict(ident,epochs=0),a.device)
        assert np.array_equal(run.predict(zero,x,base,a.device),base)
        assert len(h)==4 and len(h0)==1
        hs=[[dict(validation_person_mae=v,validation_people=3) for v in [1.,2.,3.]] for _ in range(4)]
        assert run.select_epoch(hs)[0]==0
        hs[0][1]['validation_person_mae']=0.
        hs[0][1]['validation_people']=3
        assert run.select_epoch(hs)[0]==0
        data={'fold':np.arange(10)%5,'person_id':np.array(['q%d'%i for i in range(10)])}
        for f in range(5):
            for j in range(5):
                if f!=j:run.split(data,f,j)
        data['person_id'][1]=data['person_id'][0]
        rejected=False
        try:run.split(data,0,1)
        except ValueError:rejected=True
        assert rejected
        # Fit/validation prediction cannot read a hidden outer array: it is not an argument.
        result=dict(passed=True,device=a.device,synthetic_only=True,real_GDI_optimizer_steps=0,
            resume_state_exact=True,validation_changes_do_not_change_weights_or_stats=True,
            mismatched_checkpoint_rejected=True,zero_correction_exact=True,person_overlap_rejected=True,
            correction_updates=True,epoch_zero_selection_supported=True,torch=str(torch.__version__))
        run.save(a.output/'verification.json',result)
        print(result,flush=True)


if __name__=='__main__':main()
