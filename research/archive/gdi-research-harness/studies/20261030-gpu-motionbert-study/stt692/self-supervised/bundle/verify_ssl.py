"""Synthetic SSL isolation, exact resume, masks, transfer and tamper checks."""
from pathlib import Path
import argparse,tempfile,inspect
import numpy as np
import torch
import run as r
import pretrain as s

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);r.setup(71,a.device)
    c=r.load(r.ROOT/'protocol.json');c=dict(c,ssl=dict(c['ssl'],epochs=2,batch_size=4))
    vals=np.random.default_rng(71).normal(size=(5,2,124,4,2)).astype(np.float32)
    x=r.WindowSet(vals,np.array([0,0,1,2,2]),3);ids=np.array(['p0','p1','p2'])
    target=torch.from_numpy(vals.reshape(-1,124,4,2)).to(a.device)
    local,future,lm,fm=s.masked_inputs(target,22)
    assert (lm.sum(1)==31).all() and (fm.sum(1)==31).all()
    assert torch.equal(local[~lm],target[~lm]) and (local[lm]==0).all()
    assert torch.equal(future[:,:93],target[:,:93]) and (future[:,93:]==0).all()
    changed=target.clone();changed[:,93:]+=10000
    assert torch.equal(future,s.masked_inputs(changed,22)[1])
    changed=target.clone();changed[lm]+=10000
    assert torch.equal(local,s.masked_inputs(changed,22)[0])
    net=s.SSL(r.new_model()).to(a.device).eval()
    with torch.no_grad():
        assert torch.equal(net(future,'future'),net(s.masked_inputs(target.masked_fill(fm[:,:,None,None],999),22)[1],'future'))
    assert 'y' not in inspect.signature(s.initialize).parameters
    with tempfile.TemporaryDirectory(dir=a.output) as tmp:
        root=Path(tmp);r.setup(71,a.device);m=r.new_model().to(a.device)
        initial={k:v.clone() for k,v in m.state_dict().items()}
        s.initialize(m,x,ids,c,root/'full',a.device,71)
        r.setup(71,a.device);n=r.new_model().to(a.device)
        s.initialize(n,x,ids,c,root/'resume',a.device,71,stop_after=1)
        s.initialize(n,x,ids,c,root/'resume',a.device,71)
        for k,v in m.state_dict().items():assert torch.equal(v,n.state_dict()[k])
        ck=torch.load(root/'full/latest.pt',map_location=a.device,weights_only=True)
        changed_encoder=0
        for k,v in m.state_dict().items():
            if s.encoder_key(k):
                assert torch.equal(v,ck['model']['encoder.'+k]);changed_encoder+=int(not torch.equal(v,initial[k]))
            else:assert torch.equal(v,initial[k])
        assert changed_encoder>0
        replay=s.SSL(r.new_model()).to(a.device);replay.load_state_dict(ck['model']);replay.eval()
        with torch.no_grad():
            assert torch.isfinite(replay(local,'local')).all()
            error=s.losses(replay,target,22,torch.from_numpy(x.side_weights()).to(a.device),c['ssl'])
            assert all(torch.isfinite(v) for v in error)
        old=r.sha(root/'full/latest.pt');s.initialize(m,x,ids,c,root/'full',a.device,71)
        assert r.sha(root/'full/latest.pt')==old
        wrong=r.WindowSet(vals+1,x.owner,3)
        try:s.initialize(m,wrong,ids,c,root/'full',a.device,71)
        except ValueError:pass
        else:raise AssertionError('Changed input accepted')
        (root/'full/latest.pt').write_bytes((root/'full/latest.pt').read_bytes()+b'corrupt')
        try:s.audit(x,ids,c,root/'full',71)
        except ValueError:pass
        else:raise AssertionError('Corrupt SSL checkpoint accepted')
    result=dict(passed=True,synthetic_only=True,real_GDI_optimizer_steps=0,device=a.device,
        labels_absent_from_ssl_api=True,hidden_targets_do_not_change_encoder_input=True,exact_resume=True,
        only_encoder_transferred=True,fresh_GDI_head_preserved=True,changed_encoder_tensors=changed_encoder,
        input_identity_and_corruption_rejected=True)
    r.save(a.output/'verification.json',result);print(result)
if __name__=='__main__':main()
