"""Verify zero-weight equivalence, auxiliary gradients, replay and nested selection."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
import run as r


def fake():
    rng = np.random.default_rng(31)
    n = 20
    person = np.array(['synthetic-%02d' % (i // 2) for i in range(n)])
    return dict(x=r.b.WindowSet(rng.normal(size=(n,2,124,12)).astype(np.float32),np.arange(n),n),
                y=rng.uniform(40,110,(n,2)), aux=rng.normal(15,10,(n,2)), person_id=person,
                video_id=np.array(['fake-%02d'%i for i in range(n)]), fold=np.array([(i//2)%5 for i in range(n)]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    r.verify_manifest()
    c=r.config()
    d=fake()
    tr,va,_=r.b.split(d,0,1)
    ident=r.identity(d,c,0,1,2,0.0,'synthetic_equivalence')
    original=r.b.fit(d['x'][tr],d['y'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],
                     c,2,args.output/'original',ident,args.device)
    adapted=r.fit(d['x'][tr],d['y'][tr],d['aux'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],
                  c,2,args.output/'adapted',ident,args.device)
    assert all(torch.equal(value,adapted[0].state_dict()[key]) for key,value in original[0].state_dict().items())
    for a,b in zip(original[3],adapted[3]):
        for key in ['loss','scaled_mse','train_person_mae','validation_person_mae','calibration']:
            assert a[key]==b[key],(key,a[key],b[key])
    # Auxiliary targets cannot influence the zero-weight control.
    altered=r.fit(d['x'][tr],d['y'][tr],d['aux'][tr]*1000+999,d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],
                  c,2,args.output/'changed_aux',ident,args.device)
    assert all(torch.equal(value,altered[0].state_dict()[key]) for key,value in adapted[0].state_dict().items())
    # Resume must reproduce uninterrupted optimization, including dropout RNG.
    r.fit(d['x'][tr],d['y'][tr],d['aux'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],
          c,1,args.output/'resume',ident,args.device)
    resumed=r.fit(d['x'][tr],d['y'][tr],d['aux'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],
                  c,2,args.output/'resume',ident,args.device)
    assert all(torch.equal(value,resumed[0].state_dict()[key]) for key,value in adapted[0].state_dict().items())
    r.b.setup(717,args.device)
    model=r.GdiCNN(c['dropout']).to(args.device)
    xt=r.channels(torch.from_numpy(d['x'].values[:3])).reshape(-1,12,124).to(args.device)
    target=torch.zeros(6,device=args.device)
    model.train()
    auxiliary=(model(xt)[:,1]-target).square().mean()
    auxiliary.backward()
    assert model.convs[0].weight.grad.abs().sum()>0
    assert model.output.weight.grad[1].abs().sum()>0
    assert model.output.weight.grad[0].abs().sum()==0
    assert r.select([dict(score=1.0,weight=0.3,epoch=1),dict(score=1.0,weight=0.1,epoch=3),dict(score=1.0,weight=0.1,epoch=2)])['epoch']==2
    fits=0
    if args.full:
        short=copy.deepcopy(c)
        short['max_epochs']=1
        for variant in ['gdi_only','knee_auxiliary']:
            for fold in range(5):
                r.train_fold(d,short,variant,fold,args.output/'nested',args.device)
        r.summarize(d,short,args.output/'nested')
        summary=r.b.load(args.output/'nested/summary.json')
        fits=summary['completed_fits']
        assert fits==90
        # Mutating outer outcomes cannot change inner target scaling or an inner fit.
        changed=copy.deepcopy(d)
        changed['y'][changed['fold']==0]=999999
        changed['aux'][changed['fold']==0]=-999999
        a=r.fit_subset(d,short,0,1,1,0.1,args.output/'outer_invariance_a',args.device,'invariance')
        z=r.fit_subset(changed,short,0,1,1,0.1,args.output/'outer_invariance_b',args.device,'invariance')
        assert all(torch.equal(value,z[0].state_dict()[key]) for key,value in a[0].state_dict().items())
    report=dict(passed=True,device=args.device,manifest_sha256=r.b.sha(r.ROOT/'manifest.json'),
                zero_weight_matches_original_model_and_curves_exactly=True,
                zero_weight_invariant_to_auxiliary_targets=True, auxiliary_reaches_shared_convolutions=True,
                checkpoint_resume_checked=True, tie_break_checked=True, synthetic_nested_fits=fits,
                outer_outcome_mutation_invariant=bool(args.full), real_optimizer_steps=0)
    r.b.save(args.output/'verification.json',report)
    print(r.b.json.dumps(report),flush=True)


if __name__=='__main__':main()
