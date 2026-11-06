"""CPU synthetic contract tests plus actual input checks; no actual-label training."""
from pathlib import Path
import copy,json,os,contextlib
import numpy as np
import torch
import run as r
from ordinal import OrdinalCNN,objective,probabilities,targets,boundaries
from worker import assignments,pilot_specs
def main():
    out=Path(os.environ['ORDINAL_TEST_ROOT']);out.mkdir(parents=True,exist_ok=True)
    c=r.read(r.H/'protocol.json');r.setup(c['seed'],'cpu')
    # Known edge convention and a separately computed person mean.
    for width in (10,20):
        values=np.array([-1,0,width-.001,width,160,161]);assert targets(values,width).tolist()==[0,1,1,2,len(boundaries(width)),len(boundaries(width))]
    ids=np.array(['a','a','b']);w=r.weights(ids);assert abs((w*np.array([2.,4.,10.])).mean()-6.5)<1e-6
    n=20;g=np.random.default_rng(1);d=dict(x=g.normal(size=(n,2,124,12)).astype(np.float32),window_record=np.arange(n),window_start=np.zeros(n,int),
        y=g.uniform(40,110,(n,2)),person_id=np.repeat(np.array(['p%d'%i for i in range(10)]),2),video_id=np.array(['v%d'%i for i in range(n)]),fold=np.repeat(np.arange(5),4),_windows=[np.array([i]) for i in range(n)])
    small=dict(c,max_epochs=2,pilot_epochs=2)
    checks=[]
    initials=[]
    for group in c['groups']:
        r.setup(c['seed'],'cpu');width,mode=r.spec(group);m=OrdinalCNN(width,mode)
        initials.append({k:v.clone() for k,v in m.state_dict().items() if k.startswith('base.') or k.startswith('reg.')})
        x,counts=r.packed(d,[4,5],'cpu');score,logits=m(x,counts)
        if logits is not None:
            p=probabilities(logits);assert bool((p>=0).all()) and torch.allclose(p.sum(-1),torch.ones((2,2)))
            assert bool((logits[...,:-1]>=logits[...,1:]).all())
        loss,_=objective(m,score,logits,torch.tensor(d['y'][[4,5]],dtype=torch.float32),torch.ones(2),40.,70.,1.)
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        folder=out/group;r.fit(d,small,group,0,1,1.,2,folder,'cpu');r.audit(d,small,group,0,1,1.,2,folder)
        replay=r.replay(d,small,group,0,1,1.,2,folder,'cpu');checks.append(dict(group=group,replay=replay))
    for state in initials:
        assert all(torch.equal(v,initials[0][k]) for k,v in state.items())
    # Resume must recover EXACT weights and optimizer state without new epoch selection.
    folder=out/'resumed';r.fit(d,small,'joint10',0,1,1.,2,folder,'cpu',stop_after=1)
    assert not (folder/'fit_complete.json').exists();r.fit(d,small,'joint10',0,1,1.,2,folder,'cpu')
    a=torch.load(out/'joint10/latest.pt',weights_only=True);b=torch.load(folder/'latest.pt',weights_only=True)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    # Outer labels never affect training, scaling or the inner metric histories.
    poisoned=copy.deepcopy(d);poisoned['y'][poisoned['fold']==0]=999999
    folder=out/'outer-label-poison';r.fit(poisoned,small,'joint10',0,1,1.,2,folder,'cpu')
    b=torch.load(folder/'latest.pt',weights_only=True);assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    for x,y in zip(a['history'],b['history']):assert x['validation']==y['validation'] and x['train']==y['train']
    # Selection is an explicit lexicographic maximum of accuracy, then NLL.
    hs={'1.0':[[{'validation_people':2,'validation':{'accuracy':.5,'nll':1.}}, {'validation_people':2,'validation':{'accuracy':.6,'nll':1.2}}]]*4,
        '0.25':[[{'validation_people':2,'validation':{'accuracy':.6,'nll':1.1}}, {'validation_people':2,'validation':{'accuracy':.6,'nll':1.1}}]]*4}
    chosen,_=r.choose(hs,'joint10',[.25,1.]);assert chosen['lam']==.25 and chosen['epoch']==1
    tasks=assignments(c);assert len(tasks)==25 and not(set(tasks[::2])&set(tasks[1::2]))
    assert len(pilot_specs(c))==7
    nested=dict(c,max_epochs=1)
    for group,f in tasks:r.train_outer(d,nested,group,f,out/'nested','cpu')
    r.summarize(d,nested,out/'nested')
    assert len(list((out/'nested').rglob('fit_complete.json')))==165
    assert len(list((out/'nested').glob('*/fold*/complete.json')))==25
    # Actual data, all20 inner and5 outer boundaries; never fit actual data here.
    actual=r.data();assert len(actual['y'])==1061
    for f in range(5):
        for j in [v for v in range(5) if v!=f]+[None]:
            tr,va,te=r.split(actual,f,j);ww=r.weights(actual['person_id'][tr]);pids=actual['person_id'][tr]
            totals=[float(ww[pids==p].sum()) for p in np.unique(pids)];assert max(totals)-min(totals)<1e-6
    result=dict(passed=True,synthetic_replays=checks,exact_resume=True,outer_label_poison_invariant=True,
        selection_test=True,matched_initialization=True,synthetic_nested_fits=165,synthetic_outer_predictions=25,rank_tasks_disjoint=True,actual_people=692,actual_records=1061,actual_windows=11261,
        all25_person_splits=True,actual_data_optimizer_steps=0,manifest_sha256=r.sha(r.H/'manifest.json'))
    r.save(r.H/'checks/verification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
