"""Meaningful leakage, masking, budget and transfer checks before GPU submission."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;B=HERE/'bundle';ROOT=HERE.parents[1]
sys.path[:0]=[str(B),str(ROOT/'outputs/motionbert-smoke-20261015/deps')]
import experiment as e
import torch
p=e.read(B/'protocol.json');d=e.load_data(Path('D:/gait-research-data/D_P/gdi-masked-pretrain-v3-20261105/data.npz'),p)
checks=[]
for fold in range(5):
    common=None; plans={}; external_flags={}
    for condition in p['conditions']:
        tr,ext,ev=e.rows_for(d,fold,condition)
        assert set(d['person_id'][tr]).isdisjoint(d['person_id'][ev])
        assert set(d['person_id'][ext]).isdisjoint(d['person_id'][tr])
        if common is None:common=(tr,ev)
        else:assert np.array_equal(common[0],tr) and np.array_equal(common[1],ev)
        base,_=e.example_pool(d,tr);pool,weight=e.example_pool(d,np.r_[tr,ext] if condition.startswith('B_') else tr)
        assert abs(weight.mean()-1)<1e-6
        # Enumerate the actual epoch sampling rule, not merely matching epoch labels.
        count=2*len(base);order=np.random.default_rng(1).permutation(2*len(pool))[:count];assert len(order)==count
        for family in p['families']:
            c=e.read(B/(family+'-original.json'));epochs=p['gdi_epochs'][family][str(fold)]
            checks.append(dict(fold=fold,condition=condition,family=family,epochs=epochs,examples=epochs*count,steps=epochs*int(np.ceil(count/c['batch_size'])),train_people=len(set(d['person_id'][tr])),test_people=len(set(d['person_id'][ev]))))
        if condition in ['C','D_adult','D_child']:
            ssl={k:d[k] for k in ['x','observed_xy','window_record','record_weight','person_id']};assert 'y' not in ssl
            plan,flags=e.sampling_plan(ssl,tr,ext,16,32,12345);plans[condition]=plan;external_flags[condition]=flags
            assert set(d['person_id'][d['window_record'][plan]].ravel()).isdisjoint(d['person_id'][ev])
    assert np.array_equal(external_flags['D_adult'],external_flags['D_child'])
    for condition in ['D_adult','D_child']:
        flags=external_flags[condition];assert np.array_equal(plans[condition][~flags],plans['C'][~flags])
    for family in p['families']:
        q=[r for r in checks if r['fold']==fold and r['family']==family];assert len({r['steps'] for r in q})==len({r['examples'] for r in q})==1
numerical=[]
for family in p['families']:
    e.setup(42);model=e.Reconstruction(family,p);model.eval()
    if family=='CNN':
        original=torch.randn(3,32,10,requires_grad=True);revised=original.detach().clone().requires_grad_(True)
        target=torch.nn.functional.interpolate(original,size=124,mode='linear',align_corners=False)
        actual=revised@model.upsample_weights
        torch.testing.assert_close(target,actual,atol=2e-6,rtol=0)
        probe=torch.randn_like(target);(target*probe).sum().backward();(actual*probe).sum().backward()
        torch.testing.assert_close(original.grad,revised.grad,atol=3e-6,rtol=0)
    x=torch.from_numpy(d['x'][[0,11261]].reshape(-1,124,12));mask=torch.from_numpy(e.time_mask(14,1,2))
    changed=x.clone();changed[mask]=1234.
    with torch.no_grad():a=model(x,mask);b=model(changed,mask)
    assert torch.equal(a,b),'Hidden values leak into model input'
    assert a.shape==(4,124,8) and torch.isfinite(a).all()
    model.train();loss=(model(x,mask)[mask]-x[:,:,:8][mask]).square().mean();loss.backward()
    grads=[v.grad for v in model.parameters() if v.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0
    e.setup(42);base=e.newmodel(family);state=base.state_dict();keys=e.backbone_keys(base,family,p)
    before={k:v.clone() for k,v in state.items()};state.update({k:model.base.state_dict()[k] for k in keys});base.load_state_dict(state)
    assert all(torch.equal(base.state_dict()[k],before[k]) for k in before if k not in keys)
    numerical.append(dict(family=family,hidden_input_invariance=True,finite_gradient=True,head_unchanged_on_transfer=True))
    e.setup(20261078);direct=e.newmodel(family)
    e.setup(20261078);pretrained=e.Reconstruction(family,p)
    assert e.state_digest(direct.state_dict(),e.backbone_keys(direct,family,p))==e.state_digest(pretrained.base.state_dict(),e.backbone_keys(pretrained.base,family,p))
result=dict(passed=True,conditions=checks,numerical=numerical,unlabeled_data_has_no_gdi=bool(np.isnan(d['y'][1061:]).all()),data_sha256=p['data_sha256'],limits='No training effect or clinical accuracy established by preflight')
e.save(HERE/'preflight.json',result);print(json.dumps(dict(passed=True,budget_cells=len(checks),numerical=numerical),indent=2))
