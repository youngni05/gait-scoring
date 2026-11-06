"""Small real-data CPU checks; not a performance experiment."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
import train
from model import TransferCNN
from prepare import normalize

HERE=Path(__file__).resolve().parent
BUNDLE=Path('D:/gait-research-data/D_P/healthgait-transfer-20261101/bundle')


def main():
    torch.set_num_threads(2)
    ck=torch.load(BUNDLE/'gdi.pt',map_location='cpu',weights_only=True)
    original_path=HERE.parent/'20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/model.py'
    spec=importlib.util.spec_from_file_location('original_gdi_model',original_path)
    original_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(original_module)
    original=original_module.GdiCNN();original.load_state_dict(ck['model']);original.eval()
    adapted=TransferCNN(1);copied=adapted.initialize(ck,'gdi');adapted.eval()
    assert len(copied)==44
    hooks={}
    original.hidden.register_forward_hook(lambda m,a,o:hooks.update(original=o.detach()))
    adapted.hidden.register_forward_hook(lambda m,a,o:hooks.update(adapted=o.detach()))
    torch.manual_seed(17);x=torch.randn(4,2,6,124)
    with torch.no_grad():
        original(torch.cat([x.reshape(-1,6,124),torch.zeros(8,6,124)],dim=1))
        adapted(x)
    error=float((hooks['original']-hooks['adapted']).abs().max())
    assert error<1e-5,'First-convolution channel selection changed backbone semantics'
    for blocks in (0,4):
        m=TransferCNN(1);m.freeze(blocks);m.train()
        assert all(not m.norms[i].training for i in range(blocks))
        assert m.norms[-1].training
    p=np.ones((91,2,3,2));p[:,:,0]=[2,3];p[:,:,1]=[1,2];p[:,:,2]=[1,1]
    _,why=normalize(p);assert why is None
    q=p.copy();q[:30,:,0]=np.nan;assert normalize(q)[1]=='missing_over25pct'
    q=p.copy();q[0,:,0]=np.nan;assert normalize(q)[1]=='missing_boundary_no_extrapolation'
    q=p.copy();q[30:35,:,0]=np.nan;assert normalize(q)[1] is None
    try:TransferCNN(1)(torch.zeros(1,2,12,124))
    except ValueError:pass
    else:raise AssertionError('Wrong feature shape accepted')
    train.ROOT=BUNDLE
    tiny={'ids':np.array(['a','a']),'y':np.zeros((2,1))}
    assert train.mae(tiny,np.array([[1.],[-1.]]),np.ones(2,bool))==0.0
    d=train.data('target');a,b,t=train.partition(d,0,0)
    assert not(set(train.groups(d,a))&set(train.groups(d,b)))
    r=train.fit(d,a,b,BUNDLE/'checks/cpu-real-gdi-v2',train.candidate_grid(0)[0],20261101,'gdi',BUNDLE/'gdi.pt',2,12,'cpu',train.groups(d,t))
    assert r['optimizer_steps']>0 and r['reload_max_error']<1e-6
    result=dict(passed=True,real_target_people=44,real_target_records=220,first_conv_embedding_max_error=error,copied_tensors=len(copied),real_CPU_optimizer_steps=r['optimizer_steps'],reload_max_error=r['reload_max_error'],invalid_shape_rejected=True,missing_boundary_rejected=True,internal_gap_interpolated=True,frozen_BN_stays_eval=True,scope='CPU correctness only; GPU and performance not verified here')
    (HERE/'local-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
