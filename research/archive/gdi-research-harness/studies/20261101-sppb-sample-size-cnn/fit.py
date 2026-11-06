"""Isolated GPU experiment: participant-wise source pretrain and nested target CV."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import socket
import tarfile
import time
import numpy as np
import torch
from model import TransferCNN
from bn import POLICY, reset_running_statistics, calibrate

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf8')
    tmp.replace(p)


def checkpoint(p, obj):
    p = Path(p)
    tmp = p.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(p)


def setup(seed, device):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == 'cuda':
        assert torch.cuda.is_available(), 'No silent CPU fallback'
        torch.cuda.manual_seed_all(seed)
        x = torch.ones(4, device=device)
        assert float((x*x).sum()) == 4.0
        torch.cuda.synchronize()


def data(name):
    with np.load(ROOT/(name+'.npz'), allow_pickle=False) as z:
        d = {k:z[k].copy() for k in z.files}
    n = len(d['ids'])
    assert d['x'].shape[1:] == (2,6,124) and np.isfinite(d['x']).all()
    assert d['y'].ndim == 2 and len(d['y']) == n and np.isfinite(d['y']).all()
    assert set(d['owner']) == set(range(n))
    for p in np.unique(d['ids']):
        assert len(np.unique(d['folds'][d['ids']==p])) == 1
        if name == 'target':
            assert len(np.unique(d['y'][d['ids']==p])) == 1
    return d


def groups(d, mask):
    return sorted(np.unique(d['ids'][mask]).tolist())


def partition(d, outer, inner=None):
    test = d['folds'] == outer
    tr = ~test
    va = np.zeros(len(tr),bool)
    if inner is not None:
        people = np.array(groups(d,tr))
        rng = np.random.RandomState(20261031)
        rng.shuffle(people)
        valid_people = np.array_split(people,3)[inner]
        va = np.isin(d['ids'],valid_people)
        tr &= ~va
    gg = [set(groups(d,m)) for m in (tr,va,test)]
    assert not (gg[0]&gg[1] or gg[0]&gg[2] or gg[1]&gg[2])
    return tr, va, test


def records_from_windows(pred, owner, n):
    out = np.zeros((n,pred.shape[1]),np.float64)
    np.add.at(out,owner,pred)
    counts = np.bincount(owner,minlength=n)
    return out/np.maximum(counts[:,None],1)


def person_arrays(d, pred, mask):
    people = groups(d,mask)
    return people, np.stack([d['y'][(d['ids']==p)&mask].mean(0) for p in people]), np.stack([pred[(d['ids']==p)&mask].mean(0) for p in people])


def mae(d, pred, mask, scale=None):
    if d['y'].shape[1] == 1:
        _,truth,values=person_arrays(d,pred,mask)
        return float(np.abs(values-truth).mean())
    # Source has condition-specific labels: average errors, not opposing signed errors.
    err = np.abs(pred-d['y'])
    if scale is not None:
        err = err/scale
    return float(np.mean([err[(d['ids']==p)&mask].mean() for p in groups(d,mask)]))


def predict(model,d,device,mu,sd):
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0,len(d['x']),128):
            x = torch.from_numpy(d['x'][a:a+128]).to(device)
            out.append(model(x).cpu().numpy()*sd+mu)
    return records_from_windows(np.concatenate(out),d['owner'],len(d['ids']))


def scale(d, train):
    weights = np.zeros(len(train),np.float64)
    people = groups(d,train)
    for p in people:
        mask = (d['ids']==p)&train
        weights[mask] = 1.0/mask.sum()/len(people)
    mu = (d['y']*weights[:,None]).sum(0).astype(np.float32)
    sd = np.sqrt(((d['y']-mu)**2*weights[:,None]).sum(0)).astype(np.float32)
    return mu, np.maximum(sd,1e-4)


def fit(d, train, valid, out, config, seed, source, initial, epochs, patience, device, test_people):
    identity = dict(plan_sha256=sha(ROOT/'plan.json'),bn_policy=POLICY, seed=int(seed),source=source,config=config,max_epochs=int(epochs),
                    train_people=groups(d,train),validation_people=groups(d,valid),test_people=test_people,
                    input_sha256=sha(ROOT/('source.npz' if d['y'].shape[1]>1 else 'target.npz')),
                    initialization_sha256=sha(initial) if initial else None)
    if (out/'fit_complete.json').exists():
        assert load(out/'identity.json') == identity, 'Completed fit identity mismatch'
        return load(out/'fit_complete.json')
    out.mkdir(parents=True,exist_ok=True)
    save(out/'identity.json',identity)
    setup(seed,device)
    model = TransferCNN(d['y'].shape[1],config['dropout'],config['channels'])
    copied = model.initialize(torch.load(initial,map_location='cpu',weights_only=True) if initial else {},source)
    reset_running_statistics(model)
    model.freeze(config['freeze_blocks'])
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
    mu, sd = scale(d,train)
    windows = np.flatnonzero(train[d['owner']])
    wid = d['ids'][d['owner'][windows]]
    unique, counts = np.unique(wid,return_counts=True)
    count_map = dict(zip(unique,counts))
    weights = np.array([len(windows)/(len(unique)*count_map[p]) for p in wid],np.float32)
    np.testing.assert_allclose(weights.sum(),len(windows),rtol=1e-6)
    target = (d['y'][d['owner'][windows]]-mu)/sd
    rng = np.random.RandomState(seed)
    curves = []
    best, best_epoch = float('inf'),0
    steps = 0
    started = time.monotonic()
    for epoch in range(1,epochs+1):
        model.train()
        order = rng.permutation(len(windows))
        loss_sum = 0.
        for a in range(0,len(order),32):
            b = order[a:a+32]
            xx = torch.from_numpy(d['x'][windows[b]]).to(device)
            yy = torch.from_numpy(target[b]).to(device)
            ww = torch.from_numpy(weights[b]).to(device)
            opt.zero_grad(set_to_none=True)
            pp = model(xx)
            if config['loss'] == 'mse':
                err = (pp-yy).square().mean(1)
            else:
                err = torch.nn.functional.smooth_l1_loss(pp,yy,reduction='none').mean(1)
            loss = (err*ww).mean()
            assert torch.isfinite(loss).all()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            opt.step()
            loss_sum += float(loss.detach())*len(b)
            steps += 1
        calibration = calibrate(model,d,train,device,epoch,identity['input_sha256'])
        if valid.any():
            predictions = predict(model,d,device,mu,sd)
            score = mae(d,predictions,valid,sd if d['y'].shape[1]>1 else None)
        else:
            score = loss_sum/len(windows)
        curves.append(dict(epoch=epoch,train_loss=loss_sum/len(windows),validation_MAE=score if valid.any() else None))
        if score < best:
            best,best_epoch = score,epoch
            if valid.any():
                checkpoint(out/'model.pt',dict(model=model.cpu().state_dict(),mu=torch.from_numpy(mu),sd=torch.from_numpy(sd)))
                model.to(device)
                save(out/'bn_calibration.json',calibration)
        save(out/'curve.json',curves)
        if epoch % 10 == 0:
            print(json.dumps({'fit':str(out.relative_to(ROOT)),'epoch':epoch,'score':score}),flush=True)
        if valid.any() and epoch-best_epoch >= patience:
            break
    if not valid.any():
        best_epoch = epochs
        checkpoint(out/'model.pt',dict(model=model.cpu().state_dict(),mu=torch.from_numpy(mu),sd=torch.from_numpy(sd)))
        model.to(device)
        save(out/'bn_calibration.json',calibration)
    result = dict(best_epoch=best_epoch,best_validation_MAE=best if valid.any() else None,epochs_run=len(curves),optimizer_steps=steps,
                  copied_backbone_tensors=len(copied),seconds=time.monotonic()-started,checkpoint_sha256=sha(out/'model.pt'))
    # Reload the actual selected checkpoint and validate deterministic saved inference.
    ck = torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['model'])
    model.eval()
    probe = torch.from_numpy(d['x'][:3]).to(device)
    with torch.no_grad():
        p1 = model(probe).cpu().numpy()
    other = TransferCNN(d['y'].shape[1],config['dropout'],config['channels']).to(device)
    other.load_state_dict(ck['model'])
    other.eval()
    with torch.no_grad():
        p2 = other(probe).cpu().numpy()
    np.testing.assert_allclose(p1,p2,rtol=0,atol=1e-6)
    result['reload_max_error'] = float(np.max(np.abs(p1-p2)))
    save(out/'fit_complete.json',result)
    return result

