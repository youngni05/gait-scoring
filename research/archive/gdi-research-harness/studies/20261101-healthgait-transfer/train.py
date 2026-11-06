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
    identity = dict(seed=int(seed),source=source,config=config,max_epochs=int(epochs),
                    train_people=groups(d,train),validation_people=groups(d,valid),test_people=test_people,
                    input_sha256=sha(ROOT/('source.npz' if d['y'].shape[1]>1 else 'target.npz')),
                    initialization_sha256=sha(initial) if initial else None)
    if (out/'fit_complete.json').exists():
        assert load(out/'identity.json') == identity, 'Completed fit identity mismatch'
        return load(out/'fit_complete.json')
    out.mkdir(parents=True,exist_ok=True)
    save(out/'identity.json',identity)
    setup(seed,device)
    model = TransferCNN(d['y'].shape[1],config['dropout'])
    copied = model.initialize(torch.load(initial,map_location='cpu',weights_only=True) if initial else {},source)
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
        save(out/'curve.json',curves)
        if epoch % 10 == 0:
            print(json.dumps({'fit':str(out.relative_to(ROOT)),'epoch':epoch,'score':score}),flush=True)
        if valid.any() and epoch-best_epoch >= patience:
            break
    if not valid.any():
        best_epoch = epochs
        checkpoint(out/'model.pt',dict(model=model.cpu().state_dict(),mu=torch.from_numpy(mu),sd=torch.from_numpy(sd)))
        model.to(device)
    result = dict(best_epoch=best_epoch,best_validation_MAE=best if valid.any() else None,epochs_run=len(curves),optimizer_steps=steps,
                  copied_backbone_tensors=len(copied),seconds=time.monotonic()-started,checkpoint_sha256=sha(out/'model.pt'))
    # Reload the actual selected checkpoint and validate deterministic saved inference.
    ck = torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['model'])
    model.eval()
    probe = torch.from_numpy(d['x'][:3]).to(device)
    with torch.no_grad():
        p1 = model(probe).cpu().numpy()
    other = TransferCNN(d['y'].shape[1],config['dropout']).to(device)
    other.load_state_dict(ck['model'])
    other.eval()
    with torch.no_grad():
        p2 = other(probe).cpu().numpy()
    np.testing.assert_allclose(p1,p2,rtol=0,atol=1e-6)
    result['reload_max_error'] = float(np.max(np.abs(p1-p2)))
    save(out/'fit_complete.json',result)
    return result


def source_pretrain(device,pilot=False):
    d = data('source')
    config = load(ROOT/'protocol.json')['round0_candidates'][0]
    config = dict(config, loss='smooth_l1')
    train,valid = d['folds']!=0,d['folds']==0
    root = ROOT/'runs'/('pilot/source' if pilot else 'source')
    result = fit(d,train,valid,root/'select',config,20261101,'scratch',None,3 if pilot else 60,10,device,[])
    if pilot:
        return result
    all_train = np.ones(len(train),bool)
    final = fit(d,all_train,np.zeros_like(train),root/'refit',config,20261101,'scratch',None,result['best_epoch'],10,device,[])
    save(root/'summary.json',dict(selection=result,refit=final,people=len(groups(d,all_train)),refit_all_source_people=True))
    return final


def candidate_grid(round_no):
    base = load(ROOT/'protocol.json')['round0_candidates'][0]
    if round_no == 0:
        return [base]
    if round_no == 1:
        return [dict(base,lr=lr,freeze_blocks=fr) for lr in (0.001,0.0001) for fr in (0,4)]
    return [dict(base,lr=lr,dropout=drop,loss=loss) for lr in (0.001,0.0001) for drop in (0.2,0.5) for loss in ('mse','smooth_l1')]


def run_fold_unlocked(round_no,fold,device):
    d = data('target')
    tr,_,te = partition(d,fold)
    root = ROOT/('runs/round%d/fold%d'%(round_no,fold))
    if (root/'complete.json').exists():
        return
    root.mkdir(parents=True,exist_ok=True)
    initial = {'scratch':None,'gdi':ROOT/'gdi.pt','health':ROOT/'runs/source/refit/model.pt'}
    rows=[]
    for source in ('scratch','gdi','health'):
        for seed in load(ROOT/'protocol.json')['target']['seeds']:
            scored=[]
            for ci,c in enumerate(candidate_grid(round_no)):
                inner=[]
                for j in range(3):
                    a,b,_ = partition(d,fold,j)
                    out = root/('%s-s%d/candidate%d/inner%d'%(source,seed,ci,j))
                    r = fit(d,a,b,out,c,seed+fold*100+j,source,initial[source],80,12,device,groups(d,te))
                    inner.append(r)
                # Average held-out people absolute error across the three inner partitions.
                sizes=[len(partition_people(d,fold,j)) for j in range(3)]
                score=float(np.average([r['best_validation_MAE'] for r in inner],weights=sizes))
                scored.append(dict(config=c,candidate=ci,score=score,epochs=int(np.median([r['best_epoch'] for r in inner]))))
            selected = min(scored,key=lambda r:(r['score'],r['candidate']))
            armroot = root/('%s-s%d'%(source,seed))
            save(armroot/'selection.json',dict(candidates=scored,selected=selected))
            out = armroot/'refit'
            fit(d,tr,np.zeros(len(tr),bool),out,selected['config'],seed+fold*100+50,source,initial[source],selected['epochs'],12,device,groups(d,te))
            ck = torch.load(out/'model.pt',map_location='cpu',weights_only=True)
            model = TransferCNN(1,selected['config']['dropout']).to(device)
            model.load_state_dict(ck['model'])
            pp = predict(model,d,device,ck['mu'].numpy(),ck['sd'].numpy())
            for person,y,p in zip(*person_arrays(d,pp,te)):
                rows.append(dict(id=person,fold=fold,arm=source,seed=seed,y=float(y[0]),prediction=float(p[0])))
            keep = te[d['owner']]
            model.eval()
            # Preserve per-window predictions for exact aggregation replay.
            preds=[]
            with torch.no_grad():
                for a in range(0,int(keep.sum()),128):
                    q=d['x'][keep][a:a+128]
                    preds.append(model(torch.from_numpy(q).to(device)).cpu().numpy()*ck['sd'].numpy()+ck['mu'].numpy())
            np.savez_compressed(armroot/'predictions.npz',record_indices=np.flatnonzero(te),record_predictions=pp[te],window_indices=np.flatnonzero(keep),window_predictions=np.concatenate(preds))
    with (root/'person-predictions.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(root/'complete.json',dict(fold=fold,round=round_no,people=groups(d,te),prediction_rows=len(rows),expected_final_models=9,expected_inner_fits=27*len(candidate_grid(round_no))))


def run_fold(round_no,fold,device):
    if (ROOT/('runs/round%d/fold%d/complete.json'%(round_no,fold))).exists():
        return
    lock=ROOT/('control/round%d-fold%d.lock'%(round_no,fold))
    lock.parent.mkdir(exist_ok=True)
    with lock.open('x') as f:
        f.write(json.dumps({'job':os.environ.get('SLURM_JOB_ID'),'pid':os.getpid()}))
    try:
        run_fold_unlocked(round_no,fold,device)
    finally:
        lock.unlink()


def partition_people(d,fold,j):
    return groups(d,partition(d,fold,j)[1])


def summarize_completion(round_no):
    root=ROOT/('runs/round%d'%round_no)
    if not all((root/('fold%d/complete.json'%f)).exists() for f in range(5)):
        return
    lock=root/'summary.lock'
    try:
        with lock.open('x') as f:f.write(os.environ.get('SLURM_JOB_ID','local'))
    except FileExistsError:
        return
    records=[load(root/('fold%d/complete.json'%f)) for f in range(5)]
    save(root/'summary.json',dict(round=round_no,outer_folds=5,final_models=45,inner_fits=135*len(candidate_grid(round_no)),completion_records=records,scope='Completion metadata only; metrics and replay require independent verification'))
    save(root/'summary_complete.json',dict(passed=True,summary_sha256=sha(root/'summary.json')))


def pilot(device):
    setup(20261101,device)
    src = source_pretrain(device,True)
    d=data('target');a,b,t=partition(d,0,0)
    c=candidate_grid(0)[0]
    target=fit(d,a,b,ROOT/'runs/pilot/target',c,20261101,'gdi',ROOT/'gdi.pt',3,12,device,groups(d,t))
    # Verify shape-compatible source checkpoint initialization as well.
    ck=torch.load(ROOT/'runs/pilot/source/select/model.pt',map_location='cpu',weights_only=True)
    m=TransferCNN(1);copied=m.initialize(ck,'health');assert len(copied)>20
    save(ROOT/'runs/pilot/summary.json',dict(passed=True,source=src,target=target,cuda=device=='cuda',gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,torch=str(torch.__version__),python=platform.python_version(),hostname=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),max_memory=torch.cuda.max_memory_allocated() if device=='cuda' else None))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','source','train','check'])
    ap.add_argument('--round',type=int,default=0);ap.add_argument('--folds',default='0,1,2,3,4');ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    for name,digest in load(ROOT/'manifest.json').items():
        assert sha(ROOT/name)==digest,name
    if a.mode=='check':
        source,target=data('source'),data('target')
        assert len(np.unique(target['ids']))==44 and len(target['ids'])==220
        for f in range(5):
            for j in range(3):partition(target,f,j)
        print(json.dumps({'passed':True,'source_people':len(np.unique(source['ids'])),'source_records':len(source['ids']),'source_windows':len(source['x']),'target_people':44,'target_records':220}),flush=True)
        return
    if a.mode=='pilot':pilot(a.device)
    elif a.mode=='source':source_pretrain(a.device)
    else:
        assert (ROOT/'runs/source/refit/fit_complete.json').exists()
        for f in map(int,a.folds.split(',')):run_fold(a.round,f,a.device)
        summarize_completion(a.round)
    print('HEALTHGAIT_'+a.mode.upper()+'_FINISHED',flush=True)


if __name__=='__main__':main()
