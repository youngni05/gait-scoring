"""Frozen single-window CNN plus frozen MotionBERT-feature residual model. Python 3.8+."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import tarfile
import time
import numpy as np
import torch
from base_model import GdiCNN, channels
from residual import Correction, feature_stats

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(str(temp), str(path))


def save_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(str(temp), str(path))


def setup(seed, device):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == 'cuda':
        require(torch.cuda.is_available(), 'No CUDA; no CPU fallback')
        torch.cuda.manual_seed_all(seed)
        print('CUDA allocation probe', flush=True)
        torch.empty(1, device=device)
        torch.cuda.synchronize()


def split(d, outer, inner=None):
    require(outer in range(5) and (inner is None or inner in range(5) and outer != inner), 'Invalid fold')
    te = d['fold'] == outer
    va = np.zeros(len(te), bool) if inner is None else d['fold'] == inner
    tr = ~(te | va)
    groups = [set(d['person_id'][m]) for m in (tr, va, te)]
    require(not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]), 'People overlap')
    return tr, va, te


def load_data():
    for name, digest in load(ROOT/'manifest.json').items():
        require(sha(ROOT/name) == digest, 'Bundle changed: '+name)
    with np.load(ROOT/'data.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    require(d['x'].shape == (1061,2,124,12) and d['x'].dtype == np.float32, 'Wrong geometry data')
    require(d['y'].shape == (1061,2) and len(set(d['person_id'])) == 692, 'Wrong cohort')
    require(len(set(d['video_id'])) == 1061 and len(set(d['original_row_index'])) == 1061, 'Duplicate rows')
    require(np.isfinite(d['x']).all() and np.isfinite(d['y']).all(), 'Nonfinite data')
    m = np.load(ROOT/'features.npy', mmap_mode='r', allow_pickle=False)
    require(m.shape == (1061,124,512) and m.dtype == np.float32 and np.isfinite(m).all(), 'Wrong temporal features')
    source = load(ROOT/'sources.json')
    require(sha(ROOT/'features.npy') == source['feature_cache_sha256'], 'Wrong frozen features')
    pooled = np.load(ROOT/'pooled_reference.npy', allow_pickle=False)
    require(float(np.max(np.abs(m.mean(axis=1)-pooled))) <= 1e-4, 'Pooled features mismatch')
    require(sha(ROOT/'data.npz') == source['base_data_sha256'], 'Baseline records changed')
    for f in range(5):
        tr,va,te = split(d,f)
        require(len(set(d['person_id'][te])) == [138,138,138,139,139][f], 'Wrong evaluation membership')
    return d, m


def baseline_path(outer, inner):
    return ROOT/'base'/('fold%d'%outer)/('refit' if inner is None else 'inner%d'%inner)


def baseline(d, outer, inner, device, rows):
    """Only forward inference; validate all training people before loading weights."""
    tr,va,te = split(d,outer,inner)
    folder = baseline_path(outer,inner)
    info = load(folder/'fit_complete.json')
    ident = info['identity']
    source = load(ROOT/'sources.json')
    require(ident['bundle_sha256'] == source['base_bundle_sha256'], 'Different baseline bundle')
    require(ident['outer_fold'] == outer and ident['inner_fold'] == inner, 'Wrong baseline fold')
    require(ident['purpose'] == ('outer_refit' if inner is None else 'inner_epoch_selection'), 'Wrong baseline role')
    for key, mask in [('train_people',tr),('validation_people',va),('outer_evaluation_people',te)]:
        require(ident[key] == np.unique(d['person_id'][mask]).tolist(), 'Baseline people mismatch: '+key)
    require(inner is None or info['epochs'] == 100, 'Inner baseline must use fixed 100 epochs')
    ckpath = folder/'latest.pt'
    require(sha(ckpath) == info['checkpoint_sha256'], 'Baseline checkpoint corrupt')
    ck = torch.load(ckpath, map_location='cpu', weights_only=True)
    require(ck['identity'] == ident and len(ck['history']) == info['epochs'] == ident['epochs'], 'Checkpoint identity/history mismatch')
    require(info['optimizer_steps'] == info['epochs']*math.ceil(2*int(tr.sum())/32), 'Baseline steps mismatch')
    require(float(ck['lo']) == float(np.float32(d['y'][tr].min())), 'Baseline label minimum mismatch')
    require(float(ck['span']) == float(np.float32(d['y'][tr].max()-d['y'][tr].min())), 'Baseline label scale mismatch')
    model = GdiCNN().to(device)
    model.load_state_dict(ck['model'],strict=True)
    model.requires_grad_(False)
    model.eval()
    require(not any(p.requires_grad for p in model.parameters()), 'Baseline is trainable')
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    coefficient = np.asarray(ck['history'][-1]['calibration'],dtype=np.float64)
    require(coefficient.shape == (2,) and np.isfinite(coefficient).all(), 'Invalid baseline calibration')
    result = []
    with torch.inference_mode():
        for start in range(0,len(rows),128):
            x = channels(torch.from_numpy(d['x'][rows[start:start+128]]))
            p = model(x.reshape(-1,12,124).to(device))[:,0]*ck['span'].to(device)+ck['lo'].to(device)
            result.append(p.reshape(-1,2).cpu().numpy())
    require(all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items()), 'Frozen CNN changed')
    raw = np.concatenate(result)
    return raw.astype(np.float64)*coefficient[0]+coefficient[1]


def metric(y, pred, ids):
    require(y.shape == pred.shape and y.shape == (len(ids),2) and np.isfinite(pred).all(), 'Invalid metric input')
    return float(np.mean([np.abs(pred[ids==p]-y[ids==p]).mean() for p in np.unique(ids)]))


def predict(model, features, base, device):
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0,len(features),64):
            x = torch.from_numpy(np.array(features[start:start+64],copy=True)).to(device)
            values.append(model(x).cpu().numpy())
    return base.astype(np.float64)+np.concatenate(values).astype(np.float64)


def rng_state():
    return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def fit(features, base, y, ids, val, c, epochs, folder, ident, device, stop_after=None):
    """Only current training and optional inner validation data; no outer evaluation arguments."""
    folder.mkdir(parents=True,exist_ok=True)
    setup(ident['seed'],device)
    model = Correction().to(device)
    mean, scale = feature_stats(features)
    model.mean.copy_(mean)
    model.scale.copy_(scale)
    opt = torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    ckpath = folder/'latest.pt'
    history = []
    if ckpath.exists():
        ck = torch.load(ckpath,map_location=device,weights_only=True)
        require(ck['identity'] == ident,'Resume identity mismatch')
        require(torch.equal(ck['model']['mean'].cpu(),mean) and torch.equal(ck['model']['scale'].cpu(),scale),'Training statistics changed')
        model.load_state_dict(ck['model'],strict=True)
        opt.load_state_dict(ck['optimizer'])
        history=ck['history']
        torch.set_rng_state(ck['rng']['cpu'].cpu())
        if ck['rng']['cuda']:
            torch.cuda.set_rng_state_all([x.cpu() for x in ck['rng']['cuda']])
    require(not history or history[-1]['epoch'] <= epochs,'Requested fewer epochs than checkpoint')
    def record(epoch,loss,seconds):
        row={'epoch':epoch,'loss':loss,'train_person_mae':metric(y,predict(model,features,base,device),ids),'seconds':seconds}
        if val is not None:
            vm,vb,vy,vi=val
            row.update(validation_person_mae=metric(vy,predict(model,vm,vb,device),vi),validation_people=len(np.unique(vi)))
        history.append(row)
        save_checkpoint(ckpath,dict(identity=ident,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),history=history))
        save(folder/'curve.json',history)
        print(json.dumps(dict(fit=str(folder),**row)),flush=True)
    if not history:
        zero=predict(model,features,base,device)
        require(np.array_equal(zero,base),'Zero correction changed baseline')
        record(0,0.0,0.0)
    target=torch.from_numpy(((y-base)/10.0).astype(np.float32))
    for epoch in range(history[-1]['epoch']+1,epochs+1):
        started=time.monotonic()
        model.train()
        order=np.random.default_rng(ident['seed']+epoch).permutation(len(y))
        total=0.0
        for start in range(0,len(order),c['batch_size']):
            ix=order[start:start+c['batch_size']]
            opt.zero_grad(set_to_none=True)
            output=model(torch.from_numpy(np.array(features[ix],copy=True)).to(device))/10.0
            loss=(output-target[ix].to(device)).square().mean()
            require(bool(torch.isfinite(loss)),'Nonfinite correction loss')
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Nonfinite gradient')
            opt.step()
            total+=float(loss.detach())*len(ix)
        record(epoch,total/len(y),time.monotonic()-started)
        if stop_after is not None and epoch >= stop_after:
            return model,history
    save(folder/'fit_complete.json',dict(identity=ident,epochs=epochs,optimizer_steps=epochs*math.ceil(len(y)/c['batch_size']),
        checkpoint_sha256=sha(ckpath),curve_sha256=sha(folder/'curve.json')))
    return model,history


def identity(d,c,outer,inner,epochs):
    tr,va,te=split(d,outer,inner)
    return dict(bundle_sha256=sha(ROOT/'manifest.json'),outer_fold=outer,inner_fold=inner,epochs=epochs,
        seed=c['seed']+outer*100+(50 if inner is None else inner),
        baseline_checkpoint_sha256=sha(baseline_path(outer,inner)/'latest.pt'),
        train_people=np.unique(d['person_id'][tr]).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),
        outer_evaluation_people=np.unique(d['person_id'][te]).tolist())


def select_epoch(histories):
    require(len(histories)==4 and len(set(map(len,histories)))==1,'Incomplete inner histories')
    means=np.average([[r['validation_person_mae'] for r in h] for h in histories],axis=0,
        weights=[h[0]['validation_people'] for h in histories])
    require(np.isfinite(means).all(),'Nonfinite selection')
    return int(np.argmin(means)),means.tolist()


def train_fold(d,m,c,f,out,device):
    dest=out/('fold%d'%f)
    histories=[]
    for j in range(5):
        if f==j:continue
        tr,va,te=split(d,f,j)
        rows=np.flatnonzero(tr|va)
        bp=np.full_like(d['y'],np.nan,dtype=np.float64)
        bp[rows]=baseline(d,f,j,device,rows)
        ident=identity(d,c,f,j,c['max_epochs'])
        _,hist=fit(m[tr],bp[tr],d['y'][tr],d['person_id'][tr],(m[va],bp[va],d['y'][va],d['person_id'][va]),
            c,c['max_epochs'],dest/('inner%d'%j),ident,device)
        histories.append(hist)
    epoch,means=select_epoch(histories)
    save(dest/'selection.json',dict(selected_epoch=epoch,inner_weighted_person_mae=means,includes_zero_correction=True,outer_evaluation_used=False))
    tr,va,te=split(d,f)
    base_train=baseline(d,f,None,device,np.flatnonzero(tr))
    model,hist=fit(m[tr],base_train,d['y'][tr],d['person_id'][tr],None,c,epoch,dest/'refit',identity(d,c,f,None,epoch),device)
    base_test=baseline(d,f,None,device,np.flatnonzero(te))
    reference=np.load(ROOT/'single_oof.npy',allow_pickle=False)
    require(float(np.max(np.abs(base_test-reference[te])))<=1e-4,'Frozen baseline evaluation replay mismatch')
    pred=predict(model,m[te],base_test,device)
    np.savez_compressed(dest/'predictions.npz',predictions=pred,baseline=base_test,row_index=np.flatnonzero(te),y=d['y'][te],person_id=d['person_id'][te])
    save(dest/'complete.json',dict(identity=identity(d,c,f,None,epoch),selected_epoch=epoch,
        prediction_sha256=sha(dest/'predictions.npz'),checkpoint_sha256=sha(dest/'refit/latest.pt'),
        selection_sha256=sha(dest/'selection.json'),mae=metric(d['y'][te],pred,d['person_id'][te]),
        baseline_mae=metric(d['y'][te],base_test,d['person_id'][te]),device=device,torch=str(torch.__version__)))
    print('RESIDUAL_FOLD_COMPLETE',f,flush=True)


def summarize(d,c,out):
    pred=np.full_like(d['y'],np.nan,dtype=np.float64)
    base=np.full_like(pred,np.nan)
    seen=np.zeros(len(pred),int)
    steps=0
    for f in range(5):
        dest=out/('fold%d'%f)
        complete=load(dest/'complete.json')
        require(complete['identity']==identity(d,c,f,None,complete['selected_epoch']),'Wrong outer identity')
        histories=[]
        for j in range(5):
            if j==f:continue
            p=dest/('inner%d'%j)
            done=load(p/'fit_complete.json')
            require(done['identity']==identity(d,c,f,j,c['max_epochs']),'Wrong inner identity')
            require(sha(p/'curve.json')==done['curve_sha256'] and sha(p/'latest.pt')==done['checkpoint_sha256'],'Inner artifact changed')
            history=load(p/'curve.json')
            tr,va,_=split(d,f,j)
            require(done['epochs']==c['max_epochs'] and [r['epoch'] for r in history]==list(range(c['max_epochs']+1)),'Incomplete inner epochs')
            require(done['optimizer_steps']==c['max_epochs']*math.ceil(int(tr.sum())/c['batch_size']),'Wrong optimizer count')
            require(all(r['validation_people']==len(set(d['person_id'][va])) for r in history),'Wrong validation weights')
            histories.append(history)
            steps+=done['optimizer_steps']
        chosen,means=select_epoch(histories)
        selection=load(dest/'selection.json')
        require(chosen==complete['selected_epoch']==selection['selected_epoch'],'Wrong epoch selection')
        require(means==selection['inner_weighted_person_mae'] and not selection['outer_evaluation_used'],'Selection mismatch')
        refit=load(dest/'refit/fit_complete.json')
        require(refit['identity']==complete['identity'] and refit['epochs']==chosen,'Refit identity mismatch')
        require(sha(dest/'refit/latest.pt')==refit['checkpoint_sha256']==complete['checkpoint_sha256'],'Refit checkpoint changed')
        require(sha(dest/'refit/curve.json')==refit['curve_sha256'],'Refit curve changed')
        steps+=refit['optimizer_steps']
        require(sha(dest/'selection.json')==complete['selection_sha256'] and sha(dest/'predictions.npz')==complete['prediction_sha256'],'Output changed')
        with np.load(dest/'predictions.npz',allow_pickle=False) as z:
            rows=np.flatnonzero(d['fold']==f)
            for k,expect in [('row_index',rows),('y',d['y'][rows]),('person_id',d['person_id'][rows])]:
                require(np.array_equal(z[k],expect),'Wrong prediction alignment')
            pred[rows]=z['predictions'];base[rows]=z['baseline'];seen[rows]+=1
    require((seen==1).all() and np.isfinite(pred).all(),'Incomplete evaluation')
    np.savez_compressed(out/'oof_predictions.npz',predictions=pred,baseline=base,**{k:v for k,v in d.items() if k!='x'})
    summary=dict(method=c['name'],people=len(set(d['person_id'])),records=len(d['y']),mae=metric(d['y'],pred,d['person_id']),
        baseline_mae=metric(d['y'],base,d['person_id']),fits=25,optimizer_steps=steps,
        selected_epochs=[load(out/('fold%d'%f)/'selection.json')['selected_epoch'] for f in range(5)],
        frozen_cnn=True,frozen_motionbert=True,independent_final_evaluation=False,bundle_sha256=sha(ROOT/'manifest.json'),
        oof_sha256=sha(out/'oof_predictions.npz'))
    save(out/'summary.json',summary)
    with tarfile.open(out.parent/(out.name+'-review.tar.gz'),'w:gz') as tf:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.npz'):tf.add(str(p),arcname='results/'+p.relative_to(out).as_posix())
        for name in ('run.py','residual.py','base_model.py','protocol.json','sources.json','manifest.json'):
            tf.add(str(ROOT/name),arcname='bundle/'+name)
    print(json.dumps(summary),flush=True)


def check(d,device):
    reference=np.load(ROOT/'single_oof.npy',allow_pickle=False)
    max_error=0.0
    for f in range(5):
        for j in list(k for k in range(5) if k!=f)+[None]:
            tr,va,te=split(d,f,j)
            rows=np.flatnonzero(te if j is None else va)[:2]
            pred=baseline(d,f,j,device,rows)
            require(np.isfinite(pred).all(),'Nonfinite baseline')
            if j is None:
                err=float(np.max(np.abs(pred-reference[rows])))
                require(err<=1e-4,'Baseline replay failed')
                max_error=max(max_error,err)
    result=dict(passed=True,people=692,records=1061,baseline_checkpoints=25,all_training_membership_checked=True,
        frozen_baseline_state_unchanged=True,baseline_replay_max_error=max_error,device=device,real_optimizer_steps=0)
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['check','pilot','train'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=load(ROOT/'protocol.json')
    setup(c['seed'],a.device)
    d,m=load_data()
    a.output.mkdir(parents=True,exist_ok=True)
    env=dict(hostname=socket.gethostname(),slurm_job_id=os.getenv('SLURM_JOB_ID'),device=a.device,torch=str(torch.__version__),
        cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None)
    save(a.output/'environment.json',env)
    if a.command=='check':
        save(a.output/'check.json',check(d,a.device));return
    if a.command=='pilot':
        tr,va,te=split(d,0,1)
        rows=np.flatnonzero(tr|va)
        bp=np.full_like(d['y'],np.nan,dtype=float);bp[rows]=baseline(d,0,1,a.device,rows)
        if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        model,h=fit(m[tr],bp[tr],d['y'][tr],d['person_id'][tr],(m[va],bp[va],d['y'][va],d['person_id'][va]),
            c,c['pilot_epochs'],a.output/'fit',identity(d,c,0,1,c['pilot_epochs']),a.device)
        result=dict(passed=True,pilot_only=True,epochs=c['pilot_epochs'],train_people=len(set(d['person_id'][tr])),
            validation_people=len(set(d['person_id'][va])),outer_evaluation_used=False,zero_epoch_mae=h[0]['validation_person_mae'],
            final_validation_mae=h[-1]['validation_person_mae'],seconds=time.monotonic()-started,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None)
        save(a.output/'pilot_summary.json',result);print(json.dumps(result),flush=True);return
    for f in range(5):train_fold(d,m,c,f,a.output,a.device)
    summarize(d,c,a.output)
    print('FROZEN_RESIDUAL_TRAINING_AND_EXPORT_FINISHED',flush=True)


if __name__=='__main__':main()
