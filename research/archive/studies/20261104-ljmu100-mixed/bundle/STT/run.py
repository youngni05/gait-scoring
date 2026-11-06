"""STT nested person-disjoint training with epoch-level raw prediction diagnostics."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,csv,hashlib,json,math,socket,sys,time,tarfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
import torch
from model.Transformer import SpatioTemporalTransformer
from windows import WindowSet,weighted_mse

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(str(temp),str(p))
def require(ok,message):
    if not ok:raise ValueError(message)
def setup(seed,device):
    torch.set_num_threads(4);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device=='cuda':
        require(torch.cuda.is_available(),'CUDA unavailable; no CPU fallback')
        print('CUDA allocation probe',flush=True);torch.empty(1,device=device);torch.cuda.synchronize();torch.cuda.manual_seed_all(seed)
def new_model():
    return SpatioTemporalTransformer(num_frame=124,num_joints=4,in_chans=2,joint_embed_dim=12,num_heads=2,drop_rate=0.)
def split(d,f,j):
    require(f in range(5) and (j is None or (j in range(5) and f!=j)),'Invalid folds')
    te=d['fold']==f;va=np.zeros(len(te),dtype=bool) if j is None else d['fold']==j;tr=~(te|va)
    groups=[set(d['person_id'][m]) for m in (tr,va,te)]
    require(not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]),'Person overlap')
    return tr,va,te
def data():
    for name,digest in load(ROOT/'manifest.json').items():require(sha(ROOT/name)==digest,'Bundle changed: '+name)
    with np.load(ROOT/'data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    d['x']=WindowSet(d['x'],d.pop('window_record'),len(d['y']),d.pop('window_start'),d.pop('canonical_fallback'))
    d['x'].validate()
    require(len(d['x'].values)==11261 and int(d['x'].fallback.sum())==4,'Window count/fallback mismatch')
    for record in range(len(d['y'])):
        starts=d['x'].starts[d['x'].owner==record];fallback=d['x'].fallback[d['x'].owner==record]
        require(len(np.unique(starts))==len(starts),'Duplicate windows')
        require(np.isin(starts[~fallback],np.arange(0,500-124,31)).all(),'Invalid window start')
        if fallback.any():require(len(starts)==1,'Fallback record changed')
    require(d['y'].shape==(1061,2) and np.isfinite(d['x'].values).all() and np.isfinite(d['y']).all(),'Invalid data')
    require(len(np.unique(d['person_id']))==692 and len(np.unique(d['video_id']))==1061,'Wrong cohort')
    require(len(np.unique(d['original_row_index']))==1061,'Duplicate row')
    for f in range(5):
        require(len(np.unique(d['person_id'][d['fold']==f]))==[138,138,138,139,139][f],'Wrong fold sizes')
        for j in range(5):
            if f!=j:split(d,f,j)
    return d
def aligned_mse(pred,target):
    require(pred.ndim==target.ndim==1 and pred.shape==target.shape,'Loss axes must be aligned 1D')
    return (pred-target).square().mean()
def predict(model,x,lo,span,device):
    model.eval();out=[]
    with torch.inference_mode():
        for k in range(0,len(x.values),64):
            value=torch.from_numpy(x.values[k:k+64].reshape(-1,124,4,2)).to(device)
            out.append((model(value)*span+lo).reshape(-1,2).cpu().numpy())
    return x.aggregate(np.concatenate(out))
def calibration(p,y):
    return np.linalg.lstsq(np.column_stack([p.reshape(-1).astype(float),np.ones(p.size)]),y.reshape(-1),rcond=None)[0]
def corrected(raw,coef):return raw.astype(np.float64)*coef[0]+coef[1]
def metric(y,p,ids):return float(np.mean([np.abs(y[ids==q]-p[ids==q]).mean() for q in np.unique(ids)]))
def digest_array(a):
    if isinstance(a,WindowSet):
        fields=[digest_array(v) for v in (a.values,a.owner,a.starts,a.fallback)]
        return hashlib.sha256(json.dumps([a.nrecords,fields]).encode()).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fit(x,y,ids,vx,vy,vids,c,epochs,out,identity,device,stop_after=None):
    require(vx is not None or (vy is None and vids is None),'Partial validation input')
    if vx is not None:require(not(set(ids)&set(vids)),'Training/validation people overlap')
    x.validate();require(len(x)==len(y) and y.shape==(len(ids),2),'Training shape')
    if vx is not None:
        vx.validate();require(len(vx)==len(vy) and vy.shape==(len(vids),2),'Validation shape')
    identity=dict(identity,train_x_sha=digest_array(x),train_y_sha=digest_array(y),train_ids_sha=digest_array(ids),
        validation_x_sha=None if vx is None else digest_array(vx),validation_y_sha=None if vy is None else digest_array(vy),validation_ids_sha=None if vids is None else digest_array(vids))
    setup(identity['seed'],device);out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists():require(load(out/'identity.json')==identity,'Fit identity changed')
    else:save(out/'identity.json',identity)
    model=new_model().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=40,T_mult=1,eta_min=c['eta_min'])
    lo=float(np.float32(y.min()));span=float(np.float32(max(float(y.max()-y.min()),1e-6)))
    xt=x.values.reshape(-1,124,4,2);yt=(y[x.owner].astype(np.float32).reshape(-1)-lo)/span
    weights=x.side_weights()
    history=[];path=out/'latest.pt'
    if path.exists():
        ck=torch.load(path,map_location=device,weights_only=True)
        require(ck['identity']==identity,'Checkpoint identity changed')
        require(ck['lo']==lo and ck['span']==span,'Training scale changed')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        history=ck['history'];require(len(history)<=epochs,'Checkpoint exceeds epochs')
        torch.set_rng_state(ck['rng_cpu'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng_cuda']])
    for epoch in range(len(history),epochs):
        started=time.monotonic();model.train();loss_sum=0.;lr=float(opt.param_groups[0]['lr'])
        order=np.random.default_rng(identity['seed']+epoch).permutation(len(yt))
        for start in range(0,len(order),c['batch_size']):
            ix=order[start:start+c['batch_size']];opt.zero_grad(set_to_none=True)
            pred=model(torch.from_numpy(xt[ix]).to(device));target=torch.from_numpy(yt[ix]).to(device)
            loss=weighted_mse(pred,target,torch.from_numpy(weights[ix]).to(device));require(bool(torch.isfinite(loss)),'Nonfinite loss');loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Invalid gradient')
            opt.step();require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Invalid weight')
            loss_sum+=float(loss.detach())*len(ix)
        scheduler.step()
        raw=predict(model,x,lo,span,device);coef=calibration(raw,y)
        valraw=np.empty((0,2),dtype=np.float32) if vx is None else predict(model,vx,lo,span,device)
        rawdir=out/'raw_predictions';rawdir.mkdir(exist_ok=True)
        rawfile=rawdir/('epoch%04d.npz'%(epoch+1))
        with rawfile.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,train_raw=raw,validation_raw=valraw)
        os.replace(str(rawfile.with_suffix('.tmp')),str(rawfile))
        row=dict(epoch=epoch+1,lr=lr,loss=loss_sum/len(yt),calibration=coef.tolist(),
            train_person_mae=metric(y,corrected(raw,coef),ids),
            train_raw_person_mae=metric(y,raw,ids),calibration_slope=float(coef[0]),calibration_intercept=float(coef[1]),raw_predictions_sha256=sha(rawfile),
            seconds=time.monotonic()-started)
        if vx is not None:row.update(validation_raw_person_mae=metric(vy,valraw,vids),validation_person_mae=metric(vy,corrected(valraw,coef),vids),validation_people=len(np.unique(vids)))
        history.append(row)
        ck=dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
            lo=lo,span=span,history=history,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        tmp=out/'latest.tmp';torch.save(ck,tmp);os.replace(str(tmp),str(path));save(out/'curve.json',history)
        print(json.dumps(dict(fit=out.as_posix(),**row)),flush=True)
        if stop_after is not None and len(history)>=stop_after:break
    save(out/'curve.json',history)
    if len(history)==epochs:save(out/'fit_complete.json',dict(identity=identity,epochs=epochs,optimizer_steps=epochs*math.ceil(len(yt)/c['batch_size']),checkpoint_sha256=sha(path),curve_sha256=sha(out/'curve.json')))
    return model,lo,span,history
def identity_for(d,c,f,j,epochs,purpose):
    tr,va,te=split(d,f,j)
    return dict(bundle_sha256=sha(ROOT/'manifest.json'),purpose=purpose,outer_fold=f,inner_fold=j,epochs=epochs,
        seed=c['seed']+100*f+(50 if j is None else j),
        train_people=np.unique(d['person_id'][tr]).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),
        outer_evaluation_people=np.unique(d['person_id'][te]).tolist())

def select_epoch(histories):
    require(len(histories)==4 and len(set(map(len,histories)))==1 and len(histories[0])>0,'Incomplete inner histories')
    values=np.array([[r['validation_person_mae'] for r in h] for h in histories])
    require(np.isfinite(values).all(),'Nonfinite validation scores')
    means=np.average(values,axis=0,weights=[h[0]['validation_people'] for h in histories])
    return int(np.argmin(means))+1,means.tolist()

def train_fold(d,c,f,out,device):
    dest=out/('fold%d'%f);histories=[]
    for j in range(5):
        if j==f:continue
        tr,va,_=split(d,f,j)
        _,_,_,hist=fit(d['x'][tr],d['y'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],
            c,c['max_epochs'],dest/('inner%d'%j),identity_for(d,c,f,j,c['max_epochs'],'inner_epoch_selection'),device)
        histories.append(hist)
    epoch,means=select_epoch(histories)
    save(dest/'selection.json',dict(selected_epoch=epoch,inner_mean_person_mae_by_epoch=means,outer_evaluation_used_for_selection=False))
    tr,_,te=split(d,f,None)
    m,lo,span,h=fit(d['x'][tr],d['y'][tr],d['person_id'][tr],None,None,None,c,epoch,dest/'refit',
        identity_for(d,c,f,None,epoch,'outer_refit'),device)
    coef=h[-1]['calibration'];raw=predict(m,d['x'][te],lo,span,device)
    pred=raw.astype(np.float64)*coef[0]+coef[1]
    np.savez_compressed(dest/'predictions.npz',predictions=pred[:,None,:],predictions_raw=raw[:,None,:],calibration=coef,
        row_index=np.flatnonzero(te),**{k:d[k][te] for k in ('y','person_id','video_id','fold','cohort','original_row_index')})
    save(dest/'complete.json',dict(identity=load(dest/'refit/identity.json'),selected_epoch=epoch,
        prediction_sha256=sha(dest/'predictions.npz'),checkpoint_sha256=sha(dest/'refit/latest.pt'),selection_sha256=sha(dest/'selection.json'),
        evaluation_person_mae=metric(d['y'][te],pred,d['person_id'][te]),raw_evaluation_person_mae=metric(d['y'][te],raw,d['person_id'][te]),
        calibration=coef,device=device,torch=str(torch.__version__)))
    print(json.dumps(dict(complete=str(dest),selected_epoch=epoch)),flush=True)

def metrics(y,p,ids):
    require(y.shape==p.shape==(len(ids),2) and np.isfinite(p).all(),'Metric axes/nonfinite')
    def macro(v):return float(np.mean([v[ids==q].mean() for q in np.unique(ids)]))
    bins=np.abs(np.floor(y/10)-np.floor(p/10))
    return dict(people=len(np.unique(ids)),records=len(ids),mae=metric(y,p,ids),bias=macro(p-y),
        exact_10point_bin_rate=macro(bins==0),two_or_more_bins_rate=macro(bins>=2),prediction_min=float(p.min()),prediction_max=float(p.max()))

def audit_fit(d,c,f,j,epochs,folder,purpose):
    tr,va,_=split(d,f,j);ident=identity_for(d,c,f,j,epochs,purpose)
    ident.update(train_x_sha=digest_array(d['x'][tr]),train_y_sha=digest_array(d['y'][tr]),train_ids_sha=digest_array(d['person_id'][tr]),
        validation_x_sha=None if j is None else digest_array(d['x'][va]),validation_y_sha=None if j is None else digest_array(d['y'][va]),
        validation_ids_sha=None if j is None else digest_array(d['person_id'][va]))
    complete=load(folder/'fit_complete.json');h=load(folder/'curve.json')
    require(ident==complete['identity']==load(folder/'identity.json'),'Fit identity mismatch')
    require(len(h)==complete['epochs']==epochs and [row['epoch'] for row in h]==list(range(1,epochs+1)),'Incomplete epochs')
    require(sha(folder/'latest.pt')==complete['checkpoint_sha256'] and sha(folder/'curve.json')==complete['curve_sha256'],'Fit hash mismatch')
    require(complete['optimizer_steps']==epochs*math.ceil(2*len(d['x'][tr].values)/c['batch_size']),'Optimizer steps mismatch')
    # Recompute every epoch diagnostic from stored train/validation raw predictions.
    for row in h:
        path=folder/'raw_predictions'/('epoch%04d.npz'%row['epoch'])
        require(sha(path)==row['raw_predictions_sha256'],'Raw prediction hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            raw=z['train_raw'];vr=z['validation_raw']
            require(raw.shape==(int(tr.sum()),2) and np.isfinite(raw).all(),'Raw train shape')
            coef=calibration(raw,d['y'][tr]);np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)
            require(coef[0]==row['calibration_slope'] and coef[1]==row['calibration_intercept'],'Coefficient mismatch')
            require(abs(metric(d['y'][tr],raw,d['person_id'][tr])-row['train_raw_person_mae'])<1e-10,'Train raw MAE mismatch')
            require(abs(metric(d['y'][tr],corrected(raw,coef),d['person_id'][tr])-row['train_person_mae'])<1e-10,'Train MAE mismatch')
            if j is None:require(vr.shape==(0,2) and 'validation_person_mae' not in row,'Outer refit read validation')
            else:
                require(vr.shape==(int(va.sum()),2) and row['validation_people']==len(np.unique(d['person_id'][va])),'Validation shape/count')
                require(abs(metric(d['y'][va],vr,d['person_id'][va])-row['validation_raw_person_mae'])<1e-10,'Validation raw MAE mismatch')
                require(abs(metric(d['y'][va],corrected(vr,coef),d['person_id'][va])-row['validation_person_mae'])<1e-10,'Validation MAE mismatch')
    return h,complete['optimizer_steps']

def summarize(d,c,out):
    pred=np.full_like(d['y'],np.nan,dtype=np.float64);raw=np.full_like(pred,np.nan);seen=np.zeros(len(pred),int)
    curves=[];steps=0
    for f in range(5):
        dest=out/('fold%d'%f);done=load(dest/'complete.json');histories=[]
        for j in range(5):
            if j==f:continue
            h,n=audit_fit(d,c,f,j,c['max_epochs'],dest/('inner%d'%j),'inner_epoch_selection');steps+=n;histories.append(h)
            curves.extend(dict(outer_fold=f,fit='inner%d'%j,**row) for row in h)
        chosen,means=select_epoch(histories);selection=load(dest/'selection.json')
        require(chosen==done['selected_epoch']==selection['selected_epoch'] and not selection['outer_evaluation_used_for_selection'],'Wrong selected epoch')
        np.testing.assert_array_equal(means,selection['inner_mean_person_mae_by_epoch'])
        h,n=audit_fit(d,c,f,None,chosen,dest/'refit','outer_refit');steps+=n
        curves.extend(dict(outer_fold=f,fit='refit',**row) for row in h)
        require(done['identity']==load(dest/'refit/identity.json'),'Outer identity mismatch')
        require(sha(dest/'refit/latest.pt')==done['checkpoint_sha256'],'Outer checkpoint mismatch')
        require(sha(dest/'selection.json')==done['selection_sha256'] and sha(dest/'predictions.npz')==done['prediction_sha256'],'Outer hash mismatch')
        with np.load(dest/'predictions.npz',allow_pickle=False) as z:
            ix=np.flatnonzero(d['fold']==f);np.testing.assert_array_equal(z['row_index'],ix)
            for k in ('y','person_id','video_id','fold','cohort','original_row_index'):np.testing.assert_array_equal(z[k],d[k][ix])
            require(z['predictions'].shape==z['predictions_raw'].shape==(len(ix),1,2),'Prediction axes')
            pred[ix]=z['predictions'][:,0];raw[ix]=z['predictions_raw'][:,0];seen[ix]+=1
            np.testing.assert_array_equal(z['calibration'],h[-1]['calibration'])
            np.testing.assert_allclose(pred[ix],raw[ix]*z['calibration'][0]+z['calibration'][1],rtol=0,atol=1e-10)
            require(abs(metric(d['y'][ix],pred[ix],d['person_id'][ix])-done['evaluation_person_mae'])<1e-10,'Outer MAE mismatch')
    require((seen==1).all() and np.isfinite(pred).all(),'Incomplete OOF')
    np.savez_compressed(out/'oof_predictions.npz',predictions=pred[:,None,:],predictions_raw=raw[:,None,:],**{k:v for k,v in d.items() if k!='x'})
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.writer(stream);w.writerow(['person_id','video_id','fold','side','actual_gdi','raw_prediction','corrected_prediction'])
        for i in range(len(pred)):
            for side in range(2):w.writerow([d['person_id'][i],d['video_id'][i],int(d['fold'][i]),['left','right'][side],d['y'][i,side],raw[i,side],pred[i,side]])
    # Outer predictions are deliberately absent from epoch diagnostics.
    keys=['outer_fold','fit','epoch','lr','loss','train_raw_person_mae','train_person_mae','validation_raw_person_mae',
        'validation_person_mae','calibration_slope','calibration_intercept','seconds','raw_predictions_sha256']
    with (out/'diagnostic_curves.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(curves)
    result=dict(method=c['name'],metrics=metrics(d['y'],pred,d['person_id']),raw_diagnostic_metrics=metrics(d['y'],raw,d['person_id']),
        fold_metrics=[dict(fold=f,**metrics(d['y'][d['fold']==f],pred[d['fold']==f],d['person_id'][d['fold']==f])) for f in range(5)],
        optimizer_steps=steps,fits=25,bundle_sha256=sha(ROOT/'manifest.json'),oof_sha256=sha(out/'oof_predictions.npz'),
        windows=len(d['x'].values),record_balanced_loss=True,primary_output='mean_windows_then_training_only_OLS_corrected; raw output diagnostic',independent_final_evaluation=False,
        diagnostic_rows=len(curves),all_epoch_diagnostics_recomputed=True)
    save(out/'summary.json',result);print(json.dumps(result),flush=True)

def export_review(out):
    with tarfile.open(out.parent/(out.name+'-review.tar.gz'),'w:gz') as tf:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.csv','.npz'):tf.add(str(p),arcname='results/'+p.relative_to(out).as_posix())
        for name in ('manifest.json','protocol.json','run.py','verify.py','verify_nested.py','sources.json','windows.py','worker.py','submit.slurm'):
            tf.add(str(ROOT/name),arcname='bundle/'+name)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['check','pilot','train','summarize'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--folds',type=int,nargs='+',choices=range(5))
    a=parser.parse_args();c=load(ROOT/'protocol.json');setup(c['seed'],a.device);d=data();a.output.mkdir(parents=True,exist_ok=True)
    env=dict(python=sys.version,torch=str(torch.__version__),device=a.device,hostname=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None)
    save(a.output/('environment-%s-%d.json'%(os.getenv('SLURM_JOB_ID','local'),time.time_ns())),env)
    if a.command in ('train','summarize'):
        if a.command=='train':
            require(a.device=='cuda','Real full training requires CUDA')
            require(a.folds is None or len(set(a.folds))==len(a.folds),'Duplicate worker folds')
            for outer in (range(5) if a.folds is None else a.folds):train_fold(d,c,outer,a.output,a.device)
            if a.folds is not None:
                print('STT_WORKER_FOLDS_FINISHED',flush=True);return
        summarize(d,c,a.output)
        export_review(a.output)
        print('STT_TRAINING_AND_SUMMARY_FINISHED',flush=True);return
    if a.command=='check':
        m=new_model().to(a.device).eval()
        with torch.inference_mode():p=m(torch.from_numpy(d['x'].values[:2].reshape(-1,124,4,2)).to(a.device))
        require(p.shape==(4,) and bool(torch.isfinite(p).all()),'Real input forward failed')
        result=dict(passed=True,people=692,records=1061,windows=len(d['x'].values),input_shape=list(d['x'].values.shape),parameters=sum(p.numel() for p in m.parameters()),all20_inner_splits_checked=True,real_optimizer_steps=0)
        save(a.output/'check.json',result);print(json.dumps(result));return
    require(a.device=='cuda','Real pilot requires explicit CUDA; CPU verification uses synthetic data only')
    tr,va,te=split(d,0,1);torch.cuda.reset_peak_memory_stats()
    ident=dict(bundle_sha256=sha(ROOT/'manifest.json'),seed=c['seed']+1,epochs=c['pilot_epochs'],outer_fold=0,inner_fold=1,
        train_people=np.unique(d['person_id'][tr]).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),outer_evaluation_people=np.unique(d['person_id'][te]).tolist())
    model,lo,span,h=fit(d['x'][tr],d['y'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],c,3,a.output/'fit',ident,a.device)
    raw=predict(model,d['x'][va],lo,span,a.device);coef=h[-1]['calibration'];pred=corrected(raw,coef)
    np.savez_compressed(a.output/'validation_predictions.npz',row_index=np.flatnonzero(va),predictions=pred,raw=raw,y=d['y'][va],person_id=d['person_id'][va])
    result=dict(passed=True,pilot_only=True,epochs=3,train_people=int(len(set(d['person_id'][tr]))),validation_people=int(len(set(d['person_id'][va]))),
        outer_evaluation_used=False,optimizer_steps=3*math.ceil(2*len(d['x'][tr].values)/c['batch_size']),validation_person_mae=metric(d['y'][va],pred,d['person_id'][va]),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),epoch_seconds=[v['seconds'] for v in h],parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(a.output/'fit/latest.pt'))
    save(a.output/'pilot_summary.json',result);print(json.dumps(result));print('STT_PILOT_FINISHED')

if __name__=='__main__':main()
