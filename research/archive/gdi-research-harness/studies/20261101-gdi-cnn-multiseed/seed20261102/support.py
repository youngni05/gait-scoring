"""Checks, independent recorded-epoch audit, GPU worker and checkpoint replay."""
from pathlib import Path
import argparse,contextlib,datetime,hashlib,json,os,sys,time,traceback
import numpy as np
import torch
import run as r
H=Path(__file__).resolve().parent

def audit(folder,d,c,f,j,epochs,purpose,device='cpu'):
    tr,va,_=r.split(d,f,j);ident=r.identity_for(d,c,f,j,epochs,purpose)
    done=r.load(folder/'fit_complete.json');history=r.load(folder/'curve.json')
    assert done['identity']==ident==r.load(folder/'identity.json')
    assert r.sha(folder/'latest.pt')==done['checkpoint_sha256'] and r.sha(folder/'curve.json')==done['curve_sha256']
    ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
    assert ck['identity']==ident and ck['history']==history and len(history)==epochs==done['epochs']
    lo,span=r.scale_from_train(d['y'][tr]);assert torch.equal(lo,ck['lo'].cpu()) and torch.equal(span,ck['span'].cpu())
    x=d['x'][tr];count=2*len(x.values)
    assert done['optimizer_steps']==epochs*((count+c['batch_size']-1)//c['batch_size'])
    for k,row in enumerate(history):
        assert row['epoch']==k+1
        order=np.random.default_rng(ident['fit_seed']+k).permutation(count)
        assert hashlib.sha256(order.tobytes()).hexdigest()==row['sample_sha256'] and row['sample_count']==count
        path=folder/'raw_predictions'/('epoch%04d.npz'%(k+1));assert r.sha(path)==row['raw_predictions_sha256']
        with np.load(path) as z:
            coef=r.calibrate(z['train_raw'],d['y'][tr])
            np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)
            assert abs(r.person_mae(d['y'][tr],r.corrected(z['train_raw'],coef),d['person_id'][tr])-row['train_person_mae'])<1e-10
            if j is not None:
                assert abs(r.person_mae(d['y'][va],r.corrected(z['validation_raw'],coef),d['person_id'][va])-row['validation_person_mae'])<1e-10
                assert row['validation_people']==len(np.unique(d['person_id'][va]))
            else:assert 'validation_person_mae' not in row
        assert abs(row['lr']-c['learning_rate']*c['lr_drop_factor']**((k+1)//c['lr_drop_every']))<1e-14
    return ck,history

def replay(folder,d,c,f,j,epochs,purpose,device='cpu'):
    ck,h=audit(folder,d,c,f,j,epochs,purpose)
    tr,va,te=r.split(d,f,j);model=r.GdiCNN(c['dropout']).double().to(device);model.load_state_dict(ck['model']);model.eval()
    lo,span=ck['lo'],ck['span']
    def infer(x):
        parts=[]
        with torch.inference_mode():
            for start in range(0,len(x.values),128):
                t=r.channels(torch.from_numpy(x.values[start:start+128])).reshape(-1,12,124).double().to(device)
                p=model(t)[:,0]*float(span)+float(lo);parts.append(p.cpu().numpy().reshape(-1,2))
        return x.aggregate(np.concatenate(parts))
    raw_train=infer(d['x'][tr]);last=folder/'raw_predictions'/('epoch%04d.npz'%epochs)
    with np.load(last) as z:
        error=float(abs(raw_train-z['train_raw']).max())
        coef=np.array(h[-1]['calibration'])
        error=max(error,float(abs(r.corrected(raw_train,coef)-r.corrected(z['train_raw'],coef)).max()))
        if j is not None:
            valid=infer(d['x'][va]);error=max(error,float(abs(valid-z['validation_raw']).max()),float(abs(r.corrected(valid,coef)-r.corrected(z['validation_raw'],coef)).max()))
    coef=np.array(h[-1]['calibration']);pred=infer(d['x'][va if j is not None else te])
    if j is None:
        with np.load(folder.parent/'predictions.npz') as z:
            error=max(error,float(abs(pred-z['predictions_raw'][:,0]).max()),float(abs(r.corrected(pred,coef)-z['predictions'][:,0]).max()))
    assert error<1e-4,error
    return dict(passed=True,maximum_prediction_error=error,tolerance=1e-4,checkpoint_sha256=r.sha(folder/'latest.pt'))

def verify(out,device):
    out.mkdir(parents=True,exist_ok=True);r.setup(7,device);d=r.data();c=r.config()
    assert len(d['y'])==1061 and len(np.unique(d['person_id']))==692
    r.smoke(c,out/'smoke',device)
    rng=np.random.default_rng(12);n=15
    values=rng.normal(size=(n,2,124,12)).astype(np.float32)
    x=r.WindowSet(values,np.arange(n),n);y=rng.uniform(40,100,(n,2));people=np.array(['p%d'%i for i in range(n)])
    ident=dict(fit_seed=321,initialization_seed=654,epochs=2,purpose='synthetic')
    def fit(dest,epochs,yy=y[:5]):return r.fit(x,y,people,x[:5],yy,people[:5],c,epochs,out/dest,ident,device)[0]
    with (out/'synthetic.log').open('w') as log,contextlib.redirect_stdout(log):
        full=fit('full',2);fit('resume',1);resumed=fit('resume',2);poisoned=fit('poison',2,np.ones((5,2))*999)
        for k,v in full.state_dict().items():assert torch.equal(v,resumed.state_dict()[k]) and torch.equal(v,poisoned.state_dict()[k])
        a=r.load(out/'full/curve.json');b=r.load(out/'poison/curve.json')
        assert [v['calibration'] for v in a]==[v['calibration'] for v in b]
        # Model initialization differs while dropout state and minibatch order stay fixed.
        r.setup(321,device);m0=r.GdiCNN(c['dropout']).to(device);training_rng=r.rng_state()
        r.setup(654,device);m1=r.GdiCNN(c['dropout']).to(device);r.restore_rng(training_rng)
        assert any(not torch.equal(v,m1.state_dict()[k]) for k,v in m0.state_dict().items())
        after=r.rng_state();assert torch.equal(after['cpu'],training_rng['cpu'])
        assert all(torch.equal(a,b) for a,b in zip(after['cuda'],training_rng['cuda']))
        sd=dict(x=x,y=y,person_id=people,video_id=np.array(['v%d'%i for i in range(n)]),fold=np.arange(n)%5,cohort=np.arange(n)%2,original_row_index=np.arange(n))
        sc=dict(c,max_epochs=1)
        for f in range(5):r.train_fold(sd,sc,f,out/'nested',device,None)
        r.summarize(sd,sc,out/'nested')
    proof=dict(passed=True,device=device,real_people=692,real_records=1061,real_windows=11261,real_optimizer_steps=0,synthetic_nested_fits=25,resume_exact=True,held_label_poisoning_invariant=True,initialization_differs=True,training_rng_retained=True,baseline_batch_order_retained=True,manifest_sha256=r.sha(H/'manifest.json'))
    r.save(out/'verification.json',proof);print(json.dumps(proof),flush=True)

def worker():
    c=r.config();control=H/'control';control.mkdir(exist_ok=True);(control/'worker.lock').mkdir()
    state=dict(job_id=os.environ['SLURM_JOB_ID'],seed=c['initialization_seed'],phase='checking',started_at=datetime.datetime.now().isoformat())
    def update(**kw):state.update(kw);r.save(control/'worker-primary.json',state)
    try:
        update();assert torch.cuda.is_available() and torch.cuda.device_count()==1
        r.setup(7,'cuda');torch.ones(1,device='cuda').sum().item()
        import subprocess
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
        uuids={line.split(',')[1].strip() for line in rows if line.split(',')[0].strip()==str(os.getpid())};assert len(uuids)==1
        update(gpu_uuid=next(iter(uuids)),hostname=os.uname().nodename)
        verify(H/'checks/gpu','cuda');d=r.data();update(phase='pilot')
        tr,va,_=r.split(d,0,1);ident=r.identity_for(d,c,0,1,3,'pilot_not_performance')
        _,_,_,hist=r.fit(d['x'][tr],d['y'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],c,3,H/'runs/pilot/fit',ident,'cuda')
        proof=replay(H/'runs/pilot/fit',d,c,0,1,3,'pilot_not_performance','cuda')
        r.save(H/'runs/pilot/replay-gpu64.json',proof)
        proof=replay(H/'runs/pilot/fit',d,c,0,1,3,'pilot_not_performance','cpu');r.save(H/'runs/pilot/replay-cpu64.json',proof)
        r.save(H/'runs/pilot/summary.json',dict(pilot_only=True,epochs=3,train_people=len(np.unique(d['person_id'][tr])),validation_people=len(np.unique(d['person_id'][va])),outer_evaluation_used=False,mean_epoch_seconds=float(np.mean([z['seconds'] for z in hist])),checkpoint_sha256=proof['checkpoint_sha256'],manifest_sha256=r.sha(H/'manifest.json')))
        update(phase='waiting_local_gate');start=time.monotonic();gate=control/'approve-main.json'
        while not gate.exists():
            if time.monotonic()-start>3600:raise TimeoutError('Local pilot approval timeout; main did not start')
            time.sleep(5)
        g=r.load(gate);assert g['passed'] and g['manifest_sha256']==r.sha(H/'manifest.json') and g['checkpoint_sha256']==r.sha(H/'runs/pilot/fit/latest.pt')
        update(phase='main');deadline=time.monotonic()+c['wall_time_seconds']
        for f in range(5):
            update(active_fold=f);r.train_fold(d,c,f,H/'runs/main','cuda',deadline)
        r.summarize(d,c,H/'runs/main')
        r.save(H/'runs/summary_complete.json',dict(fits=25,outer_folds=5,summary_sha256=r.sha(H/'runs/main/summary.json'),manifest_sha256=r.sha(H/'manifest.json')))
        update(phase='training_complete',exit_code=0)
    except BaseException as exc:
        update(phase='failed',exit_code=1,error=repr(exc));traceback.print_exc();raise

def review(out):
    r.setup(7,'cpu');d=r.data();c=r.config();dest=H/'runs/main';allchecks=[];epochs_audited=0
    assert r.load(H/'runs/summary_complete.json')['fits']==25
    for f in range(5):
        histories=[]
        for j in range(5):
            if j==f:continue
            _,h=audit(dest/('fold%d/inner%d'%(f,j)),d,c,f,j,100,'inner_epoch_selection');histories.append(h);epochs_audited+=len(h)
        epoch,means=r.select_epoch(histories);sel=r.load(dest/('fold%d/selection.json'%f));assert epoch==sel['selected_epoch']
        np.testing.assert_allclose(means,sel['inner_mean_person_mae_by_epoch'],rtol=0,atol=1e-10)
        allchecks.append(replay(dest/('fold%d/refit'%f),d,c,f,None,epoch,'outer_refit'));epochs_audited+=epoch
    summary=r.load(dest/'summary.json');assert r.sha(dest/'summary.json')==r.load(H/'runs/summary_complete.json')['summary_sha256']
    assert summary['fits']==25 and r.sha(dest/'oof_predictions.npz')==summary['oof_sha256']
    with np.load(dest/'oof_predictions.npz') as z:
        for k in ['y','person_id','video_id','fold','cohort']:np.testing.assert_array_equal(z[k],d[k])
        assert abs(r.person_mae(d['y'],z['predictions'][:,0],d['person_id'])-summary['metrics'][0]['mae'])<1e-10
        for f in range(5):
            with np.load(dest/('fold%d/predictions.npz'%f)) as q:np.testing.assert_array_equal(z['predictions'][d['fold']==f],q['predictions'])
    r.save(out,dict(passed=True,fits=25,epochs_audited=epochs_audited,final_cpu64_replays=allchecks,manifest_sha256=r.sha(H/'manifest.json')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['verify','worker','review','pilot-review']);ap.add_argument('--out',type=Path);ap.add_argument('--device',default='cpu');a=ap.parse_args()
    if a.command=='worker':worker()
    elif a.command=='verify':verify(a.out,a.device)
    elif a.command=='review':review(a.out)
    else:
        r.setup(7,'cpu');r.save(a.out,replay(H/'runs/pilot/fit',r.data(),r.config(),0,1,3,'pilot_not_performance'))
