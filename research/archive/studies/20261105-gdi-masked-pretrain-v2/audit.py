"""Independent checkpoint replay, calibration and budget audit of completed fits."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];B=HERE/'bundle'
sys.path[:0]=[str(B),str(ROOT/'outputs/motionbert-smoke-20261015/deps')]
import experiment as e
import torch
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','main'],required=True);ap.add_argument('--family',choices=['CNN','STT'],required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--conditions',default='A,B_adult,B_child,C,D_adult,D_child');ap.add_argument('--folds',default='0');a=ap.parse_args()
p=e.read(B/'protocol.json');d=e.load_data(Path('D:/gait-research-data/D_P/gdi-masked-pretrain-v2-20261105/data.npz'),p);e.setup(123)
audits=[]
for f in map(int,a.folds.split(',')):
 for condition in a.conditions.split(','):
    folder=a.root/a.stage/a.family/condition/('fold%d'%f);done=e.read(folder/'complete.json');ident=e.read(folder/'identity.json')
    assert done['identity']==ident and ident['manifest_sha256']==e.sha(B/'manifest.json')
    for name,key in [('model.pt','checkpoint_sha256'),('predictions.npz','prediction_sha256'),('curve.json','curve_sha256'),('model-replay.npz','replay_sha256')]:assert e.sha(folder/name)==done[key]
    train=(d['fold']>=0)&(d['fold']!=f);evaluation=d['fold']==f
    if a.stage=='pilot':train&=d['fold']!=(f+1)%5;evaluation=d['fold']==(f+1)%5
    tr=np.flatnonzero(train);ev=np.flatnonzero(evaluation)
    assert ident['train_people']==np.unique(d['person_id'][tr]).tolist() and ident['evaluation_people']==np.unique(d['person_id'][ev]).tolist()
    assert set(ident['train_people']).isdisjoint(ident['evaluation_people'])
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);assert ck['identity']==ident and ck['history']==e.read(folder/'curve.json')
    epochs=2 if a.stage=='pilot' else p['gdi_epochs'][a.family][str(f)]
    examples=2*int(np.isin(d['window_record'],tr).sum());batch=e.read(B/(a.family+'-original.json'))['batch_size']
    assert len(ck['history'])==epochs and done['epochs']==epochs
    assert done['examples']==epochs*examples and done['optimizer_steps']==epochs*math.ceil(examples/batch)
    assert ck['lo']==float(np.float32(d['y'][tr].min())) and ck['span']==float(np.float32(d['y'][tr].max()-d['y'][tr].min()))
    e.setup(ident['seed']);model=e.newmodel(a.family);keys=e.backbone_keys(model,a.family,p);initial=model.state_dict()
    head=hashlib.sha256(b''.join(initial[k].numpy().tobytes() for k in sorted(initial) if k not in keys)).hexdigest()
    initproof=e.read(HERE/'server-initialization-replay.json');assert initproof['manifest_sha256']==e.sha(B/'manifest.json')
    server_head=next(r['head_sha256'] for r in initproof['rows'] if r['family']==a.family and r['fold']==f and r['seed']==ident['seed'])
    assert server_head==done['head_initial_sha256']==ck['head_initial_sha256']
    local_initialization_matches=(head==server_head)
    model.load_state_dict(ck['model'],strict=True);model.eval()
    # Independent full window aggregation, matching family-specific batching/dtype.
    def replay(rows):
        wi=np.flatnonzero(np.isin(d['window_record'],rows));owners=d['window_record'][wi];values=[];bs=128 if a.family=='CNN' else 64
        with torch.no_grad():
            for k in range(0,len(wi),bs):
                x=torch.from_numpy(d['x'][wi[k:k+bs]].reshape(-1,124,12))
                raw=model(x.transpose(1,2))[:,0] if a.family=='CNN' else model(x[:,:,:8].reshape(-1,124,4,2))
                values.append((raw*ck['span']+ck['lo']).reshape(-1,2).numpy())
        values=np.concatenate(values)
        return np.array([values[owners==r].mean(axis=0,dtype=np.float64) for r in rows])
    with np.load(folder/'predictions.npz') as z:
        assert np.array_equal(z['train_rows'],tr) and np.array_equal(z['evaluation_rows'],ev)
        rt=replay(tr);rv=replay(ev)
        maxerr=max(float(np.max(np.abs(rt-z['train_raw']))),float(np.max(np.abs(rv-z['evaluation_raw']))));assert maxerr<.001,maxerr
        calibrated_error=float(np.max(np.abs(rv*z['calibration'][0]+z['calibration'][1]-z['prediction'])));assert calibrated_error<.001,calibrated_error
        # Closed-form covariance OLS, independent of training least-squares routine.
        x=z['train_raw'].ravel().astype(float);y=d['y'][tr].ravel();xc=x-x.mean();yc=y-y.mean()
        if np.dot(xc,xc)>1e-16:
            slope=np.dot(xc,yc)/np.dot(xc,xc);coef=np.array([slope,y.mean()-slope*x.mean()]);np.testing.assert_allclose(coef,z['calibration'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(z['evaluation_raw']*z['calibration'][0]+z['calibration'][1],z['prediction'],atol=1e-10,rtol=0)
    for row in ck['history']:
        assert row['examples']==examples and row['optimizer_steps']==math.ceil(examples/batch)
        rawpath=folder/'epoch-train'/('epoch%04d.npz'%row['epoch']);assert e.sha(rawpath)==row['train_raw_sha256']
        with np.load(rawpath) as z:
            assert np.array_equal(z['rows'],tr);coef=e.ols(z['raw'],d['y'][tr]);np.testing.assert_allclose(coef,row['calibration'],atol=1e-8,rtol=0)
        if not condition.startswith('B_'):assert row['external_examples']==0
    ssl_error=None
    if condition=='C' or condition.startswith('D_'):
        ps=e.read(folder/'pretrain-complete.json');assert ps['checkpoint_sha256']==e.sha(folder/'pretrain.pt')==done['pretrain_checkpoint_sha256']
        steps=40 if a.stage=='pilot' else p['pretrain_steps'];assert ps['steps']==steps and ps['side_examples']==steps*32*2
        pc=torch.load(folder/'pretrain.pt',map_location='cpu',weights_only=True);assert pc['identity']==ident and pc['steps']==steps
        with np.load(folder/'pretrain-plan.npz') as z:
            win=z['windows'];ext=z['external_flags'];source=d['window_record'][win]
            assert win.shape==(steps,32) and np.all(np.isin(source[~ext],tr))
            if condition=='C':assert not ext.any()
            elif condition=='D_adult':assert ((source[ext]>=1061)&(source[ext]<1073)).all()
            else:assert ((source[ext]>=1073)&(source[ext]<1085)).all()
            assert int(ext.sum())==ps['external_pair_examples']
        recon=e.Reconstruction(a.family,p);recon.load_state_dict(pc['model']);recon.eval()
        with np.load(folder/'pretrain-replay.npz') as z,torch.no_grad():
            target=recon(torch.from_numpy(d['x'][z['windows']].reshape(-1,124,12)),torch.from_numpy(z['mask'])).numpy()
            ssl_error=float(np.max(np.abs(target-z['reconstruction'])));assert ssl_error<.001,ssl_error
    row=dict(family=a.family,condition=condition,fold=f,passed=True,epochs=epochs,optimizer_steps=done['optimizer_steps'],examples=done['examples'],cpu_replay_max_error=maxerr,calibrated_cpu_replay_max_error=calibrated_error,pretrain_cpu_replay_max_error=ssl_error,server_seed_reconstruction_passed=True,local_cross_version_seed_sha_matches=local_initialization_matches,complete_sha256=e.sha(folder/'complete.json'))
    audits.append(row);print(json.dumps(row),flush=True)
e.save(HERE/'audits'/(a.stage+'-'+a.family+'-'+a.conditions.replace(',','_')+'-'+a.folds.replace(',','_')+'.json'),dict(passed=True,stage=a.stage,audits=audits))
