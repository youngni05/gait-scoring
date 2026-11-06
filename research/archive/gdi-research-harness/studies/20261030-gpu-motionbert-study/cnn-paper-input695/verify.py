"""Real input audit + synthetic training only. No actual GDI CNN fitting locally."""
from pathlib import Path
import contextlib,json,sys,time,zipfile,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
sys.path.insert(0,str(HERE/'bundle'))
import torch
import run as r
from model import GdiCNN,channels
from prepare_inputs import source_functions

def reject(fn):
    try:fn()
    except (ValueError,FileNotFoundError):return
    raise AssertionError('Invalid case accepted')

def main():
    out=HERE/'checks'/('verification-'+str(time.time_ns()));out.mkdir(parents=True)
    r.setup(7,'cpu');d=r.data();c=r.config()
    sources=json.loads((HERE/'input_sources.json').read_text())
    byindex={row['index']:row for row in sources['records']}
    for i,index in enumerate(d['original_row_index']):
        p=HERE/'records'/('%04d.npz'%index);proof=r.load(p.with_suffix('.json'));row=byindex[int(index)]
        assert r.sha(p)==proof['artifact_sha256']
        assert r.sha(ROOT/row['source_input'])==row['source_sha256']==proof['source_sha256']
        with np.load(p,allow_pickle=False) as z:
            assert np.array_equal(z['x'],d['x'][i]) and np.array_equal(z['frame_ids'],row['frames'])
    # Independent geometric identities on all retained 12-feature windows.
    x=d['x'].astype(np.float64)
    np.testing.assert_allclose(np.linalg.norm(x[...,6:8]-x[...,0:2],axis=-1),x[...,10],rtol=0,atol=5e-7)
    np.testing.assert_allclose(x[:,1,:,0]-x[:,0,:,0],x[:,0,:,11],rtol=0,atol=5e-7)
    np.testing.assert_allclose(x[:,0,:,11],-x[:,1,:,11],rtol=0,atol=1e-7)
    for a,b,center,target in [(0,4,2,8),(6,2,0,9)]:
        va=x[...,a:a+2]-x[...,center:center+2];vb=x[...,b:b+2]-x[...,center:center+2]
        cross=np.abs(va[...,0]*vb[...,1]-va[...,1]*vb[...,0]);dot=(va*vb).sum(-1)
        np.testing.assert_allclose(np.arctan2(cross,dot),x[...,target],rtol=0,atol=1e-3)
    ns=source_functions();a=np.ones((124,75))*10
    # Stationary symmetric legs give thigh length 2, ankle-knee-hip angle pi, toe-ankle-knee pi/2.
    for joint,xy in {12:(9,10),9:(11,10),13:(9,12),10:(11,12),14:(9,14),11:(11,14),19:(10,14),22:(12,14)}.items():
        a[:,3*joint:3*joint+2]=xy;a[:,3*joint+2]=1
    f=ns['process_video_and_add_cols'](a,swap_orientation=False)
    np.testing.assert_allclose(f[:,[50,51]],np.pi,rtol=0,atol=1e-7)
    np.testing.assert_allclose(f[:,[52,53]],np.pi/2,rtol=0,atol=1e-7)
    np.testing.assert_allclose(f[:,[54,55]],.5,rtol=0,atol=1e-7)
    assert ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](a),[14,11,13,10,12,9,19,22])==0
    missing=a.copy();missing[:32,57]=0
    assert ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](missing),[14,11,13,10,12,9,19,22])>0.25
    assert sum(p.numel() for p in GdiCNN().parameters())==47840
    layout=torch.arange(2*2*124*12).reshape(2,2,124,12).float()
    assert channels(layout)[1,1,8,37]==layout[1,1,37,8]
    reject(lambda:channels(torch.zeros(3,124,17,3)))
    bad=dict(d,person_id=d['person_id'].copy());i=np.flatnonzero(d['fold']==0)[0];j=np.flatnonzero(d['fold']==1)[0];bad['person_id'][i]=bad['person_id'][j]
    reject(lambda:r.split(bad,0,1))
    train=np.arange(8,dtype=np.float64).reshape(4,2)
    coef=r.calibrate(train,train*2+5)
    np.testing.assert_allclose(coef,[2,5],rtol=0,atol=1e-12)
    hs=[[{'validation_person_mae':v,'validation_people':n} for v in arr] for n,arr in [(1,[1,10]),(1,[1,10]),(1,[1,10]),(100,[10,1])]]
    assert r.select_epoch(hs)[0]==2
    print('Input provenance, all geometric identities, side layout, weighted selection and calibration: passed',flush=True)
    r.smoke(c,out/'smoke','cpu')
    rng=np.random.default_rng(5);sx=rng.normal(size=(32,2,124,12)).astype(np.float32);sy=rng.uniform(40,100,(32,2));pid=np.array(['s%d'%i for i in range(32)])
    ident={'fit_seed':321,'purpose':'synthetic_only','epochs':2}
    with (out/'resume.log').open('w') as log,contextlib.redirect_stdout(log):
        full,*_=r.fit(sx,sy,pid,sx[:8],sy[:8],pid[:8],c,2,out/'full',ident,'cpu')
        r.fit(sx,sy,pid,sx[:8],sy[:8],pid[:8],c,1,out/'resume',ident,'cpu')
        resumed,*_=r.fit(sx,sy,pid,sx[:8],sy[:8],pid[:8],c,2,out/'resume',ident,'cpu')
        poisoned,*_=r.fit(sx,sy,pid,sx[:8],np.full((8,2),9999.),pid[:8],c,2,out/'held_labels_changed',ident,'cpu')
        for k,v in full.state_dict().items():assert torch.equal(v,resumed.state_dict()[k]) and torch.equal(v,poisoned.state_dict()[k])
        a=r.load(out/'full/curve.json');b=r.load(out/'held_labels_changed/curve.json')
        assert [v['calibration'] for v in a]==[v['calibration'] for v in b]
        reject(lambda:r.fit(sx,sy,pid,None,None,None,c,2,out/'resume',dict(ident,fit_seed=99),'cpu'))
    print('Epoch resume exact; changing held labels does not change model or calibration',flush=True)
    sd={k:v for k,v in d.items() if k not in ('x','y')};sd['x']=rng.normal(size=d['x'].shape).astype(np.float32);sd['y']=rng.uniform(40,100,d['y'].shape)
    sc=dict(c,max_epochs=1);dest=out/'synthetic_nested'
    reject(lambda:r.summarize(sd,sc,dest))
    for f in range(5):
        with (out/'nested.log').open('a') as log,contextlib.redirect_stdout(log):r.train_fold(sd,sc,f,dest,'cpu',None)
        print('Synthetic fold %d/5 passed'%(f+1),flush=True)
    with (out/'nested.log').open('a') as log,contextlib.redirect_stdout(log):r.summarize(sd,sc,dest);r.export_review(dest)
    with np.load(dest/'oof_predictions.npz',allow_pickle=False) as z:
        p=z['predictions'][:,0];raw=z['predictions_raw'][:,0]
        independent=np.mean([np.abs(p[sd['person_id']==who]-sd['y'][sd['person_id']==who]).mean() for who in set(sd['person_id'])])
    assert abs(independent-r.load(dest/'summary.json')['metrics'][0]['mae'])<1e-10
    for f in range(5):
        ck=torch.load(dest/('fold%d/refit/latest.pt'%f),map_location='cpu',weights_only=True);m=GdiCNN();m.load_state_dict(ck['model'])
        tr,_,te=r.split(sd,f);pp=r.predict(m,channels(torch.from_numpy(sd['x'][te])),ck['lo'],ck['span'],'cpu')
        np.testing.assert_array_equal(pp,raw[te])
        train_pred=r.predict(m,channels(torch.from_numpy(sd['x'][tr])),ck['lo'],ck['span'],'cpu');cal=r.calibrate(train_pred,sd['y'][tr])
        np.testing.assert_array_equal(r.corrected(pp,cal),p[te])
    fp=dest/'fold0/predictions.npz';orig=fp.read_bytes();fp.write_bytes(orig+b'corruption');reject(lambda:r.summarize(sd,sc,dest));fp.write_bytes(orig)
    with zipfile.ZipFile(HERE/'gdi-cnn-paper-input.zip') as z:
        assert z.testzip() is None
        for name,digest in r.load(HERE/'bundle/manifest.json').items():assert hashlib.sha256(z.read('gdi-cnn-paper-input-20261030/'+name)).hexdigest()==digest
    proof={'passed':True,'people':692,'records':1061,'source_geometry_checks':True,'same124_frame_ids_verified':True,
           'training_only_calibration_label_poison_test':True,'side_order_and_weighted_epoch_selection':True,
           'checkpoint_resume_exact':True,'independent_raw_and_calibrated_reinference_on_synthetic':True,
           'synthetic_nested_fits':25,'synthetic_steps':r.load(dest/'summary.json')['optimizer_steps'],
           'actual_GDI_training_steps':0,'actual_GPU_execution':False,'torch':str(torch.__version__),
           'bundle_sha256':r.sha(HERE/'bundle/manifest.json'),'zip_verified':True,'details':out.relative_to(HERE).as_posix()}
    r.save(HERE/'verification.json',proof);print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
