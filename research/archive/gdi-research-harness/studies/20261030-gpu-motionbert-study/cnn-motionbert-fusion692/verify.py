"""Preparation checks only: real CPU frozen inference, synthetic CNN optimization."""
from pathlib import Path
import contextlib,json,sys,time,shutil,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];B=HERE/'bundle';BASE=HERE.parent/'cnn-paper-input695'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
sys.path.insert(0,str(B))
import torch
import run as r
import features as f
from model import GdiCNN,channels,training_feature_stats

def reject(fn):
    try:fn()
    except (ValueError,FileNotFoundError):return
    raise AssertionError('Invalid case accepted')

def synthetic(n,rng):
    x=rng.normal(size=(n,2,124,524)).astype(np.float32)
    x[:,1,:,12:]=x[:,0,:,12:]
    return x,rng.uniform(40,100,(n,2))

def main():
    out=HERE/'checks'/('verification-'+str(time.time_ns()));out.mkdir(parents=True)
    r.setup(7,'cpu');d=r.data(with_features=False);c=r.config()
    with np.load(BASE/'bundle/data.npz',allow_pickle=False) as z:
        for k in d:np.testing.assert_array_equal(d[k],z[k])
    inputs=r.load(BASE/'input_sources.json');source_by_id={row['index']:row for row in inputs['records']}
    with np.load(B/'motion_input.npz',allow_pickle=False) as z:
        for row,i in enumerate(d['original_row_index']):
            source=source_by_id[int(i)]
            assert r.sha(ROOT/source['source_input'])==source['source_sha256']
            with np.load(ROOT/source['source_input'],allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw['frame_ids124'],z['frame_ids'][row])
        selected=np.array([0,len(d['y'])-1])
        small_motion={k:z[k][selected].copy() for k in z.files}
    reject(lambda:r.data()) # Training must fail until a full, identified cache exists.
    bad={k:v.copy() for k,v in d.items()};bad['person_id'][0]=bad['person_id'][np.flatnonzero(d['fold']!=d['fold'][0])[0]]
    reject(lambda:r.validate_data(bad,fused=False))
    print('Real input alignment, same frame IDs, all person folds, and missing-cache rejection passed.',flush=True)

    f.check(B,out/'feature_check','cpu')
    # Tiny actual-pose fixture tests resumable, unnormalized fixed features without GDI labels.
    fixture=out/'feature_fixture';fixture.mkdir()
    shutil.copy2(B/'MB_lite.bin',fixture/'MB_lite.bin')
    np.savez_compressed(fixture/'motion_input.npz',**small_motion)
    f.save(fixture/'manifest.json',{k:f.sha(fixture/k) for k in ('MB_lite.bin','motion_input.npz')})
    m=f.encoder(B,'cpu');before={k:v.detach().clone() for k,v in m.state_dict().items()}
    with (out/'feature_cache.log').open('w') as log,contextlib.redirect_stdout(log):
        f.extract(fixture,fixture/'resume','cpu',batch=1,limit_batches=1,model=m)
        assert not (fixture/'resume/complete.json').exists()
        f.extract(fixture,fixture/'resume','cpu',batch=1,model=m)
        f.extract(fixture,fixture/'full','cpu',batch=1,model=m)
    cached,cache_sha=f.read_cache(fixture,fixture/'resume',2);continuous,_=f.read_cache(fixture,fixture/'full',2)
    np.testing.assert_array_equal(cached,continuous)
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    expected_temporal=np.array(cached)
    del cached,continuous,m,before
    broken=fixture/'resume/features.npy';arr=np.load(broken,mmap_mode='r+');arr[0,0,0]+=1;arr.flush();del arr
    reject(lambda:f.read_cache(fixture,fixture/'resume',2))
    print('Real frozen feature extraction and cache resume exact; corrupt cache rejected.',flush=True)

    fused=np.concatenate([d['x'][selected],np.broadcast_to(expected_temporal[:,None],(2,2,124,512))],axis=3)
    assert fused.shape==(2,2,124,524)
    np.testing.assert_array_equal(fused[:,:,:,:12],d['x'][selected])
    np.testing.assert_array_equal(fused[:,0,:,12:],fused[:,1,:,12:])
    mean,scale=training_feature_stats(fused[:1])
    np.testing.assert_allclose(mean.numpy(),expected_temporal[:1].astype(np.float64).mean((0,1)),rtol=1e-6,atol=1e-7)
    assert sum(p.numel() for p in GdiCNN().parameters())==178912
    layout=torch.arange(2*2*124*524).reshape(2,2,124,524).float()
    assert channels(layout)[1,1,400,37]==layout[1,1,37,400]
    reject(lambda:channels(torch.zeros(3,2,124,12)))
    r.smoke(c,out/'smoke','cpu')

    rng=np.random.default_rng(5);sx,sy=synthetic(12,rng);pid=np.array(['s%d'%i for i in range(12)])
    ident={'fit_seed':321,'purpose':'synthetic_only','epochs':2}
    with (out/'resume.log').open('w') as log,contextlib.redirect_stdout(log):
        full,*_=r.fit(sx,sy,pid,sx[:4],sy[:4],pid[:4],c,2,out/'full',ident,'cpu')
        r.fit(sx,sy,pid,sx[:4],sy[:4],pid[:4],c,1,out/'resume',ident,'cpu')
        resumed,*_=r.fit(sx,sy,pid,sx[:4],sy[:4],pid[:4],c,2,out/'resume',ident,'cpu')
        vx=sx[:4].copy();vx[:,:,:,12:]=777
        poisoned,*_=r.fit(sx,sy,pid,vx,np.full((4,2),9999.),pid[:4],c,2,out/'held_changed',ident,'cpu')
        for k,v in full.state_dict().items():assert torch.equal(v,resumed.state_dict()[k]) and torch.equal(v,poisoned.state_dict()[k])
        a=r.load(out/'full/curve.json');b=r.load(out/'held_changed/curve.json')
        assert [v['calibration'] for v in a]==[v['calibration'] for v in b]
        mean,scale=training_feature_stats(sx)
        assert torch.equal(full.feature_mean,mean) and torch.equal(full.feature_scale,scale)
        changed=sx.copy();changed[:,:,:,12:]+=1
        reject(lambda:r.fit(changed,sy,pid,None,None,None,c,2,out/'resume',ident,'cpu'))
    full.eval()
    original=r.predict(full,channels(torch.from_numpy(sx)),torch.tensor(0.),torch.tensor(1.),'cpu')
    perturbed=sx.copy();perturbed[:,:,:,12:]=0
    another=r.predict(full,channels(torch.from_numpy(perturbed)),torch.tensor(0.),torch.tensor(1.),'cpu')
    assert not np.array_equal(original,another),'Model ignores the MotionBERT channels'
    print('Fusion path used; train-only feature statistics, held-input/label poisoning, epoch resume passed.',flush=True)

    n=30;xx,yy=synthetic(n,rng)
    sd={'x':xx,'y':yy,'person_id':np.array(['person%02d'%i for i in range(n)]),
        'video_id':np.array(['video%02d'%i for i in range(n)]),'fold':np.arange(n)%5,
        'cohort':np.arange(n)%2,'original_row_index':np.arange(n),'_feature_cache_sha256':'synthetic-only'}
    sc=dict(c,max_epochs=1);dest=out/'synthetic_nested'
    reject(lambda:r.summarize(sd,sc,dest))
    for fold in range(5):
        with (out/'nested.log').open('a') as log,contextlib.redirect_stdout(log):r.train_fold(sd,sc,fold,dest,'cpu',None)
        print('Synthetic nested fold %d/5 passed'%(fold+1),flush=True)
    with (out/'nested.log').open('a') as log,contextlib.redirect_stdout(log):r.summarize(sd,sc,dest)
    with np.load(dest/'oof_predictions.npz',allow_pickle=False) as z:pred=z['predictions'][:,0];raw=z['predictions_raw'][:,0]
    independent=np.mean([np.abs(pred[sd['person_id']==who]-yy[sd['person_id']==who]).mean() for who in set(sd['person_id'])])
    assert abs(independent-r.load(dest/'summary.json')['metrics'][0]['mae'])<1e-10
    for fold in range(5):
        ck=torch.load(dest/('fold%d/refit/latest.pt'%fold),map_location='cpu',weights_only=True);model=GdiCNN();model.load_state_dict(ck['model'])
        tr,_,te=r.split(sd,fold);pp=r.predict(model,channels(torch.from_numpy(xx[te])),ck['lo'],ck['span'],'cpu')
        np.testing.assert_array_equal(pp,raw[te])
        train_pred=r.predict(model,channels(torch.from_numpy(xx[tr])),ck['lo'],ck['span'],'cpu');coef=r.calibrate(train_pred,yy[tr])
        np.testing.assert_array_equal(r.corrected(pp,coef),pred[te])
    changed=dict(sd,_feature_cache_sha256='different-cache')
    reject(lambda:r.summarize(changed,sc,dest))
    fp=dest/'fold0/predictions.npz';old=fp.read_bytes();fp.write_bytes(old+b'corrupt');reject(lambda:r.summarize(sd,sc,dest));fp.write_bytes(old)
    proof=dict(passed=True,people=692,records=1061,same124frames_verified=True,matching_labels_and_folds=True,
        actual_CPU_feature_check=r.load(out/'feature_check/feature_check.json'),frozen_feature_cache_resume_exact=True,
        corrupt_feature_cache_rejected=True,model_uses_added_features=True,train_only_normalization_and_calibration=True,
        held_inputs_and_labels_poison_test=True,CNN_resume_exact=True,synthetic_nested_fits=25,
        independent_checkpoint_reinference=True,synthetic_optimizer_steps=r.load(dest/'summary.json')['optimizer_steps'],
        actual_GDI_optimizer_steps=0,actual_GPU_execution=False,full_real_temporal_cache_created=False,
        trainable_parameters=178912,bundle_sha256=r.sha(B/'manifest.json'),details=out.relative_to(HERE).as_posix())
    r.save(HERE/'verification.json',proof);print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
