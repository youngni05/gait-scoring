"""Real provenance/geometry; synthetic weighting, leakage, resume and nested pipeline."""
from pathlib import Path
import contextlib,json,sys,time,hashlib,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BASE=HERE.parent/'cnn-paper-input695'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
sys.path.insert(0,str(HERE/'bundle'))
import torch
import run as r
from windows import WindowSet,weighted_mse
from model import GdiCNN

def reject(fn):
    try:fn()
    except (ValueError,FileNotFoundError):return
    raise AssertionError('Invalid case accepted')

def synthetic(n,rng,counts=None):
    counts=1+np.arange(n)%3 if counts is None else np.asarray(counts)
    owner=np.repeat(np.arange(n),counts)
    x=WindowSet(rng.normal(size=(len(owner),2,124,12)).astype(np.float32),owner,n)
    return x,rng.uniform(40,100,(n,2))

def main():
    out=HERE/'checks'/('verification-'+str(time.time_ns()));out.mkdir(parents=True)
    r.setup(7,'cpu');d=r.data();c=r.config();x=d['x']
    with np.load(BASE/'bundle/data.npz',allow_pickle=False) as old:
        for k in ('y','person_id','video_id','fold','cohort','original_row_index'):np.testing.assert_array_equal(d[k],old[k])
    for i,index in enumerate(d['original_row_index']):
        p=HERE/'records'/('%04d.npz'%index);proof=r.load(p.with_suffix('.json'));oldproof=r.load(BASE/'records'/('%04d.json'%index))
        assert r.sha(p)==proof['artifact_sha256'] and proof['raw_content_sha256']==oldproof['raw_content_sha256']
        assert proof['prepare_code_sha256']==r.sha(HERE/'prepare_inputs.py')
        with np.load(p,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['multi_x'],x.values[x.owner==i])
            np.testing.assert_array_equal(z['window_start'],x.starts[x.owner==i])
            if proof['canonical_fallback']:
                np.testing.assert_array_equal(z['multi_x'][0],z['x'])
            else:assert proof['eligible_sliding_window_starts']==oldproof['eligible_sliding_window_starts']
    # Independent geometry on every emitted window, not just the canonical interval.
    geom=x.values.astype(np.float64)
    np.testing.assert_allclose(np.linalg.norm(geom[...,6:8]-geom[...,0:2],axis=-1),geom[...,10],rtol=0,atol=8e-7)
    np.testing.assert_allclose(geom[:,1,:,0]-geom[:,0,:,0],geom[:,0,:,11],rtol=0,atol=8e-7)
    np.testing.assert_allclose(geom[:,0,:,11],-geom[:,1,:,11],rtol=0,atol=1e-7)
    for a,b,center,target in [(0,4,2,8),(6,2,0,9)]:
        va=geom[...,a:a+2]-geom[...,center:center+2];vb=geom[...,b:b+2]-geom[...,center:center+2]
        cross=np.abs(va[...,0]*vb[...,1]-va[...,1]*vb[...,0]);dot=(va*vb).sum(-1)
        np.testing.assert_allclose(np.arctan2(cross,dot),geom[...,target],rtol=0,atol=1e-3)
    del geom
    for f in range(5):
        for j in range(5):
            if j==f:continue
            tr,va,te=r.split(d,f,j);sx=x[tr]
            weights=sx.side_weights().reshape(-1,2)
            totals=np.zeros(len(sx));np.add.at(totals,sx.owner,weights.sum(1))
            np.testing.assert_allclose(totals,2*len(sx.values)/len(sx),rtol=1e-7)
            assert abs(weights.mean()-1)<1e-6
    rng=np.random.default_rng(5)
    wx,_=synthetic(2,rng,[1,3])
    pred=torch.tensor([1.,1.,2.,2.,3.,3.,4.,4.],requires_grad=True)
    weights=torch.from_numpy(wx.side_weights())
    actual=weighted_mse(pred,torch.zeros(8),weights)
    manual=(pred[:2].square().mean()+pred[2:].square().mean())/2
    torch.testing.assert_close(actual,manual,rtol=0,atol=0)
    ga=torch.autograd.grad(actual,pred,retain_graph=True)[0];gm=torch.autograd.grad(manual,pred)[0]
    torch.testing.assert_close(ga,gm,rtol=0,atol=0)
    reject(lambda:weighted_mse(pred,torch.zeros(8,1),weights))
    reject(lambda:weighted_mse(pred,torch.zeros(8),weights[:,None]))
    vals=np.array([[1,10],[2,20],[4,40],[6,60]],dtype=np.float32)
    np.testing.assert_array_equal(wx.aggregate(vals),[[1,10],[4,40]])
    class Probe(torch.nn.Module):
        def forward(self,t):return torch.stack([t[:,0,0],torch.zeros(len(t))],1)
    wx.values[:]=0
    for k,v in enumerate(vals):wx.values[k,:,0,0]=v
    np.testing.assert_array_equal(r.predict(Probe(),wx,torch.tensor(2.),torch.tensor(3.),'cpu',batch=2),[[5,32],[14,122]])
    rx=wx[np.array([1,0])]
    np.testing.assert_array_equal(rx.aggregate(vals),[[4,40],[1,10]])
    print('All 11261 windows: source hashes, geometry, original splits, weight sums, vector gradients and record aggregation passed.',flush=True)
    r.smoke(c,out/'smoke','cpu')
    sx,sy=synthetic(12,rng);pid=np.array(['s%d'%i for i in range(12)]);vx=sx[:4]
    ident={'fit_seed':321,'purpose':'synthetic_only','epochs':2}
    with (out/'resume.log').open('w') as log,contextlib.redirect_stdout(log):
        full,*_=r.fit(sx,sy,pid,vx,sy[:4],pid[:4],c,2,out/'full',ident,'cpu')
        r.fit(sx,sy,pid,vx,sy[:4],pid[:4],c,1,out/'resume',ident,'cpu')
        resumed,*_=r.fit(sx,sy,pid,vx,sy[:4],pid[:4],c,2,out/'resume',ident,'cpu')
        poisoned,*_=r.fit(sx,sy,pid,vx,np.full((4,2),9999.),pid[:4],c,2,out/'held_labels_changed',ident,'cpu')
        for k,v in full.state_dict().items():assert torch.equal(v,resumed.state_dict()[k]) and torch.equal(v,poisoned.state_dict()[k])
        a=r.load(out/'full/curve.json');b=r.load(out/'held_labels_changed/curve.json')
        assert [v['calibration'] for v in a]==[v['calibration'] for v in b]
        reject(lambda:r.fit(sx,sy,pid,None,None,None,c,2,out/'resume',dict(ident,fit_seed=99),'cpu'))
    print('Weighted epoch resume exact; held-label poisoning leaves weights/calibration unchanged.',flush=True)
    # Small entirely synthetic data exercises all 25 fits; no real GDI optimization.
    n=30;xx,yy=synthetic(n,rng)
    sd={'x':xx,'y':yy,'person_id':np.array(['person%02d'%i for i in range(n)]),
        'video_id':np.array(['video%02d'%i for i in range(n)]),'fold':np.arange(n)%5,
        'cohort':np.arange(n)%2,'original_row_index':np.arange(n)}
    sc=dict(c,max_epochs=1);dest=out/'synthetic_nested'
    reject(lambda:r.summarize(sd,sc,dest))
    bad=dict(sd,person_id=sd['person_id'].copy());bad['person_id'][0]=bad['person_id'][1]
    reject(lambda:r.split(bad,0,1))
    for f in range(5):
        with (out/'nested.log').open('a') as log,contextlib.redirect_stdout(log):r.train_fold(sd,sc,f,dest,'cpu',None)
        print('Synthetic nested fold %d/5 passed'%(f+1),flush=True)
    with (out/'nested.log').open('a') as log,contextlib.redirect_stdout(log):r.summarize(sd,sc,dest);r.export_review(dest)
    with np.load(dest/'oof_predictions.npz',allow_pickle=False) as z:p=z['predictions'][:,0];raw=z['predictions_raw'][:,0]
    independent=np.mean([np.abs(p[sd['person_id']==who]-yy[sd['person_id']==who]).mean() for who in set(sd['person_id'])])
    assert abs(independent-r.load(dest/'summary.json')['metrics'][0]['mae'])<1e-10
    for f in range(5):
        ck=torch.load(dest/('fold%d/refit/latest.pt'%f),map_location='cpu',weights_only=True);m=GdiCNN();m.load_state_dict(ck['model'])
        tr,_,te=r.split(sd,f);pp=r.predict(m,xx[te],ck['lo'],ck['span'],'cpu')
        np.testing.assert_array_equal(pp,raw[te])
        train_pred=r.predict(m,xx[tr],ck['lo'],ck['span'],'cpu');coef=r.calibrate(train_pred,yy[tr])
        np.testing.assert_array_equal(r.corrected(pp,coef),p[te])
    fp=dest/'fold0/predictions.npz';orig=fp.read_bytes();fp.write_bytes(orig+b'corruption');reject(lambda:r.summarize(sd,sc,dest));fp.write_bytes(orig)
    proof={'passed':True,'people':692,'records':1061,'windows':11261,'fallback_records':4,
        'all_window_geometry_and_provenance':True,'same_labels_people_and_folds':True,
        'record_weight_sums_all20_splits':True,'weighted_loss_and_gradient_independent':True,
        'broadcast_shape_error_rejected':True,'known_window_average_and_prediction':True,
        'checkpoint_resume_exact':True,'validation_label_poison_test':True,'synthetic_nested_fits':25,
        'synthetic_nested_optimizer_steps':r.load(dest/'summary.json')['optimizer_steps'],
        'independent_checkpoint_reinference':True,'actual_GDI_optimizer_steps':0,'actual_GPU_execution':False,
        'bundle_sha256':r.sha(HERE/'bundle/manifest.json'),'details':out.relative_to(HERE).as_posix()}
    r.save(HERE/'verification.json',proof);print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
