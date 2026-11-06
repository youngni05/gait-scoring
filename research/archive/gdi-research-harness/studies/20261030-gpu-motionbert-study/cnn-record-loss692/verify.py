"""Synthetic mathematical, resume and nested tests; real data checked without fitting."""
from pathlib import Path
import argparse,contextlib,json,tempfile
import numpy as np
import torch
import run as r
from record_loss import packed,loss_from_windows

def reject(fn):
    try:fn()
    except (ValueError,FileNotFoundError):return
    raise AssertionError('Invalid case accepted')

def synthetic(n=10):
    rng=np.random.default_rng(33);counts=1+np.arange(n)%3;owner=np.repeat(np.arange(n),counts)
    x=r.WindowSet(rng.normal(size=(len(owner),2,124,12)).astype(np.float32),owner,n)
    return dict(x=x,y=rng.uniform(40,100,(n,2)),person_id=np.array(['s%02d'%i for i in range(n)]),
        video_id=np.array(['v%02d'%i for i in range(n)]),fold=np.arange(n)%5,cohort=np.arange(n)%2,original_row_index=np.arange(n))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--quick',action='store_true');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);r.setup(7,a.device);d=r.data();base=r.config()
    assert r.sha(r.ROOT/'data.npz')==r.load(r.ROOT/'manifest.json')['data.npz']
    sizes=[]
    for f in range(5):
        for j in [None]+[i for i in range(5) if i!=f]:
            tr,va,te=r.split(d,f,j);sx=d['x'][tr]
            assert len(sx)==tr.sum() and sx.counts.sum()==len(sx.values)
            sizes.append(dict(outer=f,inner=j,records=len(sx),windows=len(sx.values),steps_per_epoch=int(np.ceil(len(sx)/base['batch_records']))))
    # Independent loss identity: window loss = recording loss + within-record variance.
    pred=torch.tensor([[1.,10.],[2.,20.],[4.,40.],[6.,60.]],device=a.device,requires_grad=True)
    target=torch.tensor([[0.,0.],[1.,2.]],device=a.device);counts=[1,3]
    rec=loss_from_windows(pred,target,counts,'record_mean');win=loss_from_windows(pred,target,counts,'window_control')
    manual=((pred[0]-target[0]).square().mean()+(pred[1:].mean(0)-target[1]).square().mean())/2
    torch.testing.assert_close(rec,manual,rtol=0,atol=0)
    g=torch.autograd.grad(rec,pred,retain_graph=True)[0];gm=torch.autograd.grad(manual,pred,retain_graph=True)[0]
    torch.testing.assert_close(g,gm,rtol=0,atol=0)
    variance=((pred[1:]-pred[1:].mean(0)).square().mean())/2
    torch.testing.assert_close(win,rec+variance,rtol=1e-6,atol=1e-6)
    reject(lambda:loss_from_windows(pred,target[:,:,None],counts,'record_mean'))
    reject(lambda:loss_from_windows(pred,target,[0,4],'record_mean'))
    sd=synthetic();x=sd['x'];y=sd['y'];ids=sd['person_id']
    values,ns=packed(x,np.array([3,1]),a.device)
    expected=np.concatenate([x.values[x.owner==q] for q in (3,1)])
    torch.testing.assert_close(values.cpu(),r.channels(torch.from_numpy(expected)).reshape(-1,12,124),rtol=0,atol=0)
    reject(lambda:packed(x,[1,1],a.device))
    proofs={}
    for variant in base['variants']:
        c=dict(base,loss_variant=variant,name='cnn_'+variant+'_packed_records');ident={'fit_seed':321,'purpose':'synthetic','epochs':2}
        out=a.output/variant;out.mkdir(exist_ok=True)
        with (out/'verification.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log):
            # Validation changes must not affect parameters, normalization or calibration.
            full,lo,span,h=r.fit(x,y,ids,x[:3],y[:3],ids[:3],c,2,out/'full',ident,a.device)
            r.fit(x,y,ids,x[:3],y[:3],ids[:3],c,1,out/'resume',ident,a.device)
            resumed,*_=r.fit(x,y,ids,x[:3],y[:3],ids[:3],c,2,out/'resume',ident,a.device)
            badx=r.WindowSet(x[:3].values+100,x[:3].owner,3)
            poisoned,plo,pspan,ph=r.fit(x,y,ids,badx,y[:3]+9999,ids[:3],c,2,out/'poison',ident,a.device)
            assert torch.equal(lo,plo) and torch.equal(span,pspan)
            for k,v in full.state_dict().items():
                assert torch.equal(v,resumed.state_dict()[k]) and torch.equal(v,poisoned.state_dict()[k])
            assert [row['calibration'] for row in h]==[row['calibration'] for row in ph]
            reject(lambda:r.fit(x,y+1,ids,None,None,None,c,2,out/'resume',ident,a.device))
            fits=0
            if not a.quick:
                sc=dict(c,max_epochs=1);dest=out/'nested'
                reject(lambda:r.summarize(sd,sc,dest))
                for f in range(5):r.train_fold(sd,sc,f,dest,a.device,None)
                r.summarize(sd,sc,dest);r.export_review(dest)
                summary=r.load(dest/'summary.json');fits=summary['fits'];assert fits==25
                with np.load(dest/'oof_predictions.npz') as z:
                    pp=z['predictions'][:,0];raw=z['predictions_raw'][:,0]
                    independent=np.mean([np.abs(pp[ids==q]-y[ids==q]).mean() for q in np.unique(ids)])
                    assert abs(independent-summary['metrics'][0]['mae'])<1e-10
                for f in range(5):
                    tr,_,te=r.split(sd,f);ck=torch.load(dest/('fold%d/refit/latest.pt'%f),map_location=a.device,weights_only=True)
                    m=r.GdiCNN(c['dropout']).to(a.device);m.load_state_dict(ck['model'])
                    rp=r.predict(m,x[te],ck['lo'],ck['span'],a.device)
                    np.testing.assert_array_equal(rp,raw[te])
                    coef=r.calibrate(r.predict(m,x[tr],ck['lo'],ck['span'],a.device),y[tr])
                    np.testing.assert_allclose(r.corrected(rp,coef),pp[te],rtol=0,atol=1e-10)
                # Outer label/input poison leaves all fold0 fitted weights and epoch choice identical.
                perturbed=dict(sd,y=y.copy(),x=r.WindowSet(x.values.copy(),x.owner.copy(),len(x)))
                outer=sd['fold']==0;perturbed['y'][outer]+=9999;perturbed['x'].values[np.isin(x.owner,np.flatnonzero(outer))]+=100
                r.train_fold(perturbed,sc,0,out/'outer_poison',a.device,None)
                assert r.load(dest/'fold0/selection.json')==r.load(out/'outer_poison/fold0/selection.json')
                for role in ['inner1','inner2','inner3','inner4','refit']:
                    ca=torch.load(dest/('fold0/'+role+'/latest.pt'),map_location='cpu',weights_only=True)
                    cb=torch.load(out/('outer_poison/fold0/'+role+'/latest.pt'),map_location='cpu',weights_only=True)
                    for k,v in ca['model'].items():assert torch.equal(v,cb['model'][k])
                fp=dest/'fold0/predictions.npz';old=fp.read_bytes();fp.write_bytes(old+b'corruption')
                reject(lambda:r.summarize(sd,sc,dest));fp.write_bytes(old)
        proofs[variant]=dict(resume_exact=True,validation_poison_invariant=True,training_change_resume_rejected=True,synthetic_nested_fits=fits,outer_poison_invariant=not a.quick,stored_model_reinference=not a.quick)
        print(json.dumps(dict(variant=variant,passed=True)),flush=True)
    report=dict(passed=True,environment=dict(torch=str(torch.__version__),numpy=np.__version__,device=a.device),people=692,records=1061,windows=11261,all_25_training_subsets_checked=True,loss_and_gradient_independent=True,variance_decomposition=True,packing_order_verified=True,variants=proofs,real_data_optimizer_steps=0,bundle_sha256=r.sha(r.ROOT/'manifest.json'),training_subsets=sizes)
    r.save(a.output/'verification.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
