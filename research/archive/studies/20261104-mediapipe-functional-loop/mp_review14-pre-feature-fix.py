"""Independent saved-model replay and person/normalization/epoch audit of Toronto pilot."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104');O=B/'first4s-models';sys.path.insert(0,str(H));from mp_compare14 import Net
VAR=['image_xy','pelvis_stable_xy','relative_mp_xyz','bones_velocity','pair_mean_absdiff'];GRID=[5,20,50]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def desc(x,m):
    out=[]
    for a,v in zip(x,m):
        mu=[];sd=[];diff=[];lag=[];freq=[]
        for j in range(a.shape[1]):
            z=a[v[:,j],j];center=float(z.mean()) if len(z) else 0.;var=float(np.mean((z-center)**2)) if len(z) else 0.;okay=v[1:,j]&v[:-1,j];mu.append(center);sd.append(np.sqrt(var));diff.append(float(np.abs(a[1:,j]-a[:-1,j])[okay].mean()) if okay.any() else 0.);lag.append(float(((a[1:,j]-center)*(a[:-1,j]-center))[okay].mean())/max(var,1e-8) if okay.any() else 0.);freq.append(float(v[:,j].mean()))
        out.append(mu+sd+diff+lag+freq)
    return np.array(out)
def audit_fit(rec,x,m,y,ids,tr,ev,seed,epochs,grid,proto):
    ctx=read(rec['context']);assert ctx['train']==tr.tolist() and ctx['eval']==ev.tolist() and ctx['train_ids']==ids[tr].tolist() and ctx['eval_ids']==ids[ev].tolist() and not set(tr)&set(ev);assert ctx['seed']==seed and ctx['epochs']==epochs and ctx['protocol_sha256']==sha(H/'first4s-protocol.json');assert [r['epoch'] for r in rec['curve']]==grid
    # Fit channel moments independently on observed training entries only.
    # The fit used a float32 masked sum divided by integer counts. A compressed
    # valid-only float32 reduction has a different summation order. Rebuild the
    # specified arithmetic on the original training tensor, then require exact
    # equality; do not relax the saved-model replay tolerance.
    count=m[tr].sum(axis=(0,1));mu=np.sum(np.where(m[tr],x[tr],np.float32(0)),axis=(0,1))/np.maximum(count,1);dev=np.where(m[tr],x[tr]-mu,0.);sd=np.sqrt(np.sum(dev*dev,axis=(0,1))/np.maximum(count,1));sd[sd<1e-8]=1
    p=ctx['prep'];assert np.array_equal(mu,p['mean']) and np.array_equal(sd,p['std']);ym=float(y[tr].mean());ys=max(float(y[tr].std()),1e-8);assert p['ym']==ym and p['ys']==ys
    # Replay the exact float32-rounded inputs that the trained device received.
    a=np.concatenate([np.where(m,(x-np.array(p['mean']))/np.array(p['std']),0),m.astype(float)],-1).transpose(0,2,1).astype(np.float32).astype(float);maxdelta=0
    for r in rec['curve']:
        assert sha(r['model'])==r['sha256'];net=Net(a.shape[1]).double();net.load_state_dict(torch.load(r['model'],map_location='cpu',weights_only=True));net.eval()
        with torch.no_grad():pr=net(torch.from_numpy(a[ev])).numpy()*ys+ym
        original=np.array(r['prediction']);assert np.allclose(pr,original,atol=1e-6,rtol=1e-6);maxdelta=max(maxdelta,float(np.max(np.abs(pr-original))))
    return maxdelta
def main():
    torch.set_num_threads(2);proto=read(H/'first4s-protocol.json');receipt=read(H/'first4s-receipt.json');assert proto['code_sha256']==sha(H/'mp_compare14.py') and proto['data_sha256']==sha(B/'bottom14-inputs.npz');assert receipt['cases']==70 and receipt['protocol_sha256']==sha(H/'first4s-protocol.json')
    for p,v in receipt['hashes'].items():assert sha(p)==v
    raw=np.load(B/'bottom14-inputs.npz');ids=np.array(sorted(set(raw['person_ids'])));ix=np.array([np.flatnonzero(raw['person_ids']==pid)[0] for pid in ids]);assert ix.tolist()==proto['selected_window_indices'] and ids.tolist()==proto['ids'];lab={r['person_id']:r['tug'] for r in read(H/'clinical-label-audit.json')['labels']};y=np.array([lab[i] for i in ids]);preds={'train_mean':np.array([np.delete(y,i).mean() for i in range(14)])};checks=[];count=0;ridgecount=0
    for vi,v in enumerate(VAR):
        x=raw[v][ix];maskname='joint_mask' if v=='image_xy' else 'bones_velocity_mask' if v=='bones_velocity' else 'pair_mask' if v=='pair_mean_absdiff' else 'pelvis_mask';m=np.repeat(raw[maskname][ix],3 if v=='relative_mp_xyz' else 2,-1);features=desc(x,m);pr=[];rp=[]
        for outer in range(14):
            rec=read(O/v/f'fold{outer}/fold.json');tr=np.array([i for i in range(14) if i!=outer]);ev=np.array([outer]);groups=np.array_split(np.random.default_rng(20261104+outer).permutation(tr),3);innerpred=np.zeros((3,14));delta=0
            for k,(r,iv) in enumerate(zip(rec['inner'],groups)):
                it=np.array([i for i in tr if i not in iv]);delta=max(delta,audit_fit(r,x,m,y,ids,it,iv,20261104+outer*100+vi+k*1000,50,GRID,proto));count+=3
                for j,row in enumerate(r['curve']):innerpred[j,iv]=row['prediction']
            scores=np.abs(innerpred[:,tr]-y[tr]).mean(1);assert np.allclose(scores,rec['inner_mae'],atol=1e-12,rtol=1e-12);epoch=GRID[int(np.argmin(scores))];assert epoch==rec['epoch'];delta=max(delta,audit_fit(rec['final'],x,m,y,ids,tr,ev,20261104+outer*100+vi+5000,epoch,[epoch],proto));count+=1;pr.append(rec['prediction']);assert rec['actual']==y[outer] and rec['train_mean']==preds['train_mean'][outer] and rec['prediction']==rec['final']['curve'][0]['prediction'][0]
            rr=rec['ridge'];assert sha(rr['path'])==rr['sha256'];r=np.load(rr['path']);assert np.array_equal(r['train'],tr) and np.array_equal(r['eval'],ev);assert np.allclose(r['features'],features,atol=1e-9,rtol=1e-9);F=r['features'];mu=F[tr].mean(0);sd=F[tr].std(0);sd[sd<1e-8]=1;assert np.array_equal(mu,r['mean']) and np.array_equal(sd,r['std']);A=(F[tr]-mu)/sd;coef=r['coef'];res=A.T@(A@coef-(y[tr]-y[tr].mean()))+10*coef;assert np.max(np.abs(res))<1e-7;assert r['intercept']==y[tr].mean();p=(F[ev]-mu)/sd@coef+r['intercept'];assert np.allclose(p,rr['prediction'],atol=1e-12,rtol=1e-12);rp.append(float(p[0]));ridgecount+=1;checks.append(dict(variant=v,outer=outer,selected_epoch=epoch,cpu64_max_delta=delta,checkpoints=10,ridge_normal_equation_passed=True))
        preds[v+'/cnn']=np.array(pr);preds[v+'/ridge']=np.array(rp);print('verified',v,flush=True)
    assert count==700 and ridgecount==70;out={};unc={};rng=np.random.default_rng(20261104);boot=rng.integers(0,14,(5000,14));baseline=np.abs(preds['train_mean']-y);held=[]
    for key,p in preds.items():
        e=np.abs(p-y);out[key]=dict(n=14,mae=float(e.mean()),p95=float(np.quantile(e,.95)),tug20plus_n=int(sum(y>=20)),tug20plus_error=float(e[y>=20][0]),bias=float((p-y).mean()),pearson=float(np.corrcoef(p,y)[0,1]) if np.std(p)>0 else None);diff=e-baseline;unc[key]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),mae_difference_mean_baseline=float(diff.mean()),difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist());held.extend(dict(id=str(i),condition=key,actual=float(a),prediction=float(pr)) for i,a,pr in zip(ids,y,p))
    save(B/'first4s-heldout.json',held);save(H/'first4s-comparison.json',out);save(H/'first4s-uncertainty.json',dict(method='5000pairedpersonbootstrap;14people/1severe;conditionalonthisexploratoryfixedpipeline,notselectionadjusted/clinical.',conditions=unc));save(H/'first4s-verification.json',dict(status='passed',people=14,cnn_fits=280,inner_checkpoint_replays=630,final_checkpoint_replays=70,ridge_models_replayed=70,checks=checks,comparison_sha256=sha(H/'first4s-comparison.json'),heldout_sha256=sha(B/'first4s-heldout.json'),review_code_sha256=sha(Path(__file__))));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
