"""Exploratory person-disjoint Toronto TUG comparison on first qualified 4s only."""
from pathlib import Path
import sys,json,hashlib,time,argparse,os
import numpy as np,torch,torch_directml
from torch import nn
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104');O=B/'first4s-models'
VAR=['image_xy','pelvis_stable_xy','relative_mp_xyz','bones_velocity','pair_mean_absdiff'];GRID=[5,20,50]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def inputs():
    d=np.load(B/'bottom14-inputs.npz');ids=np.array(sorted(set(d['person_ids'])));ix=np.array([np.flatnonzero(d['person_ids']==i)[0] for i in ids]); labels={r['person_id']:r['tug'] for r in read(H/'clinical-label-audit.json')['labels']};y=np.array([labels[i] for i in ids]);out={}
    for v in VAR:
        m=d['joint_mask'][ix] if v=='image_xy' else d['bones_velocity_mask'][ix] if v=='bones_velocity' else d['pair_mask'][ix] if v=='pair_mean_absdiff' else d['pelvis_mask'][ix]
        factor=3 if v=='relative_mp_xyz' else 2; mask=np.repeat(m,factor,axis=-1);x=d[v][ix];assert x.shape==mask.shape and np.isfinite(x).all();out[v]=(x,mask)
    return ids,ix,y,out
def prep(x,m,tr,y):
    count=m[tr].sum(axis=(0,1));mean=(x[tr]*m[tr]).sum(axis=(0,1))/np.maximum(count,1);var=(((x[tr]-mean)*m[tr])**2).sum(axis=(0,1))/np.maximum(count,1);std=np.sqrt(var);std[std<1e-8]=1
    return dict(mean=mean.tolist(),std=std.tolist(),ym=float(y[tr].mean()),ys=max(float(y[tr].std()),1e-8))
def scaled(x,m,p):return np.concatenate([np.where(m,(x-np.array(p['mean']))/np.array(p['std']),0),m.astype(float)],axis=-1).transpose(0,2,1).astype(np.float32)
class Net(nn.Module):
    def __init__(self,n):
        super().__init__();self.conv=nn.Sequential(nn.Conv1d(n,16,5,padding=2),nn.SiLU(),nn.Conv1d(16,16,5,padding=4,dilation=2),nn.SiLU());self.head=nn.Sequential(nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
    def forward(self,x):
        z=self.conv(x);return self.head(torch.cat([z.mean(-1),z.std(-1,unbiased=False)],1)).squeeze(-1)
def descriptors(x,m):
    n=np.maximum(m.sum(1),1);mu=(x*m).sum(1)/n;dev=np.where(m,x-mu[:,None,:],0);sd=np.sqrt((dev**2).sum(1)/n);pair=m[:,1:]&m[:,:-1];pn=np.maximum(pair.sum(1),1);diff=(np.abs(np.diff(x,axis=1))*pair).sum(1)/pn;lag=(dev[:,1:]*dev[:,:-1]*pair).sum(1)/pn/np.maximum(sd**2,1e-8)
    return np.concatenate([mu,sd,diff,lag,m.mean(1)],axis=1)
def ridge_fit(x,m,y,tr,ev,path):
    f=descriptors(x,m);mu=f[tr].mean(0);sd=f[tr].std(0);sd[sd<1e-8]=1;z=(f[tr]-mu)/sd;ym=y[tr].mean();dual=np.linalg.solve(z@z.T+10*np.eye(len(tr)),y[tr]-ym);coef=z.T@dual;pr=(f[ev]-mu)/sd@coef+ym;np.savez_compressed(path,mean=mu,std=sd,coef=coef,intercept=ym,train=tr,eval=ev,prediction=pr,features=f)
    return dict(path=str(path),sha256=sha(path),prediction=pr.tolist())
def predict(net,a,ev,p,device):
    net.eval()
    with torch.no_grad():r=net(torch.from_numpy(a[ev]).to(device)).cpu().numpy().astype(float)
    return r*p['ys']+p['ym']
def fit(x,m,y,ids,tr,ev,epochs,seed,path,grid):
    path=Path(path);receipt=path/'complete.json'
    if receipt.exists():return read(receipt)
    assert not(path/'context.json').exists(),'preserve incomplete fit before retry';path.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);p=prep(x,m,tr,y);a=scaled(x,m,p);device=torch_directml.device();net=Net(a.shape[1]).to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,foreach=False);t=torch.from_numpy(((y[tr]-p['ym'])/p['ys']).astype(np.float32)).to(device);xx=torch.from_numpy(a[tr]).to(device);rows=[]
    save(path/'context.json',dict(train=tr.tolist(),eval=ev.tolist(),train_ids=ids[tr].tolist(),eval_ids=ids[ev].tolist(),prep=p,epochs=epochs,seed=seed,protocol_sha256=sha(H/'first4s-protocol.json')))
    for epoch in range(1,epochs+1):
        net.train();opt.zero_grad(set_to_none=True);loss=nn.functional.smooth_l1_loss(net(xx),t,beta=.5);assert bool(torch.isfinite(loss).cpu());loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5,foreach=False);opt.step()
        if epoch in grid:
            pr=predict(net,a,ev,p,device);model=path/f'epoch{epoch}.pt';torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},model);rows.append(dict(epoch=epoch,prediction=pr.tolist(),model=str(model),sha256=sha(model)))
    r=dict(context=str(path/'context.json'),curve=rows,loss=float(loss.detach().cpu()));save(receipt,r);return r
def main():
    torch.set_num_threads(2);ids,ix,y,inp=inputs();assert len(ids)==14;assert read(H/'first-windows-visual-review.json')['status']=='sampled_first_windows_reviewed'
    proto=dict(code_sha256=sha(Path(__file__)),data_sha256=sha(B/'bottom14-inputs.npz'),label_audit_sha256=sha(H/'clinical-label-audit.json'),visual_review_sha256=sha(H/'first-windows-visual-review.json'),people=14,selected_window_indices=ix.tolist(),ids=ids.tolist(),variants=VAR,outer='Leave one entire person out,14folds; one earliest operational4secondwindow per person, frozen before reading TUG. All14 included, no outcome exclusions.',inner='For each outertrain13, permute with seed20261104+outerindex and split into3innerheldout groups4/4/5. All scaling/target normalization fit only innertrain; aggregate innerOOF choose minimum MAE,earliest tie. Final refit on13 at selectedepoch. Fixed single seed, no outcome-based representation selection.',models='Training-person mean; ridge alpha10 on masked perchannel mean/std/lag1/meanabsdiff summaries; small temporalCNN on time-ordered coordinates+explicit masks. Conv16kernel5,dilation1/2,SiLU,mean+stdpool,16unithead. AdamW.001,decay.01,fullbatch,50epochs,grid5/20/50,SmoothL1beta.5,clip5. No dropout/seed ensemble/clinical covariates.',verification='Saved contexts, training-only masked scaling, IDboundaries, all210innerfits*3checkpoints+70finalmodels CPU64 replay; ridge70models native reload, exactepochchoice, independently recomputed14personMAE/p95/score-banderrors and pairedbootstrap. Every variant same14people/windowset.',limits='14people,only1TUG>=20. Exploratory input feasibility,not clinical/external accuracy validation. Frontal/back camera,notCPGDI lateral. Releasedpresentationclock only,relativeMPznotmetres; no jointtruth/SPPB/GDI. First4sselectedfromlongerrecordings,not4sacquisitionlatency. Sparse visual samples cannot prove allframes actor/jointidentity. Ridge differs fromCNN; compare both. No true m/s video-speed calibration, so no speed-only advantage claim.')
    if(H/'first4s-protocol.json').exists():assert read(H/'first4s-protocol.json')==proto
    else:save(H/'first4s-protocol.json',proto)
    O.mkdir(exist_ok=True);start=time.perf_counter();done=0
    for v,(x,m) in inp.items():
      for outer in range(14):
        path=O/v/f'fold{outer}';receipt=path/'fold.json'
        if receipt.exists():done+=1;continue
        tr=np.array([j for j in range(14) if j!=outer]);ev=np.array([outer]);groups=np.array_split(np.random.default_rng(20261104+outer).permutation(tr),3);inner=[]
        for k,iv in enumerate(groups):inner.append(fit(x,m,y,ids,np.array([j for j in tr if j not in iv]),iv,50,20261104+outer*100+VAR.index(v)+k*1000,path/f'inner{k}',GRID))
        scores=[]
        for gi,ep in enumerate(GRID):
            pr=np.zeros(14)
            for r,iv in zip(inner,groups):pr[iv]=r['curve'][gi]['prediction']
            scores.append(float(np.abs(pr[tr]-y[tr]).mean()))
        ep=GRID[int(np.argmin(scores))];final=fit(x,m,y,ids,tr,ev,ep,20261104+outer*100+VAR.index(v)+5000,path/'final',[ep]);ridge=ridge_fit(x,m,y,tr,ev,path/'ridge.npz');save(receipt,dict(variant=v,outer=outer,inner=inner,inner_mae=scores,epoch=ep,final=final,ridge=ridge,train_mean=float(y[tr].mean()),actual=float(y[outer]),prediction=final['curve'][0]['prediction'][0]));done+=1;save(H/'first4s-progress.json',dict(status='running',complete=done,expected=70,elapsed_seconds=time.perf_counter()-start,last=v+str(outer)));print(v,outer,'epoch',ep,round(time.perf_counter()-start,1),flush=True)
    save(H/'first4s-receipt.json',dict(status='complete_pending_independent_review',cases=done,protocol_sha256=sha(H/'first4s-protocol.json'),hashes={str(p):sha(p) for p in O.rglob('*') if p.is_file()}))
if __name__=='__main__':main()
