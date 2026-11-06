"""Train-person-only masked motion pretraining, paired against same CPU CNN."""
from pathlib import Path
import sys,json,hashlib,time,os
import numpy as np,torch
from torch import nn
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104');R=H/'motion-pretrain';O=B/'motion-pretrain';sys.path.insert(0,str(H));from mp_compare14 import Net,inputs,prep,scaled
VAR=['pelvis_stable_xy','pair_mean_absdiff'];CONDS=['supervised','masked_pretrain'];GRID=[5,20,50]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def fit(x,m,y,ids,tr,ev,epochs,seed,path,grid,mode):
    path=Path(path);complete=path/'complete.json'
    if complete.exists():return read(complete)
    assert not(path/'context.json').exists();path.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);p=prep(x,m,tr,y);a=scaled(x,m,p);net=Net(a.shape[1]);pre=None;start=time.perf_counter();ctx=dict(train=tr.tolist(),eval=ev.tolist(),train_ids=ids[tr].tolist(),eval_ids=ids[ev].tolist(),prep=p,epochs=epochs,seed=seed,mode=mode,protocol_sha256=sha(R/'protocol.json'));save(path/'context.json',ctx);xx=torch.from_numpy(a[tr]);C=x.shape[-1]
    if mode=='masked_pretrain':
        decoder=nn.Conv1d(16,C,1);optimizer=torch.optim.AdamW(list(net.conv.parameters())+list(decoder.parameters()),lr=.001,weight_decay=.01,foreach=False);initial={k:v.clone() for k,v in net.conv.state_dict().items()};rng=np.random.default_rng(seed+77777);losses=[];probe=None
        for epoch in range(100):
            # Pair x/y components of a joint; only actually observed training values
            # are possible targets. Invalid detections never become pseudo truth.
            corruption=np.repeat(rng.random((len(tr),120,C//2))<.25,2,-1).transpose(0,2,1)&(a[tr,C:]>0);assert corruption.any();keep=torch.from_numpy(~corruption);noisy=xx.clone();noisy[:,:C]*=keep;noisy[:,C:]*=keep;optimizer.zero_grad(set_to_none=True);reconstruction=decoder(net.conv(noisy));target=xx[:,:C];where=torch.from_numpy(corruption);loss=((reconstruction-target)**2)[where].mean();assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(list(net.conv.parameters())+list(decoder.parameters()),5,foreach=False);optimizer.step();losses.append(float(loss.detach()))
            if epoch==99:probe=corruption
        assert any(not torch.equal(initial[k],v) for k,v in net.conv.state_dict().items());model=path/'pretrained.pt';torch.save({'encoder':net.conv.state_dict(),'decoder':decoder.state_dict()},model);np.savez_compressed(path/'pretrain-probe.npz',corruption=probe);net.eval();decoder.eval();noisy=xx.clone();keep=torch.from_numpy(~probe);noisy[:,:C]*=keep;noisy[:,C:]*=keep
        with torch.no_grad():values=decoder(net.conv(noisy)).numpy()
        np.savez_compressed(path/'pretrain-replay.npz',prediction=values);pre=dict(model=str(model),sha256=sha(model),probe=str(path/'pretrain-probe.npz'),replay=str(path/'pretrain-replay.npz'),initial_loss=losses[0],last_loss=losses[-1],epochs=100);save(path/'pretrain.json',pre)
    # Reset optimizer; both paired conditions get the same supervised updates.
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,foreach=False);target=torch.from_numpy(((y[tr]-p['ym'])/p['ys']).astype(np.float32));rows=[]
    for epoch in range(1,epochs+1):
        net.train();opt.zero_grad(set_to_none=True);loss=nn.functional.smooth_l1_loss(net(xx),target,beta=.5);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5,foreach=False);opt.step()
        if epoch in grid:
            net.eval()
            with torch.no_grad():pr=net(torch.from_numpy(a[ev])).numpy().astype(float)*p['ys']+p['ym']
            model=path/f'epoch{epoch}.pt';torch.save(net.state_dict(),model);rows.append(dict(epoch=epoch,prediction=pr.tolist(),model=str(model),sha256=sha(model)))
    rec=dict(context=str(path/'context.json'),curve=rows,pretrain=pre,elapsed_seconds=time.perf_counter()-start);save(complete,rec);return rec
def main():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);torch.set_num_threads(2);ids,ix,y,inp=inputs();proto=dict(code_sha256=sha(Path(__file__)),base_library_sha256=sha(H/'mp_compare14.py'),input_sha256=sha(B/'bottom14-inputs.npz'),label_sha256=sha(H/'clinical-label-audit.json'),baseline_verification_sha256=sha(H/'first4s-verification.json'),people=14,ids=ids.tolist(),window_indices=ix.tolist(),variants=VAR,conditions=CONDS,design='Same14people/earliestpreviouslyreviewed4s/outerleaveonepersonout/inner3exactseedpartitions fromfirst4s study. Two geometry-motivated representations, pelvisstablexy andpairmeanabsdiff,predeclaredwithoutchoosinglowestpreviousouterMAE. MatchedsameCPUNetarchitecture+seed+supervisedsteps. All scaling/unsupervised/supervised/pretraining andepochselection excludesoutertestperson; innerpretraining excludesinnervalidpeople too.',pretraining='100full-batchAdamW.001/decay.01passes;random25percentobservedjointxyhidden,preserveoriginalmissingmask;encoderconv16and1x1decoder reconstructoriginalstandardizedMediaPipeestimatedcoordinatesonlyonhiddenobservedtrainingentries. DecoderdiscardedbeforeTUGfit;supervisedoptimizerreset. No anatomicaltruth orpose-error-correctionclaim. Fixedmaskrngfitseed+77777;savepretrainedencoder/decoder andfinalcorruption/reconstruction forCPU64replay.',supervised='SamefixedCPUNet16convkernel5dilation1/2,meanstdpool,head16. FullbatchAdamW.001,decay.01,SmoothL1.5,gradclip5,50innerpasses,grid5/20/50. MininnerOOFMAE/earliesttie;finalrefitselectedepochs,with100train-onlypretrainingpasses inpretrainedcondition. No outputsclipped orrepresentationchosenfromoutertest.',verification='224fits;504innergrid+56finalcheckpointsCPU64replay,plus112pretrainedstates/probes;sourcehashes/personboundaries/exacttraining-onlyscalers/epochselection/metrics/paireduncertainty. Reused14developmentpeople,1severe;notclinicalvalidation. Original releasedcaptureclockunverified,CPGDIviewmismatchunchanged.',expected_additional_GB=.06)
    if(R/'protocol.json').exists():assert read(R/'protocol.json')==proto
    else:save(R/'protocol.json',proto)
    done=0;start=time.perf_counter();save(R/'process.json',dict(pid=os.getpid(),status='running'))
    for v in VAR:
      x,m=inp[v]
      for mode in CONDS:
       for outer in range(14):
        path=O/v/mode/f'fold{outer}';complete=path/'fold.json'
        if complete.exists():done+=1;continue
        tr=np.array([i for i in range(14) if i!=outer]);ev=np.array([outer]);groups=np.array_split(np.random.default_rng(20261104+outer).permutation(tr),3);inner=[];vi=['image_xy','pelvis_stable_xy','relative_mp_xyz','bones_velocity','pair_mean_absdiff'].index(v)
        for k,iv in enumerate(groups):inner.append(fit(x,m,y,ids,np.array([i for i in tr if i not in iv]),iv,50,20261104+outer*100+vi+k*1000,path/f'inner{k}',GRID,mode))
        scores=[]
        for j,epoch in enumerate(GRID):
            pp=np.zeros(14)
            for r,iv in zip(inner,groups):pp[iv]=r['curve'][j]['prediction']
            scores.append(float(np.abs(pp[tr]-y[tr]).mean()))
        epoch=GRID[int(np.argmin(scores))];final=fit(x,m,y,ids,tr,ev,epoch,20261104+outer*100+vi+5000,path/'final',[epoch],mode);save(complete,dict(variant=v,mode=mode,outer=outer,inner=inner,inner_mae=scores,epoch=epoch,final=final,prediction=final['curve'][0]['prediction'][0]));done+=1;save(R/'progress.json',dict(status='running',complete=done,expected=56,elapsed_seconds=time.perf_counter()-start,last=v+'/'+mode+str(outer)));print(v,mode,outer,round(time.perf_counter()-start,1),flush=True)
    save(R/'receipt.json',dict(status='complete_pending_independent_review',cases=56,protocol_sha256=sha(R/'protocol.json'),hashes={str(p):sha(p) for p in O.rglob('*') if p.is_file()}));save(R/'process.json',dict(pid=os.getpid(),status='complete'))
if __name__=='__main__':main()
