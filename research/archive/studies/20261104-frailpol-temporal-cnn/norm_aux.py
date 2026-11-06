"""Local CPU, matched norm inputs and auxiliary functional-threshold learning."""
from pathlib import Path
import sys,json,hashlib,time,argparse,os
import numpy as np,torch
from torch import nn
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'norm-aux';O=B/'norm-aux';sys.path.insert(0,str(H));from native_motion import Model as NativeModel
CONDS=['full_tail4','norm_tail4','symnorm_tail4','symnorm_tail16','symnorm_aux_tail4'];GRID=[10,25,50,100]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def data():
    original=np.load(B/'data.npz');native=np.load(B/'native-motion/data.npz');norm=np.load(B/'norm-input/inputs.npz');assert all(np.array_equal(original[k],native[k]) for k in ['ids','y','fold']);assert all(np.array_equal(norm[k],native[k]) for k in ['ids','fold','F']);return dict(ids=original['ids'],y=original['y'],fold=original['fold'],F=norm['F'],full=native['native'][:,:24],norm=norm['norm'],symnorm=norm['symmetric_norm'],valid=native['native'][:,-1:]>0)
def kind(c):return 'symnorm' if c.startswith('symnorm') else 'norm' if c.startswith('norm') else 'full'
class Model(nn.Module):
    def __init__(self,channels,aux=False):
        super().__init__();self.core=NativeModel();state=torch.get_rng_state()
        if channels!=24:self.core.conv[0]=nn.Conv1d(channels+1,16,1)
        self.aux=nn.Linear(48,2) if aux else None;torch.set_rng_state(state)
    def forward_all(self,x,f):
        mask=x[:,-1:];z=self.core.conv(x);n=mask.sum(-1).clamp_min(1);mu=(z*mask).sum(-1)/n;sd=(((z-mu[:,:,None])**2*mask).sum(-1)/n+1e-8).sqrt();a=torch.cat([self.core.manual(f),mu,sd],1);return self.core.head(a).squeeze(-1),self.aux(a) if self.aux is not None else None
    def forward(self,x,f):return self.forward_all(x,f)[0]
def preprocess(d,tr,c):
    raw=d[kind(c)];mask=d['valid'][tr];count=mask.sum();mean=(raw[tr]*mask).sum((0,2))/count;std=np.sqrt((((raw[tr]-mean[None,:,None])**2)*mask).sum((0,2))/count);std[std<1e-8]=1;fm=d['F'][tr].mean(0);fs=d['F'][tr].std(0);fs[fs<1e-8]=1;y=d['y'][tr];target=(y[:,None]>=np.array([12,20])[None,:]);positive=target.sum(0);negative=len(tr)-positive;assert np.all(positive>0) and np.all(negative>0)
    return dict(kind=kind(c),condition=c,channel_mean=mean.tolist(),channel_std=std.tolist(),manual_mean=fm.tolist(),manual_std=fs.tolist(),ym=float(y.mean()),ys=max(float(y.std()),1e-8),aux_positive_weight=(negative/positive).tolist())
def arrays(d,p):
    raw=d[p['kind']];mask=d['valid'];values=np.where(mask,(raw-np.array(p['channel_mean'])[None,:,None])/np.array(p['channel_std'])[None,:,None],0);x=np.concatenate([values,mask.astype(float)],axis=1).astype(np.float32);f=((d['F']-p['manual_mean'])/p['manual_std']).astype(np.float32);return x,f
def metric(y,p):
    e=np.abs(y-p);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),score=max(float(e.mean())/.9,float(np.quantile(e,.95))/3,float(e[y>=20].mean())/3))
def infer(net,x,f,ev,p):
    net.eval();out=[];aux=[]
    with torch.no_grad():
        for lo in range(0,len(ev),64):
            ii=ev[lo:lo+64];r,a=net.forward_all(torch.from_numpy(x[ii]),torch.from_numpy(f[ii]));out.extend(r.numpy().astype(float)*p['ys']+p['ym'])
            if a is not None:aux.extend(a.numpy().astype(float))
    return np.array(out),np.array(aux) if aux else None
def fit(d,tr,ev,c,epochs,seed,path,grid):
    path=Path(path);complete=path/'complete.json'
    if complete.exists():return read(complete)
    assert not(path/'context.json').exists(),'preserveincompletefitbeforeretry';path.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);p=preprocess(d,tr,c);x,f=arrays(d,p);use_aux='_aux_' in c;net=Model(x.shape[1]-1,use_aux);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,foreach=False);y=d['y'][tr];target=((y-p['ym'])/p['ys']).astype(np.float32);tail=16 if c.endswith('16') else 4;weight=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.)).astype(np.float32);aux_target=(y[:,None]>=np.array([12,20])[None,:]).astype(np.float32);pos=torch.tensor(p['aux_positive_weight'],dtype=torch.float32);rows=[];start=time.perf_counter();save(path/'context.json',dict(condition=c,train=tr.tolist(),eval=ev.tolist(),train_ids=d['ids'][tr].tolist(),eval_ids=d['ids'][ev].tolist(),epochs=epochs,seed=seed,prep=p,protocol_sha256=sha(R/'protocol.json')))
    for epoch in range(1,epochs+1):
        net.train();order=np.random.default_rng(seed+epoch).permutation(len(tr))
        for lo in range(0,len(tr),64):
            ii=order[lo:lo+64];ix=tr[ii];opt.zero_grad(set_to_none=True);r,a=net.forward_all(torch.from_numpy(x[ix]),torch.from_numpy(f[ix]));loss=(nn.functional.smooth_l1_loss(r,torch.from_numpy(target[ii]),beta=.5,reduction='none')*torch.from_numpy(weight[ii])).mean()
            if use_aux:loss=loss+.1*nn.functional.binary_cross_entropy_with_logits(a,torch.from_numpy(aux_target[ii]),pos_weight=pos)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5,foreach=False);opt.step()
        if epoch in grid:
            pr,aux=infer(net,x,f,ev,p);model=path/f'epoch{epoch}.pt';torch.save(net.state_dict(),model);rows.append(dict(epoch=epoch,prediction=pr.tolist(),aux_logits=aux.tolist() if aux is not None else None,model=str(model),sha256=sha(model),metrics=metric(d['y'][ev],pr)));save(path/'curve.json',rows)
    r=dict(context=str(path/'context.json'),curve=rows,elapsed_seconds=time.perf_counter()-start,epochs=epochs,seed=seed);save(complete,r);print(c,path.parent.name,path.name,round(r['elapsed_seconds'],1),flush=True);return r
def freeze():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);assert read(H/'norm-input/verification.json')['status']=='passed_input_invariance_only_no_model_fitted';base=read(H/'native-motion/protocol.json');p=dict(code_sha256=sha(Path(__file__)),native_library_sha256=sha(H/'native_motion.py'),block_library_sha256=sha(H/'bundle/train.py'),source_original_sha256=sha(B/'data.npz'),source_native_sha256=sha(B/'native-motion/data.npz'),source_norm_sha256=sha(B/'norm-input/inputs.npz'),splits=base['splits'],conditions=CONDS,design='Same594persons/labels/outer5andinner3;selectednativecoreonly. Comparefull24channels,6sensorAcc/Gyrnormchannels,and6left-rightmeanabsdiff+sacrumnormchannels,all208slotswithexplicitvalidmask. Sameobservedspeedandcoreduration;allchannel/manual/targetnormalizationonlyfittrainingpeople. Shared16channelthreeBlockmaskedTCN,fullyCPU2threads. Coreotherlayersinitializedidenticallyacrossconditionsbyseedandfull24construction;6channelprojectorreplacesonlyfirstConv,withCPU_RNGrestored. Normvssymnormsameparametercount/initialization. AuxvssymnormsamecoreinitanddropoutRNG,auxinitializedthenRNGrestored. Fullvsnormfirstprojectionparametercountdiffers,notperfectcapacityisolation.',training='AdamW.001,weight_decay.01,batch64,100innerpasses,grid10/25/50/100,clip5,SmoothL1beta.5standardTUG. Tail4or16,middle sqrt(tail). Symnormaux adds .1BCE on >=12/>=20 labels,positiveweights nnegative/npositive fromfit-trainingpeopleonly. NumericTUGheadisprimary;auxcategoriesnotnewclinicaldiagnosislabels. No targetclipping/timewarping/newpeople. Fixedseed20261106+outer100+inner1000,final+5000,allconditionssamecoreseed. CPUinitialseednotselected.',selection='MinaggregatedinnerOOFmax(MAE/.9,p95/3,slowMAE/3),earliestepoch tie. All5conditionsreported,notselectbestconditionusingoutertest. Mainstartonlyafterrealinner5epochpilotCPU64save-replaypasses.',verification='Expected75inner+25finalfits,300innergrid+25finalcheckpointCPU64replay,plusallauxlogits;sourceID/label/fold/windowalignments,training-onlynormalizersandauxclassweights,innerselection,independentlyrebuiltmetrics/paireduncertainty. Keepfailuresandoriginal1e-6absolute+relativeCPU64pointtolerance. Reuseddevelopmentcohort,severe9,singleinit,noRGB/MediaPipeaccuracyclaim. ExistingGPUstudyuntouched.',expected_additional_GB=.1)
    if(R/'protocol.json').exists():assert read(R/'protocol.json')==p
    else:save(R/'protocol.json',p)
    return p
def pilot():
    p=freeze();d=data();torch.set_num_threads(2);s=p['splits']['0']['inner'][0];tr=np.array(s['train']);ev=np.array(s['valid'][:32]);checks=[]
    for c in CONDS:
        r=fit(d,tr,ev,c,5,20261106,O/'pilot'/c,[5]);prep=read(r['context'])['prep'];x,f=arrays(d,prep);net=Model(x.shape[1]-1,'_aux_' in c).double();net.load_state_dict(torch.load(r['curve'][0]['model'],map_location='cpu',weights_only=True));net.eval()
        with torch.no_grad():raw,aux=net.forward_all(torch.from_numpy(x[ev].astype(float)),torch.from_numpy(f[ev].astype(float)));pr=raw.numpy()*prep['ys']+prep['ym']
        assert np.allclose(pr,r['curve'][0]['prediction'],atol=1e-6,rtol=1e-6)
        if aux is not None:assert np.allclose(aux.numpy(),r['curve'][0]['aux_logits'],atol=1e-6,rtol=1e-6)
        checks.append(dict(condition=c,seconds=r['elapsed_seconds'],max_cpu64_TUG_delta=float(np.max(np.abs(pr-r['curve'][0]['prediction']))),train_people=len(tr),eval_people=len(ev),no_outerpeople=True))
    save(R/'pilot-verification.json',dict(status='passed',checks=checks,protocol_sha256=sha(R/'protocol.json')))
def main():
    p=freeze();assert read(R/'pilot-verification.json')['status']=='passed' and read(R/'pilot-verification.json')['protocol_sha256']==sha(R/'protocol.json');d=data();torch.set_num_threads(2);start=time.perf_counter();done=0;save(R/'process.json',dict(pid=os.getpid(),status='running'))
    for c in CONDS:
      for fold in range(5):
        path=O/c/f'fold{fold}';complete=path/'fold.json'
        if complete.exists():done+=1;continue
        ss=p['splits'][str(fold)];tr=np.array(ss['train']);te=np.array(ss['test']);inner=[]
        for k,s in enumerate(ss['inner']):inner.append(fit(d,np.array(s['train']),np.array(s['valid']),c,100,20261106+fold*100+k*1000,path/f'inner{k}',GRID))
        scores=[]
        for j,epoch in enumerate(GRID):
            pr=np.zeros(len(tr))
            for r,s in zip(inner,ss['inner']):pr[[np.flatnonzero(tr==i)[0] for i in s['valid']]]=r['curve'][j]['prediction']
            scores.append(metric(d['y'][tr],pr))
        epoch=GRID[int(np.argmin([m['score'] for m in scores]))];final=fit(d,tr,te,c,epoch,20261106+fold*100+5000,path/'final',[epoch]);save(complete,dict(condition=c,fold=fold,inner=inner,inner_scores=scores,epoch=epoch,final=final,test_ids=d['ids'][te].tolist(),prediction=final['curve'][0]['prediction']));done+=1;save(R/'progress.json',dict(status='running',complete=done,expected=25,last=c+f'/fold{fold}',elapsed_seconds=time.perf_counter()-start))
    save(R/'receipt.json',dict(status='complete_pending_independent_review',cases=25,protocol_sha256=sha(R/'protocol.json'),hashes={str(p):sha(p) for c in CONDS for p in (O/c).rglob('*') if p.is_file()}));save(R/'process.json',dict(pid=os.getpid(),status='complete'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','pilot','main']);a=ap.parse_args();{'freeze':freeze,'pilot':pilot,'main':main}[a.mode]()
