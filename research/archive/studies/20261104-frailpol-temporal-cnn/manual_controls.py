"""Matched manual-only ablations for norm-aux; no waveform input accepted by model."""
from pathlib import Path
import sys,json,hashlib,time,argparse,os
import numpy as np,torch
from torch import nn
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'manual-controls';O=B/'manual-controls';sys.path.insert(0,str(H));from norm_aux import Model as WaveModel
CONDS=['speed_tail4','speed_tail16','speed_aux_tail4','speed_duration_tail4','speed_duration_aux_tail4'];GRID=[10,25,50,100]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
class Model(nn.Module):
    def __init__(self,aux=False):super().__init__();self.reference=WaveModel(6,aux)
    def forward_all(self,f):
        # Constant encoder summary is an intentional ablation, not a missing
        # clinical clip. The forward API accepts manual features only.
        constant=f.new_zeros((len(f),32));constant[:,16:]=1e-4;a=torch.cat([self.reference.core.manual(f),constant],1);return self.reference.core.head(a).squeeze(-1),self.reference.aux(a) if self.reference.aux is not None else None
    def forward(self,f):return self.forward_all(f)[0]
def manual(d,c):
    f=d['F'].copy()
    if 'duration' not in c:f[:,1]=0.
    return f
def prep(d,tr,c):
    f=manual(d,c);mu=f[tr].mean(0);sd=f[tr].std(0);sd[sd<1e-8]=1;y=d['y'][tr];positive=(y[:,None]>=np.array([12,20])[None,:]).sum(0);assert np.all(positive>0)
    return dict(condition=c,mean=mu.tolist(),std=sd.tolist(),ym=float(y.mean()),ys=max(float(y.std()),1e-8),positive_weight=((len(tr)-positive)/positive).tolist())
def metric(y,p):
    e=np.abs(y-p);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),score=max(float(e.mean())/.9,float(np.quantile(e,.95))/3,float(e[y>=20].mean())/3))
def infer(net,f,ev,p):
    net.eval();out=[];aux=[]
    with torch.no_grad():
        for lo in range(0,len(ev),64):
            r,a=net.forward_all(torch.from_numpy(f[ev[lo:lo+64]]));out.extend(r.numpy().astype(float)*p['ys']+p['ym'])
            if a is not None:aux.extend(a.numpy().astype(float))
    return np.array(out),np.array(aux) if aux else None
def fit(d,tr,ev,c,epochs,seed,path,grid):
    path=Path(path);complete=path/'complete.json'
    if complete.exists():return read(complete)
    assert not(path/'context.json').exists();path.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);p=prep(d,tr,c);f=((manual(d,c)-p['mean'])/p['std']).astype(np.float32);auxiliary='_aux_' in c;net=Model(auxiliary);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,foreach=False);y=d['y'][tr];target=((y-p['ym'])/p['ys']).astype(np.float32);tail=16 if c.endswith('16') else 4;w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.)).astype(np.float32);aux=(y[:,None]>=np.array([12,20])[None,:]).astype(np.float32);positive=torch.tensor(p['positive_weight'],dtype=torch.float32);rows=[];start=time.perf_counter();save(path/'context.json',dict(condition=c,train=tr.tolist(),eval=ev.tolist(),train_ids=d['ids'][tr].tolist(),eval_ids=d['ids'][ev].tolist(),seed=seed,epochs=epochs,prep=p,protocol_sha256=sha(R/'protocol.json')))
    for epoch in range(1,epochs+1):
        net.train();order=np.random.default_rng(seed+epoch).permutation(len(tr))
        for lo in range(0,len(tr),64):
            ii=order[lo:lo+64];opt.zero_grad(set_to_none=True);r,a=net.forward_all(torch.from_numpy(f[tr[ii]]));loss=(nn.functional.smooth_l1_loss(r,torch.from_numpy(target[ii]),beta=.5,reduction='none')*torch.from_numpy(w[ii])).mean()
            if auxiliary:loss=loss+.1*nn.functional.binary_cross_entropy_with_logits(a,torch.from_numpy(aux[ii]),pos_weight=positive)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5,foreach=False);opt.step()
        if epoch in grid:
            pr,logits=infer(net,f,ev,p);model=path/f'epoch{epoch}.pt';torch.save(net.state_dict(),model);rows.append(dict(epoch=epoch,prediction=pr.tolist(),aux_logits=logits.tolist() if logits is not None else None,model=str(model),sha256=sha(model),metrics=metric(d['y'][ev],pr)));save(path/'curve.json',rows)
    out=dict(context=str(path/'context.json'),curve=rows,epochs=epochs,seed=seed,elapsed_seconds=time.perf_counter()-start);save(complete,out);return out
def data():
    original=np.load(B/'data.npz');native=np.load(B/'native-motion/data.npz');assert all(np.array_equal(original[k],native[k]) for k in ['ids','y','fold']);return dict(ids=original['ids'],y=original['y'],fold=original['fold'],F=native['F'])
def freeze():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);base=read(H/'norm-aux/protocol.json');p=dict(code_sha256=sha(Path(__file__)),wave_library_sha256=sha(H/'norm_aux.py'),native_library_sha256=sha(H/'native_motion.py'),block_library_sha256=sha(H/'bundle/train.py'),wave_protocol_sha256=sha(H/'norm-aux/protocol.json'),original_data_sha256=sha(B/'data.npz'),native_data_sha256=sha(B/'native-motion/data.npz'),conditions=CONDS,splits=base['splits'],design='Separatepredeclaredcompanionbeforeviewingnorm-auxouterresults;existing5conditionstudyunchanged. Same594/frozenfolds andsourceobservedspeed/coreduration. ForwardacceptsmanualFonly: speedalonewithdurationconstant0,orspeed+observedcoreduration. Encoderpoolsummaryconstant(mu0,sd1e-4) isintentionalablation,notusingmissingactualclipasnormaldata. Every594actualsourceclipremainsmeasurable. SameWaveModel6 initialization/manual16/head48->32->1/aux48->2/coreseedanddropoutRNG;unusedconvnotexecuted,so differencescanattributeaddedwave/duration/auxconditionswithmatchedpredictionheads.',training='Exactnorm-auxCPU2threads/seed20261106+fold100+inner1000,final+5000,AdamW.001/decay.01,batch64,SmoothL1beta.5,tail4or16,middlesqrt(tail),clip5,100innerpasses,grid10/25/50/100. Aux.1BCEatTUG>=12/>=20 withfit-trainingpositiveweights. Normalizeonlyfittrainingmanualcolumns/target. AggregateinnerOOFminmax3goals,earliesttie. Nooutputclipping/conditionselectiononoutertest.',verification='SourceIDs/labels/folds/rawspeed/windowdurationexact;pointtol1e-6absolute+relativeCPU64for300innergrid+25finalstatesandauxlogits;normalizers/auxweights/innerselection;independentmetrics/uncertainty andpairedwave-vsmanual comparisonafterbothstudiesfullyverified. Save100fits. Syntheticcoreinit/zero-waveablationequivalence+realinner5epochpilotmustpassbeforemain. Reuseddevelopmentfolds/severe9/singleinit;noRGB/MediaPipeclinicalvalidation.',expected_additional_GB=.1)
    if(R/'protocol.json').exists():assert read(R/'protocol.json')==p
    else:save(R/'protocol.json',p)
    return p
def pilot():
    p=freeze();torch.set_num_threads(2);d=data();ss=p['splits']['0']['inner'][0];tr=np.array(ss['train']);ev=np.array(ss['valid'][:32]);checks=[]
    # Exactly match the source head before learning, with no motion content.
    torch.manual_seed(20261106);net=Model(True);torch.manual_seed(20261106);wave=WaveModel(6,True);f=torch.randn(8,2);x=torch.zeros(8,7,208);net.train();wave.train();torch.manual_seed(991);a=net(f);torch.manual_seed(991);b=wave(x,f);assert torch.equal(a,b)
    for c in CONDS:
        rec=fit(d,tr,ev,c,5,20261106,O/'pilot'/c,[5]);pp=read(rec['context'])['prep'];f=((manual(d,c)-pp['mean'])/pp['std']).astype(np.float32).astype(float);m=Model('_aux_' in c).double();m.load_state_dict(torch.load(rec['curve'][0]['model'],map_location='cpu',weights_only=True));m.eval()
        with torch.no_grad():r,a=m.forward_all(torch.from_numpy(f[ev]));pr=r.numpy()*pp['ys']+pp['ym']
        assert np.allclose(pr,rec['curve'][0]['prediction'],atol=1e-6,rtol=1e-6)
        if a is not None:assert np.allclose(a.numpy(),rec['curve'][0]['aux_logits'],atol=1e-6,rtol=1e-6)
        checks.append(dict(condition=c,seconds=rec['elapsed_seconds'],max_cpu64_TUG_delta=float(np.max(np.abs(pr-rec['curve'][0]['prediction']))),outerpeople_excluded=True))
    save(R/'pilot-verification.json',dict(status='passed',checks=checks,matched_zero_wave_head_and_dropout_exact=True,protocol_sha256=sha(R/'protocol.json')))
def main():
    p=freeze();assert read(R/'pilot-verification.json')['status']=='passed' and read(R/'pilot-verification.json')['protocol_sha256']==sha(R/'protocol.json');torch.set_num_threads(2);d=data();start=time.perf_counter();done=0;save(R/'process.json',dict(pid=os.getpid(),status='running'))
    for c in CONDS:
      for fold in range(5):
        path=O/c/f'fold{fold}';complete=path/'fold.json'
        if complete.exists():done+=1;continue
        ss=p['splits'][str(fold)];tr=np.array(ss['train']);te=np.array(ss['test']);inner=[]
        for k,s in enumerate(ss['inner']):inner.append(fit(d,np.array(s['train']),np.array(s['valid']),c,100,20261106+fold*100+k*1000,path/f'inner{k}',GRID))
        scores=[]
        for j,epoch in enumerate(GRID):
            pred=np.zeros(len(tr))
            for r,s in zip(inner,ss['inner']):pred[[np.flatnonzero(tr==i)[0] for i in s['valid']]]=r['curve'][j]['prediction']
            scores.append(metric(d['y'][tr],pred))
        epoch=GRID[int(np.argmin([m['score'] for m in scores]))];final=fit(d,tr,te,c,epoch,20261106+fold*100+5000,path/'final',[epoch]);save(complete,dict(condition=c,fold=fold,inner=inner,inner_scores=scores,epoch=epoch,final=final,test_ids=d['ids'][te].tolist(),prediction=final['curve'][0]['prediction']));done+=1;save(R/'progress.json',dict(status='running',complete=done,expected=25,last=c+f'/fold{fold}',elapsed_seconds=time.perf_counter()-start));print(c,fold,round(time.perf_counter()-start,1),flush=True)
    save(R/'receipt.json',dict(status='complete_pending_independent_review',cases=25,protocol_sha256=sha(R/'protocol.json'),hashes={str(p):sha(p) for c in CONDS for p in (O/c).rglob('*') if p.is_file()}));save(R/'process.json',dict(pid=os.getpid(),status='complete'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','pilot','main']);a=ap.parse_args();{'freeze':freeze,'pilot':pilot,'main':main}[a.mode]()
