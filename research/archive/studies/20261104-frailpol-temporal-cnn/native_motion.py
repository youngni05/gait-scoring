"""Matched masked temporal models: phase-normalized versus native-time/amplitude IMU."""
from pathlib import Path
import sys,json,hashlib,zipfile,time,argparse,os
import numpy as np,torch
try:import torch_directml
except ImportError:torch_directml=None
from torch import nn
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'native-motion';O=B/'native-motion';S=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');SRC=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103');sys.path.insert(0,str(S));sys.path.insert(0,str(H/'bundle'))
from train import Block
CONDS=['phase_tail4','native_tail4','native_tail16'];GRID=[10,25,50,100]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8');q.replace(p)
def sequence(raw,w):
    ref=raw['sacrum back'].PacketCounter.iloc[w['core_start']:w['core_end']].to_numpy();values=[]
    for sensor in ['left ankle','right ankle','sacrum back']:
        df=raw[sensor].set_index('PacketCounter').loc[ref]
        for kind in ['Acc','Gyr']:
            a=df[[kind+'_'+axis for axis in 'XYZ']].to_numpy(float)
            if kind=='Gyr':a=np.rad2deg(a)
            values.extend(np.column_stack([a,np.linalg.norm(a,axis=1)]).T)
    a=np.array(values);t=np.arange(len(ref))/100.;grid=np.arange(208)/50.;valid=grid<=t[-1]+1e-10;native=np.zeros((25,208));native[-1]=valid;phase=np.ones((25,208))
    for j,v in enumerate(a):
        native[j,valid]=np.interp(grid[valid],t,v);v=np.interp(np.linspace(0,1,208),np.linspace(0,1,len(v)),v);phase[j]=(v-v.mean())/max(v.std(),1e-8)
    assert native.shape==phase.shape==(25,208) and np.isfinite(native).all() and np.isfinite(phase).all() and valid.sum()>1
    return native,phase,len(ref)/100.
def prepare():
    import pandas as pd,io,re
    def filemap(z):
        result={}
        for name in z.namelist():
            match=re.match(r'(\d+)-(.+)\.csv$',Path(name).name,re.I)
            if match:result.setdefault(int(match[1]),{})[match[2].lower().replace('_',' ')]=name
        return result
    def rawread(z,m,pid):return {sensor:pd.read_csv(io.BytesIO(z.read(name))) for sensor,name in m[pid].items()}
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);base=read(H/'local-directml/base-protocol.json');d=np.load(B/'data.npz');features=read('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103/round05/features.json');byid={r['id']:r for r in features};fp=dict(code_sha256=sha(Path(__file__)),source_data_sha256=sha(B/'data.npz'),source_raw_sha256=sha(SRC/'raw/IMU_signals.zip'),source_windows_sha256=sha('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103/round05/features.json'),people=594,source='Same frozen bilateral core including originalanchor fallback. No expansion into chair/turn/global TUG duration. Raw units and100Hz source time retained;50Hz observed samples at0,.02,..4.14seconds in208slots. Some originalcore length4.15s preserved,not truncated to4s. Zero padding always paired withfalsemask. 24acc/gyrXYZ+norm channels plusvalidtime mask. Gyr rad/s converteddegrees/s; not video.',phase_control='Same raw core,208phasepoints eachchannelper-windowzscore,allvalidmask; matchednewarchitecture. Bothconditions receive speed+observedcoreduration. Nativechannelstandardization usesfit-trainingvalidframes only; no per-person amplitude normalization.',checks='All594 sequences rereadexact;first12 perturboutsidecorecannotchangeinputs; no outsidecoretime interpolation. Preserve IDs/y/fold fromverifieddata;sourcewindowmeaning unchanged.',expected_additional_GB=.12)
    if(R/'input-protocol.json').exists():assert read(R/'input-protocol.json')==fp
    else:save(R/'input-protocol.json',fp)
    if(R/'input-verification.json').exists():assert read(R/'input-verification.json')['data_sha256']==sha(O/'data.npz');return
    native=[];phase=[];duration=[]
    with zipfile.ZipFile(SRC/'raw/IMU_signals.zip') as z:
      mapping=filemap(z)
      for n,pid in enumerate(d['ids']):
        w=byid[int(pid)]['window'];raw=rawread(z,mapping,int(pid));a,b,t=sequence(raw,w);aa,bb,tt=sequence(rawread(z,mapping,int(pid)),w);assert np.array_equal(a,aa) and np.array_equal(b,bb) and t==tt
        if n<12:
            ref=raw['sacrum back'].PacketCounter.iloc[w['core_start']:w['core_end']].to_numpy();alter={s:df.copy() for s,df in raw.items()}
            for df in alter.values():df.loc[~df.PacketCounter.isin(ref),df.columns!='PacketCounter']=99999.
            aa,bb,tt=sequence(alter,w);assert np.array_equal(a,aa) and np.array_equal(b,bb) and t==tt
        native.append(a);phase.append(b);duration.append(t)
    np.savez_compressed(O/'data.npz',native=np.array(native),phase=np.array(phase),F=np.column_stack([d['F'][:,0],duration]),ids=d['ids'],y=d['y'],fold=d['fold']);save(R/'input-verification.json',dict(status='passed',data_sha256=sha(O/'data.npz'),input_protocol_sha256=sha(R/'input-protocol.json'),all594double_replay=True,outsidecore12invariance=True,shape=[594,25,208],observed_duration_range=[min(duration),max(duration)]))
    proto=dict(owner_thread=base['owner_thread'],code_sha256=sha(Path(__file__)),block_library_sha256=sha(H/'bundle/train.py'),data_sha256=sha(O/'data.npz'),input_protocol_sha256=sha(R/'input-protocol.json'),splits=base['splits'],conditions=CONDS,architecture='SharedmaskedTCN25->16,three residualBlock16,dilation1/2/4,kernel5,GroupNorm4,SiLU. Poolmean+std onlyvalidtimepositions; speed+observedcore duration2->16manualbranch; head48->32->1,dropout.15. Duration andmaskderivedsolelyselectedcore. Allconditionssamearchitecture.',training='AdamW.001,weight_decay.01,foreachFalse,SmoothL1beta.5 standardizedTUG;tail4or16,12..20sqrt(tail),<12weight1,batch64,gradclip5;100epochs eachinner,epochgrid10/25/50/100 chosenminmaxthreegoalswithinoutertrain,firsttie. CPUinit/permutationfixedseed20261105+fold100+conditionindex+inner1000,final+5000. No seedselection or clipping. DirectML may useCPUfallback.',verification='Real inner-only5epochpilot percondition,weightchange,save/CPU64replay beforemain; full expected45innerfits+15final,180innergrid+15finalcheckpointreplay;trainingonlymanual/channel/targetnormalization,exactpeopleandepochselection,independentmetricsandpaireduncertainty.',limits='594frozen/reuseddevelopmentpeople,severe9. IMUonly,notRGB/MediaPipe;singleinit;matchedphasecontrolisolatesinputhypothesisfromnewarchitecture;differentfrompreviousCNN,do notattributeallpreviousvsnewdifferenceonlytoinput. Allhuman data remainlocal.')
    save(R/'protocol.json',proto);print('native inputs verified',flush=True)
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Sequential(nn.Conv1d(25,16,1),Block(16,1),Block(16,2),Block(16,4));self.manual=nn.Sequential(nn.Linear(2,16),nn.SiLU(),nn.Dropout(.15));self.head=nn.Sequential(nn.Linear(48,32),nn.SiLU(),nn.Dropout(.15),nn.Linear(32,1))
    def forward(self,x,f):
        mask=x[:,-1:];z=self.conv(x);n=mask.sum(-1).clamp_min(1);mu=(z*mask).sum(-1)/n;sd=(((z-mu[:,:,None])**2*mask).sum(-1)/n+1e-8).sqrt();return self.head(torch.cat([self.manual(f),mu,sd],1)).squeeze(-1)
def preprocess(data,tr,c):
    raw=data['phase' if c.startswith('phase') else 'native'];mask=raw[tr,-1:,:];n=mask.sum();mean=(raw[tr,:24]*mask).sum((0,2))/n;std=np.sqrt((((raw[tr,:24]-mean[None,:,None])**2)*mask).sum((0,2))/n);std[std<1e-8]=1;F=data['F'][tr];fm=F.mean(0);fs=F.std(0);fs[fs<1e-8]=1
    return dict(channel_mean=mean.tolist(),channel_std=std.tolist(),manual_mean=fm.tolist(),manual_std=fs.tolist(),target_mean=float(data['y'][tr].mean()),target_std=max(float(data['y'][tr].std()),1e-8),condition=c)
def arrays(data,p):
    raw=data['phase' if p['condition'].startswith('phase') else 'native'];x=raw.copy();x[:,:24]=np.where(raw[:,-1:]>0,(raw[:,:24]-np.array(p['channel_mean'])[None,:,None])/np.array(p['channel_std'])[None,:,None],0);f=(data['F']-p['manual_mean'])/p['manual_std'];return x.astype(np.float32),f.astype(np.float32)
def infer(net,x,f,idx,p,device):
    net.eval();out=[]
    with torch.no_grad():
        for lo in range(0,len(idx),64):
            ii=idx[lo:lo+64];out.extend(net(torch.from_numpy(x[ii]).to(device),torch.from_numpy(f[ii]).to(device)).cpu().numpy().astype(float)*p['target_std']+p['target_mean'])
    return np.array(out)
def metric(y,p):
    e=np.abs(p-y);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),score=max(float(e.mean())/.9,float(np.quantile(e,.95))/3,float(e[y>=20].mean())/3))
def fit(data,tr,ev,c,epochs,seed,path,grid):
    path=Path(path);complete=path/'complete.json'
    if complete.exists():return read(complete)
    assert not(path/'context.json').exists(),'Preserve incomplete fit; no silentoverwrite';path.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);p=preprocess(data,tr,c);x,f=arrays(data,p);device=torch_directml.device();net=Model().to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,foreach=False);yy=data['y'][tr];tail=16 if c.endswith('16') else 4;weight=np.where(yy>=20,tail,np.where(yy>=12,np.sqrt(tail),1.)).astype(np.float32);target=((yy-p['target_mean'])/p['target_std']).astype(np.float32);rows=[];start=time.perf_counter();initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};save(path/'context.json',dict(condition=c,train=tr.tolist(),eval=ev.tolist(),train_ids=data['ids'][tr].tolist(),eval_ids=data['ids'][ev].tolist(),prep=p,seed=seed,epochs=epochs,protocol_sha256=sha(R/'protocol.json')))
    for ep in range(1,epochs+1):
        net.train();order=np.random.default_rng(seed+ep).permutation(len(tr))
        for lo in range(0,len(tr),64):
            ii=order[lo:lo+64];ix=tr[ii];opt.zero_grad(set_to_none=True);pred=net(torch.from_numpy(x[ix]).to(device),torch.from_numpy(f[ix]).to(device));t=torch.from_numpy(target[ii]).to(device);w=torch.from_numpy(weight[ii]).to(device);loss=(nn.functional.smooth_l1_loss(pred,t,beta=.5,reduction='none')*w).mean();assert bool(torch.isfinite(loss).cpu());loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5,foreach=False);opt.step()
        if ep in grid:
            pr=infer(net,x,f,ev,p,device);model=path/f'epoch{ep}.pt';torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},model);assert any(not torch.equal(initial[k],v.detach().cpu()) for k,v in net.state_dict().items());rows.append(dict(epoch=ep,prediction=pr.tolist(),model=str(model),sha256=sha(model),metrics=metric(data['y'][ev],pr)));save(path/'curve.json',rows)
    r=dict(context=str(path/'context.json'),curve=rows,elapsed_seconds=time.perf_counter()-start,epochs=epochs,seed=seed);save(complete,r);print(c,path.parent.name,path.name,round(r['elapsed_seconds'],1),flush=True);return r
def pilot():
    proto=read(R/'protocol.json');assert proto['code_sha256']==sha(Path(__file__)) and proto['data_sha256']==sha(O/'data.npz');d=np.load(O/'data.npz');torch.set_num_threads(2);ss=proto['splits']['0']['inner'][0];tr=np.array(ss['train']);ev=np.array(ss['valid'][:32]);records=[]
    for ci,c in enumerate(CONDS):
        r=fit(d,tr,ev,c,5,20261105+ci,O/'pilot'/c,[5]);p=read(r['context'])['prep'];x,f=arrays(d,p);row=r['curve'][0];net=Model().double();net.load_state_dict(torch.load(row['model'],map_location='cpu',weights_only=True));net.eval()
        with torch.no_grad():pr=net(torch.from_numpy(x[ev].astype(float)),torch.from_numpy(f[ev].astype(float))).numpy()*p['target_std']+p['target_mean']
        assert np.allclose(pr,row['prediction'],atol=1e-6,rtol=1e-6);records.append(dict(condition=c,fit=r,cpu64_max_delta=float(np.max(np.abs(pr-row['prediction']))),inner_only=True))
    estimated={r['condition']:r['fit']['elapsed_seconds']/5*100*3*5*1.18 for r in records};save(R/'pilot-verification.json',dict(status='passed',records=records,data_sha256=sha(O/'data.npz'),code_sha256=sha(Path(__file__)),total_estimated_seconds_by_model=estimated,monitor_interval_seconds=.5*min(estimated.values()),monitor_rule='Halfshortestunfinishedmodeltotalestimate,notremainingtime. LocaltrainingnoSlurmallocation; schooltransferstillpending.'));print('native real pilot passed',estimated,flush=True)
def main():
    proto=read(R/'protocol.json');pilotrec=read(R/'pilot-verification.json');assert pilotrec['status']=='passed' and proto['code_sha256']==pilotrec['code_sha256']==sha(Path(__file__)) and proto['data_sha256']==sha(O/'data.npz');d=np.load(O/'data.npz');torch.set_num_threads(2);done=0;start=time.perf_counter()
    save(R/'process.json',dict(pid=os.getpid(),status='running'))
    for ci,c in enumerate(CONDS):
      for fold in range(5):
        path=O/c/f'fold{fold}';complete=path/'fold.json'
        if complete.exists():done+=1;continue
        ss=proto['splits'][str(fold)];tr=np.array(ss['train']);te=np.array(ss['test']);inner=[]
        for k,s in enumerate(ss['inner']):inner.append(fit(d,np.array(s['train']),np.array(s['valid']),c,100,20261105+fold*100+ci+k*1000,path/f'inner{k}',GRID))
        scores=[]
        for j,ep in enumerate(GRID):
            p=np.zeros(len(tr))
            for r,s in zip(inner,ss['inner']):p[[np.flatnonzero(tr==i)[0] for i in s['valid']]]=r['curve'][j]['prediction']
            scores.append(metric(d['y'][tr],p))
        ep=GRID[int(np.argmin([m['score'] for m in scores]))];final=fit(d,tr,te,c,ep,20261105+fold*100+ci+5000,path/'final',[ep]);save(complete,dict(condition=c,fold=fold,inner=inner,inner_scores=scores,epoch=ep,final=final,test_ids=d['ids'][te].tolist(),prediction=final['curve'][0]['prediction']));done+=1;save(R/'progress.json',dict(status='running',completed_cases=done,expected_cases=15,last=c+f'/fold{fold}',elapsed_seconds=time.perf_counter()-start))
    save(R/'receipt.json',dict(status='complete_pending_independent_review',cases=15,protocol_sha256=sha(R/'protocol.json'),hashes={str(p):sha(p) for c in CONDS for p in (O/c).rglob('*') if p.is_file()}));save(R/'process.json',dict(pid=os.getpid(),status='complete'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);a=ap.parse_args();{'prepare':prepare,'pilot':pilot,'main':main}[a.mode]()
