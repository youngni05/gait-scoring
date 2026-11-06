"""Real TUG training, entirely local: CPU controls and Intel DirectML motion models."""
from pathlib import Path
import sys,os,json,time,hashlib,argparse,random
import numpy as np,torch,torch_directml
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');LOCAL=H/'local-directml';OUT=B/'local-main';sys.path.insert(0,str(H/'bundle'))
from train import Model,CONDS,preprocess
GRID=[10,25,50,100]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8');temp.replace(p)
def metric(y,p):
    e=np.abs(y-p);m=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()));m['score']=max(m['mae']/.9,m['p95']/3,m['slow_mae']/3);return m
def prepare():
    base=read(H/'protocol.json');syn=read(LOCAL/'synthetic-verification.json');real=read(LOCAL/'real-verification.json');assert syn['status']==real['status']=='passed' and real['source_data_sha256']==base['data_sha256']==sha(B/'data.npz');assert syn['model_library_sha256']==real['model_library_sha256']==sha(H/'bundle/train.py')
    protocol=dict(owner_thread=base['owner_thread'],created=time.time(),experiment='frailpol-local-temporal-20261104',base_protocol_sha256=sha(H/'protocol.json'),data_sha256=base['data_sha256'],model_library_sha256=sha(H/'bundle/train.py'),code_sha256=sha(Path(__file__)),real_pilot_sha256=sha(LOCAL/'real-verification.json'),conditions=CONDS,epoch_grid=GRID,seed_base=20261104,backend_by_condition={c:'directml' if c.startswith('motion') else 'cpu' for c in CONDS},splits=base['splits'],goal=base['goal'],training='Same594people/bilateralcore<=4s/raw24x128andF213. Architecture/source from verified pilot. AdamW lr.001,weight_decay.01,foreachFalse,SmoothL1 beta.5 in training-only standardized TUG; label weights1/<12,2/12..20,4/>=20;batch64,clipnorm5;100epochs allinnerfits. Fixed CPU initialization and CPU batch permutations derived from fitseed+epoch. No seed selection. Some DirectML AdamW ops use CPU fallback, logged. Backend RNG bitwise training determinism not asserted.',selection='Three inner fits, each100epochs and checkpoints10/25/50/100. Aggregate OOF predictions in outer-training IDorder and minimize max(MAE/.9,p95/3,slowMAE/3),earliestepoch tie. Refit on outertrain selectedepoch. No outer labels in preprocessing/training/epochselection.',verification='Save every inner grid checkpoint+normalizers+IDs+predictions and every final checkpoint. Same-backend reload replay duringfit; independent CPU64 replay and training-only normalization reconstruction for all240innergrid+20final checkpoints, exactsplits/epochchoice, metrics and pairedpersonbootstrap. Allhuman data remain local. No remote calls.',limits='Reused development folds;singleinitialization;no clinical/RGB/MediaPipe accuracy proof. Raw signals are phase-resampled and per-window zscored; native-time/amplitude input is a separate future hypothesis, not silently substituted. Motion+features includes measured timing/amplitude descriptors; motion+speed cannot independently recover absolute cadence from phase time.',estimated_models=4,expected_inner_fits=60,expected_final_fits=20,expected_saved_checkpoints=260)
    path=LOCAL/'main-protocol.json'
    if path.exists():
        old=read(path);assert old['code_sha256']==protocol['code_sha256'] and old['data_sha256']==protocol['data_sha256'] and old['model_library_sha256']==protocol['model_library_sha256']
    else:save(path,protocol)
    OUT.mkdir(exist_ok=True);return read(path)
def arrays(data,idx,prep):
    F=data['F'][:,:1] if prep['nfeatures']==1 else data['F'];a=np.where(np.isfinite(F[idx]),F[idx],np.array(prep['median']));a=(a-np.array(prep['mean']))/np.array(prep['std']);return torch.from_numpy(data['X'][idx].astype(np.float32)),torch.from_numpy(a.astype(np.float32))
def infer(model,data,idx,prep,device):
    model.eval();out=[]
    with torch.no_grad():
        for lo in range(0,len(idx),64):
            x,f=arrays(data,idx[lo:lo+64],prep);raw=model(x.to(device),f.to(device)).cpu().numpy().astype(float);out.extend(raw*prep['target_std']+prep['target_mean'])
    return np.array(out,float)
def fit(data,tr,ev,c,epochs,directory,seed_value,grid,role):
    directory=Path(directory);complete=directory/'fit-complete.json'
    if complete.exists():
        old=read(complete);assert old['epochs']==epochs and old['seed']==seed_value and old['condition']==c;return old
    assert not(directory/'context.json').exists(),'Incomplete fit present; preserve and inspect before retry instead of silently overwriting'
    directory.mkdir(parents=True,exist_ok=True);random.seed(seed_value);np.random.seed(seed_value);torch.manual_seed(seed_value);device=torch_directml.device() if c.startswith('motion') else torch.device('cpu');prep=preprocess(data,tr,c);model=Model(c,prep['nfeatures']).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False);X,F=arrays(data,tr,prep);yy=data['y'][tr];target=torch.from_numpy(((yy-prep['target_mean'])/prep['target_std']).astype(np.float32));weight=torch.from_numpy(np.where(yy>=20,4.,np.where(yy>=12,2.,1.)).astype(np.float32));rows=[];start=time.perf_counter()
    ctx=dict(condition=c,seed=seed_value,role=role,epochs=epochs,device=str(device),preprocessing=prep,train_indices=tr.tolist(),eval_indices=ev.tolist(),train_ids=data['ids'][tr].tolist(),eval_ids=data['ids'][ev].tolist(),protocol_sha256=sha(LOCAL/'main-protocol.json'),data_sha256=sha(B/'data.npz'));save(directory/'context.json',ctx)
    for epoch in range(1,epochs+1):
        model.train();order=np.random.default_rng(seed_value+epoch).permutation(len(tr));loss_sum=0.
        for lo in range(0,len(tr),64):
            ii=torch.from_numpy(order[lo:lo+64].astype(np.int64));x=X[ii].to(device);f=F[ii].to(device);t=target[ii].to(device);w=weight[ii].to(device);opt.zero_grad(set_to_none=True);p=model(x,f);loss=(torch.nn.functional.smooth_l1_loss(p,t,beta=.5,reduction='none')*w).mean();assert bool(torch.isfinite(loss).cpu());loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,foreach=False);opt.step();loss_sum+=float(loss.detach().cpu())*len(ii)
        if epoch in grid:
            prediction=infer(model,data,ev,prep,device);path=directory/f'epoch-{epoch:03d}.pt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path);rng=torch.get_rng_state();reload_model=Model(c,prep['nfeatures']).to(device);reload_model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));replayed=infer(reload_model,data,ev,prep,device);torch.set_rng_state(rng);assert np.allclose(prediction,replayed,atol=1e-6,rtol=1e-6);del reload_model
            rows.append(dict(epoch=epoch,prediction=prediction.tolist(),checkpoint=str(path),checkpoint_sha256=sha(path),metrics=metric(data['y'][ev],prediction),training_loss=loss_sum/len(tr),elapsed_seconds=time.perf_counter()-start));save(directory/'curve.json',rows);print(c,directory.parent.name,directory.name,epoch,round(time.perf_counter()-start,1),flush=True)
    rec=dict(condition=c,seed=seed_value,epochs=epochs,context=str(directory/'context.json'),curve=str(directory/'curve.json'),prediction=rows[-1]['prediction'],checkpoint=rows[-1]['checkpoint'],checkpoint_sha256=rows[-1]['checkpoint_sha256'],elapsed_seconds=time.perf_counter()-start);save(complete,rec);return rec
def worker(which):
    proto=prepare();torch.set_num_threads(2);data=np.load(B/'data.npz');conditions=[c for c in CONDS if c.startswith('motion')==(which=='motion')];save(LOCAL/(which+'-process.json'),dict(pid=os.getpid(),conditions=conditions,started=time.time(),status='running'));done=0
    for c in conditions:
      ci=CONDS.index(c)
      for f in range(5):
        directory=OUT/c/f'fold{f}';complete=directory/'complete.json'
        if complete.exists():done+=1;continue
        split=proto['splits'][str(f)];tr=np.array(split['train']);te=np.array(split['test']);records=[]
        for k,ss in enumerate(split['inner']):records.append(fit(data,np.array(ss['train']),np.array(ss['valid']),c,100,directory/f'inner{k}',20261104+f*100+ci+k*1000,GRID,'inner_validation'))
        scores=[]
        for epoch in GRID:
            p=np.zeros(len(tr));seen=[]
            for rec,ss in zip(records,split['inner']):
                r=[r for r in read(rec['curve']) if r['epoch']==epoch][0];ix=[int(np.flatnonzero(tr==i)[0]) for i in ss['valid']];p[ix]=r['prediction'];seen+=ss['valid']
            assert len(seen)==len(tr) and set(seen)==set(tr);scores.append(metric(data['y'][tr],p))
        selected=GRID[int(np.argmin([m['score'] for m in scores]))];final=fit(data,tr,te,c,selected,directory/'final',20261104+f*100+ci+5000,[selected],'outer_test');save(complete,dict(condition=c,fold=f,inner=records,inner_scores=scores,epoch_grid=GRID,selected_epoch=selected,final=final,test_ids=data['ids'][te].tolist(),prediction=final['prediction'],protocol_sha256=sha(LOCAL/'main-protocol.json')));done+=1;save(LOCAL/(which+'-progress.json'),dict(status='running',completed_cases=done,expected_cases=10,last=c+f'/fold{f}'))
    save(LOCAL/(which+'-receipt.json'),dict(status='complete_pending_independent_review',conditions=conditions,cases=done,protocol_sha256=sha(LOCAL/'main-protocol.json'),hashes={str(p):sha(p) for c in conditions for p in (OUT/c).rglob('*') if p.is_file()}));save(LOCAL/(which+'-process.json'),dict(pid=os.getpid(),conditions=conditions,status='complete'));print(which,'local training complete',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','controls','motion']);a=ap.parse_args();prepare() if a.mode=='prepare' else worker(a.mode)
