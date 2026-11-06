"""Local Intel GPU feasibility pilot. No remote process or data upload."""
from pathlib import Path
import sys,os,time,json,hashlib,argparse,random
import numpy as np,torch,torch_directml
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');sys.path.insert(0,str(H/'bundle'))
from train import Model,CONDS,preprocess
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def run(mode):
    torch.set_num_threads(2);random.seed(20261104);np.random.seed(20261104);torch.manual_seed(20261104);device=torch_directml.device();a=torch.tensor([1.,2.]).to(device);assert float(a.sum().cpu())==3.
    out=B/'local-directml-pilot'/mode;out.mkdir(parents=True,exist_ok=True);meta=H/'local-directml';meta.mkdir(exist_ok=True);runs=[]
    if mode=='real':
        p=json.loads((H/'protocol.json').read_text());assert sha(B/'data.npz')==p['data_sha256'];data=np.load(B/'data.npz');split=p['splits']['0']['inner'][0];tr=np.array(split['train']);ev=np.array(split['valid'][:32]);assert not set(tr)&set(ev);assert not set(tr)&set(p['splits']['0']['test']);assert not set(ev)&set(p['splits']['0']['test'])
    else:
        rng=np.random.default_rng(20261104);data=dict(X=rng.normal(size=(96,24,128)),F=rng.normal(size=(96,213)),y=rng.normal(12,3,96),ids=np.arange(96));data['F'][:,0]=rng.uniform(.3,1.5,96);tr=np.arange(64);ev=np.arange(64,96)
    for ci,c in enumerate(CONDS):
        torch.manual_seed(20261104+ci);prep=preprocess(data,tr,c);F=data['F'][:,:1] if prep['nfeatures']==1 else data['F'];filled=np.where(np.isfinite(F),F,np.array(prep['median']));features=(filled-np.array(prep['mean']))/np.array(prep['std']);y=(data['y']-prep['target_mean'])/prep['target_std'];model=Model(c,prep['nfeatures']).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};start=time.perf_counter();losses=[]
        for epoch in range(5 if mode=='real' else 3):
            model.train();perm=np.random.default_rng(20261104+epoch).permutation(tr);ls=0.
            for lo in range(0,len(tr),64):
                ix=perm[lo:lo+64];x=torch.from_numpy(data['X'][ix].astype(np.float32)).to(device);f=torch.from_numpy(features[ix].astype(np.float32)).to(device);target=torch.from_numpy(y[ix].astype(np.float32)).to(device);weight=torch.from_numpy(np.where(data['y'][ix]>=20,4.,np.where(data['y'][ix]>=12,2.,1.)).astype(np.float32)).to(device);opt.zero_grad(set_to_none=True);pred=model(x,f);loss=(torch.nn.functional.smooth_l1_loss(pred,target,beta=.5,reduction='none')*weight).mean();assert bool(torch.isfinite(loss).cpu());loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,foreach=False);opt.step();ls+=float(loss.detach().cpu())*len(ix)
            losses.append(ls/len(tr))
        model.eval()
        with torch.no_grad():prediction=(model(torch.from_numpy(data['X'][ev].astype(np.float32)).to(device),torch.from_numpy(features[ev].astype(np.float32)).to(device)).cpu().numpy().astype(float)*prep['target_std']+prep['target_mean'])
        elapsed=time.perf_counter()-start;state={k:v.detach().cpu() for k,v in model.state_dict().items()};assert any(not torch.equal(initial[k],state[k]) for k in state);path=out/(c+'.pt');torch.save(state,path);cpu=Model(c,prep['nfeatures']).double();cpu.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));cpu.eval()
        with torch.no_grad():rp=cpu(torch.from_numpy(data['X'][ev]).double(),torch.from_numpy(features[ev]).double()).numpy()*prep['target_std']+prep['target_mean']
        assert np.isfinite(prediction).all() and np.allclose(prediction,rp,atol=1e-6,rtol=1e-6);runs.append(dict(condition=c,epochs=len(losses),train_people=len(tr),validation_people=len(ev),elapsed_seconds=elapsed,training_losses=losses,cpu64_max_delta=float(np.max(np.abs(prediction-rp))),checkpoint_sha256=sha(path),preprocessing=prep));print(mode,c,round(elapsed,2),'seconds',runs[-1]['cpu64_max_delta'],flush=True)
    save(meta/(mode+'-verification.json'),dict(status='passed',device=str(device),device_name=torch_directml.device_name(0) if hasattr(torch_directml,'device_name') else 'Intel Arc140V from OS inventory',torch=torch.__version__,numpy=np.__version__,code_sha256=sha(Path(__file__)),model_library_sha256=sha(H/'bundle/train.py'),mode=mode,runs=runs,source_data_sha256=sha(B/'data.npz') if mode=='real' else None,scope='Pilot numerical/hardware/preprocessing only. Real mode uses first inner-training split and32inner-validation people, never outer test. No model selection or clinical MAE claim. Operator CPU fallback warnings retained in log. Real cohort data remains local.'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['synthetic','real']);run(ap.parse_args().mode)
