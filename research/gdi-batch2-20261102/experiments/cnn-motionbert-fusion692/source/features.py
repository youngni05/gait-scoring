"""Frozen temporal feature cache. No GDI labels are read by extraction."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from lib.model.DSTformer import DSTformer
ROOT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(str(tmp),str(p))
def require(ok,message):
    if not ok:raise ValueError(message)
def verify_bundle(root):
    for name,digest in load(root/'manifest.json').items():require(sha(root/name)==digest,'Bundle changed: '+name)
def identity(root):
    return dict(bundle_sha256=sha(root/'manifest.json'),pose_sha256=sha(root/'motion_input.npz'),
        encoder_sha256=sha(root/'MB_lite.bin'),pool='joint_mean_only',dtype='float32',frames=124,features=512)

def validate_motion_inputs(root,d):
    with np.load(root/'motion_input.npz',allow_pickle=False) as z:
        n=len(d['y'])
        require(z['pose'].shape==(n,124,17,3) and z['pose'].dtype==np.float32,'Wrong MotionBERT input')
        require(z['pooled_reference'].shape==(n,512),'Wrong pooled reference')
        require(np.isfinite(z['pose']).all() and np.isfinite(z['pooled_reference']).all(),'Nonfinite pose input')
        require(z['frame_ids'].shape==(n,124) and (np.diff(z['frame_ids'],axis=1)==1).all(),'Nonconsecutive frame IDs')
        for key in ('person_id','video_id','fold','original_row_index'):
            require(np.array_equal(z[key],d[key]),'Motion/CNN record alignment mismatch: '+key)

def encoder(root,device):
    torch.set_num_threads(4);torch.manual_seed(1234);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device=='cuda':
        require(torch.cuda.is_available(),'CUDA unavailable; no CPU fallback')
        torch.empty(1,device=device);torch.cuda.synchronize()
    m=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    ck=torch.load(root/'MB_lite.bin',map_location='cpu',weights_only=True)
    m.load_state_dict({k[7:] if k.startswith('module.') else k:v for k,v in ck['model_pos'].items()},strict=True)
    m.requires_grad_(False);m.eval();return m.to(device)

def temporal(m,pose,device):
    require(pose.ndim==4 and pose.shape[1:]==(124,17,3),'Expected record,time124,joints17,xyconfidence')
    require(not m.training and not any(p.requires_grad for p in m.parameters()),'Encoder must remain frozen/eval')
    with torch.inference_mode():
        rep=m.get_representation(torch.from_numpy(pose).to(device))
        require(tuple(rep.shape)==(len(pose),124,17,512),'MotionBERT representation layout changed')
        result=rep.mean(dim=2).cpu().numpy().copy()
    require(result.dtype==np.float32 and np.isfinite(result).all(),'Nonfinite temporal features')
    return result

def content_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def read_cache(root,directory,nrecords):
    proof=load(directory/'complete.json')
    require(proof['identity']==identity(root),'Feature cache belongs to another input or encoder')
    require(proof['records']==nrecords and proof['real_GDI_optimizer_steps']==0,'Incomplete feature cache')
    require(sha(directory/'features.npy')==proof['features_sha256'],'Temporal cache corrupt')
    x=np.load(directory/'features.npy',allow_pickle=False,mmap_mode='r')
    require(x.shape==(nrecords,124,512) and x.dtype==np.float32 and np.isfinite(x).all(),'Wrong temporal cache')
    return x,proof['features_sha256']

def extract(root,directory,device,batch=8,limit_batches=None,model=None):
    verify_bundle(root);directory.mkdir(parents=True,exist_ok=True);expected=identity(root)
    with np.load(root/'motion_input.npz',allow_pickle=False) as z:pose=z['pose'].copy();pooled=z['pooled_reference'].copy()
    n=len(pose)
    if (directory/'complete.json').exists():
        read_cache(root,directory,n);print('TEMPORAL_FEATURE_CACHE_ALREADY_VERIFIED',flush=True);return
    statepath=directory/'state.json';target=directory/'features.npy'
    if statepath.exists():
        state=load(statepath);require(state['identity']==expected,'Cannot resume cache after bundle/input changes')
        mm=np.load(target,allow_pickle=False,mmap_mode='r+')
        require(mm.shape==(n,124,512) and mm.dtype==np.float32,'Wrong partial cache')
        cursor=0
        for chunk in state['chunks']:
            require(chunk['start']==cursor and chunk['end']>cursor,'Broken feature-cache coverage')
            require(content_hash(mm[cursor:chunk['end']])==chunk['sha256'],'Partial feature cache corrupt')
            cursor=chunk['end']
        require(cursor==state['completed'] and cursor<=n,'Wrong cache resume offset')
    else:
        require(not target.exists(),'Cache array exists without resume metadata')
        mm=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(n,124,512))
        state={'identity':expected,'completed':0,'chunks':[],'max_pooled_reference_error':0.0}
        save(statepath,state)
    m=encoder(root,device) if model is None else model
    started=time.perf_counter();batches=0
    for start in range(state['completed'],n,batch):
        end=min(start+batch,n);out=temporal(m,pose[start:end],device)
        error=float(np.max(np.abs(out.mean(axis=1)-pooled[start:end])))
        require(error<=1e-4,'Temporal time-mean does not reproduce prior fixed pooled features: '+str(error))
        mm[start:end]=out;mm.flush()
        state['chunks'].append({'start':start,'end':end,'sha256':content_hash(out)})
        state['completed']=end;state['max_pooled_reference_error']=max(state['max_pooled_reference_error'],error)
        save(statepath,state);batches+=1
        print(json.dumps({'temporal_features_done':end,'records':n,'seconds':time.perf_counter()-started}),flush=True)
        if limit_batches is not None and batches>=limit_batches:return
    del mm
    save(directory/'complete.json',dict(identity=expected,records=n,features_sha256=sha(target),real_GDI_optimizer_steps=0,
        encoder_frozen=True,time_axis_preserved=True,max_pooled_reference_error=state['max_pooled_reference_error'],device=device,torch=str(torch.__version__)))
    print('TEMPORAL_FEATURE_CACHE_FINISHED',flush=True)

def check(root,out,device):
    verify_bundle(root)
    with np.load(root/'motion_input.npz',allow_pickle=False) as z:
        selected=np.array([0,len(z['pose'])-1]);pose=z['pose'][selected].copy();pooled=z['pooled_reference'][selected].copy()
    m=encoder(root,device);before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    start=time.perf_counter()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    f=temporal(m,pose,device);single=np.concatenate([temporal(m,row[None],device) for row in pose])
    err=float(np.max(np.abs(f.mean(axis=1)-pooled)));batcherr=float(np.max(np.abs(f-single)))
    require(err<=1e-4 and batcherr<=1e-4,'Feature identity or batch alignment mismatch')
    require(all(torch.equal(v.detach().cpu(),before[k]) for k,v in m.state_dict().items()),'Frozen encoder changed')
    require(float(np.std(f,axis=1).mean())>0,'Time variation unexpectedly lost')
    result=dict(passed=True,actual_records=selected.tolist(),shape=list(f.shape),pooled_reference_max_error=err,batch_max_error=batcherr,
        encoder_unchanged=True,requires_grad_parameters=0,real_GDI_optimizer_steps=0,device=device,seconds=time.perf_counter()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,torch=str(torch.__version__),python=platform.python_version())
    save(out/'feature_check.json',result);print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','extract']);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--output',type=Path);p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();require(a.batch_size>0,'Positive batch size required')
    if a.command=='check':check(ROOT,a.output or ROOT/'checks/features',a.device)
    else:extract(ROOT,a.output or ROOT/'cache/temporal',a.device,a.batch_size)
if __name__=='__main__':main()
