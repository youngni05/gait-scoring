"""Frozen HMR2 image-to-local-rotation inference; no SMPL mesh model needed."""
import os, sys, json, time, hashlib, types, importlib.util, ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'deps'))
import numpy as np
import torch
import cv2
import yaml
from yacs.config import CfgNode

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def modules(repo):
    # Skip package initializers which construct/render a licensed mesh. The
    # released backbone, decoder and rotation conversion themselves are unmodified.
    for name in ['hmr2','hmr2.models','hmr2.models.components','hmr2.models.heads','hmr2.utils']:
        m=types.ModuleType(name);m.__path__=[str(repo/Path(*name.split('.')))];sys.modules[name]=m
    from hmr2.models.backbones import create_backbone
    source=repo/'hmr2/models/heads/smpl_head.py'
    code=source.read_text();old='transformer_args = (transformer_args | dict(cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER))'
    assert code.count(old)==1
    # Python3.8 compatibility only: dictionary union and update are identical here.
    code=code.replace(old,'transformer_args.update(dict(cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER))')
    m=types.ModuleType('hmr2.models.heads.smpl_head');m.__package__='hmr2.models.heads'
    exec(compile(code,str(source),'exec'),m.__dict__);sys.modules[m.__name__]=m
    return create_backbone,m.build_smpl_head

def crop_tools(repo):
    # Execute only the official pure crop helpers, not training/dataset imports.
    names={'expand_to_aspect_ratio','rotate_2d','gen_trans_from_patch_cv','generate_image_patch_cv2'}
    tree=ast.parse((repo/'hmr2/datasets/utils.py').read_text())
    fns=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in fns}==names
    from typing import Optional,Tuple,List
    env=dict(np=np,cv2=cv2,Optional=Optional,Tuple=Tuple,List=List)
    exec(compile(ast.Module(body=fns,type_ignores=[]),'official_crop_helpers','exec'),env)
    return env

def main():
    mode=sys.argv[1];assert mode in ['pilot','main','cpu-smoke']
    small=mode!='main'
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.benchmark=False
    if mode!='cpu-smoke':assert torch.cuda.is_available()
    device='cpu' if mode=='cpu-smoke' else 'cuda';start=time.time();repo=HERE/'4D-Humans';out=HERE/mode;out.mkdir(exist_ok=True)
    cfgpath=next((HERE/'weights').rglob('model_config.yaml'))
    cfg=CfgNode(yaml.safe_load(cfgpath.read_text()))
    meanpath=next((HERE/'weights').rglob('smpl_mean_params.npz'));cfg.SMPL.MEAN_PARAMS=str(meanpath)
    cfg.MODEL.BBOX_SHAPE=[192,256]
    create_backbone,build_head=modules(repo)
    model=torch.nn.Module();model.backbone=create_backbone(cfg);model.smpl_head=build_head(cfg)
    ckpt=next((HERE/'weights').rglob('*.ckpt'))
    state=torch.load(str(ckpt),map_location='cpu',weights_only=False)['state_dict']
    selected={k:v for k,v in state.items() if k.startswith('backbone.') or k.startswith('smpl_head.')}
    model.load_state_dict(selected,strict=True);del state,selected
    model.eval().to(device)
    helper=crop_tools(repo)
    protocol=json.loads((HERE/'protocol.json').read_text());clips=json.loads((HERE/'clips.json').read_text())
    source_fingerprint={str(p.relative_to(HERE)):sha(p) for p in [HERE/'infer.py',HERE/'protocol.json',HERE/'clips.json',ckpt,cfgpath,meanpath]}
    source_fingerprint.update({str(p.relative_to(HERE)):sha(p) for p in (repo/'hmr2').rglob('*.py')})
    mean=np.array(cfg.MODEL.IMAGE_MEAN,dtype='float32')[None,None,:]*255
    std=np.array(cfg.MODEL.IMAGE_STD,dtype='float32')[None,None,:]*255
    checks=[];count=0
    for clip in clips[:1] if small else clips:
        folder=HERE/'clips'/clip['key'];assert sha(folder/'boxes.npz')==clip['boxes_sha256']
        boxes=np.load(folder/'boxes.npz')['boxes'];images=[]
        for k in range(150):
            p=folder/('%03d.jpg'%k);assert sha(p)==clip['frame_sha256'][k];images.append(cv2.imread(str(p)))
        for condition in protocol['conditions'][:1] if small else protocol['conditions']:
            key=clip['key']+'_'+condition['name'];rng=np.random.RandomState(condition['seed'])
            noise=np.clip(rng.normal(size=(150,3)),-3,3)*condition['sigma'];patches=[]
            for k,(im,box) in enumerate(zip(images,boxes)):
                center=(box[:2]+box[2:])/2;size=box[2:]-box[:2]
                center=center+noise[k,:2]*size[1];size=size*np.exp(noise[k,2])
                edge=helper['expand_to_aspect_ratio'](size,target_aspect_ratio=[192,256]).max()
                # Same anti-alias rule as official ViTDetDataset.
                if edge/256/2>1.1:
                    from skimage.filters import gaussian
                    im=gaussian(im,sigma=(edge/256/2-1)/2,channel_axis=2,preserve_range=True)
                patch,_=helper['generate_image_patch_cv2'](im,center[0],center[1],edge,edge,256,256,False,1.,0,border_mode=cv2.BORDER_CONSTANT)
                patches.append(((patch[:,:,::-1].astype('float32')-mean)/std).transpose(2,0,1).copy())
            patches=np.stack(patches[:16] if small else patches);rot=[]
            runstart=time.time()
            with torch.inference_mode():
                for k in range(0,len(patches),8):
                    batch=torch.from_numpy(patches[k:k+8]).to(device)
                    params,_,_=model.smpl_head(model.backbone(batch[:,:,:,32:-32]))
                    rot.append(params['body_pose'].cpu().numpy())
                if small:
                    batch=torch.from_numpy(patches[:8]).to(device)
                    replay=model.smpl_head(model.backbone(batch[:,:,:,32:-32]))[0]['body_pose'].cpu().numpy()
                    replay_error=float(abs(replay-rot[0]).max());assert replay_error<=1e-6
            r=np.concatenate(rot);eye=np.eye(3);orth=float(abs(r.transpose(0,1,3,2)@r-eye).max());det=float(abs(np.linalg.det(r)-1).max())
            assert r.shape==(len(patches),23,3,3) and np.isfinite(r).all() and orth<=1e-5 and det<=1e-5
            path=out/(key+'.npz');np.savez_compressed(path,rotations=r,noise=noise[:len(r)],fps=np.array(30))
            row=dict(key=key,frames=len(r),elapsed_seconds=time.time()-runstart,orthogonality_error=orth,determinant_error=det,sha256=sha(path));checks.append(row);count+=len(r)
            save(out/'progress.json',dict(mode=mode,completed=len(checks),expected=1 if small else 70,frames=count,last=row))
            print(json.dumps(row),flush=True)
    result=dict(passed=True,mode=mode,frames=count,outputs=checks,elapsed_seconds=time.time()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,source_sha256=source_fingerprint,rotation_only=True,mesh_generated=False)
    if small:result['repeat_max_error']=replay_error
    assert count==(16 if small else 10500)
    save(out/'complete.json',result);print('ROTATION_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
