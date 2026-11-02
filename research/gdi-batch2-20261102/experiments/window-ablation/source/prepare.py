"""Approved 2x3 ablation, fixed existing development rows, no labels."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parents[1]
ROOT=H.parent
sys.path.insert(0,str(H))
from short_window_contract import source_functions
from temporal_readiness import DATA, MB, EXPECTED_WEIGHT_SHA256
DEV=H/'jobs/development-baseline/runs/20261025T044558572569Z-71adb0'
SHORT=H/'jobs/native124-cache-v1'
OLD=H/'control/desktop-sessions/20261026T130506Z-e876e7'
VARIANTS=('front_global','middle_global','last_global','middle_local','last_local')

def require(ok,message):
    if not ok: raise ValueError(message)

def load(p): return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def context():
    with np.load(DEV/'development_packet.npz',allow_pickle=False) as z:
        videos,people,split=(z[k].copy() for k in ('video_ids','patient_ids','split'))
    require(set(split)=={'train','validation'} and len(set(videos))==243,'Wrong cohort')
    require((sum(split=='train'),sum(split=='validation'),len(set(people[split=='train'])),len(set(people[split=='validation'])))==(203,40,175,37),'Changed split')
    require(not set(people[split=='train']) & set(people[split=='validation']),'Person overlap')
    exclusions=load(H/'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json')
    require(not set(people)&set(exclusions['original_test_patient_ids']),'Original test included')
    with np.load(SHORT/'development_features124.npz',allow_pickle=False) as z:
        for key,value in (('video_ids',videos),('patient_ids',people),('split',split)):
            np.testing.assert_array_equal(z[key],value)
        features=z['X'].copy()
    dev_receipt=load(DEV/'receipt.json')
    require(sha(DEV/'result.json')==dev_receipt['result_sha256'],'Development receipt changed')
    require(sha(DEV/'development_packet.npz')==load(DEV/'result.json')['artifacts'][0]['sha256'],'Packet changed')
    receipt=load(OLD/'receipt.json')
    for path in (SHORT/'development_features124.npz',OLD/'length_models.npz',OLD/'length_comparison.json',OLD/'length_internal_oof.csv'):
        entry=next(a for a in receipt['artifacts'] if a['path']==path.relative_to(H).as_posix())
        require(sha(path)==entry['sha256'],'Prior result changed')
    paths=[Path(__file__),HERE/'plan.md',DEV/'development_packet.npz',SHORT/'development_features124.npz',
           MB/'assets/MB_lite.bin',H/'short_window_contract.py',ROOT/'tools/stanford-sample-check/prepare_ridge_data.py',
           MB/'assets/mmpose_inference_3d.py']
    paths+=sorted((MB/'vendor/lib/model').glob('*.py'))
    paths+=[DATA/'windows'/(str(v)+'.npz') for v in videos]
    paths+=[SHORT/f'records/row_{i:04d}.npz' for i in range(243)]
    fingerprints={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    require(fingerprints[(MB/'assets/MB_lite.bin').relative_to(ROOT).as_posix()]==EXPECTED_WEIGHT_SHA256,'Weights changed')
    config={'sha256':fingerprints,'variants':VARIANTS,'offsets':{'front':0,'middle':59,'last':119},
            'frames':124,'threads':4,'maxlen':243,'labels_used':False,'new_people':0}
    dest=HERE/'sources.json'
    if dest.exists(): require(load(dest)==json.loads(json.dumps(config)),'Source/config drift')
    else: save(dest,config)
    return videos,people,split,features

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    require((HERE/'plan.md').is_file(),'Missing plan')
    start=time.monotonic()
    videos,people,split,old_features=context()
    indexes=np.flatnonzero(split=='train')[:3].tolist() if args.pilot else list(range(243))
    if not args.pilot: require(load(HERE/'pilot.json')['verified'],'Pilot required')
    for path in (MB/'vendor',MB/'deps',DATA/'deps'): sys.path.insert(0,str(path))
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,
                    mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    cp=torch.load(MB/'assets/MB_lite.bin',map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in cp['model_pos'].items()},strict=True)
    model.requires_grad_(False);model.eval()
    normalize,_,_=source_functions()
    computed=reused=0
    for index in indexes:
        dest=HERE/f'records/row_{index:04d}.npz'; proof=dest.with_suffix('.json')
        if dest.exists():
            require(proof.is_file() and sha(dest)==load(proof)['sha256'],'Partial or changed record')
            reused+=1;continue
        require(not proof.exists(),'Partial proof')
        with np.load(DATA/'windows'/(str(videos[index])+'.npz'),allow_pickle=False) as z:
            long=z['input'].copy();frames=z['frame_ids'].copy()
        with np.load(SHORT/f'records/row_{index:04d}.npz',allow_pickle=False) as z:
            front_local=z['input124'].copy();reference=z['feature'].copy()
            np.testing.assert_array_equal(z['frame_ids124'],frames[:124])
        require(long.shape==(243,17,3) and long.dtype==np.float32 and np.isfinite(long).all(),'Input shape/value')
        require((np.diff(frames)==1).all() and (long[:,1:7,2]>0).all(),'Observation rule changed')
        reconstructed=normalize(long[:124])
        np.testing.assert_allclose(reconstructed,front_local,rtol=1e-5,atol=2e-6)
        np.testing.assert_array_equal(reference,old_features[index])
        inputs={}
        for name,offset in (('front',0),('middle',59),('last',119)):
            x=long[offset:offset+124].copy();inputs[name+'_global']=x
            if name!='front': inputs[name+'_local']=normalize(x)
        outputs={}; details={}
        with torch.inference_mode():
            # Pilot checks chained normalization against a truly raw-prepared front input.
            if args.pilot:
                p=model.get_representation(torch.from_numpy(reconstructed[None])).mean(dim=(1,2)).numpy()[0].copy()
                np.testing.assert_allclose(p,reference,rtol=1e-5,atol=2e-6)
                details['front_reconstruction_feature_max_abs_difference']=float(np.max(np.abs(p-reference)))
            for name in VARIANTS:
                x=inputs[name]
                require(np.isfinite(x).all() and x.shape==(124,17,3),'Invalid ablation input')
                offset=0 if name.startswith('front') else 59 if name.startswith('middle') else 119
                np.testing.assert_array_equal(x[...,2],long[offset:offset+124,...,2])
                rep=model.get_representation(torch.from_numpy(x[None]))
                require(tuple(rep.shape)==(1,124,17,512) and torch.isfinite(rep).all().item(),'Invalid feature')
                outputs[name]=rep.mean(dim=(1,2)).numpy()[0].copy()
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,**outputs)
        save(proof,{'index':index,'sha256':sha(dest),'source_window_sha256':sha(DATA/'windows'/(str(videos[index])+'.npz')),
                    'front_reconstruction_input_max_abs_difference':float(np.max(np.abs(reconstructed-front_local))),
                    'front_local_feature_reused':True,'confidence_preserved':True,'details':details})
        computed+=1
        if computed%20==0: print(json.dumps({'computed':computed,'reused':reused,'total':len(indexes),'elapsed':time.monotonic()-start}),flush=True)
    result={'verified':True,'records':len(indexes),'new_records_computed':computed,'reused':reused,
            'new_variants':len(VARIANTS),'labels_used':False,'new_people':0,'elapsed_seconds':time.monotonic()-start}
    if args.pilot: save(HERE/'pilot.json',result)
    else:
        all_features={name:[] for name in VARIANTS}
        for index in range(243):
            p=HERE/f'records/row_{index:04d}.npz'
            require(sha(p)==load(p.with_suffix('.json'))['sha256'],'Feature digest mismatch')
            with np.load(p,allow_pickle=False) as z:
                for name in VARIANTS:
                    require(z[name].shape==(512,) and np.isfinite(z[name]).all(),'Invalid saved feature')
                    all_features[name].append(z[name].copy())
        np.savez_compressed(HERE/'features.npz',**{k:np.stack(v).astype(np.float64) for k,v in all_features.items()},
                            front_local=old_features,video_ids=videos,patient_ids=people,split=split)
        result['consolidated_sha256']=sha(HERE/'features.npz')
        save(HERE/'preparation.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
