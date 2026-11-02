"""Recovery: construct each local window directly from raw coordinates."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
from prepare import HERE,H,ROOT,DATA,MB,SHORT,VARIANTS,context,load,save,sha,require
sys.path.insert(0,str(H))
from native124_cache import read_poses
from short_window_contract import prepare_candidate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    start=time.monotonic();videos,people,split,features=context()
    require((HERE/'recovery_plan.md').is_file(),'Missing recovery plan')
    tasks=load(H/'control/desktop-sessions/20261026T100042Z-9d41cd/native124_tasks.json')['tasks']
    members=load(SHORT/'zip_members.json')
    require([t['video_id'] for t in tasks]==list(videos),'Raw task order mismatch')
    require(sha(SHORT/'zip_members.json')==load(SHORT/'state.json')['member_index_sha256'],'Raw index changed')
    spec=importlib.util.spec_from_file_location('archive_reader',ROOT/'tools/stanford-sample-check/inspect_archive.py')
    arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)
    st=arc.ARCHIVE.stat()
    stat={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    require(stat==load(SHORT/'config.json')['archive_stat'],'Archive metadata drift')
    tracked=[Path(__file__),HERE/'recovery_plan.md',HERE/'sources.json',SHORT/'zip_members.json',
             H/'native124_cache.py',H/'short_window_contract.py',HERE/'prepare.py']
    source={'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in tracked},'archive_stat':stat,
            'normalization':'raw window local, stored243 global','pilot_limit':3,'original_rules_unchanged':True}
    source_file=HERE/'raw_sources.json'
    if source_file.exists(): require(load(source_file)==source,'Recovery source changed')
    else:save(source_file,source)
    indexes=np.flatnonzero(split=='train')[:3].tolist() if args.pilot else list(range(243))
    if not args.pilot: require(load(HERE/'raw_pilot.json')['verified'],'Raw pilot required')
    for p in (MB/'vendor',MB/'deps',DATA/'deps'):sys.path.insert(0,str(p))
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,
                    mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    cp=torch.load(MB/'assets/MB_lite.bin',map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in cp['model_pos'].items()},strict=True)
    model.requires_grad_(False);model.eval()
    computed=reused=global_reused=inferences=0
    for i in indexes:
        dest=HERE/f'raw_records/row_{i:04d}.npz';proof_path=dest.with_suffix('.json')
        if dest.exists():
            require(proof_path.is_file() and sha(dest)==load(proof_path)['sha256'],'Partial or modified recovered record')
            reused+=1;continue
        require(not proof_path.exists(),'Partial raw proof')
        task=tasks[i];poses,ids,raw_proof=read_poses(arc.ARCHIVE,members[task['raw_folder']])
        long=prepare_candidate(poses,ids,243);front=prepare_candidate(poses,ids,124)
        with np.load(DATA/'windows'/(str(videos[i])+'.npz'),allow_pickle=False) as z:
            original=z['input'].copy();original_ids=z['frame_ids'].copy()
        require(np.allclose(original,long['input'],rtol=1e-5,atol=2e-6),'Original243 reconstruction mismatch')
        require(np.array_equal(original_ids,long['frame_ids']),'Original frame IDs changed')
        with np.load(SHORT/f'records/row_{i:04d}.npz',allow_pickle=False) as z:
            require(np.array_equal(front['input'],z['input124']),'Canonical raw124 input changed')
            require(np.array_equal(z['feature'],features[i]),'Canonical124 feature changed')
        windows={};offsets={'front':0,'middle':59,'last':119}
        for name,offset in offsets.items():
            start_index=long['start_index']+offset
            local=prepare_candidate(poses[start_index:start_index+124],ids[start_index:start_index+124],124)
            require(np.array_equal(local['frame_ids'],original_ids[offset:offset+124]),'Window position changed')
            require(np.array_equal(local['input'][...,2],original[offset:offset+124,...,2]),'Confidence changed')
            if name!='front':windows[name+'_local']=local['input']
            windows[name+'_global']=original[offset:offset+124].copy()
        output={};old=HERE/f'records/row_{i:04d}.npz'
        if old.exists():
            require(sha(old)==load(old.with_suffix('.json'))['sha256'],'Original temporary record changed')
            with np.load(old,allow_pickle=False) as z:
                for name in ('front_global','middle_global','last_global'):output[name]=z[name].copy()
            global_reused+=1
        with torch.inference_mode():
            for name in VARIANTS:
                if name in output:continue
                rep=model.get_representation(torch.from_numpy(windows[name][None]))
                require(tuple(rep.shape)==(1,124,17,512) and torch.isfinite(rep).all().item(),'Invalid recovered feature')
                output[name]=rep.mean(dim=(1,2)).numpy()[0].copy();inferences+=1
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,**output)
        raw_path=HERE/f'raw_member_proofs/row_{i:04d}.json';save(raw_path,raw_proof)
        save(proof_path,{'index':i,'sha256':sha(dest),'raw_proof_sha256':sha(raw_path),'raw_json_checked':len(raw_proof),
              'original243_max_abs_difference':float(np.max(np.abs(original-long['input']))),
              'canonical_raw124_exactly_reproduced':True,'confidence_preserved':True,
              'global_features_reused':old.exists(),'old_temporary_sha256':sha(old) if old.exists() else None})
        computed+=1
        if computed%20==0:print(json.dumps({'raw_computed':computed,'reused':reused,'global_reused':global_reused,'total':len(indexes),'elapsed':time.monotonic()-start}),flush=True)
    require(arc.ARCHIVE.stat().st_size==st.st_size and arc.ARCHIVE.stat().st_mtime_ns==st.st_mtime_ns,'Archive metadata changed')
    result={'verified':True,'records':len(indexes),'computed':computed,'reused':reused,'global_feature_records_reused':global_reused,
            'new_inferences':inferences,'labels_used':False,'raw_local_windows':True,'elapsed_seconds':time.monotonic()-start}
    if args.pilot:save(HERE/'raw_pilot.json',result)
    else:
        arrays={name:[] for name in VARIANTS};raw_count=0
        for i in range(243):
            p=HERE/f'raw_records/row_{i:04d}.npz';proof=load(p.with_suffix('.json'))
            require(sha(p)==proof['sha256'],'Recovered feature digest mismatch')
            rp=HERE/f'raw_member_proofs/row_{i:04d}.json'
            require(sha(rp)==proof['raw_proof_sha256'],'Raw proof changed')
            raw_count+=proof['raw_json_checked']
            with np.load(p,allow_pickle=False) as z:
                for name in VARIANTS:
                    require(z[name].shape==(512,) and np.isfinite(z[name]).all(),'Invalid feature table row')
                    arrays[name].append(z[name].copy())
        np.savez_compressed(HERE/'raw_features.npz',**{k:np.stack(v).astype(np.float64) for k,v in arrays.items()},
                            front_local=features,video_ids=videos,patient_ids=people,split=split)
        result['consolidated_sha256']=sha(HERE/'raw_features.npz');result['raw_json_checked']=raw_count
        save(HERE/'raw_preparation.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
