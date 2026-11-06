import json,hashlib,sys,time
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;OLD=H/'studies/20261028-window-cause-controls';MB=R/'outputs/motionbert-smoke-20261015';DATA=R/'outputs/gdi-ridge-experiment-20261016'
sys.stdout.reconfigure(encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def main():
    start=time.perf_counter();assert not (S/'features.npz').exists();manifest=load(S/'cycle_manifest.json');ready=load(S/'readiness.json');assert ready['eligible_records']>0
    with np.load(S/'identities.npz',allow_pickle=False) as z:inds=z['eligible_rows'].copy();ids=z['patient_ids'][inds].copy();vid=z['video_ids'][inds].copy()
    with np.load(OLD/'features.npz',allow_pickle=False) as z:base={p:z[p][inds].copy() for p in ['front','last']}
    assert sha(OLD/'features.npz')==load(S/'sources.json')[(OLD/'features.npz').relative_to(R).as_posix()]
    tracked={p.relative_to(R).as_posix():sha(p) for p in [Path(__file__),S/'plan.md',S/'cycles.py',S/'cycle_manifest.json',S/'identities.npz',S/'readiness.json',MB/'assets/MB_lite.bin',MB/'vendor/lib/model/DSTformer.py']}
    oldsources=load(OLD/'sources.json')['sha256']
    for p in [MB/'assets/MB_lite.bin',MB/'vendor/lib/model/DSTformer.py']:assert sha(p)==oldsources[p.relative_to(R).as_posix()]
    for p in [MB/'vendor',MB/'deps',DATA/'deps']:sys.path.insert(0,str(p))
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    cp=torch.load(MB/'assets/MB_lite.bin',map_location='cpu',weights_only=True);model.load_state_dict({k.removeprefix('module.'):v for k,v in cp['model_pos'].items()},strict=True);model.eval();model.requires_grad_(False)
    def encode(q):
        with torch.inference_mode():return model.get_representation(torch.from_numpy(q[None])).mean((1,2)).numpy()[0].copy()
    out={k:[] for k in ['control_front','control_last','aligned_front','aligned_last']};maxreplay=0.;calls=0
    for j,i in enumerate(inds):
        path=S/f'inputs/row_{i:04d}.npz';assert sha(path)==manifest[i]['input_sha256'];tracked[path.relative_to(R).as_posix()]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            for k in out:
                q=z[k];assert q.shape==(124,17,3) and np.isfinite(q).all();out[k].append(encode(q));calls+=1
        if j<3:
            p=OLD/f'records/row_{i:04d}.npz';assert sha(p)==load(S/'sources.json')[p.relative_to(R).as_posix()]
            with np.load(p,allow_pickle=False) as z:
                for pos in ['front','last']:
                    err=float(np.max(np.abs(encode(z[pos+'_input'])-base[pos][j])));assert err<1e-5;maxreplay=max(maxreplay,err);calls+=1
        if (j+1)%20==0:print(json.dumps({'completed':j+1,'seconds':time.perf_counter()-start}),flush=True)
    arrays={k:np.stack(v).astype(float) for k,v in out.items()};assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(S/'features.npz',**arrays,original_front=base['front'],original_last=base['last'],patient_ids=ids,video_ids=vid,source_rows=inds)
    assert all(sha(R/p)==h for p,h in tracked.items());save('inference_sources.json',tracked)
    result={'verified':True,'people':len(set(ids)),'records':len(inds),'encoder_calls':calls,'baseline_replay_max':maxreplay,'new_model_fits':0,'features_sha256':sha(S/'features.npz'),'seconds':time.perf_counter()-start};save('inference.json',result);print(json.dumps(result))
if __name__=='__main__':main()
