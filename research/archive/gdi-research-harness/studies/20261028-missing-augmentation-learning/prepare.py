import hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;PREV=H/'studies/20261028-rhythm-versus-missing-pooling';SHORT=H/'jobs/native124-cache-v1';MB=R/'outputs/motionbert-smoke-20261015';DATA=R/'outputs/gdi-ridge-experiment-20261016'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
GROUPS=((9,10),(13,),(16,))
def hide(a,index,kind):
    assert a.shape==(124,17,3) and np.isfinite(a).all()
    length=31 if kind=='training' else 62;seed=2026102801 if kind=='training' else 2026102802
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(index)]));group=int(rng.integers(0,3));start=int(rng.integers(0,125-length))
    selected=np.zeros((124,17),dtype=bool);selected[start:start+length,list(GROUPS[group])]=True
    active=selected&(a[...,2]>0);b=a.copy();b[active]=0
    np.testing.assert_array_equal(a[~active],b[~active]);np.testing.assert_array_equal(a[:,1:7],b[:,1:7]);np.testing.assert_array_equal(a[a[...,2]==0],b[a[...,2]==0])
    assert np.all(b[active]==0)
    return b,dict(group=group,joints=list(GROUPS[group]),start=start,length=length,new_missing_cells=int(active.sum()),selected_cells=int(selected.sum()),original_missing_cells=int((a[...,2]==0).sum()))
def checks():
    a=np.random.default_rng(5).uniform(-1,1,(124,17,3)).astype(np.float32);a[...,2]=1;a[:,9]=[-1,1,0]
    for kind in ('training','stress'):
        b,p=hide(a,3,kind);bb,pp=hide(a,3,kind);np.testing.assert_array_equal(b,bb);assert p==pp
    a[:,[9,10,13,16],2]=0
    for kind in ('training','stress'):
        b,p=hide(a,8,kind);np.testing.assert_array_equal(a,b);assert p['new_missing_cells']==0
    return dict(nonselected_and_lower_body_unchanged=True,preexisting_missing_unchanged=True,deterministic_masks=True,all_target_joints_already_missing_is_identity=True)
def main():
    st=time.perf_counter();assert not (S/'preparation.json').exists();save('mask_checks.json',checks())
    sealed={p['path']:p['sha256'] for p in load(PREV/'study_receipt.json')['artifacts']}
    for n in ('features.npz','sources.json'):
        p=PREV/n;assert sha(p)==sealed[p.relative_to(H).as_posix()]
    with np.load(PREV/'features.npz',allow_pickle=False) as z:
        ids,vid,indices=(z[k].copy() for k in ('patient_ids','video_ids','original_indices'));original={p:z[p+'_baseline'].copy() for p in ('front','middle','last')}
    assert len(vid)==203 and len(set(ids))==175
    source=load(PREV/'sources.json')['sha256'];tracked={}
    for p in (Path(__file__),S/'plan.md',PREV/'features.npz',MB/'assets/MB_lite.bin',MB/'vendor/lib/model/DSTformer.py'):
        d=sha(p);tracked[p.relative_to(R).as_posix()]=d
        if p.relative_to(R).as_posix() in source:assert d==source[p.relative_to(R).as_posix()]
    for p in (MB/'vendor',MB/'deps',DATA/'deps'):sys.path.insert(0,str(p))
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    cp=torch.load(MB/'assets/MB_lite.bin',map_location='cpu',weights_only=True);model.load_state_dict({k.removeprefix('module.'):v for k,v in cp['model_pos'].items()},strict=True);model.eval();model.requires_grad_(False)
    def feature(x):
        with torch.inference_mode():return model.get_representation(torch.from_numpy(x[None])).mean((1,2)).numpy()[0].copy()
    new={'training':[],'stress':[]};diag=[];maxreplay=0.;calls=0;(S/'records').mkdir(exist_ok=True)
    for j,i in enumerate(indices):
        p=SHORT/f'records/row_{i:04d}.npz';d=sha(p);assert d==source[p.relative_to(R).as_posix()];tracked[p.relative_to(R).as_posix()]=d
        with np.load(p,allow_pickle=False) as z:
            a=z['input124'].copy();np.testing.assert_array_equal(z['feature'],original['front'][j]);assert (np.diff(z['frame_ids124'])==1).all()
        if j<3:
            got=feature(a);calls+=1;err=float(np.max(np.abs(got-original['front'][j])));assert err<1e-5;maxreplay=max(maxreplay,err)
        vals={};info={}
        for kind in new:
            changed,details=hide(a,int(i),kind);f=feature(changed) if details['new_missing_cells'] else original['front'][j].copy();calls+=int(details['new_missing_cells']>0)
            assert f.shape==(512,) and np.isfinite(f).all();new[kind].append(f);vals[kind]=f;info[kind]=details
        path=S/f'records/row_{j:04d}.npz';np.savez_compressed(path,**vals);save(f'records/row_{j:04d}.json',dict(original_index=int(i),sha256=sha(path),masks=info));diag.append(info)
        if (j+1)%40==0:print(json.dumps(dict(completed=j+1,seconds=time.perf_counter()-st)),flush=True)
    np.savez_compressed(S/'features.npz',**original,training_aug=np.stack(new['training']).astype(np.float64),stress=np.stack(new['stress']).astype(np.float64),patient_ids=ids,video_ids=vid,original_indices=indices)
    assert all(sha(R/p)==d for p,d in tracked.items());save('sources.json',tracked);save('mask_manifest.json',diag)
    summary={kind:dict(records=203,changed_records=sum(d[kind]['new_missing_cells']>0 for d in diag),new_missing_cells=sum(d[kind]['new_missing_cells'] for d in diag),group_counts={str(g):sum(d[kind]['group']==g for d in diag) for g in range(3)}) for kind in new}
    result=dict(verified=True,people=175,records=203,encoder_calls=calls,original_replay_max=maxreplay,labels_used=False,validation37_used=False,features_sha256=sha(S/'features.npz'),masks=summary,seconds=time.perf_counter()-st)
    save('preparation.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
