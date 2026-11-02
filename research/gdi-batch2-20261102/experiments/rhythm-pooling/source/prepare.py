import sys,json,hashlib,time,importlib.util
from pathlib import Path
import numpy as np
from temporal_features import features,FEATURE_NAMES,checks
sys.stdout.reconfigure(encoding='utf-8')
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;OLD=H/'studies/20261027-window-ablation';SHORT=H/'jobs/native124-cache-v1';DATA=R/'outputs/gdi-ridge-experiment-20261016';MB=R/'outputs/motionbert-smoke-20261015'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    started=time.perf_counter();assert not (S/'preparation.json').exists();save('synthetic_checks.json',checks())
    source=load(OLD/'sources.json')['sha256'];sealed={a['path']:a['sha256'] for a in load(OLD/'study_receipt.json')['artifacts']}
    assert sha(OLD/'raw_features.npz')==sealed[(OLD/'raw_features.npz').relative_to(H).as_posix()]
    with np.load(OLD/'raw_features.npz',allow_pickle=False) as z:
        inds=np.flatnonzero(z['split']=='train');vid=z['video_ids'][inds].copy();people=z['patient_ids'][inds].copy()
        baseline={m:z[m+'_local'][inds].copy() for m in ('front','middle','last')}
    assert len(inds)==203 and len(set(people))==175
    assert not set(people)&set(load(H/'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json')['original_test_patient_ids'])
    tasks=load(H/'control/desktop-sessions/20261026T100042Z-9d41cd/native124_tasks.json')['tasks']
    assert [tasks[i]['video_id'] for i in inds]==list(vid)
    members=load(SHORT/'zip_members.json');assert sha(SHORT/'zip_members.json')==load(SHORT/'state.json')['member_index_sha256']
    sys.path.insert(0,str(H));from native124_cache import read_poses
    from short_window_contract import prepare_candidate
    spec=importlib.util.spec_from_file_location('arc',R/'tools/stanford-sample-check/inspect_archive.py');arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)
    stat={'size':arc.ARCHIVE.stat().st_size,'mtime_ns':arc.ARCHIVE.stat().st_mtime_ns};assert stat==load(SHORT/'config.json')['archive_stat']
    tracked={}
    for p in (Path(__file__),S/'temporal_features.py',S/'plan.md',OLD/'raw_features.npz',SHORT/'zip_members.json',H/'native124_cache.py',H/'short_window_contract.py',MB/'assets/MB_lite.bin',MB/'vendor/lib/model/DSTformer.py'):
        d=sha(p);tracked[p.relative_to(R).as_posix()]=d
        if p.relative_to(R).as_posix() in source:assert d==source[p.relative_to(R).as_posix()]
    for p in (MB/'vendor',MB/'deps',DATA/'deps'):sys.path.insert(0,str(p))
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    cp=torch.load(MB/'assets/MB_lite.bin',map_location='cpu',weights_only=True);model.load_state_dict({k.removeprefix('module.'):v for k,v in cp['model_pos'].items()},strict=True);model.eval();model.requires_grad_(False)
    out={m:{'valid_pool':[],'rhythm':[]} for m in baseline};diag=[];replay=[];pooldiff=[];nopoints=0;time_rhythm=0.;time_encoder=0.
    (S/'records').mkdir(exist_ok=True)
    for j,i in enumerate(inds):
        motion,ids,proof=read_poses(arc.ARCHIVE,members[tasks[i]['raw_folder']]);long=prepare_candidate(motion,ids,243)
        p=DATA/'windows'/(str(vid[j])+'.npz');assert sha(p)==source[p.relative_to(R).as_posix()];tracked[p.relative_to(R).as_posix()]=sha(p)
        with np.load(p,allow_pickle=False) as z:
            np.testing.assert_allclose(long['input'],z['input'],rtol=1e-5,atol=2e-6);np.testing.assert_array_equal(long['frame_ids'],z['frame_ids'])
        vals={};info={}
        for m,offset in (('front',0),('middle',59),('last',119)):
            k=long['start_index']+offset;d=prepare_candidate(motion[k:k+124],ids[k:k+124],124);x=d['input']
            np.testing.assert_array_equal(d['frame_ids'],long['frame_ids'][offset:offset+124]);assert (x[:,1:7,2]>0).all()
            if m=='front':
                p=SHORT/f'records/row_{i:04d}.npz';assert sha(p)==source[p.relative_to(R).as_posix()];tracked[p.relative_to(R).as_posix()]=sha(p)
                with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(x,z['input124'])
            ts=time.perf_counter();rh,quality=features(x);time_rhythm+=time.perf_counter()-ts
            ts=time.perf_counter()
            with torch.inference_mode():
                rep=model.get_representation(torch.from_numpy(x[None]));original=rep.mean(dim=(1,2)).numpy()[0]
                mask=torch.from_numpy((x[...,2]>0).astype(np.float32))[None,...,None]
                pooled=((rep*mask).sum(dim=(1,2))/mask.sum(dim=(1,2))).numpy()[0]
            time_encoder+=time.perf_counter()-ts
            delta=float(np.max(np.abs(original-baseline[m][j])));assert delta<=1e-5;replay.append(delta)
            if j<3:
                alt=rep.numpy()[0][x[...,2]>0].astype(np.float64).mean(axis=0);err=float(np.max(np.abs(alt-pooled)));assert err<1e-5;pooldiff.append(err)
            if (x[...,2]>0).all():np.testing.assert_allclose(pooled,original,rtol=0,atol=1e-6);nopoints+=1
            vals[m+'_valid_pool']=pooled;vals[m+'_rhythm']=rh
            out[m]['valid_pool'].append(pooled);out[m]['rhythm'].append(rh)
            info[m]=dict(missing_cells=int((x[...,2]==0).sum()),**quality)
        dest=S/f'records/row_{j:04d}.npz';np.savez_compressed(dest,**vals)
        save(f'records/row_{j:04d}.json',dict(original_index=int(i),sha256=sha(dest),raw_proofs=proof,details=info));diag.append(info)
        if (j+1)%40==0:print(json.dumps(dict(completed=j+1,seconds=time.perf_counter()-started)),flush=True)
    arr={m+'_'+v:np.stack(out[m][v]).astype(np.float64) for m in out for v in ('valid_pool','rhythm')}
    arr.update({m+'_baseline':b for m,b in baseline.items()});np.savez_compressed(S/'features.npz',**arr,patient_ids=people,video_ids=vid,original_indices=inds,feature_names=np.array(FEATURE_NAMES))
    assert all(sha(R/p)==v for p,v in tracked.items());assert stat=={'size':arc.ARCHIVE.stat().st_size,'mtime_ns':arc.ARCHIVE.stat().st_mtime_ns}
    save('sources.json',dict(sha256=tracked,archive_stat=stat));save('diagnostics.json',diag)
    result=dict(verified=True,people=175,records=203,positions=3,encoder_calls=609,baseline_replay_max=max(replay),independent_pool_max=max(pooldiff),no_missing_windows=nopoints,rhythm_dimensions=10,rhythm_cpu_seconds=time_rhythm,encoder_and_pool_cpu_seconds=time_encoder,elapsed_seconds=time.perf_counter()-started,labels_used=False,validation_records_computed=0,features_sha256=sha(S/'features.npz'))
    save('preparation.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
