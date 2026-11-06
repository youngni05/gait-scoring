import sys,json,hashlib,time,importlib.util
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;PREV=H/'studies/20261028-rhythm-versus-missing-pooling';BASE=H/'studies/20261028-missing-augmentation-learning';SHORT=H/'jobs/native124-cache-v1';MB=R/'outputs/motionbert-smoke-20261015';DATA=R/'outputs/gdi-ridge-experiment-20261016'
sys.stdout.reconfigure(encoding='utf-8');sys.path.insert(0,str(H))
from short_window_contract import prepare_candidate,source_functions
from native124_cache import read_poses
normalize,convert,mapping=source_functions()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def knee(x):
    vals=[]
    for a,b,c in [(1,2,3),(4,5,6)]:
        u=x[:,a,:2].astype(float)-x[:,b,:2];v=x[:,c,:2].astype(float)-x[:,b,:2];den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1);assert (den>0).all();vals.append(np.degrees(np.arccos(np.clip((u*v).sum(1)/den,-1,1))))
    return np.stack(vals,1)
def trajectory(x):
    q=x[:,:,:2].astype(float);length=np.stack([np.linalg.norm(q[:,a]-q[:,b],axis=1) for a,b in [(1,2),(2,3),(4,5),(5,6)]],1);assert (length>0).all()
    scale=np.median(length,axis=1);smooth=np.array([np.median(scale[max(0,i-15):min(len(scale),i+16)]) for i in range(len(scale))]);return q[:,0:1],smooth[:,None,None]
def transplant(source,donor,kind):
    rs,ss=trajectory(source);rd,sd=trajectory(donor);out=source.astype(float).copy();q=out[:,:,:2]
    if kind=='translation':out[:,:,:2]=q-rs+rd
    elif kind=='scale':out[:,:,:2]=(q-rs)*(sd/ss)+rs
    elif kind=='both':out[:,:,:2]=(q-rs)*(sd/ss)+rd
    else:raise ValueError(kind)
    angle=float(np.max(np.abs(knee(out)-knee(source))));assert angle<1e-3
    result=normalize(out);original=normalize(source);missing=source[:,:,2]==0;result[missing,:2]=original[missing,:2]
    np.testing.assert_array_equal(result[:,:,2],original[:,:,2]);np.testing.assert_array_equal(result[missing],original[missing]);assert np.isfinite(result).all()
    return result,angle

def checks():
    rng=np.random.default_rng(17);x=rng.uniform(10,30,(124,17,3));x[:,:,2]=.7;x[:,9]=[0,0,0]
    for kind in ['translation','scale','both']:
        got,_=transplant(x,x,kind);np.testing.assert_allclose(got,normalize(x),atol=2e-6,rtol=0)
    y=x.copy();y[:,:,:2]=x[:,:,:2]*2.3+[31,-12];obs=x[:,:,2]>0
    np.testing.assert_allclose(normalize(y)[obs],normalize(x)[obs],atol=2e-6,rtol=0)
    got,err=transplant(x,y,'both');np.testing.assert_allclose(got,normalize(x),atol=2e-6,rtol=0)
    z=x.copy();z[:,:,:2]*=np.linspace(.8,1.2,124)[:,None,None];z[:,:,0]+=np.linspace(-5,10,124)[:,None]
    got,err=transplant(x,z,'both');assert np.max(np.abs(got-normalize(x)))>.01
    return dict(self_transplant_identity=True,constant_transform_invariance_observed=True,missing_coordinates_preserved=True,time_varying_transform_changes_input=True,angle_preserved=True)
def main():
    st=time.perf_counter();assert not (S/'features.npz').exists();save('synthetic_checks.json',checks());(S/'records').mkdir(exist_ok=True)
    seals={a['path']:a['sha256'] for a in load(PREV/'study_receipt.json')['artifacts']};bs={a['path']:a['sha256'] for a in load(BASE/'study_receipt.json')['artifacts']}
    for name in ['features.npz','sources.json']:
        p=PREV/name;assert sha(p)==seals[p.relative_to(H).as_posix()]
    assert sha(BASE/'features.npz')==bs[(BASE/'features.npz').relative_to(H).as_posix()]
    with np.load(BASE/'features.npz',allow_pickle=False) as z:ids=z['patient_ids'].copy();vid=z['video_ids'].copy();indices=z['original_indices'].copy();original={p:z[p].copy() for p in ['front','last']}
    assert len(indices)==203 and len(set(ids))==175
    tasks=load(H/'control/desktop-sessions/20261026T100042Z-9d41cd/native124_tasks.json')['tasks'];assert [tasks[i]['video_id'] for i in indices]==list(vid)
    members=load(SHORT/'zip_members.json');assert sha(SHORT/'zip_members.json')==load(SHORT/'state.json')['member_index_sha256']
    spec=importlib.util.spec_from_file_location('arc',R/'tools/stanford-sample-check/inspect_archive.py');arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)
    archive_stat={'size':arc.ARCHIVE.stat().st_size,'mtime_ns':arc.ARCHIVE.stat().st_mtime_ns};assert archive_stat==load(SHORT/'config.json')['archive_stat']
    tracked={};prior=load(PREV/'sources.json')['sha256']
    for p in [Path(__file__),S/'plan.md',BASE/'features.npz',PREV/'features.npz',SHORT/'zip_members.json',H/'short_window_contract.py',H/'native124_cache.py',R/'tools/stanford-sample-check/prepare_ridge_data.py',MB/'assets/mmpose_inference_3d.py',MB/'assets/MB_lite.bin',MB/'vendor/lib/model/DSTformer.py']:
        tracked[p.relative_to(R).as_posix()]=sha(p)
        if p.relative_to(R).as_posix() in prior:assert sha(p)==prior[p.relative_to(R).as_posix()]
    for p in [MB/'vendor',MB/'deps',DATA/'deps']:sys.path.insert(0,str(p))
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    cp=torch.load(MB/'assets/MB_lite.bin',map_location='cpu',weights_only=True);model.load_state_dict({k.removeprefix('module.'):v for k,v in cp['model_pos'].items()},strict=True);model.eval();model.requires_grad_(False)
    def encode(q):
        with torch.inference_mode():return model.get_representation(torch.from_numpy(q[None])).mean((1,2)).numpy()[0].copy()
    out={k:[] for k in [f'{kind}_{direction}' for kind in ['translation','scale','both'] for direction in ['FL','LF']]+['confidence_front','confidence_last']};info=[];maxreplay=0.;calls=0
    for j,index in enumerate(indices):
        motion,frames,proof=read_poses(arc.ARCHIVE,members[tasks[index]['raw_folder']]);proof_path=PREV/f'records/row_{j:04d}.json';assert sha(proof_path)==seals[proof_path.relative_to(H).as_posix()];assert proof==load(proof_path)['raw_proofs'];tracked[proof_path.relative_to(R).as_posix()]=sha(proof_path)
        long=prepare_candidate(motion,frames,243);q={};raw={}
        for p,offset in [('front',0),('last',119)]:
            start=long['start_index']+offset;raw[p]=convert(motion[start:start+124].astype(np.float32)[:,mapping],'coco','h36m');q[p]=normalize(raw[p]);checked=prepare_candidate(motion[start:start+124],frames[start:start+124],124)
            np.testing.assert_array_equal(q[p],checked['input']);np.testing.assert_array_equal(checked['frame_ids'],long['frame_ids'][offset:offset+124]);assert (q[p][:,1:7,2]>0).all()
            if j<3:
                delta=float(np.max(np.abs(encode(q[p])-original[p][j])));assert delta<=1e-5;maxreplay=max(maxreplay,delta);calls+=1
        rec={};metadata={'max_angle_error_deg':0.,'common_observed_cells':0};inputs={}
        for kind in ['translation','scale','both']:
            for direction,a,b in [('FL','front','last'),('LF','last','front')]:
                key=f'{kind}_{direction}';inputs[key],angle=transplant(raw[a],raw[b],kind);metadata['max_angle_error_deg']=max(metadata['max_angle_error_deg'],angle)
        common=(q['front'][:,:,2]>0)&(q['last'][:,:,2]>0);conf=np.sqrt(q['front'][:,:,2].astype(float)*q['last'][:,:,2]);metadata['common_observed_cells']=int(common.sum())
        for p in ['front','last']:
            v=q[p].copy();v[:,:,2][common]=conf[common].astype(np.float32);np.testing.assert_array_equal(v[:,:,:2],q[p][:,:,:2]);np.testing.assert_array_equal(v[:,:,2]==0,q[p][:,:,2]==0);inputs['confidence_'+p]=v
        for key,value in inputs.items():
            f=encode(value);calls+=1;assert f.shape==(512,) and np.isfinite(f).all();rec[key]=f;out[key].append(f)
        rf,sf=trajectory(raw['front']);rl,sl=trajectory(raw['last']);kf,kl=knee(raw['front']),knee(raw['last'])
        metadata.update(knee_unaligned_abs_deg=float(np.abs(kl-kf).mean()),knee_distribution_abs_deg=float(np.abs(np.sort(kl,axis=0)-np.sort(kf,axis=0)).mean()),scale_last_over_front_median=float(np.median(sl)/np.median(sf)),scale_ratio_p10_p90=[float(np.percentile(sl/sf,10)),float(np.percentile(sl/sf,90))],root_front_span=float(np.linalg.norm(np.ptp(rf,axis=0))/np.median(sf)),root_last_span=float(np.linalg.norm(np.ptp(rl,axis=0))/np.median(sf)))
        path=S/f'records/row_{j:04d}.npz';np.savez_compressed(path,**rec,front_raw=raw['front'],last_raw=raw['last'],front_input=q['front'],last_input=q['last'],**{k+'_input':a for k,a in inputs.items()});metadata['sha256']=sha(path);metadata['original_index']=int(index);save(f'records/row_{j:04d}.json',metadata);info.append(metadata)
        if (j+1)%20==0:print(json.dumps({'done':j+1,'seconds':time.perf_counter()-st}),flush=True)
    np.savez_compressed(S/'features.npz',**{k:np.stack(v).astype(float) for k,v in out.items()},**original,patient_ids=ids,video_ids=vid,original_indices=indices)
    assert all(sha(R/p)==h for p,h in tracked.items());assert archive_stat=={'size':arc.ARCHIVE.stat().st_size,'mtime_ns':arc.ARCHIVE.stat().st_mtime_ns}
    save('sources.json',{'sha256':tracked,'archive_stat':archive_stat});save('diagnostic_rows.json',info)
    result={'verified':True,'people':175,'records':203,'encoder_calls':calls,'new_training':0,'baseline_replay_max':maxreplay,'max_angle_error_deg':max(x['max_angle_error_deg'] for x in info),'features_sha256':sha(S/'features.npz'),'seconds':time.perf_counter()-st,'validation37_or_final38_223_used':False}
    save('preparation.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
