"""Prepare single-window frozen MotionBERT time features + existing CNN, no GPU launch."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;BASE=STUDY/'cnn-paper-input695';MB=STUDY/'expansion695/gpu-bundle';B=HERE/'bundle'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def replace(s,old,new,count=1):
    assert s.count(old)==count,(old,s.count(old))
    return s.replace(old,new)

def main():
    B.mkdir(parents=True,exist_ok=True)
    for source in (BASE/'bundle',MB):
        for name,digest in json.loads((source/'manifest.json').read_text()).items():assert sha(source/name)==digest,name
    shutil.copytree(BASE/'bundle/sources',B/'sources',dirs_exist_ok=True)
    shutil.copytree(MB/'lib',B/'lib',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('MB_lite.bin','MOTIONBERT_LICENSE'):shutil.copy2(MB/name,B/name)
    shutil.copy2(BASE/'bundle/data.npz',B/'data.npz')
    shutil.copy2(BASE/'bundle/model.py',B/'baseline_model.py')
    with np.load(B/'data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files if k!='x'}
    idx=d['original_row_index']
    with np.load(MB/'train_only.npz',allow_pickle=False) as z:
        for k in ('y','person_id','video_id','fold','cohort'):np.testing.assert_array_equal(d[k],z[k][idx])
        pose=z['x'][idx,0].copy();pooled=z['features'][idx,0].copy()
    frames=[]
    for i in idx:
        with np.load(BASE/'records'/('%04d.npz'%i),allow_pickle=False) as z:frames.append(z['frame_ids'].copy())
    frames=np.stack(frames)
    assert pose.shape==(1061,124,17,3) and pooled.shape==(1061,512) and np.isfinite(pose).all()
    np.savez_compressed(B/'motion_input.npz',pose=pose,pooled_reference=pooled,frame_ids=frames,
        **{k:d[k] for k in ('person_id','video_id','fold','original_row_index')})
    with np.load(BASE/'bundle/reference_predictions.npz',allow_pickle=False) as z:refs={k:z[k].copy() for k in z.files}
    proof=json.loads((BASE/'analysis/verification.json').read_text());assert proof['passed']
    with np.load(BASE/'analysis/cnn_oof.npz',allow_pickle=False) as z:
        for k in d:np.testing.assert_array_equal(d[k],z[k])
        refs['single_cnn']=z['predictions'][:,0].copy()
    np.savez_compressed(B/'reference_predictions.npz',**refs)
    metrics=json.loads((BASE/'bundle/reference_metrics.json').read_text());metrics['single_cnn_mae']=float(np.mean([np.abs(refs['single_cnn'][d['person_id']==p]-d['y'][d['person_id']==p]).mean() for p in np.unique(d['person_id'])]))
    save(B/'reference_metrics.json',metrics)

    model=(BASE/'bundle/model.py').read_text()
    model=model.replace('12 if i == 0 else 32','524 if i == 0 else 32').replace('(12, 124)','(524, 124)').replace('(2,124,12)','(2,124,524)')
    model=model.replace('B,12,124; 12 is channels','B,524,124; 524 is channels').replace('features12','features524')
    model=replace(model,'        self.output = nn.Linear(10, 2)','        self.output = nn.Linear(10, 2)\n        self.register_buffer("feature_mean",torch.zeros(512))\n        self.register_buffer("feature_scale",torch.ones(512))')
    model=replace(model,'        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):', '        motion=(x[:,12:,:]-self.feature_mean[None,:,None])/self.feature_scale[None,:,None]\n        x=torch.cat([x[:,:12,:],motion],dim=1)\n        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):')
    model=model.replace('12 channels per side;', '524 channels per side: unchanged 12 geometry channels plus 512 frozen temporal features;')
    model+='''

def training_feature_stats(x):
    # Both sides receive the same full-body feature; use one copy, training records only.
    if x.ndim!=4 or x.shape[1:]!=(2,124,524):raise ValueError('Wrong fusion axes')
    v=x[:,0,:,12:].astype('float64')
    mean=v.mean(axis=(0,1));scale=v.std(axis=(0,1))
    scale[scale<1e-6]=1.0
    return torch.from_numpy(mean.astype('float32')),torch.from_numpy(scale.astype('float32'))
'''
    (B/'model.py').write_text(model,encoding='utf-8')
    source=(BASE/'bundle/run.py').read_text()
    source=replace(source,'from model import GdiCNN, channels','from model import GdiCNN, channels, training_feature_stats\nfrom features import validate_motion_inputs, read_cache')
    source=replace(source,'def validate_data(d):','def validate_data(d, fused=True):')
    source=replace(source,"d['x'].shape==(n,2,124,12)","d['x'].shape==(n,2,124,524 if fused else 12)")
    source=replace(source,'def data():','def data(with_features=True):')
    source=replace(source,'    validate_data(d)\n    return d', '''    validate_data(d,fused=False)
    validate_motion_inputs(ROOT,d)
    if with_features:
        features,cache_hash=read_cache(ROOT,ROOT/'cache/temporal',len(d['y']))
        tiled=np.broadcast_to(features[:,None,:,:],(len(features),2,124,512))
        d['x']=np.concatenate([d['x'],tiled],axis=3)
        d['_feature_cache_sha256']=cache_hash
        validate_data(d)
    return d''')
    source=source.replace('paper_feature_matched_window_no_search','frozen_motionbert_temporal_fusion_no_search').replace('matched124_official12_separate_sides','matched124_geometry12_plus_motionbert512')
    source=source.replace('reshape(-1,12,124)','reshape(-1,524,124)').replace('torch.rand(c[\'batch_size\']//2,2,124,12)','torch.rand(c[\'batch_size\']//2,2,124,524)')
    source=replace(source,"model = GdiCNN(c['dropout']).to(device)\n    opt = make_optimizer(model, c)", "model = GdiCNN(c['dropout']).to(device)\n    mean,scale=training_feature_stats(x)\n    model.feature_mean.copy_(mean);model.feature_scale.copy_(scale)\n    opt = make_optimizer(model, c)")
    source=replace(source,"model.load_state_dict(ck['model'], strict=True)", "require(torch.equal(ck['model']['feature_mean'].cpu(),mean) and torch.equal(ck['model']['feature_scale'].cpu(),scale),'Feature normalization changed')\n        model.load_state_dict(ck['model'], strict=True)")
    source=replace(source,"return {'bundle_sha256': sha(ROOT / 'manifest.json'), 'purpose': purpose,", "return {'bundle_sha256': sha(ROOT / 'manifest.json'), 'feature_cache_sha256':d['_feature_cache_sha256'], 'purpose': purpose,")
    source=replace(source,"'primary_output':'training_only_OLS_corrected; raw is diagnostic',", "'primary_output':'training_only_OLS_corrected; raw is diagnostic',\n               'single_cnn_mae_reduction':reference['single_cnn_mae']-rows[0]['mae'],\n               'frozen_motionbert':True, 'feature_cache_sha256':d['_feature_cache_sha256'],")
    source=replace(source,"ROOT / 'run.py', ROOT / 'model.py'", "ROOT / 'run.py', ROOT / 'model.py', ROOT / 'features.py', ROOT / 'input_contract.json', ROOT / 'cache/temporal/complete.json'")
    source=replace(source,'    d = data()','    d = data(with_features=args.command != \'check\')')
    source=replace(source,"'cnn_input_shape': [len(d['y'])*2, 12, 124]", "'cnn_input_shape_after_feature_extraction': [len(d['y'])*2, 524, 124], 'temporal_cache_required_for_training':True")
    source=source.replace('Official-feature matched-window CNN','Frozen MotionBERT temporal + geometry single-window CNN')
    (B/'run.py').write_text(source,encoding='utf-8')
    cfg=json.loads((BASE/'bundle/protocol.json').read_text())
    cfg.update(name='cnn_geometry12_frozen_motionbert512_temporal_v1',stage='frozen_motionbert_temporal_fusion_no_search',input='matched124_geometry12_plus_motionbert512',
        feature_normalization='512-feature mean/std over current training records and time only; raw12 unchanged',
        encoder='original MB_lite frozen, eval, no GDI fine-tuned weights',feature_pool='joint mean only; preserve all124timesteps',
        input_channels=524,wall_time_seconds=14400,note='Same single window and folds; extra channels also increase trainable CNN capacity; not a capacity-matched ablation.')
    save(B/'protocol.json',cfg)
    save(B/'input_contract.json',dict(people=692,records=1061,raw_shape=[1061,2,124,12],pose_shape=list(pose.shape),temporal_shape=[1061,124,512],
        fused_shape=[1061,2,124,524],same_frame_ids=True,same_labels_and_folds=True,feature_pool='17-joint mean only; do not pool time',
        side_handling='same full-body temporal features appended to left and right geometry separately',
        normalization='per fit training only; cached raw frozen features are never normalized across all people',
        single_cnn_manifest_sha256=sha(BASE/'bundle/manifest.json'),motionbert_checkpoint_sha256=sha(MB/'MB_lite.bin'),
        old_pose_bundle_sha256=sha(MB/'manifest.json'),old_pose_data_sha256=sha(MB/'train_only.npz'),
        baseline_predictions_sha256=sha(BASE/'analysis/cnn_oof.npz'),no_protected_labels=True,actual_gpu_execution=False))
    print('Fusion bundle inputs and runner generated; temporal GPU cache intentionally not precomputed.')

if __name__=='__main__':main()
