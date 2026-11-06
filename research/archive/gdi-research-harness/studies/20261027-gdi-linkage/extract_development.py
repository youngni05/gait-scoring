"""Resumable frozen-encoder features for development only, no final inference."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from audit_labels import HERE,H,ROOT,load,sha,save

MB=ROOT/'outputs/motionbert-smoke-20261015'
DATA=ROOT/'outputs/gdi-ridge-experiment-20261016'
SHORT=H/'jobs/native124-cache-v1'
WEIGHT_SHA='6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461'

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
    tick=time.monotonic()
    assert (HERE/'transfer_plan.md').is_file()
    rows=load(HERE/'development_linkage.json')['records']
    assert len(rows)==861 and len({r['patient_id'] for r in rows})==520
    assert all(r['eligible'] and r['reserved_role']=='development_candidate' for r in rows)
    weight=MB/'assets/MB_lite.bin';assert sha(weight)==WEIGHT_SHA
    source_paths=[Path(__file__),HERE/'audit_labels.py',HERE/'transfer_plan.md',HERE/'development_linkage.json',
                  HERE/'linkage_summary.json',MB/'vendor/lib/model/DSTformer.py',MB/'vendor/lib/model/drop.py',weight]
    sources={'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
             'model':'frozen MotionBERT-Lite; original architecture; mean(time,joints); no fine-tuning',
             'scope':'development520 only; final223 not read by extractor'}
    manifest=HERE/'feature_sources.json'
    if manifest.exists():assert load(manifest)==sources
    else:save(manifest.name,sources)
    for dep in (MB/'vendor',MB/'deps',DATA/'deps'):sys.path.insert(0,str(dep))
    import torch
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,
                    mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    cp=torch.load(weight,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in cp['model_pos'].items()},strict=True)
    model.requires_grad_(False);model.eval()
    def feature(motion):
        assert motion.shape==(124,17,3) and motion.dtype==np.float32
        assert np.isfinite(motion).all() and np.max(np.abs(motion))<=1.0
        with torch.inference_mode():v=model.get_representation(torch.from_numpy(motion[None])).mean(dim=(1,2)).numpy()[0]
        assert v.shape==(512,) and np.isfinite(v).all()
        return v
    pilot=HERE/'feature_pilot.json'
    if a.pilot:
        differences=[]
        for i in range(3):
            with np.load(SHORT/f'records/row_{i:04d}.npz',allow_pickle=False) as z:
                v=feature(z['input124'].copy());d=float(np.max(np.abs(v-z['feature'])))
                np.testing.assert_allclose(v,z['feature'],rtol=0,atol=1e-6);differences.append(d)
        assert not pilot.exists()
        targets=rows[:3]
    else:
        assert load(pilot)['verified']
        targets=rows
    (HERE/'development_features').mkdir(exist_ok=True)
    computed=reused=0
    for row in targets:
        dest=HERE/f"development_features/row_{row['sequence']:04d}.npz";proof=dest.with_suffix('.json')
        source=H/row['artifact'];assert sha(source)==row['sha256']
        if dest.exists():
            p=load(proof)
            assert sha(dest)==p['sha256'] and p['input_sha256']==row['sha256']
            assert p['feature_sources_sha256']==sha(manifest)
            reused+=1;continue
        assert not proof.exists()
        with np.load(source,allow_pickle=False) as z:
            motion=z['input124'].copy();ids=z['frame_ids124'].copy()
        assert ids.shape==(124,) and np.all(np.diff(ids)==1)
        v=feature(motion)
        np.savez_compressed(dest,feature=v)
        proof.write_text(json.dumps({'sequence':row['sequence'],'sha256':sha(dest),'input_sha256':row['sha256'],
            'feature_sources_sha256':sha(manifest)},indent=2),encoding='utf-8')
        computed+=1
        if computed%50==0:print(json.dumps({'computed':computed,'reused':reused,'target':len(targets),'elapsed_seconds':round(time.monotonic()-tick,1)}),flush=True)
    if a.pilot:
        save(pilot.name,{'verified':True,'old_training_feature_max_abs_differences':differences,
                         'new_development_records':3,'feature_sources_sha256':sha(manifest)})
        print(json.dumps(load(pilot)),flush=True);return
    values=[]
    for row in rows:
        dest=HERE/f"development_features/row_{row['sequence']:04d}.npz"
        assert sha(dest)==load(dest.with_suffix('.json'))['sha256']
        with np.load(dest,allow_pickle=False) as z:values.append(z['feature'].copy())
    assert not (HERE/'development_features.npz').exists()
    np.savez_compressed(HERE/'development_features.npz',X=np.asarray(values,dtype=np.float64),
        Y=np.array([r['GDI_LR'] for r in rows],dtype=np.float64),
        patient_ids=np.array([r['patient_id'] for r in rows]),video_ids=np.array([r['video_id'] for r in rows]),
        sequence=np.array([r['sequence'] for r in rows]))
    assert all(sha(ROOT/p)==s for p,s in sources['sha256'].items())
    save('feature_preparation.json',{'verified':True,'people':520,'records':861,'computed_this_run':computed,
         'reused_pilot_or_completed':reused,'elapsed_seconds':time.monotonic()-tick,'final_records_processed':0,
         'sha256':sha(HERE/'development_features.npz'),'feature_sources_sha256':sha(manifest)})
    print(json.dumps(load(HERE/'feature_preparation.json')),flush=True)

if __name__=='__main__':main()
