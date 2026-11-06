"""Prepare a pinned official STT model with the existing single-window cohort."""
from pathlib import Path
import hashlib,json,shutil,subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
B=HERE/'bundle'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def main():
    original=HERE/'official'
    commit=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],text=True).strip()
    assert commit=='549d58bc32449b24a0eae0c37ca573f9aa58a1dd'
    assert not subprocess.check_output(['git','-C',str(original),'status','--porcelain'],text=True).strip()
    B.mkdir(exist_ok=True);(B/'model').mkdir(exist_ok=True);(B/'sources').mkdir(exist_ok=True)
    hashes={}
    for p in sorted(original.rglob('*.py')):
        if '.git' in p.parts:continue
        rel=p.relative_to(original);hashes[rel.as_posix()]=sha(p)
        dest=B/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        if rel.parts[0]=='model':shutil.copyfile(p,B/rel)
    shutil.copyfile(original/'README.md',B/'sources/README.md')
    base=HERE.parent/'cnn-paper-input695/bundle'
    assert sha(base/'data.npz')=='1448059f493156c0194a9d2518f5c6cf04e40c4bb2a39472d5d0c35885ab8967'
    with np.load(base/'data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    x=d['x'][...,:8].reshape(1061,2,124,4,2).copy()
    assert np.isfinite(x).all() and len(np.unique(d['person_id']))==692
    d['x']=x;np.savez_compressed(B/'data.npz',**d)
    for f in range(5):
        assert len(np.unique(d['person_id'][d['fold']==f]))==[138,138,138,139,139][f]
    save(B/'sources.json',dict(repository='https://github.com/vinuni-vishc/transformer-gait-analysis',commit=commit,
        upstream_files=hashes,base_data_sha256=sha(base/'data.npz'),data_sha256=sha(B/'data.npz'),
        model_source_unmodified=True,input_coordinates_exact_existing_cnn_first8=True,
        joint_order=['ankle','knee','hip','big_toe'],sides=['left','right'],frames=124,
        normalization='Existing pelvis-center/mean-thigh-length; full-clip interpolation/smoothing preserved',
        independent_final_data_used=False,license_file_present=any(p.name.lower().startswith('license') for p in original.iterdir())))
    save(B/'protocol.json',dict(name='official_STT_single124_pilot692',seed=20261028,pilot_epochs=3,
        batch_size=128,learning_rate=0.0003,eta_min=0.00008,T_0=40,T_mult=1,
        optimizer='Adam',weight_decay=0.,dropout=0.,joint_embed_dim=12,num_heads=2,
        pilot_outer_fold=0,pilot_inner_fold=1,target_scale='current_training_pooled_sides_minmax',
        loss='aligned_1d_scaled_MSE_single_window',calibration='training_only_shared_slope_intercept',
        full_training_implemented=False,official_full_epochs=200,
        proposed_full_selection='existing_nested_person_weighted_calibrated_MAE',
        note='Pilot readiness only; no STT performance estimate or official-paper reproduction'))
    save(HERE/'input_verification.json',dict(passed=True,people=692,records=1061,shape=list(x.shape),
        source_sha256=sha(base/'data.npz'),prepared_sha256=sha(B/'data.npz'),same_people_folds_targets=True,
        coordinate_values_exact=True,real_optimizer_steps=0))
    print(json.dumps(dict(commit=commit,input_shape=list(x.shape),source_model_files=len(list((B/'model').glob('*.py'))))))

if __name__=='__main__':main()
