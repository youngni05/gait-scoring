"""Freeze existing datasets/splits and extract deterministic coordinate summaries."""
from common import HERE, ROOT, CPU, GPU, arrays, errors, load, save, sha, split
import numpy as np
import shutil


def summaries(x):
    n = len(x)
    valid = x[...,2] > 0
    features, names = [], []
    for joint in range(17):
        m = valid[:,:,joint]
        denom = np.maximum(m.sum(1),1)
        for axis in range(2):
            v = x[:,:,joint,axis].astype(float)
            mean = (v*m).sum(1)/denom
            std = np.sqrt(((v-mean[:,None])**2*m).sum(1)/denom)
            paired = m[:,1:] & m[:,:-1]
            speed = (abs(np.diff(v,axis=1))*paired).sum(1)/np.maximum(paired.sum(1),1)
            features += [mean,std,speed]
            names += [f'joint{joint}_axis{axis}_{s}' for s in ['mean','std','mean_abs_change']]
        features += [x[:,:,joint,2].mean(1), m.mean(1)]
        names += [f'joint{joint}_mean_confidence',f'joint{joint}_observed_fraction']
    for label, hip, knee, ankle in [('right',1,2,3),('left',4,5,6)]:
        a = x[:,:,hip,:2]-x[:,:,knee,:2]
        b = x[:,:,ankle,:2]-x[:,:,knee,:2]
        denom = np.linalg.norm(a,axis=2)*np.linalg.norm(b,axis=2)
        ok = denom > 1e-12
        angle = np.arccos(np.clip((a*b).sum(2)/np.maximum(denom,1e-12),-1,1))*180/np.pi
        mean = (angle*ok).sum(1)/np.maximum(ok.sum(1),1)
        std = np.sqrt(((angle-mean[:,None])**2*ok).sum(1)/np.maximum(ok.sum(1),1))
        features += [mean,std,ok.mean(1)]
        names += [f'{label}_2d_knee_angle_mean',f'{label}_2d_knee_angle_std',f'{label}_2d_knee_angle_valid_fraction']
    rel = x[:,:,3,:2]-x[:,:,6,:2]
    for axis in range(2):
        v = rel[:,:,axis]
        features += [v.mean(1),v.std(1),abs(v).mean(1),v.max(1)-v.min(1)]
        names += [f'ankle_relative_axis{axis}_{s}' for s in ['mean','std','mean_absolute','range']]
    out = np.stack(features,axis=1).astype(np.float64)
    assert out.shape == (n,len(names)) and np.isfinite(out).all()
    return out,names


def main():
    assert not (HERE/'data.npz').exists()
    parent = CPU/'expanded695-pls-study'
    pls = CPU/'expanded695-pls64-study'
    sources = [GPU/'data.npz',GPU/'input_contract.json',GPU/'run.py',GPU/'model.py',GPU/'protocol.json',
               parent/'development695.npz',pls/'oof_predictions.npz',HERE/'protocol.json']
    sources += [pls/f'models/fold{f}_inner.npz' for f in range(5)]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    raw, mb, old = arrays(GPU/'data.npz'),arrays(parent/'development695.npz'),arrays(pls/'oof_predictions.npz')
    assert raw['x'].shape == (1064,124,17,3) and mb['X'].shape == (1064,512)
    for key in ['person_id','video_id','fold','y']:
        np.testing.assert_array_equal(raw[key],mb[key])
        if key in old: np.testing.assert_array_equal(raw[key],old[key])
    k,names = summaries(raw['x'])
    d = {**raw,'kinematic': k,'motionbert': mb['X'], 'pls': old['predictions'][old['methods'].tolist().index('pls64')]}
    assert abs(errors(d['y'],d['pls'],d['person_id']).mean()-7.974896350374014)<1e-5
    (HERE/'splits').mkdir()
    for f in range(5):
        inner = arrays(pls/f'models/fold{f}_inner.npz')
        tr = np.flatnonzero(d['fold'] != f)
        np.testing.assert_array_equal(tr,inner['outer_train_indices'])
        np.savez_compressed(HERE/f'splits/outer{f}.npz',inner_assignment=inner['inner_assignment'])
        for j in range(4):split(d,f,j)
    np.savez_compressed(HERE/'data.npz',**d)
    (HERE/'cnn_source').mkdir()
    for name in ['run.py','model.py','protocol.json']:
        shutil.copyfile(GPU/name,HERE/'cnn_source'/name)
    save('feature_names.json',{'count':len(names),'names':names,'physical_units':False})
    save('sources.json',{'sha256':hashes,'dataset_people':695,'dataset_records':1064,
                         'local_dataset_sha256':sha(HERE/'data.npz'),'protected_numeric_data_read':False})
    save('existing_experiment_audit.json',{'same695_CNN_completed_results_found':False,'same695_RF_completed_results_found':False,
        'CNN_available':'Prepared six-convolution source adaptation and synthetic-only workflow, not real-person training results',
        'RF_available':'No RandomForestRegressor/Classifier training code found in searched project code',
        'search_scope':['gdi-research-harness','tools','2026-ITEC0401','ai-measurement-review'],
        'teammate_experiment_status':'No teammate code or predictions available; completion cannot be confirmed',
        'existing_completed_GDI_methods':['frozen MotionBERT Ridge/PCA/PLS','MotionBERT partial training','PLS/GPU fixed equal prediction average']})
    print({'people':695,'records':1064,'coordinate_summary_features':len(names),'status':'prepared'})


if __name__=='__main__':main()
