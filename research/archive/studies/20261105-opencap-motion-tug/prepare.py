"""Rebuild exactly the old 297 three-second windows as phone-joint-compatible series."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,csv,hashlib,importlib.util
import numpy as np
from scipy.signal import savgol_filter
H=Path(__file__).resolve().parent
B=Path('D:/gait-research-data/D_P/opencap-motion-tug-20261105')
OLD=Path('D:/Projects/D_P/studies/20261104-opencap-camera-tug')
DATA=Path('D:/gait-research-data/D_P/opencap-camera-tug-20261104')
JOINTS=['Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist','midHip','RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','LHeel','RHeel']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def source_module():
    spec=importlib.util.spec_from_file_location('original_extractor',OLD/'experiment.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def transform(xyz):
    pelvis=xyz[:,JOINTS.index('midHip')];direction=(pelvis[-1]-pelvis[0])[[0,2]];distance=np.linalg.norm(direction);assert distance>.1;direction/=distance
    q=np.stack([xyz[:,:,[0,2]]@direction,xyz[:,:,1]],axis=-1)
    leg=np.median(np.concatenate([np.linalg.norm(q[:,JOINTS.index(s+'Hip')]-q[:,JOINTS.index(s+'Knee')],axis=1)+np.linalg.norm(q[:,JOINTS.index(s+'Knee')]-q[:,JOINTS.index(s+'Ankle')],axis=1) for s in ['L','R']]))
    assert .2<leg<1.7
    root=q[:,JOINTS.index('midHip')].copy();relative=(q-root[:,None])/leg
    dynamics=np.stack([(root[:,0]-root[0,0])/leg,(root[:,1]-root[:,1].mean())/leg],axis=1)
    sequence=np.c_[relative.reshape(91,-1),dynamics].T.astype(np.float32)
    speed=float(np.mean(abs(np.diff(root[:,0])))*30)
    return sequence,speed,float(leg)

def main():
    B.mkdir(parents=True,exist_ok=True);original=source_module();d=np.load(DATA/'features.npz');audit=read(OLD/'source-audit.json');meta=read(OLD/'metadata.json');prep=read(OLD/'preparation.json');verification=read(OLD/'verification.json')
    assert verification['passed'] and sha(DATA/'features.npz')==prep['features_sha256']==verification['features_sha256']
    assert sha(OLD/'results.json')==verification['results_sha256'] and sha(OLD/'protocol.json')==verification['protocol_sha256']
    prot=read(OLD/'protocol.json');assert sha(prot['labels'])==prot['labels_sha256'];labels={(row['subid'],int(row['visit'])):row for row in csv.DictReader(Path(prot['labels']).open())}
    sequences=[];records=[];maxfeature=0.;maxspeed=0.
    for i,row in enumerate(audit):
        assert row['id']==d['ids'][i]==meta[i]['id'];path=DATA/'raw'/row['member'];assert sha(path)==row['sha256']
        assert abs(float(labels[(row['id'],row['visit'])]['tft_tug'])-d['y'][i])<1e-5
        t,a,names=original.readtrc(path);grid=row['start_time']+np.arange(91)/30
        assert grid[0]>=t[0]-1e-6 and grid[-1]<=t[-1]+1e-5
        xyz=np.stack([np.stack([np.interp(grid,t,a[:,names.index(name),axis]) for axis in range(3)],axis=1) for name in JOINTS],axis=1)
        xyz=savgol_filter(xyz,7,2,axis=0);S,speed,leg=transform(xyz);assert S.shape==(34,91) and np.isfinite(S).all()
        assert abs(speed-d['X'][i,0])<1e-10;maxspeed=max(maxspeed,abs(speed-d['X'][i,0]))
        ff,_=original.extract(path,row['start_time']);rebuilt=np.array([ff[name] for name in d['names']]);np.testing.assert_allclose(rebuilt,d['X'][i],rtol=0,atol=1e-10,equal_nan=True)
        maxfeature=max(maxfeature,float(np.nanmax(abs(rebuilt-d['X'][i]))))
        moved,_,_=transform(xyz+np.array([2.,-1.,3.]));np.testing.assert_allclose(S,moved,rtol=0,atol=2e-6)
        sequences.append(S);records.append({'id':row['id'],'source_sha256':sha(path),'start_time':row['start_time'],'end_time':row['end_time'],'leg_length_m':leg})
    assert len(sequences)==297 and sum(r['age']>=65 for r in meta)==34
    np.savez_compressed(B/'input.npz',ids=d['ids'],y=d['y'],X=d['X'],names=d['names'],sequence=np.array(sequences),age=np.array([r['age'] for r in meta]),diagnosis=np.array([r['diagnosis'] for r in meta]))
    save(H/'input-audit.json',{'status':'passed','people':297,'older65':34,'TUG20plus':int(sum(d['y']>=20)),'sequence_shape':[297,34,91],'source_all297_SHA_checked':True,'same_person_visit_TUG_checked':True,'all64_features_rebuilt':True,'max_feature_difference':maxfeature,'max_speed_difference':maxspeed,'translation_invariance_passed':True,'input_sha256':sha(B/'input.npz'),'source_features_sha256':sha(DATA/'features.npz'),'source_splits_sha256':sha(OLD/'effective-splits.json'),'source_extractor_sha256':sha(OLD/'experiment.py'),'joints':JOINTS,'records':records})
    print('PREPARED_297',maxfeature,maxspeed,flush=True)

if __name__=='__main__':main()
