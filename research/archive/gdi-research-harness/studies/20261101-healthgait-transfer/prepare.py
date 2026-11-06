"""Prepare immutable common-coordinate arrays; keep large artifacts on D."""
import argparse
import csv
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
OLD = HERE.parent/'20261030-older44-sppb-timeseries'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')


def normalize(points):
    # Points are T,side2,joint3(ankle,knee,hip),xy2 in image coordinates.
    missing = (~np.isfinite(points)).reshape(len(points),-1).mean(0)
    if missing.max() > .25:
        return None, 'missing_over25pct'
    flat = points.reshape(len(points),-1).copy()
    for j in range(flat.shape[1]):
        good = np.flatnonzero(np.isfinite(flat[:,j]))
        if len(good) < 2 or good[0] != 0 or good[-1] != len(flat)-1:
            return None, 'missing_boundary_no_extrapolation'
        flat[:,j] = np.interp(np.arange(len(flat)),good,flat[good,j])
    p = gaussian_filter1d(flat,1,axis=0).reshape(points.shape)
    hip = p[:,:,2].mean(1)
    scale = np.linalg.norm(p[:,:,2]-p[:,:,1],axis=-1).mean(1)
    if not np.isfinite(scale).all() or (scale<=1e-6).any():
        return None,'invalid_thigh_scale'
    p = (p-hip[:,None,None,:])/scale[:,None,None,None]
    a = p.reshape(len(p),12)
    result = np.stack([np.interp(np.linspace(0,len(p)-1,124),np.arange(len(p)),a[:,j]) for j in range(12)],1)
    result = result.reshape(124,2,6).transpose(1,2,0).astype(np.float32)
    assert result.shape == (2,6,124) and np.isfinite(result).all()
    return result,None


def target(bundle):
    spec=importlib.util.spec_from_file_location('original44prepare',OLD/'prepare.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest=list(csv.DictReader((OLD/'manifest.csv').open(encoding='utf8')))
    xs=[];ys=[];ids=[];folds=[];proof=[]
    for r in manifest:
        path=module.RAW/r['file']
        assert sha(path)==r['sha256']
        frames,p,fps=module.read_walk(path)
        a=int(r['run_start_row0']);q=p[a:a+301]
        assert len(q)==301 and np.isfinite(q).all() and (np.diff(frames[a:a+301])==1).all()
        pelvis=q[:,:4].mean(1)
        forward=pelvis[-1]-pelvis[0];forward[2]=0;forward/=np.linalg.norm(forward)
        projected=np.stack([q@forward,-q[:,:,2]],-1)
        selected=np.stack([projected[:,[6,4,0]],projected[:,[7,5,1]]],axis=1)
        x,reason=normalize(selected)
        assert reason is None,r['file']
        xs.append(x);ys.append([float(r['target'])]);ids.append(r['id']);folds.append(int(r['fold']))
        proof.append(dict(file=r['file'],sha256=r['sha256'],person=r['id'],fold=int(r['fold']),start_frame=int(frames[a]),end_frame=int(frames[a+300]),fps=fps,forward=forward.tolist()))
    with np.load(OLD/'input.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.array(ids),z['ids'])
        np.testing.assert_array_equal(np.array(ys)[:,0],z['y'])
        np.testing.assert_array_equal(np.array(folds),z['folds'])
    np.savez_compressed(bundle/'target.npz',x=np.array(xs),y=np.array(ys,np.float32),ids=np.array(ids),folds=np.array(folds),owner=np.arange(len(xs)),records=np.array([r['file'] for r in manifest]))
    save(HERE/'target-audit.json',dict(people=len(set(ids)),records=len(xs),windows=len(xs),input_sha256=sha(bundle/'target.npz'),source_manifest_sha256=sha(OLD/'manifest.csv'),labels_sha256=sha(module.LABELS),score_counts={str(s):sum(float(ys[ids.index(p)][0])==s for p in set(ids)) for s in sorted(set(y[0] for y in ys))},records_proof=proof))


def source(raw,bundle):
    root=raw/'Health_Gait'
    rows=list(csv.DictReader((root/'gait_parameters.csv').open(encoding='utf-8-sig')))
    print(json.dumps({'csv_columns':list(rows[0]),'first_row':rows[0]}),flush=True)
    # Header must be inspected against the official released file before fitting.
    byid={r['ID']:r for r in rows}
    names=['l_ankle','l_knee','l_hip','r_ankle','r_knee','r_hip']
    xs=[];owner=[];ys=[];ids=[];records=[];meta=[];reject=[]
    archives=sorted((raw/'pose-archives').glob('*.zip'))
    readers={}
    if archives:
        for archive in archives:
            z=zipfile.ZipFile(io.BytesIO(archive.read_bytes()))
            for n in z.namelist():
                assert n not in readers,'Duplicate archived pose record'
                readers[n]=z
        pose_files=sorted(root/Path(n).relative_to('Health_Gait') for n in readers)
    else:
        index=json.loads((raw/'directory.json').read_text())
        pose_files=sorted(raw/e['name'] for e in index if '/pose/' in e['name'] and e['name'].endswith('.json'))
    assert pose_files,'No public pose JSON files'
    for path in pose_files:
        rel=path.relative_to(root)
        person,gait=rel.parts[1:3]
        assert gait in ('UGS','FGS') and person in byid,(path,person)
        row=byid[person]
        keys=[k+'_'+gait for k in ('Velocity','Cadence','Step','Stride')]
        if any(k not in row for k in keys):
            keys=[k+'_'+gait for k in ('Speed','Cadence','Step','Stride')]
        try:
            y=np.array([float(row[k]) for k in keys],np.float32)
        except (ValueError,KeyError):
            reject.append(dict(file=str(rel),reason='missing_measured_label'));continue
        if not np.isfinite(y).all() or (y<=0).any():
            reject.append(dict(file=str(rel),reason='invalid_measured_label'));continue
        member='Health_Gait/'+rel.as_posix()
        data=json.loads(readers[member].read(member)) if readers else json.loads(path.read_text())
        assert isinstance(data,list) and len(data)>0
        frame_ids=np.array([int(frame['frame']) for frame in data])
        assert frame_ids[0]==0 and (np.diff(frame_ids)==1).all(), 'Nonconsecutive public frame IDs'
        p=np.full((len(data),2,3,2),np.nan)
        for i,frame in enumerate(data):
            joints=frame['joints']
            if joints is None:continue
            for j,name in enumerate(names):
                point=joints.get(name)
                if point is not None:
                    p[i,j//3,j%3]=[point['x'],point['y']]
        p[p==0]=np.nan
        good=[]
        observed=np.flatnonzero(np.isfinite(p).all((1,2,3)))
        starts=range(int(observed[0]),int(observed[-1])-90+1,30) if len(observed) else []
        for start in starts:
            x,reason=normalize(p[start:start+91])
            if reason is None:good.append((start,x))
        if not good:
            reject.append(dict(file=str(rel),reason='no_complete_3s_window',frames=len(p)));continue
        record=len(ys)
        for start,x in good:
            xs.append(x);owner.append(record)
            meta.append(dict(record=record,start_frame=start,end_frame=start+90,fps=30))
        ys.append(y);ids.append(person);records.append(str(rel))
    people=np.array(sorted(set(ids)))
    assert len(people)>=300,'Do not silently turn a few public samples into full pretraining'
    shuffled=people.copy();np.random.RandomState(20261101).shuffle(shuffled)
    lookup={p:f for f,chunk in enumerate(np.array_split(shuffled,5)) for p in chunk}
    folds=np.array([lookup[p] for p in ids])
    np.savez_compressed(bundle/'source.npz',x=np.array(xs),y=np.array(ys),ids=np.array(ids),folds=folds,owner=np.array(owner),records=np.array(records))
    save(HERE/'source-audit.json',dict(people=len(people),records=len(ys),windows=len(xs),downloaded_pose_files=len(pose_files),measured_label_columns=keys,source_csv_sha256=sha(root/'gait_parameters.csv'),input_sha256=sha(bundle/'source.npz'),frame_ids_contiguous_all_files=True,fps_evidence='Dataset paper reports30Hz video, released frame indices are contiguous; no independent per-file timestamps',rejected=reject,window_manifest=meta))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,default=Path('D:/gait-research-data/D_P/datasets/HealthGait'));ap.add_argument('--bundle',type=Path,default=Path('D:/gait-research-data/D_P/healthgait-transfer-20261101/bundle'));ap.add_argument('--target-only',action='store_true')
    a=ap.parse_args();a.bundle.mkdir(parents=True,exist_ok=True)
    assert str(a.bundle.resolve()).lower().startswith('d:\\gait-research-data\\d_p\\')
    target(a.bundle)
    if not a.target_only:source(a.raw,a.bundle)
    for n in ('model.py','train.py','protocol.json','verify.py','launch.sh','submit.slurm','MODEL_LICENSE'):
        if (HERE/n).exists():shutil.copy2(HERE/n,a.bundle/n)
    original=WORKSPACE/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle/weights/fold0.pt'
    assert sha(original)=='7853024d521f0915e3013927feed83fe7a7b6afb09d726799f4f9b70627a81ab'
    shutil.copy2(original,a.bundle/'gdi.pt')
    save(HERE/'gdi-provenance.json',dict(source=str(original),sha256=sha(original),source_original_checkpoint_sha256='b1ec9ec0c6fea97032e2f4e306f3d469ea4be8189b4329649981a2a7e596b80a',selection='Fixed fold0 before target training; no choice by target score'))
    if not a.target_only:
        manifest={p.name:sha(p) for p in a.bundle.iterdir() if p.is_file() and p.name!='manifest.json'}
        save(a.bundle/'manifest.json',manifest)
        save(HERE/'bundle-receipt.json',dict(path=str(a.bundle.resolve()),files=manifest))


if __name__=='__main__':main()
