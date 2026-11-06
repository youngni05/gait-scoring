"""Inventory original F-round MAT files and geometrically screen sagittal walking."""
import sys, io, json, tarfile, hashlib, re
from pathlib import Path
sys.path.insert(0,'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')
import numpy as np
from scipy.io import loadmat
import cv2
HERE=Path(__file__).resolve().parent
ROOT=Path('D:/gait-research-data/MoVi')
META=Path('D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261029-movi-expansion/movi-metadata-current.json')

def digest(p,alg='sha256'):
    h=hashlib.new(alg)
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def runs(mask,frames):
    edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
    return [{'start_one_based':int(frames[a]),'end_one_based':int(frames[b-1]),'frames':int(b-a)} for a,b in zip(starts,ends)]

def main():
    provider=json.loads(META.read_text(encoding='utf-8'))['data']['latestVersion']['files']
    files={x['label']:x for x in provider}
    cal={}
    with tarfile.open(ROOT/'raw/Camera Parameters.tar') as t:
        for camera in ['PG1','PG2']:
            cal[camera]=[dict(np.load(io.BytesIO(t.extractfile('Calib/'+n+'_'+camera+'.npz').read()))) for n in ['cameraParams','Extrinsics']]
    need=['LFWT','RFWT','LBWT','RBWT','LKNE','LKNEIN','RKNE','RKNEIN','LANK','RANK','LHEE','RHEE','LTOE','RTOE','LMT1','RMT1','LMT5','RMT5']
    viewnames=['LSJC','RSJC','LHIP','RHIP','LKJC','RKJC','LAJC','RAJC','LHEE','RHEE','LTOE','RTOE']
    rows=[];archive_checks=[]
    for archive in ['F_Subjects_1_45.tar','F_Subjects_46_90.tar']:
        p=ROOT/'raw'/archive;actual=digest(p,'md5');assert actual==files[archive]['dataFile']['checksum']['value']
        archive_checks.append({'name':archive,'bytes':p.stat().st_size,'md5':actual,'provider_match':True})
        with tarfile.open(p,'r:') as t:
            for member in t:
                match=re.fullmatch(r'F_v3d_Subject_(\d+)\.mat',Path(member.name).name)
                if not match:continue
                sid=int(match[1]);raw=t.extractfile(member).read();obj=loadmat(io.BytesIO(raw),simplify_cells=True)[f'Subject_{sid}_F'];m=obj['move']
                names=list(map(str,m['markerName'])); motions=list(map(str,m['motions_list']))
                row={'id':sid,'age':int(obj['subject']['age']),'sex':str(obj['subject']['sex']),'member':member.name,'sha256':hashlib.sha256(raw).hexdigest(),'missing_marker_names':[n for n in need if n not in names],
                     'fields':list(m),'segment_names':list(map(str,m['segmentName']))}
                if 'walking' not in motions:row['walking']=False;rows.append(row);continue
                ix=motions.index('walking');a,b=map(int,m['flags30'][ix]);frames=np.arange(a,b+1);indices=(frames-1)*4
                assert np.array_equal(m['flags120'].astype(int),(m['flags30'].astype(int)-1)*4+1)
                xyz=m['markerLocation'][indices];gaps=m['markerGaps'][indices].astype(bool)
                pelvis=xyz[:,[names.index(n) for n in ['LFWT','RFWT','LBWT','RBWT']]].mean(axis=1)
                vel=np.zeros_like(pelvis)
                for k in range(len(pelvis)):
                    l=max(0,k-5);r=min(len(pelvis)-1,k+5);vel[k]=(pelvis[r]-pelvis[l])*30/(r-l)
                # World z is vertical; source coordinates are millimetres.
                speed=np.linalg.norm(vel[:,:2],axis=1)/1000
                needidx=[names.index(n) for n in need if n in names]
                marker_valid=np.isfinite(xyz[:,needidx]).all(axis=(1,2)) & ~gaps[:,needidx].any(axis=1)
                aff=m['jointsAffine_v3d'][indices];segvalid=np.isfinite(aff).all(axis=(1,2,3)) & ~m['jointsGaps_v3d'][indices].astype(bool).any(axis=1)
                row.update(walking=True,walking_start=a,walking_end=b,walking_frames=len(frames),marker_valid_frames=int(marker_valid.sum()),segment_valid_frames=int(segvalid.sum()),views=[])
                for cam,(intr,ext) in cal.items():
                    R,T=ext['rotationMatrix'],ext['translationVector'];center=-T@R.T
                    ray=pelvis[:,:2]-center[:2];dot=np.sum(ray*vel[:,:2],axis=1)
                    denom=np.linalg.norm(ray,axis=1)*np.linalg.norm(vel[:,:2],axis=1)
                    deviation=np.degrees(np.arcsin(np.clip(np.abs(dot)/np.maximum(denom,1e-8),0,1)))
                    row['missing_view_markers']=[n for n in viewnames if n not in names]
                    points=np.stack([xyz[:,names.index(n)] if n in names else np.full((len(frames),3),np.nan) for n in viewnames],axis=1)
                    viewgaps=np.stack([gaps[:,names.index(n)] if n in names else np.ones(len(frames),bool) for n in viewnames],axis=1).any(axis=1)
                    camera3=points@R+T;xy=camera3[...,:2]/camera3[...,2,None];r2=(xy**2).sum(axis=-1);k1,k2=intr['RadialDistortion'];xy*= (1+k1*r2+k2*r2*r2)[...,None]
                    uv=(np.concatenate([xy,np.ones(xy.shape[:-1]+(1,))],axis=-1)@intr['IntrinsicMatrix'])[...,:2]-1
                    inside=(camera3[...,2]>0).all(axis=1)&(uv[...,0]>=0).all(axis=1)&(uv[...,0]<800).all(axis=1)&(uv[...,1]>=0).all(axis=1)&(uv[...,1]<600).all(axis=1)&~viewgaps
                    mask=marker_valid&inside&(speed>=.3)&(deviation<=15)
                    spans=runs(mask,frames);vname=f'F_{cam}_Subject_{sid}_L.avi';local=ROOT/'raw'/vname
                    view={'camera':cam,'video':vname,'listed':vname in files,'local':local.exists(),'side15_frames':int(mask.sum()),'max_contiguous_frames':max([v['frames'] for v in spans],default=0),'intervals':spans,
                          'side30_frames':int((marker_valid&inside&(speed>=.3)&(deviation<=30)).sum()),'median_deviation_moving':float(np.median(deviation[speed>=.3])) if (speed>=.3).any() else None}
                    if local.exists():
                        assert local.stat().st_size==files[vname]['dataFile']['filesize']
                        cap=cv2.VideoCapture(str(local));view.update(fps=cap.get(cv2.CAP_PROP_FPS),video_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                        cap.set(cv2.CAP_PROP_POS_FRAMES,a-1);decoded=0
                        for _ in frames:
                            ok,image=cap.read()
                            if not ok:break
                            decoded+=1
                        cap.release();view['walking_decoded_frames']=decoded
                    row['views'].append(view)
                rows.append(row)
                if len(rows)%10==0: print('MoVi inspected',len(rows),flush=True)
                (HERE/'movi-audit.json').write_text(json.dumps({'screen':{'round':'F','side_deviation_max_deg':15,'minimum_speed_m_s':.3,'moving_direction_window_frames':11,'in_frame_points':viewnames,'required_markers':need,'side_is_geometric_not_visual_or_clinical_validation':True},'archives':archive_checks,'subjects':rows},indent=2)+'\n')
    assert len(rows)==90
    print(json.dumps({'subjects':len(rows),'walking':sum(x['walking'] for x in rows),'view_thresholds':{str(n):sum(v['listed'] and v['max_contiguous_frames']>=n for r in rows for v in r.get('views',[])) for n in [30,60,124]}}))

if __name__=='__main__':main()
