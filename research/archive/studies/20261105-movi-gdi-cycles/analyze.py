import sys,io,json,csv,tarfile,hashlib,struct
from pathlib import Path
sys.path.insert(0,'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt,find_peaks
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
import cv2
HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent/'20261105-kuopio-movi-audit'
ROOT=Path('D:/gait-research-data/MoVi/raw')
BIG=Path('D:/gait-research-data/D_P/kuopio-movi-audit-20261105/movi-video-ranges')
REF=np.load('outputs/gdi-reference-investigation-20261030/gdi_original_active166_reference.npz')
FPS=120
def dump(name,obj): (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def unit(v):
    norm=np.linalg.norm(v,axis=-1,keepdims=True)
    return v/np.where(norm>1e-9,norm,np.nan)
def frame_from_yz(y,z):
    z=unit(z);y=unit(y-(y*z).sum(-1,keepdims=True)*z);x=unit(np.cross(y,z));return np.stack([x,y,z],axis=-1)
def frame_from_xy(x,y):
    x=unit(x);y=unit(y-(y*x).sum(-1,keepdims=True)*x);z=unit(np.cross(x,y));return np.stack([x,y,z],axis=-1)
def filter_xyz(x):
    out=x.copy();sos=butter(4,6,fs=FPS,output='sos')
    for j in range(x.shape[1]):
        valid=np.isfinite(x[:,j]).all(-1);edges=np.diff(np.r_[False,valid,False].astype(int));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
        for a,b in zip(starts,ends):
            if b-a>27:out[a:b,j]=sosfiltfilt(sos,x[a:b,j],axis=0)
    return out
def euler(mat,seq):
    good=np.isfinite(mat).all(axis=(-2,-1));out=np.full((len(mat),3),np.nan)
    out[good]=Rotation.from_matrix(mat[good]).as_euler(seq,degrees=True)
    return out
def relative(a,b):return np.einsum('nji,njk->nik',a,b)
def nine_angles(p,side,progress):
    sign=1 if side=='L' else -1
    front=(p('LFWT')+p('RFWT'))/2;back=(p('LBWT')+p('RBWT'))/2
    pelvis=frame_from_xy(front-back,p('LFWT')-p('RFWT'))
    thigh=frame_from_yz(sign*(p(side+'KNE')-p(side+'KNEIN')),p(side+'HIP')-p(side+'KJC'))
    shank=frame_from_yz(thigh[:,:,1],p(side+'KJC')-p(side+'AJC'))
    foot=frame_from_xy(p(side+'TOE')-p(side+'HEE'),sign*(p(side+'MT5')-p(side+'MT1')))
    foot_alt=frame_from_xy(p(side+'TOE')-p(side+'HEE'),shank[:,:,1])
    global_frame=np.stack([progress,np.cross([0.,0.,1.],progress),[0.,0.,1.]],axis=-1)
    pe=euler(np.einsum('ij,njk->nik',global_frame.T,pelvis),'YXZ');hi=euler(relative(pelvis,thigh),'YXZ');kn=euler(relative(thigh,shank),'YXZ');an=euler(relative(shank,foot),'YZX');aa=euler(relative(shank,foot_alt),'YZX')
    foot_forward=foot[:,:,0];yaw=np.degrees(np.arctan2(foot_forward@global_frame[:,1],foot_forward@global_frame[:,0]))
    angles=np.column_stack([pe[:,0],sign*pe[:,1],-sign*pe[:,2],-hi[:,0],-sign*hi[:,1],-sign*hi[:,2],kn[:,0],-an[:,0],-sign*yaw])
    alt=angles.copy();alt[:,7]=-aa[:,0]
    orth=max(float(np.nanmax(np.abs(relative(z,z)-np.eye(3)))) for z in [pelvis,thigh,shank,foot])
    return angles,alt,orth
def score(curve):
    f=curve.T.reshape(459)@REF['feature_basis'];d=np.linalg.norm(f-REF['control_feature_center'])
    return float(100-10*(np.log(d)-REF['raw_distance_log_mean'])/REF['raw_distance_log_sample_sd'])
def resample(x,a,b):
    t=np.linspace(a,b,51);return np.column_stack([np.interp(t,np.arange(len(x)),x[:,j]) for j in range(9)])
def source_frame(sid,frame):
    head=(BIG/f'{sid:02d}-header.bin').read_bytes();tail=(BIG/f'{sid:02d}-tail.bin').read_bytes();movi=head.find(b'movi');ix=tail.rfind(b'idx1');size=struct.unpack_from('<I',tail,ix+4)[0]
    entries=[struct.unpack_from('<4sIII',tail,ix+8+i) for i in range(0,size,16)];entries=[x for x in entries if x[0]==b'00dc'];e=entries[frame-1]
    meta=json.loads((BIG/f'{sid:02d}-verification.json').read_text());blob=(BIG/f'{sid:02d}-walking.bin').read_bytes();off=movi+e[2]-meta['walking_range'][0]
    assert blob[off:off+4]==b'00dc'
    img=cv2.imdecode(np.frombuffer(blob[off+8:off+8+e[3]],np.uint8),cv2.IMREAD_COLOR);assert img is not None
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
def main():
    old=json.loads((AUDIT/'movi-audit.json').read_text());selected={r['id']:r for r in old['subjects'] if next(v for v in r['views'] if v['camera']=='PG1')['listed'] and next(v for v in r['views'] if v['camera']=='PG1')['max_contiguous_frames']>=60};assert len(selected)==28
    allcycles=[];subjects=[];curves=[];alternative=[];contacts=[];signals={};transform_audit=[]
    for archive in ['F_Subjects_1_45.tar','F_Subjects_46_90.tar']:
        with tarfile.open(ROOT/archive) as tar:
            for member in tar:
                if not member.name.endswith('.mat'):continue
                sid=int(Path(member.name).stem.split('_')[-1])
                if sid not in selected:continue
                src=tar.extractfile(member).read();assert hashlib.sha256(src).hexdigest()==selected[sid]['sha256']
                m=loadmat(io.BytesIO(src),simplify_cells=True)[f'Subject_{sid}_F']['move'];names=list(m['markerName']);r=selected[sid];view=next(v for v in r['views'] if v['camera']=='PG1');run=next(s for s in view['intervals'] if s['frames']>=60)
                offset=(r['walking_start']-1)*4;end=(r['walking_end']-1)*4
                raw=m['markerLocation'][offset:end+1].copy();gaps=m['markerGaps'][offset:end+1].astype(bool);raw[gaps]=np.nan;xyz=filter_xyz(raw);p=lambda name:xyz[:,names.index(name)]
                lo=(run['start_one_based']-1)*4-offset;hi=(run['end_one_based']-1)*4-offset
                pel=np.nanmean(xyz[:,[names.index(n) for n in ['LFWT','RFWT','LBWT','RBWT']]],axis=1)
                direction=pel[hi]-pel[lo];direction[2]=0;direction=unit(direction)
                events={};angles={};alts={};maxorth=0
                for side in ['L','R']:
                    heel_ap=(p(side+'HEE')-pel)@direction;toe_ap=(p(side+'TOE')-pel)@direction
                    hs=find_peaks(heel_ap,distance=42,prominence=50)[0];to=find_peaks(-toe_ap,distance=42,prominence=50)[0]
                    hs=hs[(hs>=lo)&(hs<=hi)];to=to[(to>=lo)&(to<=hi)]
                    events[side]={'hs':hs.tolist(),'to':to.tolist()};angles[side],alts[side],orth=nine_angles(p,side,direction);maxorth=max(maxorth,orth)
                    signals[f'{sid}_{side}']=np.column_stack([np.arange(len(xyz))+offset+1,heel_ap,toe_ap,p(side+'HEE')[:,2],p(side+'TOE')[:,2]])
                required=['LFWT','RFWT','LBWT','RBWT']+[s+n for s in ['L','R'] for n in ['HIP','KJC','AJC','KNE','KNEIN','HEE','TOE','MT1','MT5']]
                q=[names.index(n) for n in required];valid=np.isfinite(raw[:,q]).all(axis=(1,2))
                person={'id':sid,'age':r['age'],'run':run,'mocap_start_one_based':offset+lo+1,'mocap_end_one_based':offset+hi+1,'direction':direction.tolist(),'events_relative_to_walking':events,'orthogonality_max_error':maxorth,'candidate_cycles':0,'accepted_cycles':0}
                for side in ['L','R']:
                    hs=events[side]['hs'];other=events['R' if side=='L' else 'L']['hs']
                    for a,b in zip(hs[:-1],hs[1:]):
                        tos=[x for x in events[side]['to'] if a<x<b];opp=[x for x in other if a<x<b];duration=(b-a)/FPS;exc=float(np.nanmax(p(side+'HEE')[a:b+1,2])-np.nanmin(p(side+'HEE')[a:b+1,2]));advance=float((p(side+'HEE')[b]-p(side+'HEE')[a])@direction)
                        reasons=[]
                        if not .6<=duration<=1.6:reasons.append('duration')
                        if len(tos)!=1:reasons.append('toe_off_count')
                        if len(opp)!=1:reasons.append('opposite_contact_count')
                        if len(tos)==1 and not .2<(tos[0]-a)/(b-a)<.85:reasons.append('toe_off_phase')
                        if exc<20:reasons.append('heel_excursion')
                        if advance<200:reasons.append('stride_progression')
                        if not valid[a:b+1].all():reasons.append('missing_marker')
                        if not np.isfinite(angles[side][a:b+1]).all():reasons.append('nonfinite_angle')
                        cycle={'id':sid,'side':side,'start_mocap_one_based':offset+a+1,'end_mocap_one_based':offset+b+1,'start_video_nearest':int(round((offset+a)/4))+1,'end_video_nearest':int(round((offset+b)/4))+1,'duration_s':duration,'toe_off_percent':100*(tos[0]-a)/(b-a) if len(tos)==1 else None,'heel_excursion_mm':exc,'heel_progression_mm':advance,'accepted':not reasons,'reasons':reasons}
                        person['candidate_cycles']+=1
                        if not reasons:
                            curve=resample(angles[side],a,b);alt=resample(alts[side],a,b);cycle['curve_index']=len(curves);curves.append(curve);alternative.append(alt);cycle['exploratory_gdi']=score(curve);cycle['alternative_ankle_gdi']=score(alt)
                            cycle['phase_shift_gdi']={str(delta):score(resample(angles[side],a+delta,b+delta)) for delta in [-8,8] if a+delta>=lo and b+delta<=hi and valid[a+delta:b+delta+1].all()}
                            person['accepted_cycles']+=1
                            # Kinematic corroboration around each heel-position maximum, not a forceplate validation.
                            diagnostics=[]
                            for h in [a,b]:
                                z=p(side+'HEE')[:,2];s=max(0,h-12);e=min(len(z),h+13)
                                diagnostics.append({'heel_height_mm':float(z[h]),'height_above_local_min_mm':float(z[h]-np.nanmin(z[s:e])),'heel_speed_after_100ms_mm_s':float(np.linalg.norm(p(side+'HEE')[min(h+12,len(z)-1)]-p(side+'HEE')[h])*10)})
                            cycle['contact_diagnostics']=diagnostics
                            contacts.append({'id':sid,'side':side,'curve_index':cycle['curve_index'],'frames':[cycle['start_video_nearest'],int(round((offset+tos[0])/4))+1,cycle['end_video_nearest']]})
                        allcycles.append(cycle)
                subjects.append(person)
                # Directly test generic FK export interpretations against supplied global joint centres.
                segments=list(m['segmentName']);parents=m['segmentParent'].astype(int)-1;af=m['jointsAffine_v3d'][offset+lo:offset+hi+1]
                ta={'id':sid,'interpretations':{}}
                for transposed in [False,True]:
                    memo={}
                    def fk(j):
                        if j not in memo:
                            local=af[:,j].copy()
                            if transposed:local[:,:3,:3]=local[:,:3,:3].transpose(0,2,1)
                            memo[j]=local if parents[j]<0 else fk(parents[j])@local
                        return memo[j]
                    err=[]
                    for seg,name in [('LTH','LHIP'),('LSK','LKJC'),('LFT','LAJC'),('RTH','RHIP'),('RSK','RKJC'),('RFT','RAJC')]:
                        delta=fk(segments.index(seg))[:,:3,3]-m['markerLocation'][offset+lo:offset+hi+1,names.index(name)];err.extend(np.linalg.norm(delta,axis=1).tolist())
                    x=np.array(err);ta['interpretations']['rotation_transposed' if transposed else 'as_stored']={'median_mm':float(np.nanmedian(x)),'max_mm':float(np.nanmax(x))}
                transform_audit.append(ta)
                print('person',sid,'cycles',person['accepted_cycles'],'of',person['candidate_cycles'],flush=True)
    assert len(subjects)==28
    subjects.sort(key=lambda s:s['id']);dump('subjects.json',subjects);dump('cycles.json',allcycles);dump('transform-audit.json',transform_audit);dump('contact-review-index.json',contacts)
    np.savez_compressed(HERE/'angle-curves.npz',curves=np.array(curves),alternative_ankle=np.array(alternative),reference_curves=REF['control_angle_curves'],cycle_percent=REF['gait_cycle_percent'])
    np.savez_compressed(HERE/'event-signals.npz',**signals)
    cols=['id','side','start_video_nearest','end_video_nearest','duration_s','toe_off_percent','accepted','reasons','exploratory_gdi','alternative_ankle_gdi']
    with (HERE/'cycles.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(allcycles)
    qa=HERE/'qa';qa.mkdir(exist_ok=True)
    for page in range((len(contacts)+7)//8):
        group=contacts[page*8:page*8+8];sheet=Image.new('RGB',(960,len(group)*265),'white');draw=ImageDraw.Draw(sheet)
        for row,c in enumerate(group):
            for col,fr in enumerate(c['frames']):
                im=source_frame(c['id'],fr).resize((320,240));sheet.paste(im,(col*320,row*265));draw.text((col*320+3,row*265+241),f"ID {c['id']} {c['side']} | cycle {c['curve_index']} | {['HS','TO','HS'][col]} {fr}",fill='black')
        sheet.save(qa/f'contacts-{page+1}.jpg',quality=85)
    accepted=[c for c in allcycles if c['accepted']]
    summary={'people_screened':28,'people_with_accepted_cycles':sum(s['accepted_cycles']>0 for s in subjects),'people_with_bilateral_cycles':sum(any(c['id']==s['id'] and c['side']=='L' for c in accepted) and any(c['id']==s['id'] and c['side']=='R' for c in accepted) for s in subjects),'candidate_cycles':len(allcycles),'accepted_cycles':len(accepted),'left_cycles':sum(c['side']=='L' for c in accepted),'right_cycles':sum(c['side']=='R' for c in accepted),'no_cycle_ids':[s['id'] for s in subjects if not s['accepted_cycles']],'numeric_gdi_mean':float(np.mean([c['exploratory_gdi'] for c in accepted])),'numeric_gdi_range':[float(min(c['exploratory_gdi'] for c in accepted)),float(max(c['exploratory_gdi'] for c in accepted))],'training_labels_accepted':0,'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()}
    dump('summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
