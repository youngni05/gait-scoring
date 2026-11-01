from pathlib import Path
import sys,io,json,hashlib,re,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
SOURCE=HERE.parent/'20261030-older44-sppb-timeseries/candidate-audit/nmd_opencap_participant_info.csv'
ANGLES=['pelvis_tilt','pelvis_list']+[f'{a}_{side}' for side in ['r','l'] for a in ['hip_flexion','hip_adduction','hip_rotation','knee_angle','ankle_angle','subtalar_angle']]+['lumbar_extension','lumbar_bending','lumbar_rotation']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d): (HERE/n).write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')
def read(p):
    text=p.read_text();assert 'inDegrees=yes' in text
    return np.genfromtxt(io.StringIO(text.split('endheader',1)[1].strip()),names=True)
def active(a):
    t=a['time'];assert np.all(np.diff(t)>0)
    dt=float(np.median(np.diff(t)));v=np.hypot(np.gradient(a['pelvis_tx'],t),np.gradient(a['pelvis_tz'],t))
    width=max(1,round(.25/dt));smooth=np.convolve(np.pad(v,(width//2,width-1-width//2),mode='edge'),np.ones(width)/width,mode='valid')
    mask=smooth>.2
    for i in range(1,len(mask)-1):
        if not mask[i] and mask[i-1]:
            end=i
            while end<len(mask) and not mask[end]:end+=1
            if end<len(mask) and t[end]-t[i]<=.25:mask[i:end]=True
    for n in ANGLES+['pelvis_tx','pelvis_ty','pelvis_tz']:mask&=np.isfinite(a[n])
    cuts=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(cuts==1);ends=np.flatnonzero(cuts==-1)-1
    return [(int(s),int(e),float(t[e]-t[s])) for s,e in zip(starts,ends)]
def window(a,start,seconds):
    t=a['time'];end=np.searchsorted(t,t[start]+seconds,side='left');assert end<len(t)
    b=a[start:end+1];tt=b['time'];grid=tt[0]+np.arange(round(seconds*30)+1)/30
    angles=np.stack([np.interp(grid,tt,b[n]) for n in ANGLES])
    pos=np.stack([np.interp(grid,tt,b[n]) for n in ['pelvis_tx','pelvis_ty','pelvis_tz']])
    vel=np.gradient(pos,1/30,axis=1);x=np.concatenate([angles,np.hypot(vel[0],vel[2])[None],vel[1,None]],axis=0)
    assert x.shape==(19,round(seconds*30)+1) and np.isfinite(x).all()
    return x.astype(np.float32)
def main():
    selected=json.loads((HERE/'selected-files.json').read_text());assert all((HERE/'raw'/x['name']).exists() for x in selected)
    labels=pd.read_csv(SOURCE);assert not labels.duplicated(['subid','visit']).any()
    lookup=labels.set_index(['subid','visit']);records=[];cache={}
    for x in selected:
        name=x['name'];pid,visit,ses=re.search(r'(sub-\d+)/visit-(\d+)/ses-(\d+)',name).groups();a=read(HERE/'raw'/name);segs=active(a);cache[name]=(a,segs)
        y=lookup.loc[(pid,int(visit)),'tft_tug'] if (pid,int(visit)) in lookup.index else np.nan
        records.append({'id':pid,'visit':int(visit),'session':int(ses),'file':name,'source_seconds':float(a['time'][-1]-a['time'][0]),'active_max_seconds':max([s[2] for s in segs],default=0),'target':float(y) if np.isfinite(y) else None})
    save('record-audit.json',records)
    for secs in [1,3,6]:
        eligible=[r for r in records if r['active_max_seconds']>=secs and r['target'] is not None and r['target']>0]
        print('ELIGIBLE',secs,'records',len(eligible),'people',len(set(r['id'] for r in eligible)),flush=True)
    if '--audit' in sys.argv:return
    eligible=[r for r in records if r['active_max_seconds']>=6 and r['target'] is not None and r['target']>0]
    chosen={}
    for r in sorted(eligible,key=lambda r:(r['id'],r['visit'],r['session'])):chosen.setdefault(r['id'],r)
    rows=list(chosen.values());ids=np.array([r['id'] for r in rows]);folds=np.zeros(len(rows),int)
    for fold,(_,test) in enumerate(KFold(5,shuffle=True,random_state=20261030).split(ids)):folds[test]=fold
    for secs in [1,3,6]:
        out=HERE/f'{secs}s';out.mkdir(exist_ok=True);xs=[];man=[]
        for r in rows:
            a,segs=cache[r['file']];start=next(s for s,e,dur in segs if dur>=6)
            xs.append(window(a,start,secs));man.append(dict(r,start_frame=start,start_time=float(a['time'][start]),seconds=secs,source_sha256=sha(HERE/'raw'/r['file'])))
        np.savez_compressed(out/'input.npz',x=np.stack(xs),y=np.array([r['target'] for r in rows]),ids=ids,folds=folds,frames=np.full(len(rows),secs*30+1))
        protocol={'target':'clinician TUG seconds','seconds':secs,'people':len(rows),'channels':ANGLES+['pelvis_horizontal_speed','pelvis_vertical_velocity'],'cohort':'same people; first eligible visit/session with6s active span','active_rule':'0.25s mean horizontal pelvis speed >0.2m/s; close gaps<=0.25s; earliest continuous6s span; all conditions same start','normalization':'train-person only; velocities from selected window only','outer_seed':20261030,'inner_seed':20261031,'source_label_sha256':sha(SOURCE),'scope':'OpenCap processed kinematics from2-3RGB cameras; original IK/scaling may use full capture; not raw single-camera or prospective clinical validation'}
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2));pd.DataFrame(man).to_csv(out/'manifest.csv',index=False)
        (out/'input-audit.json').write_text(json.dumps({'input_sha256':sha(out/'input.npz'),'protocol_sha256':sha(out/'protocol.json')},indent=2))
        for n in ['common.py','cpu.py','gpu.py']:
            text=(HERE.parent/'20261030-older44-sppb-timeseries'/n).read_text();text=text.replace('cin=39','cin=19').replace('OLDER44_SPPB','OPENCAP415_TUG')
            if n=='common.py':text=text.replace('ROOT = HERE.parents[2]','ROOT = HERE.parents[3]')
            (out/n).write_text(text)
    save('comparison-protocol.json',{'primary':[1,3,6],'same_cohort_and_splits':True,'people':len(rows),'unit':'first eligible visit per person','selection_uses':'finite inputs, lengths and presence ofpositiveTUG; no performance selection','baseline':'Mean Median Ridge RF plus small CNN TCN','reuse_129':'not appended; fresh fits on expanded release; overlap subgroup descriptive only'})
    print('PREPARED',len(rows),flush=True)
if __name__=='__main__':main()
