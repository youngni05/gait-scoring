"""Source-only feature extraction. No clinical label is used to choose a feature."""
from common import *
import csv, ast
from scipy.signal import savgol_filter

SOURCES={}
def track(p):
    p=Path(p);SOURCES[str(p)]=sha(p);return p
def legacy_functions():
    # Reuse only pure, previously audited CSV/event parsers, not its main experiment.
    path=track(OLD/'prepare.py');tree=ast.parse(path.read_text(encoding='utf8'))
    names=['section','load_trial','align','event_features','pooled_cv']
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAMES' for t in n.targets)]
    ns=dict(np=np,csv=csv)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)
    return ns
def spread(x):return float(np.quantile(x,.95)-np.quantile(x,.05))
def angle(u,v):
    return np.rad2deg(np.arccos(np.clip(np.sum(u*v,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)),-1,1)))
def older():
    old=legacy_functions();pp=read(track(OLD/'people.json'));audits=read(track(OLD/'trial-audit.json'))
    raw=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-sppb-timeseries/raw')
    cycles_by_person={p['id']:[] for p in pp};checks=[]
    for ri,r in enumerate(audits):
        path=track(raw/r['file']);assert SOURCES[str(path)]==r['source_sha256']
        fr,q,ev,params,dup=old['load_trial'](path)
        vals,cycles,rejected,speed,local=old['event_features'](fr,q,ev)
        assert abs(speed-r['speed'])<1e-12 and cycles==r['cycles']
        pelvis=local[:,10:14].mean(1);time=fr/100
        for c in cycles:
            left=c['side']=='Left';h,k,a,t,heel,opp=(0,2,4,6,8,9) if left else (1,3,5,7,9,8)
            ix=np.flatnonzero((time>=c['t0'])&(time<=c['t1']))
            base=dict(stride_time=c['stride'],step_length=c['length'],swing_percent=c['swing'],cadence=120/c['stride'])
            pos=c['t0']*100-fr[0]
            lo=int(np.floor(pos+1e-8));hi=int(np.ceil(pos-1e-8));w=pos-lo
            at=local[lo] if lo==hi else (1-w)*local[lo]+w*local[hi]
            base['step_width']=float(abs(at[heel,1]-at[opp,1]))
            # All coordinates must be observed for this cycle. No interpolation across gaps.
            v=local[ix];good=len(ix)>=max(11,.9*(c['t1']-c['t0'])*100) and np.isfinite(v).all()
            if good:
                pelvis_c=pelvis[ix];thigh=v[:,k]-v[:,h];foot=v[:,t]-v[:,heel]
                knee=180-angle(v[:,h]-v[:,k],v[:,a]-v[:,k])
                thigh_pitch=np.rad2deg(np.unwrap(np.arctan2(thigh[:,0],-thigh[:,2])))
                foot_pitch=np.rad2deg(np.unwrap(np.arctan2(foot[:,2],foot[:,0])))
                asi=v[:,10]-v[:,11];obliq=np.rad2deg(np.arctan2(asi[:,2],np.linalg.norm(asi[:,:2],axis=1)))
                rel=v[:,a,0]-pelvis_c[:,0]
                velocity=savgol_filter(rel,11,2,deriv=1,delta=.01)
                base.update(knee_ROM=spread(knee),thigh_sagittal_ROM=spread(thigh_pitch),foot_pitch_ROM=spread(foot_pitch),toe_vertical_excursion=spread(v[:,t,2]),ankle_relative_forward_speed_p95=float(np.quantile(abs(velocity),.95)),pelvis_vertical_excursion=spread(pelvis_c[:,2]),pelvis_obliquity_ROM=spread(obliq))
            cycles_by_person[r['id']].append(dict(side=c['side'],file=r['file'],t0=c['t0'],t1=c['t1'],features=base,kinematic_valid=bool(good)))
        if (ri+1)%55==0:print('older raw trials',ri+1,flush=True)
    names=sorted(set(k for cc in cycles_by_person.values() for c in cc for k in c['features']))
    gdi={p['id']:p['scores']['primary'] for p in read(track(OLD/'gdi-scores.json'))}
    rows=[]
    for p in pp:
        cc=cycles_by_person[p['id']];f={'speed':p['features']['speed']}
        for n in names:
            means=[np.mean([c['features'][n] for c in cc if c['side']==side and n in c['features']]) for side in ['Left','Right']]
            f[n+'_mean']=float(np.mean(means));f[n+'_asym_abs']=float(abs(means[0]-means[1]))
        assert all(np.isfinite(list(f.values())))
        rows.append(dict(id=p['id'],dataset='older',sppb=p['sppb'],age=p['age'],fold=p['fold'],strict=p['event_eligible'],features=f,extra={k:v for k,v in p['features'].items() if k!='speed'},gdi=gdi.get(p['id'])))
    save(BIG/'older-cycle-features.json',cycles_by_person)
    return rows

IMU_NAMES=['CycleDuration[s]','Cadence[steps/min]','StrideLength[m]','PathLength3D[%SL]','PathLength2D[%SL]','Clearance[m]','Stride_Velocity[m/s]','Per_Swing[%CD]','Per_Stance[%CD]','Per_Loading[%Stance]','Per_FootFlat[%Stance]','Per_Pushing[%Stance]','PitchTO[degrees]','PitchHS[degrees]','PeakAngleVelocity[degrees/s]','RoMPitch[degrees]']
def imu():
    pp=read(track(LARGE/'people.json'));rows=[];manifest={}
    for name in ['gait2care-manifest.json','gstride-manifest.json']:
        for r in read(track(LARGE/name)):manifest[r['file']]=r['sha256']
    for p in pp:
        path=track(p['source_file']);assert sha(path)==manifest[str(path)]
        df=pd.read_csv(path);df.columns=[c.replace(' ','') for c in df.columns]
        sides=['R_','L_'] if p['dataset']=='gait2care' else [''];groups={};masks={}
        for side in sides:
            a=df[[side+k for k in ['CycleDuration[s]','StrideLength[m]','Per_Swing[%CD]','Stride_Velocity[m/s]','TurningAngle[degrees]']]].to_numpy(float)
            mask=np.isfinite(a).all(1)&(a[:,0]>0)&(a[:,1]>0)&(a[:,2]>0)&(a[:,2]<100)&(a[:,3]>0)&(abs(a[:,4])<20)
            groups[side]=a[mask];masks[side]=mask
        f=dict(p['features']);assert abs(np.concatenate([v[:,3] for v in groups.values()]).mean()-f['speed'])<1e-12
        if p['dataset']=='gstride':f.pop('PCI')
        for j,key in enumerate(IMU_NAMES):
            arrays=[df.loc[masks[s],s+key].to_numpy(float) for s in sides]
            arrays=[v[np.isfinite(v)] for v in arrays]
            valid=min(map(len,arrays))>=15
            mean=float(np.concatenate(arrays).mean()) if valid else None
            sd=float(np.sqrt(sum(np.sum((a-a.mean())**2) for a in arrays)/sum(len(a)-1 for a in arrays))) if valid else None
            f[f'imu{j:02d}_mean']=mean;f[f'imu{j:02d}_SD']=sd
            if len(arrays)==2:f[f'imu{j:02d}_asym_abs']=float(abs(arrays[0].mean()-arrays[1].mean())) if valid else None
        rows.append(dict(id=p['id'],dataset=p['dataset'],sppb=p['sppb'],age=p['age'],community=p['community'],features=f))
    return rows

def main():
    BIG.mkdir(parents=True,exist_ok=True)
    data=older()+imu();cohorts={}
    for name in ['older37','older24','older44','gait2care','gstride']:
        if name.startswith('older'):
            people=[dict(p) for p in data if p['dataset']=='older' and (name=='older44' or p['gdi'] is not None) and (name!='older24' or p['strict'])]
            if name=='older24':
                for p in people:p['features']={**p['features'],**p['extra']}
        else:people=[dict(p) for p in data if p['dataset']==name]
        people=sorted(people,key=lambda p:p['id'])
        if not name.startswith('older'):
            for fold,(_,te) in enumerate(KFold(5,shuffle=True,random_state=20261103).split(people)):
                for i in te:people[i]['fold']=fold
        cohorts[name]=people
    weekly=ROOT/'gdi-research-harness/studies/20261031-mobilised-uncertainty-review/first-eligible-per-target.csv'
    w=pd.read_csv(track(weekly));w=w[w.audit_target=='SPPB'];assert len(w)==w.participantid.nunique()==2232
    inventory={n:dict(n=len(pp),features=len(pp[0]['features']),age65plus=sum(p['age']>=65 for p in pp),sppb_range=[min(p['sppb'] for p in pp),max(p['sppb'] for p in pp)],missing={k:sum(p['features'][k] is None for p in pp) for k in pp[0]['features'] if any(p['features'][k] is None for p in pp)}) for n,pp in cohorts.items()}
    inventory['mobilised_inventory_only']=dict(n=2232,source_rows=len(w),unique_people=int(w.participantid.nunique()),modality='weekly sensor summaries, not short video')
    save(BIG/'cohorts.json',cohorts);save(HERE/'inventory.json',inventory)
    save(HERE/'source-manifest.json',SOURCES)
    save(HERE/'preparation-receipt.json',dict(protocol_sha256=sha(HERE/'protocol.json'),cohorts_sha256=sha(BIG/'cohorts.json'),sources=len(SOURCES),code_sha256=sha(Path(__file__))))
    print(json.dumps(inventory,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
