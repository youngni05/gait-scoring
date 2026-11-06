"""Outcome-blind short-stride extraction. Clinical labels are never opened here."""
from signal_pilot import *
import argparse,hashlib,warnings
from scipy.ndimage import gaussian_filter1d
from gaitmap.stride_segmentation import BarthOriginalTemplate
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory
from gaitmap.parameters import SpatialParameterCalculation

FS=100
PILOT=[1,9,63,100,190,479,482,483,621,622,655,763]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def heading(raw):
    s=raw['sacrum back'];r=Rotation.from_euler('xyz',s[['Roll','Pitch','Yaw']],degrees=True)
    f=r.apply(np.tile([0,0,-1],(len(s),1)))
    return gaussian_filter1d(np.rad2deg(np.unwrap(np.arctan2(f[:,1],f[:,0]))),5),np.linalg.norm(f[:,:2],axis=1)

def validate(raw):
    chosen={s:raw[s] for s in ['left ankle','right ankle','sacrum back']}
    ref=next(iter(chosen.values())).PacketCounter.to_numpy()
    for s,d in chosen.items():
        if not np.array_equal(ref,d.PacketCounter):raise ValueError('sensor_timing_mismatch')
        if not np.all(np.mod(np.diff(ref),65536)==1):raise ValueError('missing_or_duplicate_packets')
        if not np.isfinite(d.to_numpy(float)).all():raise ValueError('nonfinite_raw_signal')
    units={}
    for s,d in chosen.items():
        r=Rotation.from_euler('xyz',d[['Roll','Pitch','Yaw']],degrees=True)
        w=(r[:-1].inv()*r[1:]).as_rotvec()*FS
        g=d[['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy();g=(g[:-1]+g[1:])/2;mask=np.linalg.norm(g,axis=1)>.2
        slope=float(np.sum(w[mask]*g[mask])/np.sum(g[mask]**2));units[s]=slope
        if not .8<slope<1.2:raise ValueError('gyro_unit_not_confirmed')
    return units

def calculate(raw,ev):
    # Each reconstruction receives only the selected short window plus 80 ms alignment margins.
    core_start=int(min(e['start'] for e in ev.values()));core_end=int(max(e['end'] for e in ev.values()))
    lo=core_start-8;hi=core_end+9
    a=fsf({s:d.iloc[lo:hi].reset_index(drop=True) for s,d in raw.items()})
    events={s:pd.DataFrame([{k:v-lo for k,v in e.items() if k!='s_id'}],index=pd.Index([0],name='s_id')) for s,e in ev.items()}
    tr=StrideLevelTrajectory().estimate({s:a[s] for s in events},stride_event_list=events,sampling_rate_hz=FS)
    sp=SpatialParameterCalculation().calculate(stride_event_list=events,positions=tr.position_,orientations=tr.orientation_,sampling_rate_hz=FS)
    b=convert_to_fbf(a,left_like='left',right_like='right');per={};waves=[]
    for s,e in ev.items():
        v=sp.parameters_[s].iloc[0];duration=(e['end']-e['start'])/FS
        q=dict(speed=float(v.stride_length/duration),stride_time=duration,swing_fraction=(e['ic']-e['tc'])/(e['end']-e['start']))
        for k in ['stride_length','max_sensor_lift','max_lateral_excursion','max_orientation_change','ic_angle','tc_angle']:q[k]=float(v[k])
        d=b[s].iloc[int(e['start'])-lo:int(e['end'])-lo]
        q['ankle_gyro_rms']=float(np.sqrt(np.mean(d.gyr_ml**2)));q['ankle_acc_std']=float(np.std(np.linalg.norm(d[['acc_pa','acc_ml','acc_si']],axis=1)))
        waves.append(np.interp(np.linspace(0,1,101),np.linspace(0,1,len(d)),d.gyr_ml))
        if not(.05<=q['speed']<=2.5 and .1<=q['stride_length']<=3 and 0<=q['max_sensor_lift']<=1):raise ValueError('implausible_trajectory')
        per[s]=q
    features={}
    for k in next(iter(per.values())):
        v=[p[k] for p in per.values()];features[k]=float(np.mean(v))
    # Trunk descriptors use this short core window only; axis labels retain sensor coordinates.
    sac=raw['sacrum back'].iloc[core_start:core_end]
    for kind in ['Acc','Gyr']:
        x=sac[[kind+'_'+a for a in ['X','Y','Z']]].to_numpy()
        if kind=='Gyr':x=np.rad2deg(x)
        for j,axis in enumerate(['X','Y','Z']):features[f'sacrum_{kind}_{axis}_std']=float(np.std(x[:,j]))
        features[f'sacrum_{kind}_norm_std']=float(np.std(np.linalg.norm(x,axis=1)))
    if not np.isfinite(list(features.values())).all():raise ValueError('nonfinite_features')
    return features,dict(core_start=core_start,core_end=core_end,support_start=lo,support_end_exclusive=hi,support_seconds=(hi-lo)/FS,per_side=per)

def extract(raw):
    units=validate(raw);a=fsf(raw);bf=convert_to_fbf(a,left_like='left',right_like='right')
    with warnings.catch_warnings(record=True) as ws:
        dtw=BarthDtw(template=BarthOriginalTemplate(use_cols=['gyr_ml'])).segment(bf,sampling_rate_hz=FS)
        ed=HerzerEventDetection(min_vel_search_win_size_ms=130,mid_swing_peak_prominence=.03*180/np.pi,mid_swing_n_considered_peaks=2).detect(bf,stride_list=dtw.stride_list_,sampling_rate_hz=FS)
    events={s:d.reset_index().to_dict('records') for s,d in ed.min_vel_event_list_.items()}
    def valid(e):return e['start']<e['tc']<e['ic']<e['end'] and .6<=(e['end']-e['start'])/FS<=3
    valid_events={s:[e for e in v if valid(e)] for s,v in events.items()}
    h,horizontal=heading(raw);pairs=[];counts=dict(events={s:len(v) for s,v in events.items()},overlap=0,short=0,straight=0)
    for l in valid_events['left_sensor']:
        for r in valid_events['right_sensor']:
            if min(l['end'],r['end'])<=max(l['start'],r['start']):continue
            counts['overlap']+=1;lo=int(min(l['start'],r['start']));hi=int(max(l['end'],r['end']))
            if (hi-lo+17)>400 or lo<8 or hi+9>len(h):continue
            counts['short']+=1
            if np.ptp(h[lo:hi])>20 or np.min(horizontal[lo:hi])<.5:continue
            counts['straight']+=1;pairs.append((hi,lo,l,r))
    audit=dict(unit_slopes=units,candidates=counts,warnings=[str(w.message) for w in ws])
    singles=[]
    for s,ee in valid_events.items():
        for e in ee:
            lo=int(e['start']);hi=int(e['end'])
            if lo<8 or hi+9>len(h):continue
            if np.ptp(h[lo:hi])>20 or np.min(horizontal[lo:hi])<.5:continue
            singles.append((hi,lo,s,e))
    audit['single_candidates']=len(singles)
    if not singles:return dict(status='unmeasurable',reason='no_short_straight_cycle',audit=audit)
    hi,lo,s,e=sorted(singles,key=lambda p:(p[0],p[1],p[2],p[3]['s_id']))[0]
    ev={s:e}
    try:features,window=calculate(raw,ev)
    except Exception as e:return dict(status='unmeasurable',reason=str(e),audit=audit,selected_events=ev)
    return dict(status='ok',features=features,window=window,selected_events=ev,audit=audit)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    records=[];t=time.monotonic()
    with zipfile.ZipFile(BIG/'raw/IMU_signals.zip') as z:
        m=filemap(z)
        for i,pid in enumerate(PILOT if args.pilot else sorted(m)):
            try:r=extract(rawread(z,m,pid))
            except Exception as e:r=dict(status='unmeasurable',reason=str(e))
            r['id']=pid;records.append(r)
            if args.pilot or (i+1)%50==0:print(i+1,pid,r['status'],r.get('reason',''),r.get('window',{}).get('support_seconds'),flush=True)
    path=BIG/('features-single-pilot.json' if args.pilot else 'features.json');save(path,records)
    save(HERE/('feature-single-pilot-receipt.json' if args.pilot else 'feature-receipt.json'),dict(people=len(records),measurable=sum(r['status']=='ok' for r in records),seconds=time.monotonic()-t,path=str(path),sha256=sha(path),code_sha256=sha(Path(__file__)),pilot_helper_sha256=sha(HERE/'signal_pilot.py')))
if __name__=='__main__':main()
