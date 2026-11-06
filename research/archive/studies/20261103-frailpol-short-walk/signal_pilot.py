from pathlib import Path
import os,sys,json,zipfile,io,re,time,traceback
HERE=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk')
BIG=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103')
os.environ['NUMBA_CACHE_DIR']=str(BIG/'cache/numba');os.environ['MPLCONFIGDIR']=str(BIG/'cache/mpl');os.environ['OMP_NUM_THREADS']='2'
sys.stdout.reconfigure(encoding='utf8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.event_detection import HerzerEventDetection
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from scipy.spatial.transform import Rotation
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def filemap(z):
    result={}
    for n in z.namelist():
        m=re.match(r'(\d+)-(.+)\.csv$',Path(n).name,re.I)
        if m:result.setdefault(int(m[1]),{})[m[2].lower().replace('_',' ')]=n
    return result
def rawread(z,m,pid):return {s:pd.read_csv(io.BytesIO(z.read(n))) for s,n in m[pid].items()}
def fsf(raw):
    result={}
    for side in ['left','right']:
        d=raw[side+' ankle'];sign=1 if side=='left' else -1
        x={}
        for k in ['Acc','Gyr']:
            for dest,source,sgn in [('x','Y',sign),('y','Z',sign),('z','X',1)]:x[k.lower()+'_'+dest]=d[k+'_'+source].to_numpy()*sgn*(180/np.pi if k=='Gyr' else 1)
        result[side+'_sensor']=pd.DataFrame(x)
    return result
def main():
    with zipfile.ZipFile(BIG/'raw/IMU_signals.zip') as z:
        m=filemap(z);ids=sorted(m);print('people',len(ids),'positions',m[ids[0]],flush=True)
        pilot=[1,9,63]+sorted(np.random.RandomState(20261103).choice([p for p in ids if p not in [1,9,63]],9,replace=False).tolist())
        fig,ax=plt.subplots(len(pilot),2,figsize=(16,2.6*len(pilot)),squeeze=False);report=[]
        for row,pid in enumerate(pilot):
            t=time.monotonic();raw=rawread(z,m,pid);r=dict(id=pid,sensors={s:dict(n=len(d),columns=list(d),missing=int(d.isna().sum().sum()),first_counter=float(d.PacketCounter.iloc[0]),last_counter=float(d.PacketCounter.iloc[-1]),counter_steps=np.unique(np.diff(d.PacketCounter),return_counts=True)[0].tolist(),acc_norm_median=float(np.median(np.linalg.norm(d[['Acc_X','Acc_Y','Acc_Z']],axis=1)))) for s,d in raw.items()})
            try:
                a=fsf(raw);fbf=convert_to_fbf(a,left_like='left',right_like='right');dtw=BarthDtw();dtw.segment(fbf,sampling_rate_hz=100)
                ed=HerzerEventDetection(min_vel_search_win_size_ms=130,mid_swing_peak_prominence=.03*180/np.pi,mid_swing_n_considered_peaks=2).detect(fbf,stride_list=dtw.stride_list_,sampling_rate_hz=100)
                r['strides']={s:d.replace({np.nan:None}).reset_index().to_dict('records') for s,d in ed.min_vel_event_list_.items()}
                for s,c in [('left_sensor','tab:blue'),('right_sensor','tab:orange')]:
                    ax[row,0].plot(np.arange(len(fbf[s]))/100,fbf[s].gyr_ml,c=c,lw=.6,label=s)
                    for _,v in ed.min_vel_event_list_[s].iterrows():ax[row,0].axvspan(v.start/100,v.end/100,color=c,alpha=.12)
                sac=raw.get('sacrum back',raw.get('hips back'))
                if sac is not None:
                    rot=Rotation.from_euler('xyz',sac[['Roll','Pitch','Yaw']],degrees=True)
                    forward=rot.apply(np.tile([0,0,-1],(len(sac),1)));heading=np.rad2deg(np.unwrap(np.arctan2(forward[:,1],forward[:,0])))
                    ax[row,1].plot(np.arange(len(sac))/100,heading,label='projected body heading')
                    ax[row,1].plot(np.arange(len(sac))/100,np.rad2deg(rot.apply(sac[['Gyr_X','Gyr_Y','Gyr_Z']])[:,2]),alpha=.6,label='earth up angular velocity (deg/s)')
                r['gyro_unit_checks']={}
                for sensor,d in raw.items():
                    rot=Rotation.from_euler('xyz',d[['Roll','Pitch','Yaw']],degrees=True);omega=(rot[:-1].inv()*rot[1:]).as_rotvec()*100;g=d[['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy();g=(g[:-1]+g[1:])/2;mask=np.linalg.norm(g,axis=1)>.2
                    r['gyro_unit_checks'][sensor]=dict(orientation_rad_per_s_vs_raw_slope=float(np.sum(omega[mask]*g[mask])/np.sum(g[mask]**2)),expected_if_raw_deg_per_s=float(np.pi/180),magnitude_correlation=float(np.corrcoef(np.linalg.norm(omega[mask],axis=1),np.linalg.norm(g[mask],axis=1))[0,1]))
                r['seconds']=time.monotonic()-t
            except Exception as e:r['error']=repr(e);traceback.print_exc()
            for col in range(2):ax[row,col].set_title(f'Participant {pid}');ax[row,col].grid(alpha=.2);ax[row,col].legend(fontsize=6)
            report.append(r);save(HERE/'signal-pilot-corrected.json',report);print(pid,'seconds',round(time.monotonic()-t,2),'events',{k:len(v) for k,v in r.get('strides',{}).items()},r.get('error',''),flush=True)
        fig.tight_layout();fig.savefig(HERE/'signal-pilot-corrected.png',dpi=110);plt.close(fig)
if __name__=='__main__':main()
