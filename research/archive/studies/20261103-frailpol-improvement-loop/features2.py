from pathlib import Path
import sys,os,json,zipfile,io,hashlib
sys.path.insert(0,'D:/Projects/D_P/studies/20261103-frailpol-short-walk')
from signal_pilot import filemap,rawread,save
import numpy as np
from scipy.spatial.transform import Rotation
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');SRC=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):
    z=v-v.mean();f=np.fft.rfftfreq(len(z),.01);power=abs(np.fft.rfft(z))**2;total=power[(f>=.5)&(f<=20)].sum()
    return dict(std=float(np.std(v)),iqrange=float(np.quantile(v,.9)-np.quantile(v,.1)),jerk_rms=float(np.sqrt(np.mean(np.diff(v)**2))*100),high_frequency_fraction=float(power[(f>=5)&(f<=20)].sum()/total) if total>1e-16 else 0.)
def features(raw,window):
    lo,hi=window['core_start'],window['core_end'];ref=raw['sacrum back'].PacketCounter.iloc[lo:hi].to_numpy();result={};waves={};invalid=[]
    for sensor in ['left ankle','right ankle','sacrum back','left wrist','right wrist']:
        d=raw[sensor];valid=d.PacketCounter.is_unique
        if valid:d=d.set_index('PacketCounter').reindex(ref);valid=bool(np.isfinite(d.to_numpy(float)).all())
        if valid:
            r=Rotation.from_euler('xyz',d[['Roll','Pitch','Yaw']],degrees=True);w=(r[:-1].inv()*r[1:]).as_rotvec()*100;g=d[['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy();gg=(g[:-1]+g[1:])/2;mask=np.linalg.norm(gg,axis=1)>.2;den=np.sum(gg[mask]**2)
            valid=den>1e-10 and .8<np.sum(w[mask]*gg[mask])/den<1.2
        if not valid:invalid.append(sensor)
        for kind in ['Acc','Gyr']:
            x=d[[kind+'_'+a for a in ['X','Y','Z']]].to_numpy() if valid else None
            if valid and kind=='Gyr':x=np.rad2deg(x)
            for axis in ['X','Y','Z','norm']:
                v=(np.linalg.norm(x,axis=1) if axis=='norm' else x[:,['X','Y','Z'].index(axis)]) if valid else None
                for name,value in (stats(v).items() if valid else [(s,None) for s in ['std','iqrange','jerk_rms','high_frequency_fraction']]):result[f'{sensor}_{kind}_{axis}_{name}']=value
            waves[sensor+'_'+kind]=np.linalg.norm(x,axis=1) if valid else None
    for kind in ['Acc','Gyr']:
      for left,right in [('left ankle','right ankle'),('left wrist','right wrist'),('left ankle','left wrist'),('right ankle','right wrist')]:
        a=waves[left+'_'+kind];b=waves[right+'_'+kind];value=None
        if a is not None and b is not None and np.std(a)>1e-10 and np.std(b)>1e-10:value=float(np.corrcoef(a,b)[0,1])
        result[f'coordination_{left}_{right}_{kind}']=value
    return result,invalid
def main():
    rows=json.loads((SRC/'cohorts.json').read_text());rows=[r for r in rows if r['status']=='ok' and r['tug'] is not None];h=H/'round02';h.mkdir(exist_ok=True);b=B/'round02';b.mkdir(exist_ok=True)
    proto=dict(code_sha256=sha(Path(__file__)),source_sha256=sha(SRC/'cohorts.json'),raw_zip_sha256=sha(SRC/'raw/IMU_signals.zip'),design='Outcome-blind168 descriptors, same original core windows, align each sensor viaPacketCounter; orientation-derived rad/s check; invalid optional sensor features missing, retain all594. No whole-record statistics, window location,clinical covariates or target information. Stats standard deviation,10-90range,jerkRMS,5-20Hzpowerfraction;8 norm correlations. These are single-cycle descriptors, not validated gait variability.',verification='Two independent extraction passes all594; compare exact. Perturb all sensor samples outside core window for first12 and retain invariance. Missing values imputed training-only during modeling.')
    if not(h/'feature-protocol.json').exists():save(h/'feature-protocol.json',proto)
    else:assert json.loads((h/'feature-protocol.json').read_text())==proto
    outputs=[];audit=[]
    with zipfile.ZipFile(SRC/'raw/IMU_signals.zip') as z:
      mapping=filemap(z)
      for n,row in enumerate(rows):
        raw=rawread(z,mapping,row['id']);f,bad=features(raw,row['window']);again,_=features(rawread(z,mapping,row['id']),row['window']);assert f==again
        if n<12:
            ref=raw['sacrum back'].PacketCounter.iloc[row['window']['core_start']:row['window']['core_end']].to_numpy();alter={s:d.copy() for s,d in raw.items()}
            for d in alter.values():d.loc[~d.PacketCounter.isin(ref),d.columns!='PacketCounter']=99999.
            af,_=features(alter,row['window']);assert af==f
        outputs.append(dict(id=row['id'],features={**row['features'],**f}));audit.append(dict(id=row['id'],invalid_optional_sensors=bad))
        if (n+1)%100==0:print('verified extraction',n+1,flush=True)
    save(b/'features.json',outputs);save(h/'feature-verification.json',dict(status='passed',people=len(rows),feature_count=len(outputs[0]['features']),replay_count=len(rows),perturbation_count=12,features_sha256=sha(b/'features.json'),audit=audit))
if __name__=='__main__':main()
