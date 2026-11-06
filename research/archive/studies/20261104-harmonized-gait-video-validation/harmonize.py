"""Match one-foot HS-to-HS quantities, preserving detector/domain differences."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,zipfile,warnings,csv,time
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104')
OLD=Path('D:/Projects/D_P/studies/20261104-planar-tug-improvement')
G=Path('D:/gait-research-data/D_P/planar-tug-improvement-20261104/gstride')
FSOURCE=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104/data.npz')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def freeze():
    assert not (HERE/'harmonization-protocol.json').exists()
    save(HERE/'harmonization-protocol.json',dict(code_sha256=sha(__file__),
      question='Do matched one-foot HS-to-HS speed/time/swing quantities reduce between-cohort transfer errors?',
      harmonized='One earliest-ending valid complete same-foot heel-strike-to-next-heel-strike cycle, duration0.6..3s, speed0.05..2.5m/s,length0.1..3m,swing0..1. Each person one cycle,one foot,no bilateral or stride-count/missingness inputs. Unitsm/s,seconds,fraction. Speed=horizontal endpoint distance/cycle duration; swing=(nextHS-toeoff)/duration. Core3=speed,time,swing. Length recorded but redundant and not primary input.',
      FRAILPOL='Detect min-velocity cycles with unchanged Barth/Herzer settings; require exact contiguous pair on same foot with start<toeoff<HS<end. Reconstruct each adjacent cycle with same original StrideLevelTrajectory and80ms margins. Concatenate local positions at shared min-velocity boundary after yaw-aligning foot trajectories from sourceorientation metadata. Compute HS-to-HS displacement across seam. Require full two-cycle support core<=4s and original trunk headingrange<=20deg,horizontalforward>=0.5. Choose earliest-ending valid event pair before trajectory measurement; if trajectory fails markunmeasurable,no search for favorable feature. Original594 source people only.',
      GSTRIDE='Use independent source HS events and processed displacement, choose first chronological valid straight HS cycle (absTurningAngle<20deg), no later averaging. Source speed==length/time and swing==(end-toe)/(end-start) checked. Source uses different INS/eventdetector; no claim identical measurement error.',
      comparisons='Before=old bilateral FRAILPOL and old mean GSTRIDE quantities;after=one-foot HS quantities. Same retained people and originalfold assignments for both. Fixed ExtraTrees128/leaf10/seed20261104 from prior candidate, same train-only median preprocessing and inverse-speed augmentation. No model selection/tuning. SourceF5foldCV; externalG uses mean fiveFmodels and noGlabels; localG5fold; pooled excludes eachfold from bothdatasets. G data previously used in development, so external is cross-cohort diagnostic, not untouched confirmatory test.',
      metrics='TUG MAE,bias,p95,>=20MAE;5000pairedpersonbootstrap95CI. Everyconditionreported. Foldcoverage/personseparation/savedmodelreplay checked. Old andnew pooled changes reported percohort. No arbitrary clinical pass threshold.',
      limitations='Short TUG-contained walking versus independent longwalking remains. Different devices,placement,detectors and spatialreconstruction remain. Strict contiguouspair may exclude slower/shorter recordings; report coverage and excludedTUGdistribution. No RGB or mobile accuracy inferred.',
      resources='CPU,noGPU,newsourcefeatures<20MB,models<200MB underlargepath; existingrawF andprocessedG reused.',
      sources={str(p):sha(p) for p in [FSOURCE,OLD/'gstride-short-window-audit.json',G/'short-window-input.npz',Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk/extract.py')]}))
def prepare_f(pilot=False):
    import numpy as np
    sys.path.insert(0,'D:/Projects/D_P/studies/20261103-frailpol-short-walk')
    import extract as e
    # signal_pilot keeps its historical OMP setting; restore this study's single thread.
    os.environ['OMP_NUM_THREADS']='1'
    prot=json.loads((HERE/'harmonization-protocol.json').read_text());assert prot['code_sha256']==sha(__file__)
    ids=np.load(FSOURCE)['ids'].tolist();ids=ids[:12] if pilot else ids
    out=[];started=time.monotonic()
    with zipfile.ZipFile(e.BIG/'raw/IMU_signals.zip') as z:
      mapping=e.filemap(z)
      for no,pid in enumerate(ids):
        record={'id':pid}
        try:
          raw=e.rawread(z,mapping,pid);e.validate(raw)
          a=e.fsf(raw);bf=e.convert_to_fbf(a,left_like='left',right_like='right')
          with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dtw=e.BarthDtw(template=e.BarthOriginalTemplate(use_cols=['gyr_ml'])).segment(bf,sampling_rate_hz=100)
            ed=e.HerzerEventDetection(min_vel_search_win_size_ms=130,mid_swing_peak_prominence=.03*180/np.pi,mid_swing_n_considered_peaks=2).detect(bf,stride_list=dtw.stride_list_,sampling_rate_hz=100)
          h,horiz=e.heading(raw);pairs=[]
          for side,df in ed.min_vel_event_list_.items():
            ee=df.reset_index().to_dict('records')
            for first in ee:
              for second in ee:
                if first['end']!=second['start']:continue
                lo=int(first['start']);hi=int(second['end']);hs0=int(first['ic']);hs1=int(second['ic']);toe=int(second['tc'])
                if not all(q['start']<q['tc']<q['ic']<q['end'] and .6<=(q['end']-q['start'])/100<=3 for q in [first,second]):continue
                if not(.6<=(hs1-hs0)/100<=3 and hs0<toe<hs1 and hi-lo<=400 and lo>=8 and hi+9<=len(h)):continue
                if np.ptp(h[lo:hi])>20 or np.min(horiz[lo:hi])<.5:continue
                pairs.append((hs1,hs0,side,first,second))
          if not pairs:raise ValueError('no_contiguous_straight_HS_pair_within_4s_support')
          _,_,side,first,second=min(pairs,key=lambda q:(q[0],q[1],q[2]))
          positions=[];checks=[]
          for ev in [first,second]:
            lo=int(ev['start'])-8;hi=int(ev['end'])+9
            aa=e.fsf({k:d.iloc[lo:hi].reset_index(drop=True) for k,d in raw.items()})
            events={side:e.pd.DataFrame([{k:v-lo for k,v in ev.items() if k!='s_id'}],index=e.pd.Index([0],name='s_id'))}
            tr=e.StrideLevelTrajectory().estimate({side:aa[side]},stride_event_list=events,sampling_rate_hz=100)
            pp=tr.position_[side].xs(0,level='s_id').to_numpy().copy()
            # Initial gravity alignment leaves heading zero. Rotate each reconstruction into
            # a common heading using the same physical foot local axis in source orientation.
            orient=e.Rotation.from_euler('xyz',raw[side.replace('_sensor',' ankle')][['Roll','Pitch','Yaw']].iloc[int(ev['start'])],degrees=True)
            # fsf x is +/-rawY, consistent with sensor frame mapping used by trajectory.
            axis=orient.apply([0,1 if side=='left_sensor' else -1,0])
            local_q=tr.orientation_[side].xs(0,level='s_id').iloc[0].to_numpy()
            local_axis=e.Rotation.from_quat(local_q).apply([1,0,0])
            yaw=np.arctan2(axis[1],axis[0])-np.arctan2(local_axis[1],local_axis[0])
            rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            pp[:,:2]=pp[:,:2]@rot.T
            positions.append(pp)
            checks.append({'samples':len(pp),'expected_samples':int(ev['end']-ev['start'])+1,'heading_rad':float(yaw)})
            assert len(pp)==checks[-1]['expected_samples']
          p0=positions[0];p1=positions[1]
          hs0=int(first['ic']-first['start']);hs1=int(second['ic']-second['start'])
          displacement=(p0[-1]-p0[hs0])+(p1[hs1]-p1[0]);length=float(np.linalg.norm(displacement[:2]))
          duration=(second['ic']-first['ic'])/100;swing=(second['ic']-second['tc'])/100;speed=length/duration
          if not(.05<=speed<=2.5 and .1<=length<=3 and 0<=swing/duration<=1):raise ValueError('implausible_HS_trajectory')
          record.update(status='ok',X=[speed,duration,swing/duration],stride_length=length,side=side,events=[first,second],checks=checks,
            support_start=int(first['start'])-8,support_end=int(second['end'])+9)
        except Exception as ex:record.update(status='unmeasurable',reason=str(ex))
        out.append(record)
        if pilot or (no+1)%50==0:print('F_HS',no+1,len(ids),sum(r['status']=='ok' for r in out),round(time.monotonic()-started,1),flush=True)
    BIG.mkdir(parents=True,exist_ok=True);save(BIG/('F-pilot.json' if pilot else 'F-features.json'),out)
    save(HERE/('F-pilot-receipt.json' if pilot else 'F-receipt.json'),{'people':len(out),'measurable':sum(r['status']=='ok' for r in out),'sha256':sha(BIG/('F-pilot.json' if pilot else 'F-features.json')),'seconds':time.monotonic()-started})
def prepare_g():
    import numpy as np
    audit=json.loads((OLD/'gstride-short-window-audit.json').read_text());out=[]
    for r in audit['included']:
        pid=r['id'];ii=r['stride_indices'][0]
        params={int(float(x['index'])):x for x in csv.DictReader((G/'processed/_GAIT/_PARAMETERS'/f'{pid}_GAIT_PARAMETERS.csv').open(encoding='utf-8-sig'))}
        events={int(float(x['index'])):x for x in csv.DictReader((G/'processed/_GAIT/_SEGMENTATION'/f'{pid}_GAIT_SEGMENTATION.csv').open(encoding='utf-8-sig'))}
        q=params[ii];ev=events[ii];length=float(q['StrideLength[m]']);dt=float(q['CycleDuration[s]']);sp=float(q['Stride_Velocity[m/s]']);sw=float(q['Per_Swing[%CD]'])/100
        assert abs(sp-length/dt)<1e-7 and abs(sw-(float(ev['HS_fin[sample]'])-float(ev['TO[sample]']))/(float(ev['HS_fin[sample]'])-float(ev['HS_ini[sample]'])))<1e-6
        out.append({'id':pid,'X':[sp,dt,sw],'stride_length':length,'index':ii,'events':ev})
    save(BIG/'G-features.json',out);print('G_HS',len(out),flush=True)
if __name__=='__main__':
    {'freeze':freeze,'pilot':lambda:prepare_f(True),'F':prepare_f,'G':prepare_g}[sys.argv[1]]()
