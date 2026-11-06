from pathlib import Path
import json,sys,io,importlib.util
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261031-opencap-directional-pace'
spec=importlib.util.spec_from_file_location('pace_prior',PRIOR/'study.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
np,pd,read,sha,joblib=old.np,old.pd,old.read,old.sha,old.joblib
long=old.long
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ALPHAS=[.1,1.,10.,100.]
def save(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def fit(x,y,tr,te,alpha,view):return old.fit(x,y,tr,te,alpha,view)
def trc(path):
    lines=path.read_text().splitlines();meta=lines[2].split();assert meta[4]=='m';names=[x for x in lines[3].split('\t')[2:] if x]
    d=np.loadtxt(io.StringIO('\n'.join(lines[6:])));xyz=d[:,2:].reshape(len(d),len(names),3)
    assert len(d)==int(meta[2]) and len(names)==int(meta[3]) and np.isfinite(xyz).all()
    return d[:,1],{n:xyz[:,i] for i,n in enumerate(names)}
def keep_runs(mask,seconds=.5):
    mask=mask.copy()
    for lo,hi in long.runs(mask):
        if hi-lo<int(np.ceil(seconds*30-1e-9)):mask[lo:hi]=False
    return mask
def candidates(m,phase,scout):
    out={}
    for kind,pk,leg in [('departure','rise','pre'),('return','sit','post')]:
        if 'window' not in phase[pk] or not scout[leg]:continue
        w=[phase[pk]['window'][1],scout[leg][0]] if kind=='departure' else [scout[leg][1],phase[pk]['window'][0]]
        if w[1]-w[0]>=.5:out[kind]=w
    return out
def features(a,m,s,phase,marker):
    prior=m['budgets']['20'];duration=prior['observed_seconds'];cand=candidates(m,phase,s)
    mandatory_end=max(m['turn_end'],max(w[1] for w in m['tug_walk_windows']))
    left=prior['window'][0]
    if 'departure' in cand:left=max(cand['departure'][0],mandatory_end-duration)
    left=min(left,prior['window'][0])
    base,delay,rec,traces=long.extract(a,m,{**s,'start':left},duration)
    assert abs(rec['observed_seconds']-duration)<1e-6
    t=traces['time'];mid=(t[:-1]+t[1:])/2;n=len(mid);masks={};audit={'candidates':cand,'original_window':prior['window'],'window':rec['window'],'seconds':duration,'shift_seconds':prior['window'][0]-rec['window'][0], 'baseline_record':rec}
    for strict in [False,True]:
        suffix='strict' if strict else 'main';dep=np.zeros(n,bool);ret=dep.copy();rot=dep.copy();reasons={}
        if marker is not None:
            mt,mk=marker
            assert mt[0]<=t[0]+1e-6 and mt[-1]>=t[-1]-1e-6
            def interp(values):return np.column_stack([np.interp(t,mt,values[:,j]) for j in range(3)])
            native_center=(mk['RHip']+mk['LHip'])/2
            center=uniform_filter1d(interp(native_center),7,axis=0,mode='nearest');cv=np.gradient(center,1/30,axis=0)
            root=np.column_stack([np.interp(t,a['time'],a[k]) for k in ['pelvis_tx','pelvis_ty','pelvis_tz']]);root=uniform_filter1d(root,7,axis=0,mode='nearest');rv=np.gradient(root,1/30,axis=0)
            bad=np.zeros(n,bool);bad_parts={}
            for part,nt,pos in [('MOT_jump',a['time'],np.column_stack([a[k] for k in ['pelvis_tx','pelvis_ty','pelvis_tz']])),('marker_jump',mt,native_center)]:
                thisbad=np.zeros(n,bool)
                v=np.diff(pos,axis=0)/np.diff(nt)[:,None]
                for j in np.flatnonzero((np.linalg.norm(v[:,[0,2]],axis=1)>6)|(abs(v[:,1])>3)):thisbad|=(mid>=nt[j]-.15)&(mid<=nt[j+1]+.15)
                bad_parts[part]=thisbad;bad|=thisbad
            widthbad=np.zeros(n,bool)
            for names in [('RHip','LHip'),('RShoulder','LShoulder')]:
                widths=np.linalg.norm(interp(mk[names[0]])-interp(mk[names[1]]),axis=1);med=float(np.median(widths));widthbad|=(abs(widths[:-1]/max(med,1e-8)-1)>.25)
            bad_parts['marker_width']=widthbad;bad|=widthbad
            # Suppress interpolated/smoothed edge values; no claiming support outside the observed window.
            bad|=(mid<t[0]+.15)|(mid>t[-1]-.15)
            agreement=np.linalg.norm(rv[:,:][:,[0,2]]-cv[:,[0,2]],axis=1)[:-1]
            vh=.03 if strict else .05;tol=.03 if strict else .05;agree=.2 if strict else .3;minrun=.75 if strict else .5
            knee=np.interp(mid,a['time'],(a['knee_angle_r']+a['knee_angle_l'])/2);hip=np.interp(mid,a['time'],(a['hip_flexion_r']+a['hip_flexion_l'])/2)
            stable=(abs(rv[:-1,1])<=vh)&(abs(cv[:-1,1])<=vh)&(agreement<=agree)&(knee<60)&(abs(hip)<60)&~bad
            for kind,w in cand.items():
                candidate=(t[:-1]>=w[0]-1e-8)&(t[1:]<=w[1]+1e-8);leg=s['pre'] if kind=='departure' else s['post']
                refw=[leg[0],leg[0]+.5] if kind=='departure' else [leg[1]-.5,leg[1]]
                ref=(mid>=refw[0])&(mid<refw[1]);safe=np.zeros(n,bool)
                if sum(ref)>=10:
                    rootlevel=float(np.median(root[:-1,1][ref]));markerlevel=float(np.median(center[:-1,1][ref]))
                    height=(root[:-1,1]>=rootlevel-tol)&(center[:-1,1]>=markerlevel-tol)
                    safe=keep_runs(candidate&stable&height,minrun)
                if kind=='departure':dep=safe
                else:ret=safe
                reasons[kind]={'candidate_observed':float(candidate.sum()/30),'tracking_bad':float((candidate&bad).sum()/30),'bad_components':{key:float((candidate&val).sum()/30) for key,val in bad_parts.items()},'stable_before_run_filter':float((candidate&stable).sum()/30),'accepted':float(safe.sum()/30),'reference_samples':int(ref.sum())}
            hipline=uniform_filter1d(interp(mk['RHip'])-interp(mk['LHip']),7,axis=0,mode='nearest');shoulderline=uniform_filter1d(interp(mk['RShoulder'])-interp(mk['LShoulder']),7,axis=0,mode='nearest')
            angles=[np.unwrap(np.arctan2(v[:,2],v[:,0])) for v in [hipline,shoulderline]]
            omega=[np.gradient(v,1/30)*180/np.pi for v in angles]
            rotation=(abs(omega[0][:-1])>=20)&(abs(omega[0][:-1])<=180)&(omega[0][:-1]*omega[1][:-1]>0)&(abs(omega[0][:-1]-omega[1][:-1])<=45)
            rot=keep_runs(ret&rotation,.25)
        assert not (dep&ret).any() and not ((dep|ret)&traces['valid']).any()
        audit[suffix]={'departure_seconds':float(dep.sum()/30),'return_seconds':float(ret.sum()/30),'rotation_seconds':float(rot.sum()/30),'reasons':reasons}
        masks[suffix]={'departure':dep,'return':ret,'rotation':rot}
    matched=np.r_[base,delay['10'][2]];main=audit['main'];strict=audit['strict'];dd=main['departure_seconds'];rr=main['return_seconds'];rot=main['rotation_seconds']
    def add(seconds):return np.r_[base,delay['10'][2]+seconds]
    views={'Matched':matched,'AddDeparture':add(dd),'AddReturn':add(rr),'AddRotation':add(rot),'AddBoth':add(dd+rr),
           'LearnBoth':np.r_[base,dd,rr,delay['10'][2]],'AddBothStrict':add(strict['departure_seconds']+strict['return_seconds'])}
    return views,audit,{'time':t,'valid':traces['valid'],**{f'{k}_{kind}':v for k,mask in masks.items() for kind,v in mask.items()}}

def prepare():
    assert not (HERE/'protocol.json').exists()
    prior=PRIOR/'exclude283';data=np.load(prior/'input.npz');rows=json.loads((prior/'manifest.json').read_text());audits={r['id']:r for r in json.loads((HERE.parent/'20261031-opencap-tail-diagnosis/person-audit.json').read_text())}
    scouts={r['id']:r for r in json.loads((long.HERE/'scout.json').read_text())};receipts={r['name']:r for r in json.loads((HERE/'acquisition.json').read_text())}
    arrays={'Original':data['Baseline_20']};records=[];maskstore={}
    for m in rows:
        path=long.TURN/'raw'/m['turn_file'];assert sha(path)==m['turn_sha256'];a=read(path);s=scouts[m['id']];phase=audits[m['id']]['phases'];name=m['turn_file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc');marker=None
        if candidates(m,phase,s):
            assert name in receipts and sha(HERE/'raw'/name)==receipts[name]['sha256'];marker=trc(HERE/'raw'/name)
        views,audit,masks=features(a,m,s,phase,marker)
        for name,value in views.items():arrays.setdefault(name,[]).append(value)
        for name,value in masks.items():maskstore[f'{m["id"]}_{name}']=value
        records.append({**m,'near_chair':audit})
    arrays={k:np.asarray(v) for k,v in arrays.items()};out=HERE/'cohort176';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'input.npz',ids=data['ids'],y=data['y'],**arrays);np.savez_compressed(HERE/'masks.npz',**maskstore)
    save('cohort176/manifest.json',records);(out/'splits.json').write_bytes((prior/'splits.json').read_bytes())
    save('protocol.json',{'cohorts':{'cohort176':{'people':176,'long_people':8,'views':list(arrays),'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json']}}},
      'primary':'AddBoth versus Matched: same176, original person splits, same relocated maximum20s observation, same per-person actual observation duration as old20 cap. Original distinguishes relocation from added information. Large-under4 predefined246,266,310,406; all176 and long8 also reported.',
      'window':'shift original window backward toward automatic rise-end only when departure candidate>=0.5s; preserve original full turn+adjacent3s, retain exact original duration; never use clinical score or clock onset; no padding; full-record motion selection, not online validation',
      'candidates':'after automatic core-rise END before prior away-from-chair start; after prior away-from-chair end before automatic sit START; at least0.5s; missing boundary means no added evidence, not true zero delay',
      'quality':'native MOT pelvis and TRC RHip/LHip midpoint jumps horizontal>6m/s or vertical>3m/s masked with0.15s margin; widths deviate>25percent masked;30Hz7point smoothing within observed window; MOT/TRC horizontal velocity vector difference<=0.3m/s; abs vertical speed both<=0.05m/s; both heights>=adjacent gait0.5s median minus0.05m; knee mean<60deg and abs hip mean<60deg; valid runs>=0.5s without bridging; window edge0.15s excluded',
      'rotation':'subset of quality-passed return: hip angular speed20..180deg/s, hip/shoulder same sign, difference<=45deg/s, runs>=0.25s; includes no seated frames; surrogate rotation, not manually labelled final turn',
      'strict_sensitivity':'before training: vertical<=0.03m/s, height tolerance0.03m, velocity agreement<=0.2m/s, contiguous support>=0.75s; same window and baseline',
      'model':'same4 baseline speed/turn inputs, fixed observed gait-stop offset; Add views add accepted extra duration with coefficient1 while learning remainder. LearnBoth learns two duration coefficients; same train-only scaler/Ridge alpha0.1/1/10/100 inner3fold MAE,5partitionsx5outerfold; floor residual0, no upper cap',
      'boundary_limit':'computational quality checks using same OpenCap pipeline are not independent RGB/clinical confirmation; conservative height/velocity gates reduce but cannot prove zero residual rise; no preparation time, clinical-derived onset, rise or sit duration feature',
      'fallback':'retain all176 and score same test people; zero added accepted time if no confirmed support is an observation omission, not a statement that true near-chair time is zero; no new failure-based exclusions',
      'source_hashes':{str(p.relative_to(HERE.parent)):sha(p) for p in [prior/'input.npz',prior/'splits.json',prior/'manifest.json',prior/'predictions.csv',HERE.parent/'20261031-opencap-tail-diagnosis/person-audit.json',long.HERE/'study.py',HERE/'acquisition.json']},'code_sha256':sha(HERE/'study.py')})
    save('quality-summary.json',{'candidate_people':sum(bool(r['near_chair']['candidates']) for r in records),'shifted_people':sum(r['near_chair']['shift_seconds']>1e-6 for r in records),'main_people':sum(r['near_chair']['main']['departure_seconds']+r['near_chair']['main']['return_seconds']>0 for r in records),'strict_people':sum(r['near_chair']['strict']['departure_seconds']+r['near_chair']['strict']['return_seconds']>0 for r in records),'rotation_people':sum(r['near_chair']['main']['rotation_seconds']>0 for r in records),'cases':[{k:r[k] for k in ['id','target','near_chair']} for r in records if r['id'] in ['sub-246','sub-266','sub-310','sub-406']]})
    print(json.dumps(json.loads((HERE/'quality-summary.json').read_text())),flush=True)
if __name__=='__main__':prepare()
