"""Same-person, same-duration gait versus observed preparation offsets."""
from pathlib import Path
import sys,json,importlib.util
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261031-opencap-additive-stop'
CHAIR=HERE.parent/'20261031-opencap-chair'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module('additive_prior',PRIOR/'study.py')
np,pd,read,sha,long,TURN=old.np,old.pd,old.read,old.sha,old.long,old.TURN
sys.path.insert(0,str(CHAIR))
segment=module('rise_segment',CHAIR/'segment.py')
prep=module('prep_segment',CHAIR/'preparation_phase.py')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
ALPHAS=[.1,1.,10.,100.]
KINDS=['Gait','Reallocated','Core','Prep','LearnPrep']

def save(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def fit(x,y,tr,te,alpha,view):
    # All views store their predetermined observed-time offset in the last column.
    offset=x[:,-1];xx=x[:,:-1];target=y-offset
    sc=StandardScaler().fit(xx[tr]);model=Ridge(alpha=alpha,solver='svd').fit(sc.transform(xx[tr]),target[tr])
    part=model.predict(sc.transform(xx[te]))
    return np.maximum(part,0)+offset[te],part+offset[te],(sc,model)

def intersect(a,b):return max(0.,min(a[1],b[1])-max(a[0],b[0]))

def get_phase(a,m):
    r=segment.transition(a,m,'rise');core=r['window'];extended=prep.extend(a,core)
    t=a['time'];q=a[(t>=extended[0]-1e-7)&(t<=extended[1]+1e-7)]
    vertical=float(max(abs(np.diff(q['pelvis_ty'])/np.diff(q['time']))))
    angles=np.column_stack([q[k] for k in ['pelvis_tilt','lumbar_extension','hip_flexion_l','hip_flexion_r']])
    angular=float(np.max(abs(np.diff(angles,axis=0))))
    if vertical>3:raise ValueError('preparation_vertical_jump')
    if angular>45:raise ValueError('preparation_angular_jump')
    pre,post,_=long.old.eligible(a,m);start=pre[0] if pre else m['tug_walk_windows'][0][0]
    return {'core':core,'extended':extended,'vertical_max':vertical,'angular_step_max':angular},dict(id=m['id'],start=start,pre=pre,post=post)

def extract(a,m,phase,s,b):
    base,delay,record,traces=long.extract(a,m,s,b)
    assert record=={k:v for k,v in m['budgets'][str(b)].items() if k!='moving_feature_seconds'}
    # Preserve the original contiguous 3s gait + full turn, including any gaps.
    core_left=min(w[0] for w in m['tug_walk_windows']);core_right=max(w[1] for w in m['tug_walk_windows'])
    core_left=min(core_left,m['turn_start']);core_right=max(core_right,m['turn_end'])
    if phase['extended'][1]>core_left+1e-6:raise ValueError('rise_overlaps_mandatory3s_gait')
    mandatory_frames=round((core_right-core_left)*30)
    total_frames=round(record['observed_seconds']*30)
    phase_frames=round((phase['extended'][1]-phase['extended'][0])*30)
    n=min(phase_frames,total_frames-mandatory_frames)
    if n<1:raise ValueError('no_spare_observation_for_rise')
    pw=[phase['extended'][1]-n/30,phase['extended'][1]]
    after_rise={**s,'start':max(s['start'],phase['extended'][1])}
    short,sd,sr,st=long.extract(a,m,after_rise,(total_frames-n)/30)
    assert abs(sr['observed_seconds']+n/30-record['observed_seconds'])<1e-6
    assert sr['window'][0]<=core_left+1e-6 and sr['window'][1]>=core_right-1e-6
    assert intersect(pw,sr['window'])<1e-6
    core_seconds=intersect(pw,phase['core']);pre_seconds=n/30-core_seconds
    stop=sd['10'][2]
    views={'Gait':np.r_[base,delay['10'][2]],'Reallocated':np.r_[short,stop],
           'Core':np.r_[short,stop+core_seconds],'Prep':np.r_[short,stop+n/30],
           'LearnPrep':np.r_[short,core_seconds,pre_seconds,stop]}
    audit={'original':record,'reallocated_gait':sr,'rise_clip':pw,'rise_observed':n/30,
           'core_observed':core_seconds,'preparation_observed':pre_seconds,
           'unobserved_rise':phase_frames/30-n/30,'total_observed':record['observed_seconds'],
           'elapsed_span_with_gaps':sr['window'][1]-pw[0],
           'full_preparation_rise_outside_original':phase_frames/30-intersect(phase['extended'],record['window'])}
    return views,audit

def prepare():
    assert not (HERE/'protocol.json').exists()
    source=json.loads((PRIOR/'include283/manifest.json').read_text());retained=[];excluded=[];arrays=[];audits=[]
    for i,m in enumerate(source):
        a=read(TURN/'raw'/m['turn_file']);assert sha(TURN/'raw'/m['turn_file'])==m['turn_sha256']
        try:
            phase,s=get_phase(a,m);features={};budgets={}
            for b in [10,20,30]:
                vv,rr=extract(a,m,phase,s,b);features.update({f'{k}_{b}':v for k,v in vv.items()});budgets[str(b)]=rr
            retained.append({**m,'source_index':i,'phase':phase,'comparison_budgets':budgets});arrays.append(features)
            audits.append({'id':m['id'],'target':m['target'],'status':'retained','phase':phase,'budgets':budgets})
        except ValueError as e:
            excluded.append({'id':m['id'],'target':m['target'],'reason':str(e)})
    save('phase-audit.json',audits);save('excluded.json',excluded)
    print(json.dumps({'retained':len(retained),'long_retained':[m['id'] for m in retained if m['target']>=20],'excluded':excluded}),flush=True)
    assert len(retained)>=25 and any(m['id']=='sub-283' for m in retained) and any(m['id']=='sub-397' for m in retained)
    views=list(arrays[0]);cohorts={}
    for name in ['exclude283','include283']:
        chosen=[i for i,m in enumerate(retained) if name=='include283' or m['id']!='sub-283'];rows=[retained[i] for i in chosen]
        out=HERE/name;out.mkdir(exist_ok=True)
        np.savez_compressed(out/'input.npz',ids=np.array([m['id'] for m in rows]),y=np.array([m['target'] for m in rows]),**{v:np.stack([arrays[i][v] for i in chosen]) for v in views})
        save(name+'/manifest.json',rows)
        mapping={m['source_index']:i for i,m in enumerate(rows)};remap=lambda xs:[mapping[j] for j in xs if j in mapping]
        plans=json.loads((PRIOR/'include283/splits.json').read_text())
        for rep in plans:
            for f in rep['splits']:
                for k in ['train','test']:f[k]=remap(f[k])
                for v in f['inner']:
                    for k in ['train','validation']:v[k]=remap(v[k])
        save(name+'/splits.json',plans)
        cohorts[name]={'people':len(rows),'long_people':sum(m['target']>=20 for m in rows),'views':views,'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json']}}
    save('phase-audit.json',audits);save('excluded.json',excluded)
    sources=[PRIOR/'study.py',long.HERE/'study.py',CHAIR/'segment.py',CHAIR/'preparation_phase.py']+[PRIOR/'include283'/n for n in ['manifest.json','input.npz','splits.json']]
    protocol={'cohorts':cohorts,'primary':'maximum20s, Prep versus Gait; Core and Reallocated diagnose allocation and rise-only effects; include283 and exclude283 both reported; 10/30 caps sensitivity',
      'selection':'common people passing input-only existing rise detector; vertical<=3m/s and angle step<=45deg; phase ends before mandatory3s gait; >=1 frame spare at every budget; no target or model performance exclusions',
      'observation':'exact same per-person previous actual duration at each10/20/30 cap; allocate earliest discarded gait context to latest part of prep-rise, retain original contiguous3s+turn; reduced gait begins no earlier than rise end to avoid double counting; summed nonoverlapping clips, NOT contiguous capture time; full-record segmentation retained',
      'models':'all Ridge SVD alpha .1,1,10,100 train-only scaling, inner finalTUG MAE; subtract last-column fixed offset in training and add to max(predicted remainder,0); no upper clamp',
      'views':{'Gait':'previous full gait window, observed gait stop added','Reallocated':'shorter gait plus selected prep clip, ignore prep time value (ablation)','Core':'same as Reallocated plus observed core rise duration fixed1','Prep':'same as Reallocated plus observed prep+rise duration fixed1','LearnPrep':'same clips, model learns core and preparatory seconds coefficients; gaitstop fixed1'},
      'split':'same previous5x5 outer and3inner memberships restricted to retained people, no reshuffle',
      'metrics':'person equal MAE/under/over/bias; >=20s descriptive group; paired conditional10000person bootstrap seed20261031; 28.03 and42.59 prediction mean SD range; no independent test set',
      'limits':['automatic prep includes possible unrelated posture changes and instruction wait','exact clinically timed trial and command timestamp unverified','repeated development-set analyses; no confirmatory p-values','observed rise duration includes previous boundary margins','residual is not independently annotated pure moving time'],
      'source_hashes':{str(p.relative_to(HERE.parent)):sha(p) for p in sources},'code_sha256':sha(HERE/'study.py')}
    save('protocol.json',protocol)
    print(json.dumps({'cohorts':cohorts,'excluded':excluded,'long_phases':[{'id':m['id'],'target':m['target'],'phase':m['phase'],'budget20':m['comparison_budgets']['20']} for m in retained if m['target']>=20]}),flush=True)
if __name__=='__main__':prepare()
