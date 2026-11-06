import json,hashlib,math,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;OLD=H/'studies/20261028-window-cause-controls'
sys.stdout.reconfigure(encoding='utf-8');sys.path.insert(0,str(H))
from short_window_contract import source_functions
normalize,_,_=source_functions()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def filtered(x):
    a=np.asarray(x,dtype=float);a=np.convolve(np.pad(a,(2,2),mode='edge'),np.array([1,2,3,2,1])/9,mode='valid');t=np.arange(len(a),dtype=float);t-=t.mean();return a-a.mean()-t*(t@a)/(t@t)
def detect(signal):
    x=filtered(signal);sd=float(np.std(x))
    if sd<1e-8:return {'status':'unresolved_flat','period':None,'peaks':[],'cycles':[]}
    ac=[None]
    for lag in range(1,92):
        a=x[:-lag];b=x[lag:];a=a-a.mean();b=b-b.mean();den=float(np.linalg.norm(a)*np.linalg.norm(b));ac.append(float(a@b/den) if den>1e-16 else 0.)
    candidates=[k for k in range(16,91) if ac[k]>=.35 and ac[k]>ac[k-1] and ac[k]>=ac[k+1]]
    if not candidates:return {'status':'unresolved_period','period':None,'peaks':[],'cycles':[]}
    period=candidates[0];radius=period//2;minimum=math.ceil(period/2);candidates=[]
    for t in range(1,len(x)-1):
        if x[t]>x[t-1] and x[t]>=x[t+1]:
            prom=float(x[t]-max(x[max(0,t-radius):t+1].min(),x[t:min(len(x),t+radius+1)].min()))
            if prom>=.25*sd:candidates.append(t)
    picked=[]
    for t in sorted(candidates,key=lambda q:(-x[q],q)):
        if all(abs(t-u)>=minimum for u in picked):picked.append(t)
    peaks=sorted(picked);cycles=[[a,b] for a,b in zip(peaks,peaks[1:]) if .65*period<=b-a<=1.35*period]
    return {'status':'estimated','period':period,'autocorrelation':ac[period],'peaks':peaks,'cycles':cycles,'signal_std':sd}
def contained(d,start):return [c for c in d['cycles'] if start<=c[0]<c[1]<=start+123]
def choose(d,start):
    cc=contained(d,start);pairs=[]
    for a,b in zip(cc,cc[1:]):
        if a[1]==b[0]:pairs.append([a[0],a[1],b[1]])
    return min(pairs,key=lambda a:(abs((a[0]+a[2])/2-(start+61.5)),a[0])) if pairs else None
def grid(bounds):
    a,b,c=bounds;assert 0<=a<b<c<=242;v=np.r_[np.linspace(a,b,62,endpoint=False),np.linspace(b,c,62)];assert len(v)==124 and (np.diff(v)>0).all() and v[62]==b;return v
def resample(raw,times):
    assert raw.shape==(243,17,3) and np.isfinite(raw).all();assert len(times)==124 and (np.diff(times)>0).all();out=np.zeros((124,17,3),np.float64)
    for i,t in enumerate(times):
        assert 0<=t<=242
        near=int(round(float(t)))
        if abs(t-near)<1e-10:out[i]=raw[near];continue
        left=int(np.floor(t));right=left+1;w=float(t-left);valid=(raw[left,:,2]>0)&(raw[right,:,2]>0)
        out[i,valid,:2]=raw[left,valid,:2]*(1-w)+raw[right,valid,:2]*w
        out[i,valid,2]=np.minimum(raw[left,valid,2],raw[right,valid,2])
    assert (out[:,1:7,2]>0).all();return out

def checks():
    t=np.arange(243);x=np.cos(2*np.pi*(t-10)/40);d=detect(x);assert d['status']=='estimated' and abs(d['period']-40)<=1
    assert len(d['peaks'])==6 and max(abs(a-b) for a,b in zip(d['peaks'],[10,50,90,130,170,210]))<=1
    assert len(contained(d,0))==2 and len(contained(d,119))==2
    assert detect(np.ones(243))['status']=='unresolved_flat';other=detect(x*3+17);assert d['period']==other['period'] and d['peaks']==other['peaks']
    raw=np.ones((243,17,3),np.float32);raw[:,:,0]=t[:,None];raw[:,:,1]=2*t[:,None];raw[:,:,2]=.7
    np.testing.assert_array_equal(resample(raw,np.arange(124)),raw[:124])
    raw[11,9,2]=0;q=resample(raw,np.linspace(0,123,124)+.5);assert q[10,9,2]==0 and q[11,9,2]==0 and np.all(q[10,9]==0)
    g=grid([10,50,90]);assert g[0]==10 and g[62]==50 and g[-1]==90
    return {'known_period':d['period'],'known_full_cycles_front':2,'known_full_cycles_last':2,'peak_tolerance_frames':1,'flat_rejected':True,'positive_scale_and_offset_invariant':True,'integer_resampling_identity':True,'missing_conservative':True,'two_cycle_phase_midpoint':True}
def main():
    assert not (S/'readiness.json').exists();save('synthetic_checks.json',checks())
    sealed={a['path']:a['sha256'] for a in load(OLD/'study_receipt.json')['artifacts']};tracked={}
    p=OLD/'features.npz';assert sha(p)==sealed[p.relative_to(H).as_posix()];tracked[p.relative_to(R).as_posix()]=sha(p)
    with np.load(p,allow_pickle=False) as z:ids=z['patient_ids'].copy();vid=z['video_ids'].copy();indices=z['original_indices'].copy()
    assert len(vid)==203 and len(set(ids))==175
    (S/'inputs').mkdir(exist_ok=True);rows=[];eligible=[]
    for j in range(203):
        p=OLD/f'records/row_{j:04d}.npz';assert sha(p)==sealed[p.relative_to(H).as_posix()];tracked[p.relative_to(R).as_posix()]=sha(p)
        with np.load(p,allow_pickle=False) as z:
            f=z['front_raw'].copy();l=z['last_raw'].copy();np.testing.assert_array_equal(f[-5:],l[:5]);raw=np.concatenate([f,l[5:]]);np.testing.assert_array_equal(normalize(f),z['front_input']);np.testing.assert_array_equal(normalize(l),z['last_input'])
        assert raw.shape==(243,17,3) and (raw[:,1:7,2]>0).all()
        leg=np.stack([np.linalg.norm(raw[:,a,:2].astype(float)-raw[:,b,:2],axis=1) for a,b in [(1,2),(2,3),(4,5),(5,6)]],1);scale=float(np.median(leg));assert scale>0
        right=(raw[:,3,0].astype(float)-raw[:,0,0])/scale;left=(raw[:,6,0].astype(float)-raw[:,0,0])/scale;dr,dl=detect(right),detect(left)
        chosen={p:choose(dr,start) for p,start in [('front',0),('last',119)]};ok=all(v is not None for v in chosen.values());agree=None if dr['period'] is None or dl['period'] is None else abs(dr['period']-dl['period'])/dr['period']<=.25
        row={'row':j,'status':dr['status'],'right_detection':dr,'left_period':dl['period'],'period_agreement25':agree,'front_full_cycles':len(contained(dr,0)) if dr['status']=='estimated' else None,'last_full_cycles':len(contained(dr,119)) if dr['status']=='estimated' else None,'eligible':ok,'aligned_bounds':chosen};arrays={};nmiss={}
        if ok:
            eligible.append(j);row['control_bounds']={};row['duration_frames']={};row['start_shift_frames']={}
            for p,start in [('front',0),('last',119)]:
                a,b,c=chosen[p];D=c-a;control_start=int(np.floor(start+(123-D)/2));control=[control_start,control_start+b-a,control_start+D];assert start<=control[0]<control[1]<control[2]<=start+123
                row['control_bounds'][p]=control;row['duration_frames'][p]=D;row['start_shift_frames'][p]=a-control_start
                for condition,bounds in [('aligned',chosen[p]),('control',control)]:
                    times=grid(bounds);q=resample(raw,times);key=condition+'_'+p;arrays[key]=normalize(q);arrays[key+'_times']=times;nmiss[key]=float((q[:,:,2]==0).mean())
            path=S/f'inputs/row_{j:04d}.npz';np.savez_compressed(path,**arrays);row['input_sha256']=sha(path);row['missing_fraction']=nmiss
        rows.append(row)
    for p in [Path(__file__),S/'plan.md',H/'short_window_contract.py',R/'tools/stanford-sample-check/prepare_ridge_data.py']:tracked[p.relative_to(R).as_posix()]=sha(p)
    assert all(sha(R/p)==h for p,h in tracked.items());save('sources.json',tracked);save('cycle_manifest.json',rows)
    np.savez_compressed(S/'identities.npz',patient_ids=ids,video_ids=vid,original_indices=indices,eligible_rows=np.array(eligible,dtype=int))
    histogram={p:{str(n):sum(r[p+'_full_cycles']==n for r in rows) for n in sorted(set(r[p+'_full_cycles'] for r in rows if r[p+'_full_cycles'] is not None))} for p in ['front','last']}
    status={s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))}
    result={'people':175,'records':203,'status_counts':status,'complete_cycle_histograms':histogram,'eligible_records':len(eligible),'eligible_people':len(set(ids[eligible])),'not_in_pair_records':203-len(eligible),'both_periods_available':sum(r['period_agreement25'] is not None for r in rows),'periods_agree25':sum(r['period_agreement25'] is True for r in rows),'primary_period_median':float(np.median([r['right_detection']['period'] for r in rows if r['right_detection']['period'] is not None])),'detector_is_not_clinical_event_truth':True,'labels_used':False,'protected_used':False}
    save('readiness.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
