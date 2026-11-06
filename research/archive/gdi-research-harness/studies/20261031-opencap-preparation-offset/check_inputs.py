from study import *
from scipy.ndimage import uniform_filter1d

def independent_gait(a,m,r):
    left,right=r['window'];t=left+np.arange(round((right-left)*30)+1)/30;mid=(t[1:]+t[:-1])/2
    pre,post,_=long.old.eligible(a,m);valid=np.array([any(w and w[0]-1e-6<=s<=w[1]+1e-6 for w in [pre,post]) for s in mid])
    initial=np.median(a['pelvis_ty'][a['time']<=a['time'][0]+.5])
    seated=(np.interp(mid,a['time'],a['pelvis_ty'])<initial+.12)&(np.interp(mid,a['time'],(a['knee_angle_l']+a['knee_angle_r'])/2)>60)&(np.interp(mid,a['time'],(a['hip_flexion_l']+a['hip_flexion_r'])/2)>30)
    pp=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);nt=a['time'];native=np.linalg.norm(np.diff(pp,axis=0),axis=1)/np.diff(nt);bad=np.zeros(len(mid),bool)
    for j in np.flatnonzero(native>6):bad|=(mid>=nt[j]-.15)&(mid<=nt[j+1]+.15)
    valid&=~seated&~bad;pos=np.column_stack([np.interp(t,nt,a[k]) for k in ['pelvis_tx','pelvis_tz']]);speed=np.linalg.norm(np.gradient(pos,1/30,axis=0),axis=1)[:-1]
    base=np.array([speed[valid].mean(),speed[valid].std(),abs(np.diff(speed))[valid[:-1]&valid[1:]].mean(),m['turn_observed_seconds']])
    smooth=uniform_filter1d(pos,7,axis=0,mode='nearest');hv=np.linalg.norm(np.gradient(smooth,1/30,axis=0),axis=1)[:-1];mask=(hv<.10)&valid
    # Independent run enumeration, bridging only valid gaps <=6 samples.
    def spans(z):
        result=[];left=None
        for i,value in enumerate(list(z)+[False]):
            if value and left is None:left=i
            if not value and left is not None:result.append((left,i));left=None
        return result
    for lo,hi in spans(~mask):
        if lo>0 and hi<len(mask) and hi-lo<=6 and valid[lo:hi].all():mask[lo:hi]=True
    for lo,hi in spans(mask):
        if hi-lo<15:mask[lo:hi]=False
    stop=float(mask.sum()/30)
    assert abs(stop-r['delay_features']['10'][2])<1e-10
    assert abs(valid.sum()/30-r['valid_nonturn_seconds'])<1e-10
    return base,stop

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for n,h in protocol['source_hashes'].items():assert sha(HERE.parent/n)==h
    source=dict(np.load(PRIOR/'include283/input.npz'));plans=json.loads((PRIOR/'include283/splits.json').read_text())
    audit={r['id']:r for r in json.loads((CHAIR/'transition-audit.json').read_text())}
    rows=json.loads((HERE/'include283/manifest.json').read_text());d=np.load(HERE/'include283/input.npz');maxdiff=0.
    for i,m in enumerate(rows):
        a=read(TURN/'raw'/m['turn_file']);assert sha(TURN/'raw'/m['turn_file'])==m['turn_sha256'];phase=m['phase'];assert phase['core']==audit[m['id']]['rise']['window']
        t=a['time'];dt=np.median(np.diff(t));ch=np.column_stack([a['pelvis_tilt'],a['lumbar_extension'],(a['hip_flexion_l']+a['hip_flexion_r'])/2]);smooth=uniform_filter1d(ch,max(3,round(.25/dt)),axis=0,mode='nearest')
        baseline=np.median(smooth[t<=t[0]+.75],axis=0);changed=(abs(smooth-baseline)>=8).sum(1)>=2
        q=changed&(t>=t[0]+.75)&(t<phase['core'][0]);hold=max(1,round(.25/dt));first=next((j for j in range(len(q)-hold+1) if q[j:j+hold].all()),None)
        expected=phase['core'][0] if first is None else min(phase['core'][0],max(np.ceil(t[0]*30-1e-6)/30,np.floor((t[first]-.25)*30+1e-6)/30))
        assert abs(expected-phase['extended'][0])<1e-10
        part=a[(t>=phase['extended'][0]-1e-7)&(t<=phase['extended'][1]+1e-7)]
        assert max(abs(np.diff(part['pelvis_ty'])/np.diff(part['time'])))<=3
        angles=np.column_stack([part[k] for k in ['pelvis_tilt','lumbar_extension','hip_flexion_l','hip_flexion_r']]);assert np.max(abs(np.diff(angles,axis=0)))<=45
        for b in [10,20,30]:
            r=m['comparison_budgets'][str(b)];assert np.array_equal(d[f'Gait_{b}'][i],source[f'Add10_{b}'][m['source_index']])
            base,stop=independent_gait(a,m,r['reallocated_gait']);pw=r['rise_clip'];gw=r['reallocated_gait']['window']
            assert pw[1]<=gw[0]+1e-6 and phase['extended'][0]-1e-6<=pw[0]<pw[1]<=phase['extended'][1]+1e-6
            assert abs(pw[1]-pw[0]+gw[1]-gw[0]-r['original']['observed_seconds'])<1e-6
            for w in m['tug_walk_windows']+[[m['turn_start'],m['turn_end']]]:assert gw[0]<=w[0]+1e-6 and gw[1]>=w[1]-1e-6
            full=pw[1]-pw[0];core=max(0,min(pw[1],phase['core'][1])-max(pw[0],phase['core'][0]));prepseconds=full-core
            expected={'Reallocated':np.r_[base,stop],'Core':np.r_[base,stop+core],'Prep':np.r_[base,stop+full],'LearnPrep':np.r_[base,core,prepseconds,stop]}
            for kind,value in expected.items():
                diff=float(max(abs(value-d[f'{kind}_{b}'][i])));maxdiff=max(diff,maxdiff);assert diff<1e-9
            assert abs(r['unobserved_rise']+full-(phase['extended'][1]-phase['extended'][0]))<1e-9
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort;data=np.load(out/'input.npz');ms=json.loads((out/'manifest.json').read_text());sp=json.loads((out/'splits.json').read_text());mapping={m['source_index']:i for i,m in enumerate(ms)}
        assert len(set(data['ids']))==info['people'] and sum(data['y']>=20)==info['long_people'];assert ('sub-283' in data['ids'])==(cohort=='include283')
        ix=[m['source_index'] for m in ms];assert np.array_equal(data['ids'],source['ids'][ix]) and np.array_equal(data['y'],source['y'][ix])
        for r,pr in zip(sp,plans):
            seen=[]
            for f,pf in zip(r['splits'],pr['splits']):
                for k in ['train','test']:assert f[k]==[mapping[j] for j in pf[k] if j in mapping]
                assert not set(f['train'])&set(f['test']);assert set(f['train'])|set(f['test'])==set(range(len(ms)));seen+=f['test'];seeninner=[]
                for v,pv in zip(f['inner'],pf['inner']):
                    for k in ['train','validation']:assert v[k]==[mapping[j] for j in pv[k] if j in mapping]
                    assert not set(v['train'])&set(v['validation']);assert set(v['train'])|set(v['validation'])==set(f['train']);seeninner+=v['validation']
                assert sorted(seeninner)==sorted(f['train'])
            assert sorted(seen)==list(range(len(ms)))
        for n,h in info['hashes'].items():assert sha(out/n)==h
        for j,m in enumerate(ms):
            i=list(d['ids']).index(m['id'])
            for view in info['views']:assert np.array_equal(data[view][j],d[view][i]) and np.isfinite(data[view][j]).all()
    save('input-verification.json',{'passed':True,'same_previous_memberships':True,'same_actual_total_duration':True,'nonoverlapping_clips':True,'original3s_and_turn_retained':True,'raw_features_max_difference':maxdiff,'phase_detection_independently_checked':True,'source_labels_and_hashes':True})
    print('INPUT VERIFIED',len(rows),maxdiff,flush=True)
if __name__=='__main__':main()
