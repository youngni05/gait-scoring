from study import *
def main():
    protocol=json.loads((HERE/'protocol.json').read_text());assert sha(HERE/'study.py')==protocol['code_sha256']
    for n,h in protocol['source_hashes'].items():assert sha(HERE.parent/n)==h
    original=np.load(PRIOR/'capped176/input.npz');maxdiff=0.
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort;d=np.load(out/'input.npz');rows=json.loads((out/'manifest.json').read_text())
        assert len(set(d['ids']))==info['people'] and sum(d['y']>=20)==info['long_people']
        assert np.array_equal(d['ids'][:176],original['ids']) and np.array_equal(d['y'][:176],original['y'])
        assert ('sub-283' in d['ids'])==(cohort=='include283')
        for b in [10,20,30]:
            assert np.array_equal(d[f'Speed_{b}'][:176],original[f'Speed_{b}'])
            for th in ['10','15']:assert np.array_equal(d[f'Learned{th}_{b}'][:176],original[f'Delay{th}_{b}'])
        for rep in json.loads((out/'splits.json').read_text()):
            seen=[]
            for f in rep['splits']:
                tr,te=set(f['train']),set(f['test']);assert not tr&te and tr|te==set(range(info['people']));seen+=f['test'];valid=[]
                for v in f['inner']:
                    it,iv=set(v['train']),set(v['validation']);assert not it&iv and it|iv==tr;valid+=v['validation']
                assert sorted(valid)==sorted(tr)
            assert sorted(seen)==list(range(info['people']))
        for i,m in enumerate(rows):
            a=read(TURN/'raw'/m['turn_file']);assert sha(TURN/'raw'/m['turn_file'])==m['turn_sha256']
            for b in [10,20,30]:
                r=m['budgets'][str(b)];l,h=r['window'];t=l+np.arange(round((h-l)*30)+1)/30;mid=(t[1:]+t[:-1])/2
                s=long.old.eligible(a,m);valid=np.array([any(w and w[0]-1e-6<=x<=w[1]+1e-6 for w in s[:2]) for x in mid])
                initial=np.median(a['pelvis_ty'][a['time']<=a['time'][0]+.5])
                seated=(np.interp(mid,a['time'],a['pelvis_ty'])<initial+.12)&(np.interp(mid,a['time'],(a['knee_angle_l']+a['knee_angle_r'])/2)>60)&(np.interp(mid,a['time'],(a['hip_flexion_l']+a['hip_flexion_r'])/2)>30)
                pp=np.stack([a['pelvis_tx'],a['pelvis_tz']],axis=1);native=np.linalg.norm(np.diff(pp,axis=0),axis=1)/np.diff(a['time']);bad=np.zeros(len(mid),bool)
                for j in np.flatnonzero(native>6):bad|=(mid>=a['time'][j]-.15)&(mid<=a['time'][j+1]+.15)
                valid&=~seated&~bad;pos=np.stack([np.interp(t,a['time'],a[k]) for k in ['pelvis_tx','pelvis_tz']],axis=1);speed=np.linalg.norm(np.gradient(pos,1/30,axis=0),axis=1)[:-1]
                for th in ['10','15']:
                    stopped=np.zeros(len(mid),bool)
                    for e in r['events'][th]['horizontal']:stopped|=(mid>=e['start']-1e-7)&(mid<=e['end']+1e-7)
                    assert not any(stopped&~valid);pause=stopped.sum()/30
                    for prefix in ['Add','Move']:
                        assert abs(d[f'{prefix}{th}_{b}'][i,-1]-pause)<1e-10 and d['y'][i]-pause>0
                    assert abs(pause-d[f'Learned{th}_{b}'][i,6])<1e-10
                    assert np.array_equal(d[f'Add{th}_{b}'][i,:4],d[f'Speed_{b}'][i])
                    mask=valid&~stopped;values=speed[mask];change=abs(np.diff(speed))[mask[:-1]&mask[1:]]
                    expected=np.array([values.mean(),values.std(),change.mean(),m['turn_observed_seconds']]);diff=float(max(abs(expected-d[f'Move{th}_{b}'][i,:4])));assert diff<1e-9;maxdiff=max(maxdiff,diff)
                    assert abs(mask.sum()/30-r['moving_feature_seconds'][th])<1e-10
        for n,h in info['hashes'].items():assert sha(out/n)==h
    originalplans=json.loads((PRIOR/'capped176/splits.json').read_text());exclude=json.loads((HERE/'exclude283/splits.json').read_text());include=json.loads((HERE/'include283/splits.json').read_text())
    assert exclude==originalplans
    for er,ir in zip(exclude,include):
        for ef,inf in zip(er['splits'],ir['splits']):
            for k in ['train','test']:assert [i for i in inf[k] if i!=176]==ef[k]
            for ev,iv in zip(ef['inner'],inf['inner']):
                for k in ['train','validation']:assert [i for i in iv[k] if i!=176]==ev[k]
    save('input-verification.json',{'passed':True,'prior176_baseline_inputs_exact':True,'same_people_splits_and_windows':True,'283_restored_without_reshuffle':True,'offsets_verified_from_observed_events':True,'all_residual_targets_positive':True,'moving_features_independent_max_difference':maxdiff})
    print('INPUT VERIFIED',maxdiff,flush=True)
if __name__=='__main__':main()
