from study import *

def main():
    import scout
    from prepare import old
    protocol=json.loads((HERE/'protocol.json').read_text());original=np.load(PRIOR/'input.npz');oldplans=json.loads((PRIOR/'splits.json').read_text());result={}
    for name,h in protocol['source_hashes'].items():assert sha(HERE.parent/name)==h
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort;d=np.load(out/'input.npz');ix=info['source_indices'];man=json.loads((out/'manifest.json').read_text())
        for name,h in info['hashes'].items():assert sha(out/name)==h
        assert np.array_equal(d['ids'],original['ids'][ix]) and np.array_equal(d['y'],original['y'][ix])
        assert 'sub-283' not in d['ids'] and len(set(d['ids']))==info['people'] and sum(d['y']>=20)==8
        assert np.array_equal(d['Current'],original['TurnTimeOnly'][ix])
        assert np.array_equal(d['Inverse'][:,1:],d['Current'][:,1:]) and np.array_equal(d['Inverse'][:,0],1/d['Current'][:,0])
        for rep,prior in zip(json.loads((out/'splits.json').read_text()),oldplans):
            seen=[]
            for f,g in zip(rep['splits'],prior['splits']):
                for name in ['train','test']:assert list(d['ids'][f[name]])==[str(original['ids'][i]) for i in g[name] if i in ix]
                assert not set(f['train'])&set(f['test']);seen+=f['test'];seeninner=[]
                for a,b in zip(f['inner'],g['inner']):
                    for name in ['train','validation']:assert list(d['ids'][a[name]])==[str(original['ids'][i]) for i in b[name] if i in ix]
                    assert not set(a['train'])&set(a['validation']) and set(a['train'])|set(a['validation'])==set(f['train']);seeninner+=a['validation']
                assert sorted(seeninner)==sorted(f['train'])
            assert sorted(seen)==list(range(info['people']))
        if cohort=='common176':
            for i,m in enumerate(man):
                path=scout.TURN/'raw'/m['turn_file'];assert sha(path)==m['turn_sha256'];a=scout.read(path)
                selected=scout.choose(a,m);assert selected['status']=='passed' and selected['windows']==m['tug_walk_windows']
                durations=[right-left for left,right in m['tug_walk_windows']];assert abs(sum(durations)-3)<1e-9
                segments=[]
                for left,right in m['tug_walk_windows']:
                    assert right<=m['turn_start']+1e-6 or left>=m['turn_end']-1e-6
                    mask=(a['time']>=left-1e-7)&(a['time']<=right+1e-7);tt=a['time'][mask];grid=tt[0]+np.arange(round((right-left)*30)+1)/30
                    assert grid[-1]<=tt[-1]+1e-6
                    pos=np.stack([np.interp(grid,tt,a[k][mask]) for k in ['pelvis_tx','pelvis_tz']]);v=np.gradient(pos,1/30,axis=1)
                    segments.append(np.hypot(v[0],v[1]).astype(np.float32))
                feat=pooled_speed(segments);assert np.allclose(feat,d['SameTUG'][i,:3],atol=2e-6,rtol=0)
                assert d['SameTUG'][i,3]==d['Current'][i,3]
                assert abs(sum(durations)+m['turn_observed_seconds']-m['matched_input_seconds'])<1e-8
                # Pooling itself is exactly equivalent for two pieces of the same speed trace.
                wp=scout.TURN.parent/'20261030-opencap415-tug-timeseries/raw'/m['walk_file'];assert sha(wp)==m['walk_sha256'];wa=scout.read(wp)
                trace=old.window(wa,int(np.searchsorted(wa['time'],m['walk_start']-1e-7)),3)[17]
                n=round(durations[0]*30)
                if len(durations)==2:
                    expected=np.array([trace.mean(),trace.std(),abs(np.diff(trace)).mean()])
                    assert np.array_equal(pooled_speed([trace[:n+1],trace[n:]]),expected)
        result[cohort]={'people':info['people'],'long_people':8,'same_original_person_memberships':True,'original_labels':True,'equal_input_seconds':True,'excluded_subject_absent':True}
    save('input-verification.json',{'passed':True,'cohorts':result,'sameTUG_raw_features_independently_recomputed':True});print(result,flush=True)
if __name__=='__main__':main()
