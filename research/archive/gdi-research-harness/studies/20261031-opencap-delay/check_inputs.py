from study import *
def main():
    protocol=json.loads((HERE/'protocol.json').read_text());assert sha(HERE/'study.py')==protocol['code_sha256']
    assert sha(HERE/'audit.py')==protocol['audit_code_sha256'] and sha(HERE/'audit.json')==protocol['audit_sha256']
    for n,h in protocol['source_hashes'].items():assert sha(HERE.parent/n)==h
    out=HERE/'common176';d=dict(np.load(out/'input.npz'));old=dict(np.load(PREV/'common176/input.npz'));rows=json.loads((out/'manifest.json').read_text())
    assert np.array_equal(d['ids'],old['ids']) and np.array_equal(d['y'],old['y'])
    assert len(set(d['ids']))==176 and sum(d['y']>=20)==8 and 'sub-283' not in d['ids']
    for v in ['Current','SameTUG']:assert np.array_equal(d[v],old[v])
    assert (out/'splits.json').read_bytes()==(PREV/'common176/splits.json').read_bytes()
    maxerror=0.;times=[]
    for i,m in enumerate(rows):
        assert m['id']==d['ids'][i];a=read(TURN/'raw'/m['turn_file']);assert sha(TURN/'raw'/m['turn_file'])==m['turn_sha256']
        windows=m['observed_windows'];assert abs(sum(r-l for l,r in windows)-3)<1e-6
        for j,w in enumerate(windows):
            assert w[0]>=a['time'][0] and w[1]<=a['time'][-1]
            assert overlaps(w,[m['turn_start'],m['turn_end']])<1e-6
            assert all(overlaps(w,v)<1e-6 for v in windows[j+1:])
        ss=[];hh=[];xx=[]
        for l,r in windows:
            t=np.linspace(l,r,round((r-l)*30)+1)
            pos=np.stack([np.interp(t,a['time'],a[k]) for k in ['pelvis_tx','pelvis_tz']],axis=1)
            vel=np.gradient(pos,1/30,axis=0);ss.append(np.sqrt(np.square(vel).sum(1)).astype(np.float32))
            smooth=np.stack([np.convolve(np.pad(pos[:,j],(3,3),mode='edge'),np.ones(7)/7,mode='valid') for j in range(2)],axis=1)
            v=np.gradient(smooth,1/30,axis=0);hh.extend(np.sqrt((v*v).sum(1))[:-1]);xx.extend(abs(v@np.array(m['axis']))[:-1])
        joined=np.concatenate([s[:-1] for s in ss[:-1]]+[ss[-1]]);diff=np.concatenate([abs(np.diff(s)) for s in ss])
        assert len(joined)==91 and len(diff)==90
        expected=np.array([joined.mean(),joined.std(),diff.mean(),old['SameTUG'][i,3]],float)
        error=float(abs(expected-d['ResampledSpeed'][i]).max());assert error<1e-6;maxerror=max(maxerror,error)
        for name,th in [('Delay',.1),('Delay05',.05),('Delay15',.15)]:
            assert np.array_equal(d[name][i,:4],d['ResampledSpeed'][i])
            target=np.array([sum(np.array(xx)<th),sum(np.array(hh)<th)])/30
            assert np.array_equal(target,d[name][i,4:])
        clean=m['clean_windows'];assert abs(sum(r-l for l,r in clean)-3)<1e-6
        cs=[];ch=[];cx=[]
        for j,(l,r) in enumerate(clean):
            assert overlaps([l,r],[m['turn_start'],m['turn_end']])<1e-6
            assert all(overlaps([l,r],other)<1e-6 for other in clean[j+1:])
            t=np.linspace(l,r,round((r-l)*30)+1)
            pos=np.stack([np.interp(t,a['time'],a[k]) for k in ['pelvis_tx','pelvis_tz']],axis=1)
            vel=np.gradient(pos,1/30,axis=0);cs.append(np.sqrt(np.square(vel).sum(1)).astype(np.float32))
            smooth=np.stack([np.convolve(np.pad(pos[:,j],(3,3),mode='edge'),np.ones(7)/7,mode='valid') for j in range(2)],axis=1)
            v=np.gradient(smooth,1/30,axis=0);ch.extend(np.sqrt((v*v).sum(1))[:-1]);cx.extend(abs(v@np.array(m['axis']))[:-1])
        cv=np.concatenate([v[:-1] for v in cs[:-1]]+[cs[-1]]);cd=np.concatenate([abs(np.diff(v)) for v in cs])
        assert len(cv)==91 and len(cd)==len(ch)==len(cx)==90
        expected=np.array([cv.mean(),cv.std(),cd.mean(),old['SameTUG'][i,3]])
        assert np.allclose(expected,d['CleanSpeed'][i],atol=1e-6,rtol=0)
        assert np.array_equal(d['CleanSpeed'][i],d['CleanDelay'][i,:4])
        assert np.array_equal(np.array([sum(np.array(cx)<.1),sum(np.array(ch)<.1)])/30,d['CleanDelay'][i,4:])
        times.append(3+m['turn_observed_seconds'])
    for rep in json.loads((out/'splits.json').read_text()):
        seen=[]
        for f in rep['splits']:
            tr,te=set(f['train']),set(f['test']);assert not tr&te and tr|te==set(range(176));seen+=f['test'];valid=[]
            for v in f['inner']:
                it,iv=set(v['train']),set(v['validation']);assert not it&iv and it|iv==tr and not te&(it|iv);valid+=v['validation']
            assert sorted(valid)==sorted(tr)
        assert sorted(seen)==list(range(176))
    for n,h in protocol['cohorts']['common176']['hashes'].items():assert sha(out/n)==h
    assert sha(HERE/'quality_prepare.py')==protocol['input_only_amendment']['code_sha256']
    save('input-verification.json',{'passed':True,'people':176,'long':8,'partitions':5,'independent_feature_max_difference':maxerror,'exact_original_baselines':True,'same_people_and_memberships':True,'nonoverlapping_exact3seconds':True,'observation_range': [min(times),max(times)],'no_fullrecord_time_feature':True})
    print((HERE/'input-verification.json').read_text(),flush=True)
if __name__=='__main__':main()
