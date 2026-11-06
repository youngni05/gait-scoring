from study import *

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for name,h in protocol['source_hashes'].items():assert sha(HERE.parent/name)==h
    scouts={r['id']:r for r in json.loads((long.HERE/'scout.json').read_text())}
    d=np.load(HERE/'include283/input.npz');rows=json.loads((HERE/'include283/manifest.json').read_text());maxdiff=0.;windows=0
    for i,m in enumerate(rows):
        a=read(long.TURN/'raw'/m['turn_file']);assert sha(long.TURN/'raw'/m['turn_file'])==m['turn_sha256']
        s=scouts.get(m['id'])
        if s is None:
            pre,post,_=long.old.eligible(a,m);s={'start':pre[0] if pre else m['tug_walk_windows'][0][0],'pre':pre,'post':post}
        for b in [10,20,30]:
            base,delay,r,tr=long.extract(a,m,s,b);previous=m['budgets'][str(b)]
            for k,v in r.items():assert v==previous[k]
            assert r['observed_seconds']<=b+1e-6
            t=tr['time'];mid=(t[:-1]+t[1:])/2;moving=tr['valid'].copy()
            for event in r['events']['10']['horizontal']:
                moving[(mid>=event['start']-1e-8)&(mid<event['end']-1e-8)]=False
            speed=tr['speed'][:-1];pool=float(np.mean(speed[moving]));paces=[]
            for name,sel in [('outbound',mid<m['turn_start']),('return',mid>m['turn_end'])]:
                good=moving&sel;seconds=float(sum(good)/30);v=float(np.mean(speed[good])) if seconds>=.5 else pool
                assert abs(v-m['pace'][str(b)][name]['speed'])<1e-12
                assert seconds==m['pace'][str(b)][name]['moving_seconds']
                paces.append(1/v)
            expected={'Baseline':np.r_[base,delay['10'][2]],'Pooled':np.r_[base,1/pool,delay['10'][2]],'Directional':np.r_[base,paces,delay['10'][2]]}
            for prefix,x in expected.items():
                delta=float(max(abs(d[f'{prefix}_{b}'][i]-x)));assert delta<1e-12;maxdiff=max(maxdiff,delta)
            windows+=1
        if (i+1)%50==0:print('RAW_INPUTS',i+1,flush=True)
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort;new=np.load(out/'input.npz');previous=np.load(PRIOR/cohort/'input.npz')
        for n,h in info['hashes'].items():assert sha(out/n)==h
        assert sha(out/'splits.json')==sha(PRIOR/cohort/'splits.json')
        assert np.array_equal(new['ids'],previous['ids']) and np.array_equal(new['y'],previous['y'])
        assert len(set(new['ids']))==info['people']
        for b in [10,20,30]:assert np.array_equal(new[f'Baseline_{b}'],previous[f'Add10_{b}'])
        for view in info['views']:assert np.isfinite(new[view]).all() and np.array_equal(new[view],d[view][:info['people']])
        for rep in json.loads((out/'splits.json').read_text()):
            tests=[]
            for f in rep['splits']:
                train=set(f['train']);test=set(f['test']);assert not train&test and train|test==set(range(info['people']));tests+=f['test']
                inner=[]
                for v in f['inner']:
                    assert not set(v['train'])&set(v['validation']) and set(v['train'])|set(v['validation'])==train
                    inner+=v['validation']
                assert sorted(inner)==sorted(train)
            assert sorted(tests)==list(range(info['people']))
    result={'passed':True,'source_people':177,'raw_windows_reextracted':windows,'max_feature_difference':maxdiff,'same_person_splits':True,'same_clinical_targets':True,'same_windows_and_stop_offsets':True,'all_legs_supported_without_fallback':all(not r['pace'][str(b)][k]['fallback'] for r in rows for b in [10,20,30] for k in ['outbound','return'])}
    save('input-verification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
