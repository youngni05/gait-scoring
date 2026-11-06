from study import *

def main():
    protocol=json.loads((HERE/'protocol.json').read_text()); results={}
    for path,h in protocol['source_hashes'].items(): assert sha(HERE.parent/path)==h
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort;src=HERE.parent/info['source'];old=dict(np.load(src/'input.npz'));d=dict(np.load(out/'input.npz'))
        keep=np.flatnonzero(old['ids']!=REMOVED)
        assert np.array_equal(d['ids'],old['ids'][keep]) and np.array_equal(d['y'],old['y'][keep])
        assert len(set(d['ids']))==len(d['ids'])==info['people'] and REMOVED not in d['ids']
        for v,m in info['mapping'].items():
            if v=='CoreRise':
                core=np.load(HERE.parent/'20261031-opencap-chair/event_strict/input.npz');ix=[list(core['ids']).index(pid) for pid in d['ids']]
                expected=core['ChairAll'][ix]
            else: expected=old[m['source_view']][keep][:,m['columns']]
            assert np.array_equal(d[v],expected) and np.isfinite(d[v]).all()
        om=json.loads((src/'manifest.json').read_text());nm=json.loads((out/'manifest.json').read_text());assert nm==[om[i] for i in keep]
        for r,o in zip(json.loads((out/'splits.json').read_text()),json.loads((src/'splits.json').read_text())):
            assert r['repeat']==o['repeat'];seen=[]
            for f,g in zip(r['splits'],o['splits']):
                for key in ['train','test']:
                    assert list(d['ids'][f[key]])==[str(old['ids'][i]) for i in g[key] if old['ids'][i]!=REMOVED]
                assert not set(f['train'])&set(f['test']);seen+=f['test'];iv=[]
                for a,b in zip(f['inner'],g['inner']):
                    for key in ['train','validation']:
                        assert list(d['ids'][a[key]])==[str(old['ids'][i]) for i in b[key] if old['ids'][i]!=REMOVED]
                    assert not set(a['train'])&set(a['validation'])
                    assert set(a['train'])|set(a['validation'])==set(f['train']);iv+=a['validation']
                assert sorted(iv)==sorted(f['train'])
            assert sorted(seen)==list(range(len(d['ids'])))
        for m in nm:
            if cohort=='prep':
                augmented=3+m['turn_observed_seconds']+sum(b-a for a,b in m['phase_windows'].values())
                control=3+m['context_window'][1]-m['context_window'][0]
                assert abs(augmented-m['observed_seconds'])<1e-7 and abs(augmented-control)<1e-7
            else: assert abs(m['walk_end']-m['walk_start']-(3+m['turn_observed_seconds']))<1e-7
        results[cohort]={'people':len(d['ids']),'long_people':int(sum(d['y']>=20)),'unchanged_participant_assignment':True,'exact_original_features_and_windows':True,'matched_context_seconds':True}
    save('input-verification.json',{'passed':True,'cohorts':results});print(results)
if __name__=='__main__':main()
