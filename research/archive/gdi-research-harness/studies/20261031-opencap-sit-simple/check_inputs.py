from study import *

def main():
    p=json.loads((HERE/'protocol.json').read_text());d=dict(np.load(HERE/'input.npz'));old=np.load(PRIOR/'input.npz')
    for n,h in p['source_hashes'].items():assert sha(HERE.parent/n)==h
    for n,h in p['input_hashes'].items():assert sha(HERE/n)==h
    assert sha(HERE/'splits.json')==sha(PRIOR/'splits.json') and sha(HERE/'manifest.json')==sha(PRIOR/'manifest.json')
    assert np.array_equal(d['ids'],old['ids']) and np.array_equal(d['y'],old['y'])
    assert 'sub-283' not in d['ids'] and len(set(d['ids']))==100 and sum(d['y']>=20)==3
    cols={'ShortSpeedTurn':list(range(4)),'PlusSitTime':[0,1,2,3,21],'PlusSitMotion':list(range(4))+list(range(22,29)),'PlusSitAll':list(range(4))+list(range(21,29))}
    for v,ix in cols.items():assert np.array_equal(d[v],old['Full'][:,ix])
    for v in VIEWS:assert np.isfinite(d[v]).all()
    common,prepare,geometry=imports();manifest=json.loads((HERE/'manifest.json').read_text());audit=json.loads((HERE/'observation-audit.json').read_text())
    for i,(m,r) in enumerate(zip(manifest,audit)):
        assert m['id']==r['id']==d['ids'][i]
        assert abs(r['control_input_seconds']-r['sitting_input_seconds'])<1e-10
        assert r['control_walk_end']<=m['walk_end']+1e-7
        assert abs(r['control_walk_seconds']-(3+m['phase_windows']['sit'][1]-m['phase_windows']['sit'][0]))<1e-9
        path=common.SOURCE/'raw'/m['walk_file'];assert sha(path)==m['walk_sha256'];a=common.read(path)
        expected=prepare.walk_window(a,m['walk_start'],r['control_walk_seconds'])[0]
        assert np.allclose(expected,d['MatchedSpeedTurn'][i,:3],atol=1e-12,rtol=0)
        assert d['MatchedSpeedTurn'][i,3]==d['ShortSpeedTurn'][i,3]
    for rep in json.loads((HERE/'splits.json').read_text()):
        seen=[]
        for f in rep['splits']:
            assert not set(f['train'])&set(f['test']);assert set(f['train'])|set(f['test'])==set(range(100));seen+=f['test'];iv=[]
            for inn in f['inner']:
                assert not set(inn['train'])&set(inn['validation']) and set(inn['train'])|set(inn['validation'])==set(f['train']);iv+=inn['validation']
            assert sorted(iv)==sorted(f['train'])
        assert sorted(seen)==list(range(100))
    save('input-verification.json',{'passed':True,'people':100,'long_people':3,'sub283_absent':True,'exact_previous_person_folds':True,'exact_previous_labels':True,
        'original_sit_turn_boundaries_unchanged':True,'control_walk_features_recomputed':True,'control_within_previous_clean_walk_window':True,'equal_model_input_seconds_every_person':True})
    print('INPUT_VERIFICATION_PASSED',flush=True)
if __name__=='__main__':main()
