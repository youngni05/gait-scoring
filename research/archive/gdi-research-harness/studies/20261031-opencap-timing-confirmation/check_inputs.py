from study import *
def main():
    p=json.loads((HERE/'protocol.json').read_text());assert p['code_sha256']==sha(HERE/'study.py')
    for n,h in p['source_hashes'].items():assert sha(HERE.parent/n)==h
    for c,info in p['cohorts'].items():
        olddata=np.load(PRIOR/c/'input.npz');data=np.load(HERE/c/'input.npz')
        for k in ['ids','y']:assert np.array_equal(data[k],olddata[k])
        for b in [10,20,30]:
            for kind,prior in [('Gait','Gait'),('Withheld','Reallocated'),('Candidate','Prep')]:assert np.array_equal(data[f'{kind}_{b}'],olddata[f'{prior}_{b}'])
            assert np.array_equal(data[f'Withheld_{b}'][:,:4],data[f'Candidate_{b}'][:,:4])
        assert (HERE/c/'splits.json').read_bytes()==(PRIOR/c/'splits.json').read_bytes()
        assert (HERE/c/'manifest.json').read_bytes()==(PRIOR/c/'manifest.json').read_bytes()
        for rep in json.loads((HERE/c/'splits.json').read_text()):
            seen=[]
            for f in rep['splits']:
                assert not set(f['train'])&set(f['test']);assert set(f['train'])|set(f['test'])==set(range(len(data['y'])));seen+=f['test'];ivs=[]
                for v in f['inner']:
                    assert not set(v['train'])&set(v['validation']);assert set(v['train'])|set(v['validation'])==set(f['train']);ivs+=v['validation']
                assert sorted(ivs)==sorted(f['train'])
            assert sorted(seen)==list(range(len(data['y'])))
        for n,h in info['hashes'].items():assert sha(HERE/c/n)==h
    annotations=json.loads((HERE/'timing-annotations.json').read_text());assert len(annotations)==83 and all(r['timer_start'] is None and r['timer_end'] is None and r['confirmed_preparation_intervals'] is None for r in annotations)
    save('input-verification.json',{'passed':True,'same_people_labels_folds_clips':True,'confirmed_times_not_invented':True,'verified_clinical_onsets':0,'source_hashes_unchanged':True})
    print('CONSERVATIVE INPUTS VERIFIED; CONFIRMED-TIME EXPERIMENT UNAVAILABLE',flush=True)
if __name__=='__main__':main()
