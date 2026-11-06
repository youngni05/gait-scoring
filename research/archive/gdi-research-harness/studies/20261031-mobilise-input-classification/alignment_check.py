"""Explicit visit-level identity checks, including reused OOF reference arrays."""
from study import *

def run():
    old.init(); reference={}; checks=0; populations={}
    for target,band,k in itertools.product(['sppb','tug'],['short','long'],[1,3,5]):
        held=[]
        for fold in range(5):
            tr,te,inner,_=old.cell_data(target,band,k,fold)
            col='total_sppb_score' if target=='sppb' else 'tugtime'
            held.append(te[KEY+[col,'fold']])
            for seed in SEEDS:
                tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
                new=np.load(STORE/'folds'/f'{tag}-inner.npz')
                previous=np.load(old.STORE/'folds'/f'{tag}-inner.npz')
                expected_visits=tr.visit_number.to_numpy(dtype='U')
                np.testing.assert_array_equal(new['visits'],expected_visits)
                np.testing.assert_array_equal(previous['visits'],expected_visits)
                np.testing.assert_array_equal(new['ids'],previous['ids'])
                np.testing.assert_array_equal(new['y'],old.classes(tr[col].to_numpy(),target))
                np.testing.assert_array_equal(new['y'],previous['y'])
                checks+=1
        full=pd.concat(held).sort_values(KEY).reset_index(drop=True)
        assert not full.duplicated(KEY).any()
        if target in reference:pd.testing.assert_frame_equal(full,reference[target])
        else:reference[target]=full
        assert len(full)==(7356 if target=='sppb' else 5688)
        assert full.participantid.nunique()==(2174 if target=='sppb' else 1939)
        if band=='long' and k==5:
            low=full[col]<=3 if target=='sppb' else full[col]>20
            populations[target]={'people':int(full.participantid.nunique()),'visits':len(full),
                                 'low_people':int(full.loc[low,'participantid'].nunique()),'low_visits':int(low.sum())}
    save(HERE/'alignment-verification.json',{'passed':True,'fold_seed_arrays':checks,'populations':populations,
        'checks':['same participant+visit sequence in reused and new inner OOF arrays','labels match source',
                  'identical participant visits labels and folds across all six conditions'],
        'code_sha256':sha(__file__)})
    print('ALIGNMENT_PASSED',checks,populations,flush=True)

if __name__=='__main__':run()
