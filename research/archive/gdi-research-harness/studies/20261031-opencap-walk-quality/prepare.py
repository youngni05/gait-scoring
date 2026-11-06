from common import *
import shutil

def main():
    assert not (HERE/'preparation.json').exists(),'Preparation already frozen'
    verified=json.loads((PRIOR/'verification.json').read_text());assert verified['passed']
    protocol={'primary':'STS_common6; RF; quality+actual pattern versus quality control',
        'secondary':'TUG_common6; Ridge; robust extraction; stable/flagged held-out subgroups',
        'people':'same prior 179 STS and 195 TUG people, visit/start/exact6s; sub-283 excluded everywhere',
        'views':{'speed':'A: same3 speed summaries','quality':'B: A +31 consistency/missingness variables',
                 'pattern':'C: B +14 unchanged prior pattern values','robust':'D: B +median14 pattern values across9 fixed extraction settings'},
        'quality_variables':['bad_fraction','fraction_of_9_variants_with_cycles','cycle_duration_range_across_variants']+
                            ['original_missing_'+s for s in PATTERN_NAMES]+['robust_missing_'+s for s in PATTERN_NAMES],
        'variants':VARIANTS,'variant_axes':'smoothing points3/5/7 and knee peak prominence6/8/10deg; other feature definitions frozen',
        'stable_definition':'bad_fraction<=0.05; original>=2cycles/leg; >=7/9variants return cycle duration; duration range<=0.15s; all14original+robust values finite',
        'subgroups':'same FULL-cohort fitted models evaluated on stable/flagged people; no quality exclusion from training; no easier-cohort claim',
        'split':'exact prior5 outer partitions and each inner3fold; same across A/B/C/D',
        'models':{'RF':{'trees':128,'leaf_grid':[2,5],'seeds':SEEDS},'Ridge':{'alpha_grid':[1,10,100],'seeds':[-1]}},
        'preprocessing':'train-only median imputation without automatic indicators; identical31explicit quality variables in B/C/D; train-only scaling',
        'metrics':'person-equal held-out MAE, underprediction=max(y-p,0), overprediction=max(p-y,0); >=20s descriptive tail',
        'comparison':'B-A consistency contribution; C-B original value contribution; D-B robust value contribution; D-C extraction sensitivity',
        'uncertainty':'10000 paired person resamples of repeat-averaged error; correlated repeats not independent; exploratory reused data',
        'interpretation':'consistency is NOT ground-truth pose accuracy; may encode genuine pathology; residual errors cannot prove gait lacks information',
        'resources':'local CPU, existing packages, no original RGB/manual contact labels available'}
    save('protocol.json',protocol)
    sources={str(PRIOR/'common.py'):sha(PRIOR/'common.py'),str(PRIOR/'verification.json'):sha(PRIOR/'verification.json')}
    cohorts={};cache={};quality_rows=[]
    for name in ['STS_common6','TUG_common6']:
        for f in ['input.npz','splits.json','manifest.csv']:sources[str(PRIOR/name/f)]=sha(PRIOR/name/f)
        d=dict(np.load(PRIOR/name/'input.npz')); assert 'sub-283' not in d['ids']
        old=[];rob=[];qc=[];stable=[]
        for pid,x in zip(d['ids'],d['x']):
            if pid not in cache:cache[pid]=extract(x)
            o,r,q,s,detail=cache[pid]
            np.testing.assert_allclose(o,prior.patterns(x)[0],equal_nan=True,rtol=0,atol=0)
            old.append(o);rob.append(r);qc.append(q);stable.append(s);quality_rows.append(dict(dataset=name,id=str(pid),target=float(d['y'][len(stable)-1]),**detail))
        old=np.stack(old);rob=np.stack(rob);qc=np.stack(qc);stable=np.array(stable)
        np.testing.assert_allclose(old,d['pattern'][:,3:],equal_nan=True,rtol=0,atol=0)
        qview=np.c_[d['speed'],qc]
        out=HERE/name;out.mkdir()
        np.savez_compressed(out/'input.npz',ids=d['ids'],y=d['y'],x=d['x'],speed=d['speed'],
                            quality=qview,pattern=np.c_[qview,old],robust=np.c_[qview,rob],stable=stable)
        for f in ['manifest.csv','splits.json']:shutil.copy2(PRIOR/name/f,out/f)
        cohorts[name]={'people':len(d['ids']),'stable':int(stable.sum()),'flagged':int((~stable).sum()),
                       'tail_ge20':int((d['y']>=20).sum()),'stable_tail':int(((d['y']>=20)&stable).sum()),
                       'input_sha256':sha(out/'input.npz'),'splits_sha256':sha(out/'splits.json')}
    pd.DataFrame(quality_rows).to_csv(HERE/'quality-audit.csv',index=False)
    save('preparation.json',{'cohorts':cohorts,'sources':sources,'protocol_sha256':sha(HERE/'protocol.json'),'common_sha256':sha(HERE/'common.py')})
    print(json.dumps(cohorts,indent=2),flush=True)

if __name__=='__main__':main()
