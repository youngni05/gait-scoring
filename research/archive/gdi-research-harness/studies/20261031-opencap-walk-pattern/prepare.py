from common import *

def main():
    assert not (HERE/'preparation.json').exists(), 'Frozen preparation already exists.'
    labels=pd.read_csv(LABELS); assert not labels.duplicated(['subid','visit']).any()
    lookup=labels.set_index(['subid','visit'])
    original=dict(np.load(STABILITY/'input.npz')); base_ids=original['ids']
    old_splits=json.loads((STABILITY/'splits.json').read_text())
    six=dict(np.load(SOURCE/'6s/input.npz')); common=set(six['ids'])
    protocol={'primary':'STS_common6: speed versus speed+patterns, RF',
        'secondary':['STS_common3','STS_expanded3','TUG_common6','TUG_common3','TUG_expanded3'],
        'exclusion':'sub-283 (42.59 s TUG) excluded from every training, validation and test set',
        'cohort':'existing frozen person/visit/session/start; actual positive same-visit target; no new visit selection',
        'split':'restrict all five existing outer partitions AND each existing inner fold by person ID; same across views and common3/common6',
        'views':['speed (mean, SD, mean absolute adjacent change)','pattern (same speed plus 14 predeclared features)'],
        'pattern_features':PATTERN_NAMES,
        'cycle_proxy':'5-point moving mean; knee flexion peaks distance>=0.4s prominence>=8deg; complete max-to-max cycles 0.6..2.5s; both legs >=1 cycle; CV requires >=2 per leg',
        'quality':'any 17-angle change >1000deg/s masks both frames and +/-3frames; exclude intersecting cycles; noncycle ROM requires >=80% clean samples',
        'missing':'no person dropped for pattern availability; training-only median imputation plus missing indicators; no outcome supplied to extraction',
        'models':{'RF':{'trees':128,'leaf_grid':[2,5],'seeds':SEEDS},'Ridge':{'alpha_grid':[1,10,100],'seeds':[-1]}},
        'selection':'mean absolute error on pooled inner validation people only; no tail-based tuning; no clipping or output calibration',
        'tail':'actual test score >=20 seconds, descriptive subgroup for both outcomes, not a clinical threshold',
        'uncertainty':'paired person bootstrap of repeat-averaged absolute errors; conditional on fitted models, not independent repeated cohorts',
        'success_interpretation':'report paired overall error, tail underprediction AND overprediction, repeat consistency; no improvement inferred from correlation alone',
        'scope':'exploratory reused OpenCap multi-camera kinematics; not single-camera accuracy, no SPPB conversion, missing STS cause unknown'}
    save('protocol.json',protocol)
    summary={}; quality=[]; source_hashes={str(LABELS):sha(LABELS),str(STABILITY/'splits.json'):sha(STABILITY/'splits.json')}
    for seconds,cohort in [(6,'common'),(3,'common'),(3,'expanded')]:
        src=SOURCE/('6s' if seconds==6 else 'all3s')
        d=dict(np.load(src/'input.npz')); manifest=pd.read_csv(src/'manifest.csv')
        assert list(d['ids'])==manifest.id.tolist()
        source_hashes[str(src/'input.npz')]=sha(src/'input.npz')
        source_hashes[str(src/'manifest.csv')]=sha(src/'manifest.csv')
        speed=np.stack([speed_features(x) for x in d['x']]); pf=[]
        for pid,x in zip(d['ids'],d['x']):
            v,q=patterns(x);pf.append(v)
            quality.append(dict(id=pid,seconds=seconds,cohort=cohort,**q))
        pattern=np.concatenate([speed,np.stack(pf)],axis=1)
        for outcome,col in [('STS','tft_5xsts'),('TUG','tft_tug')]:
            y=np.array([lookup.loc[(r.id,int(r.visit)),col] for r in manifest.itertuples()])
            keep=np.isfinite(y)&(y>0)&(d['ids']!='sub-283')
            if cohort=='common':keep&=np.isin(d['ids'],list(common))
            ix=np.flatnonzero(keep);ids=d['ids'][ix]; name=f'{outcome}_{cohort}{seconds}'
            out=HERE/name;out.mkdir()
            np.savez_compressed(out/'input.npz',ids=ids,y=y[ix],speed=speed[ix],pattern=pattern[ix],x=d['x'][ix])
            man=manifest.iloc[ix].copy();man['outcome']=outcome;man['label']=y[ix]
            man.to_csv(out/'manifest.csv',index=False)
            local={pid:i for i,pid in enumerate(ids)}
            def remap(index):return [local[base_ids[i]] for i in index if base_ids[i] in local]
            plan=[]
            for rep in old_splits:
                ss=[]
                for s in rep['splits']:
                    ss.append({'fold':s['fold'],'train':remap(s['train']),'test':remap(s['test']),
                               'inner':[{'train':remap(i['train']),'validation':remap(i['validation'])} for i in s['inner']]})
                plan.append({'repeat':rep['repeat'],'splits':ss})
            save(f'{name}/splits.json',plan)
            summary[name]={'people':len(ids),'tail_ge20':int((y[ix]>=20).sum()),
                           'range':[float(y[ix].min()),float(y[ix].max())],
                           'cycle_available':int(np.isfinite(pattern[ix,3]).sum()),
                           'cv_available':int(np.isfinite(pattern[ix,4]).sum()),
                           'input_sha256':sha(out/'input.npz'),'split_sha256':sha(out/'splits.json')}
    save('quality.json',quality)
    # Audit missing labels separately: they are not zero or proof of inability.
    man=pd.read_csv(STABILITY/'manifest.csv')
    man=man.merge(labels,left_on=['id','visit'],right_on=['subid','visit'],validate='one_to_one')
    man[['id','visit','target','tft_5xsts']].to_csv(HERE/'label-audit.csv',index=False)
    save('preparation.json',{'cohorts':summary,'source_hashes':source_hashes,
        'protocol_sha256':sha(HERE/'protocol.json'),'feature_code_sha256':sha(HERE/'common.py')})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
