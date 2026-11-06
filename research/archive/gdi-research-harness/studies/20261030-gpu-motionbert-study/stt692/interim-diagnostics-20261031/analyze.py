"""Descriptive diagnostics only; no training decisions or outer-score comparisons."""
import hashlib,json,math,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
source=root/'snapshot.json'
d=json.loads(source.read_text(encoding='utf-8-sig'))
def mean(rows,key):return statistics.mean(r[key] for r in rows)
out={'checked_at':d['checked_at'],'snapshot_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'experiments':{}}
for name,e in d['experiments'].items():
    diagnostics=[]
    for fit in e['fits']:
        h=fit['rows'];last=h[-20:];slopes=[r['calibration_slope'] for r in h]
        assert [r['epoch'] for r in h]==list(range(1,len(h)+1))
        assert all(math.isfinite(r[k]) for r in h for k in ('train_person_mae','train_raw_person_mae','calibration_slope','calibration_intercept'))
        v={'fit':fit['fit'],'epochs':len(h),'complete':fit['complete'],
           'slope_all_range':[min(slopes),max(slopes)],'negative_slope_epochs':[r['epoch'] for r in h if r['calibration_slope']<0],
           'slope_last20_range':[min(r['calibration_slope'] for r in last),max(r['calibration_slope'] for r in last)],
           'intercept_last20_range':[min(r['calibration_intercept'] for r in last),max(r['calibration_intercept'] for r in last)],
           'train_last20':mean(last,'train_person_mae'),'probes':fit['probes']}
        if 'validation_person_mae' in h[0]:
            best=min(h,key=lambda r:r['validation_person_mae'])
            v.update(best_epoch=best['epoch'],best_validation=best['validation_person_mae'],
                     validation_last20=mean(last,'validation_person_mae'),
                     raw_validation_last20=mean(last,'validation_raw_person_mae'),
                     correction_delta_last20=mean(last,'validation_person_mae')-mean(last,'validation_raw_person_mae'),
                     train_validation_gap_last20=mean(last,'validation_person_mae')-mean(last,'train_person_mae'),
                     last20_above_best=mean(last,'validation_person_mae')-best['validation_person_mae'])
            v['blocks']=[{'start':block[0]['epoch'],'end':block[-1]['epoch'],
                         'train':mean(block,'train_person_mae'),'val':mean(block,'validation_person_mae'),
                         'raw_train':mean(block,'train_raw_person_mae'),'raw_val':mean(block,'validation_raw_person_mae'),
                         'slope':mean(block,'calibration_slope')}
                        for i in range(0,len(h),20) if (block:=h[i:i+20])]
        diagnostics.append(v)
    for fold,selection in e['selections'].items():
        inner=[f for f in e['fits'] if f['fit'].startswith(fold+'/inner')]
        assert len(inner)==4 and all(len(f['rows'])==200 for f in inner)
        weights=[f['rows'][0]['validation_people'] for f in inner]
        values=[sum(f['rows'][epoch]['validation_person_mae']*w for f,w in zip(inner,weights))/sum(weights) for epoch in range(200)]
        assert min(range(200),key=lambda i:values[i])+1==selection['selected_epoch']
        assert max(abs(a-b) for a,b in zip(values,selection['inner_mean_person_mae_by_epoch']))<1e-12
    common=[f for f in diagnostics if f['fit'].startswith('fold0/inner')]
    assert len(common)==4 and all(f['complete'] and f['epochs']==200 for f in common)
    comparison={str(start):{k:statistics.mean(next(b for b in f['blocks'] if b['start']==start)[k] for f in common)
               for k in ('train','val','raw_train','raw_val','slope')} for start in (81,101,181)}
    out['experiments'][name]={'fits':diagnostics,'selected_epochs':{k:v['selected_epoch'] for k,v in e['selections'].items()},
                            'fold0_four_inner_equal_fit_mean':comparison}
(root/'diagnostics.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf8')
print('Snapshot:',d['checked_at'])
for name,e in out['experiments'].items():
    print(name,'selected epochs',e['selected_epochs'])
    for f in e['fits']:
        print(json.dumps({k:v for k,v in f.items() if k not in ('blocks','probes')},ensure_ascii=False))
        if f['fit'].startswith('fold0/inner'):
            print('BLOCKS',f['blocks']);print('PROBES',f['probes'])
