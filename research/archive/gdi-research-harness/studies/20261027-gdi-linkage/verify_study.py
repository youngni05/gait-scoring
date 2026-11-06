"""Independent join and CSV checks; final numeric outcomes remain unused."""
import csv
import importlib.util
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from audit_labels import HERE,H,ROOT,OLD,LEDGER,load,sha,save

def main():
    summary=load(HERE/'linkage_summary.json')
    result=load(HERE/'transfer_result.json')
    assert summary['verified'] and result['verified']
    for bundle in (summary['sources'],load(HERE/'feature_sources.json')['sha256'],load(HERE/'transfer_sources.json')['sha256']):
        assert all(sha(ROOT/p)==s for p,s in bundle.items())
    dev=load(HERE/'development_linkage.json')['records']
    final=load(HERE/'final_linkage_presence_only.json')['records']
    assert len(dev)==861 and len(final)==378
    assert len({r['patient_id'] for r in dev})==520 and len({r['patient_id'] for r in final})==223
    assert not {r['patient_id'] for r in dev}&{r['patient_id'] for r in final}
    frozen={r['sequence']:r for r in load(OLD/'new_record_split.json')['records']}
    ledger={r['sequence']:r for r in load(LEDGER)['records']}
    for r in dev+final:
        for k in ('patient_id','video_id','artifact','sha256','reserved_role'):
            assert r[k]==frozen[r['sequence']][k]
        assert r['exam_id']==ledger[r['sequence']]['exam_id']
        assert sha(H/r['artifact'])==r['sha256']
    # Read the raw member independently; retain only development keys before
    # selecting numeric values. No reserved-final GDI field is selected here.
    spec=importlib.util.spec_from_file_location('verify_arc',ROOT/'tools/stanford-sample-check/inspect_archive.py')
    arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)
    payload=arc.read_member(summary['archive_member'])
    import hashlib
    assert hashlib.sha256(payload).hexdigest()==summary['archive_member_sha256']
    wanted={(r['patient_id'],r['exam_id']) for r in dev}
    raw=defaultdict(list)
    for r in csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))):
        if (r['Patient_ID'],r['examid']) not in wanted:continue
        raw[(r['Patient_ID'],r['examid'],r['side'])].append(r['GDI'])
    for r in dev:
        for i,side in enumerate(('L','R')):
            values=raw[(r['patient_id'],r['exam_id'],side)]
            assert len(values)==1 and math.isfinite(float(values[0]))
            assert r['GDI_LR'][i]==float(values[0])
    assert all('GDI_LR' not in r and 'eligible' not in r and r['linkage_present'] for r in final)
    for r in dev:
        p=HERE/f"development_features/row_{r['sequence']:04d}.npz"
        proof=load(p.with_suffix('.json'))
        assert sha(p)==proof['sha256'] and proof['input_sha256']==r['sha256']
        assert proof['feature_sources_sha256']==sha(HERE/'feature_sources.json')
    assert len(list((HERE/'development_features').glob('*.npz')))==861
    groups=defaultdict(lambda:defaultdict(list))
    with (HERE/'development_predictions.csv').open(encoding='utf-8') as f:pred=list(csv.DictReader(f))
    assert len(pred)==861 and len({r['patient_id'] for r in pred})==520
    for p,r in zip(pred,dev,strict=True):
        assert p['patient_id']==r['patient_id'] and p['video_id']==r['video_id'] and int(p['sequence'])==r['sequence']
        for i,s in enumerate(('L','R')):assert float(p['actual_'+s])==r['GDI_LR'][i]
        for method in ('frozen124','trainmean'):
            e=(abs(float(p[method+'_L'])-float(p['actual_L']))+abs(float(p[method+'_R'])-float(p['actual_R'])))/2
            groups[method][p['patient_id']].append(e)
    independently={m:{p:statistics.mean(es) for p,es in v.items()} for m,v in groups.items()}
    means={m:statistics.mean(v.values()) for m,v in independently.items()}
    assert abs(means['frozen124']-result['frozen124']['person_mae'])<1e-10
    assert abs(means['trainmean']-result['old_train_mean_baseline']['person_mae'])<1e-10
    differences=[independently['trainmean'][p]-independently['frozen124'][p] for p in sorted(independently['trainmean'])]
    assert sum(d>0 for d in differences)==result['people_improved']
    assert sum(d<0 for d in differences)==result['people_worsened']
    assert sha(HERE/'development_predictions.csv')==result['predictions_sha256']
    save('independent_verification.json',{'verified':True,'development_labels_independently_rejoined':861,
        'frozen_inputs_sha256_checked':1239,'completed_features_sha256_checked':861,
        'prediction_csv_independently_checked':861,'person_overlap':0,'final_numeric_labels_accessed_by_verifier':False,
        'frozen124_person_mae':means['frozen124'],'trainmean_person_mae':means['trainmean'],
        'final_score_or_feature_files_created':False,'source_hashes_unchanged':True,
        'verification_script_sha256':sha(Path(__file__))})
    print(json.dumps(load(HERE/'independent_verification.json'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
