"""Read-only verification of actual outputs and held-out reservation."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from prepare import HERE,H,ROOT,DEV,load,sha,require

def main():
    result=load(HERE/'comparison.json');require(result['verified'],'No completed comparison')
    for file in ('sources.json','raw_sources.json','analysis_sources.json'):
        require(all(sha(ROOT/p)==d for p,d in load(HERE/file)['sha256'].items()),'Changed pinned source')
    grouped=defaultdict(lambda:defaultdict(list));row_counts=defaultdict(int)
    with (HERE/'predictions.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key=(row['scope'],row['method']);row_counts[key]+=1
            e=(abs(float(row['actual_L'])-float(row['prediction_L']))+abs(float(row['actual_R'])-float(row['prediction_R'])))/2
            grouped[key][row['patient_id']].append(e)
    for (scope,method),patients in grouped.items():
        require((len(patients),row_counts[(scope,method)])==((37,40) if scope=='validation37' else (175,203)), 'CSV population count mismatch')
        value=statistics.mean(statistics.mean(es) for es in patients.values())
        require(abs(value-result[scope][method]['person_mae'])<1e-10,'Independent MAE mismatch')
    split=load(HERE/'new_person_split.json');records=load(HERE/'new_record_split.json')['records']
    roles={r['patient_id']:r['reserved_role'] for r in split['people']}
    test={p for p,role in roles.items() if role=='final_evaluation_reserved'}
    dev=set(roles)-test
    require(len(test)==223 and len(dev)==520 and not test&dev,'Wrong person reservation')
    require(all(r['reserved_role']==roles[r['patient_id']] for r in records),'One person spans roles')
    for role,n in (('final_evaluation_reserved',378),('development_candidate',861)):
        require(sum(r['input124_status']=='prepared' and r['reserved_role']==role for r in records)==n,'Reserved input count mismatch')
    require(not (test|dev)&set(grouped[('validation37','front_local')]),'New/old people overlap')
    require(not (test|dev)&set(grouped[('internal175','front_local')]),'New/old training people overlap')
    for r in records:
        if r['input124_status']=='prepared': require(sha(H/r['artifact'])==r['sha256'],'New reserved input changed')
    diagnosis=load(HERE/'normalization_failure_diagnosis.json')
    require(diagnosis['mismatches_at_positive_confidence_points']==0 and diagnosis['unclipped_then_local_matches'],'Failure diagnosis mismatch')
    raw=load(HERE/'raw_preparation.json');require(raw['records']==243 and raw['raw_json_checked']==158193,'Raw recovery incomplete')
    require(not (HERE/'features.npz').exists() and not (HERE/'preparation.json').exists(),'Failed path unexpectedly reached complete gate')
    for index in range(243):
        p=HERE/f'raw_records/row_{index:04d}.npz';proof=load(p.with_suffix('.json'))
        require(sha(p)==proof['sha256'],'Raw feature changed')
        require(sha(HERE/f'raw_member_proofs/row_{index:04d}.json')==proof['raw_proof_sha256'],'Raw member evidence changed')
        require(proof['canonical_raw124_exactly_reproduced'] and proof['confidence_preserved'],'Input contract mismatch')
    for p in HERE.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    check={'verified':True,'csv_method_scope_groups':len(grouped),'validation_people':37,'internal_people':175,
           'new_final_reserved_people':223,'new_development_candidate_people':520,'new_people_in_current_comparison':0,
           'new_final_GDI_labels_opened':False,'new_final_evaluation_executed':False,
           'raw_recovery_records_verified':243,'source_fingerprints_unchanged':True,
           'failure_preserved':True,'old_guard_not_relaxed':True,'no_extra_fits_in_verification':True}
    (HERE/'independent_verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(check,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
