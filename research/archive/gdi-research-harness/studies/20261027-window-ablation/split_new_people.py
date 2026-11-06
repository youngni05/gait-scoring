"""Freeze a person-level split of already prepared new inputs, before GDI access."""
import json
from pathlib import Path
import numpy as np
from prepare import HERE,H,ROOT,load,save,sha,require

def main():
    require((HERE/'new_cohort_plan.md').is_file(),'Missing approved direction and pre-label plan')
    require(not (HERE/'new_person_split.json').exists(),'New split is already fixed; never resample')
    latest=H/'control/desktop-sessions/20261027T065618Z-8aab63'
    receipt=load(latest/'receipt.json'); require(receipt['status']=='needs_user','Prior state changed')
    require(sha(latest/'response.json')==receipt['response_sha256'],'Latest response changed')
    ledger=latest/'input_readiness_ledger.json'
    require(sha(ledger)==load(latest/'checkpoint_summary.json')['ledger_sha256'],'Input ledger drift')
    rows=load(ledger)['records'];ready=[r for r in rows if r['input124_status']=='prepared']
    people=sorted({r['patient_id'] for r in ready});require(len(people)==743 and len(ready)==1239,'Input cohort drift')
    exclusions=load(H/'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json')
    old=set(exclusions['development_patient_ids']+exclusions['original_test_patient_ids'])
    require(not set(people)&old,'Existing people overlap')
    ordered=np.array(people)[np.random.default_rng(20261027).permutation(len(people))].tolist()
    test=set(ordered[:223]);dev=set(ordered[223:]);require(len(test)==223 and len(dev)==520 and not test&dev,'Split error')
    assignments={p:'final_evaluation_reserved' if p in test else 'development_candidate' for p in people}
    records=[]
    for r in rows:
        if r['patient_id'] not in assignments:continue
        item={k:r[k] for k in ('sequence','patient_id','video_id','input124_status','input243_status')}
        item['reserved_role']=assignments[r['patient_id']]
        if r['input124_status']=='prepared':
            p=H/r['artifact'];require(sha(p)==r['sha256'],'Prepared new input changed')
            item.update({'artifact':r['artifact'],'sha256':r['sha256']})
        records.append(item)
    summary={'verified':True,'seed':20261027,'ready_people':743,'ready_records':1239,
             'development_candidate_people':520,'final_evaluation_reserved_people':223,
             'development_ready_records':sum(r['reserved_role']=='development_candidate' and r['input124_status']=='prepared' for r in records),
             'final_evaluation_ready_records':sum(r['reserved_role']=='final_evaluation_reserved' and r['input124_status']=='prepared' for r in records),
             'not_included_unprepared_people':370,'person_overlap':0,'overlap_existing250':0,
             'GDI_labels_accessed':False,'confirmed_complete_GDI_evaluation_people':None,
             'original_175_37_split_changed':False,'new_training_or_evaluation_run':False,
             'source_ledger_sha256':sha(ledger),'plan_sha256':sha(HERE/'new_cohort_plan.md')}
    save(HERE/'new_person_split.json',{'summary':summary,'people':[{'patient_id':p,'reserved_role':assignments[p]} for p in sorted(people)]})
    save(HERE/'new_record_split.json',{'records':records,'summary':summary})
    # Reproduce the random split independently from persisted lists, without labels.
    saved=load(HERE/'new_person_split.json')['people']
    require({r['patient_id'] for r in saved if r['reserved_role']=='final_evaluation_reserved'}==test,'Persisted split drift')
    save(HERE/'new_cohort_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
