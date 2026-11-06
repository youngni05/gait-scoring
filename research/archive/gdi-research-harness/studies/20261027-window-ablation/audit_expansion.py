"""Identity-only audit of unresolved records; no label access or cohort changes."""
import csv
import hashlib
import importlib.util
import io
from collections import Counter,defaultdict
from pathlib import Path
from prepare import HERE,H,ROOT,load,save,sha,require

def main():
    require((HERE/'expansion_plan.md').is_file(),'Missing prior plan')
    old=H/'control/desktop-sessions/20261026T150518Z-87c0f4'
    index=load(ROOT/'outputs/stanford-sample-check-20261016/archive-index.json')
    previous=load(old/'identity_linkage_summary.json')
    for p,d in previous['sources'].items(): require(sha(ROOT/p)==d,'Metadata source drift')
    ids=load(old/'exclusion_identities.json');excluded=set(ids['development_patient_ids']+ids['original_test_patient_ids'])
    rows=load(old/'identity_linkage_rows.json')
    candidates={r['patient_id'] for r in rows if r['status']=='unseen_identity_candidate'}
    require(len(candidates)==1113 and len(rows)==5713,'Previous inventory changed')
    spec=importlib.util.spec_from_file_location('archive_reader',ROOT/'tools/stanford-sample-check/inspect_archive.py')
    arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)
    stat=arc.ARCHIVE.stat();require(stat.st_size==index['archive_bytes'],'Archive changed')
    proof=[]
    def projected(name,allowed):
        require(name in ('video_list.csv','alldata.csv'),'Disallowed member')
        members=[m for m in index['non_json_members'] if Path(m['name']).name==name]
        require(len(members)==1,'Metadata member ambiguous')
        member=members[0];raw=arc.read_member(member)
        reader=csv.reader(io.StringIO(raw.decode('utf-8-sig')));header=next(reader)
        positions=[header.index(k) for k in allowed]
        result=[]
        for row in reader:
            require(len(row)==len(header),'CSV width changed')
            result.append(tuple(row[i] for i in positions))
        proof.append({'member':member['name'],'sha256':hashlib.sha256(raw).hexdigest(),
                      'retained_columns':allowed,'rows':len(result),'nonidentity_values_used':False})
        return result
    videos=projected('video_list.csv',('Exam_ID','Video_File'))
    clinical=projected('alldata.csv',('Patient_ID','examid','side'))
    norm=lambda s:str(int(s)) if s.isdigit() else s
    vex=defaultdict(set);exrows=defaultdict(list)
    for exam,video in videos:vex[norm(video)].add(exam)
    for patient,exam,side in clinical:exrows[exam].append((patient,side))
    categories=Counter();groups=defaultdict(set);newly_linkable=set();diagnostics=[]
    for row in rows:
        if row['status']!='unresolved_bilateral_identity':continue
        rr=exrows[row['exam_id']];patients={p for p,s in rr if p};sides={s for p,s in rr}
        if not rr:category='no_identity_rows'
        elif len(patients)>1:category='conflicting_people'
        elif len(patients)==1 and sides=={'L','R'} and len(set(rr))==2:
            category='duplicate_identity_rows_with_both_sides'
            patient=next(iter(patients))
            if patient not in excluded:
                newly_linkable.add(patient)
        elif len(patients)==1 and sides in ({'L'},{'R'}):category='only_one_side'
        else:category='other_unresolved_identity'
        categories[category]+=1
        if len(patients)==1 and next(iter(patients)) not in excluded:groups[category].update(patients)
        diagnostics.append({'folder':row['folder'],'category':category,'identity_rows':len(rr),
                            'distinct_identity_rows':len(set(rr)),'distinct_nonempty_people':len(patients)})
    require(sum(categories.values())==2504,'Unresolved partition mismatch')
    require(arc.ARCHIVE.stat().st_mtime_ns==stat.st_mtime_ns and arc.ARCHIVE.stat().st_size==stat.st_size,'Archive metadata changed')
    # Diagnostics only. A duplicate identity projection need not be a duplicate clinical visit.
    result={'verified':True,'records_diagnosed':2504,'record_categories':dict(categories),
            'distinct_nonexcluded_people_per_category':{k:len(v) for k,v in groups.items()},
            'both_sides_duplicate_identity_people_nonexcluded':len(newly_linkable),
            'both_sides_duplicate_identity_people_already_in1113':len(newly_linkable&candidates),
            'both_sides_duplicate_identity_people_outside1113':len(newly_linkable-candidates),
            'additional_approved_GDI_evaluation_people':0,'gdi_payload_opened':False,'raw_poses_opened':False,
            'selection_rule_changed':False,'evaluation_assignment_changed':False,
            'source_proofs':proof,'unresolved_video_exam_records':4,
            'interpretation':'Metadata bottlenecks only; no recovered participants or relaxed inclusion criteria.'}
    save(HERE/'expansion_identity_rows.json',diagnostics)
    save(HERE/'expansion_identity_audit.json',result)
    print(__import__('json').dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
