"""Audit frozen cohort linkage; never numerically parse reserved-final outcomes."""
import csv
import hashlib
import importlib.util
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = HERE.parents[1]
ROOT = H.parent
OLD = H / 'studies/20261027-window-ablation'
LEDGER = H / 'control/desktop-sessions/20261027T065618Z-8aab63/input_readiness_ledger.json'
EXCLUDE = H / 'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json'

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, value):
    p = HERE / name
    assert not p.exists(), 'Refuse overwriting evidence'
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    assert (HERE / 'plan.md').is_file()
    split = load(OLD / 'new_person_split.json')
    summary = split['summary']
    assert sha(LEDGER) == summary['source_ledger_sha256']
    receipt = load(OLD / 'study_receipt.json')
    assert receipt['status'] == 'completed'
    sealed = {r['path']: r['sha256'] for r in receipt['artifacts']}
    for name in ('new_person_split.json', 'new_record_split.json', 'new_cohort_summary.json'):
        p = OLD / name
        assert sha(p) == sealed[p.relative_to(H).as_posix()]
    assert summary == load(OLD / 'new_cohort_summary.json')
    roles = {r['patient_id']: r['reserved_role'] for r in split['people']}
    assert len(roles) == 743
    exc = load(EXCLUDE)
    excluded = set(exc['development_patient_ids']) | set(exc['original_test_patient_ids'])
    assert len(excluded) == 250 and not set(roles) & excluded
    counts = Counter(roles.values())
    assert counts == {'development_candidate': 520, 'final_evaluation_reserved': 223}
    ledger = {r['sequence']: r for r in load(LEDGER)['records']}
    records = []
    for frozen in load(OLD / 'new_record_split.json')['records']:
        if frozen['input124_status'] != 'prepared': continue
        row = ledger[frozen['sequence']]
        for key in ('patient_id', 'video_id', 'input124_status'):
            assert frozen[key] == row[key]
        assert frozen['reserved_role'] == roles[row['patient_id']]
        artifact = H / frozen['artifact']
        assert artifact.is_file() and sha(artifact) == frozen['sha256']
        records.append(dict(frozen, exam_id=row['exam_id']))
    assert len(records) == 1239
    assert Counter(r['reserved_role'] for r in records) == {
        'development_candidate': 861, 'final_evaluation_reserved': 378}
    needed = {(r['patient_id'], r['exam_id'], side): roles[r['patient_id']]
              for r in records for side in ('L', 'R')}
    spec = importlib.util.spec_from_file_location('bounded_archive', ROOT / 'tools/stanford-sample-check/inspect_archive.py')
    arc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arc)
    cfg = load(H / 'jobs/native124-cache-v1/config.json')
    stat = arc.ARCHIVE.stat()
    assert {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns} == cfg['archive_stat']
    member, = [m for m in load(ROOT / 'outputs/stanford-sample-check-20261016/archive-index.json')['non_json_members']
               if m['name'] == 'annotations/gdi.csv']
    payload = arc.read_member(member)
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    assert reader.fieldnames == ['examid', 'Patient_ID', 'side', 'GDI'], 'Unexpected schema'
    dev = defaultdict(list)
    final = defaultdict(lambda: [0, 0])
    skipped_old = skipped_other = 0
    for row in reader:
        person = row['Patient_ID']
        if person in excluded:
            skipped_old += 1
            continue
        key = (person, row['examid'], row['side'])
        role = needed.get(key)
        if role is None:
            skipped_other += 1
            continue
        if role == 'development_candidate':
            dev[key].append(row['GDI'])
        else:
            final[key][0] += 1
            final[key][1] += int(bool(row['GDI'].strip()))
    dev_rows, final_rows = [], []
    for record in records:
        base = {k: record[k] for k in ('sequence','patient_id','video_id','exam_id','reserved_role','artifact','sha256')}
        statuses, labels = {}, []
        for side in ('L', 'R'):
            key = (record['patient_id'], record['exam_id'], side)
            if record['reserved_role'] == 'development_candidate':
                values = dev[key]
                if len(values) != 1:
                    status = 'missing' if not values else 'duplicate'
                else:
                    try: value = float(values[0])
                    except (ValueError, TypeError): value = float('nan')
                    status = 'finite_unique' if math.isfinite(value) else 'nonfinite_or_nonnumeric'
                    if status == 'finite_unique': labels.append(value)
            else:
                n, nonempty = final[key]
                status = 'unique_nonempty_unvalidated' if n == 1 and nonempty == 1 else (
                    'missing' if n == 0 else 'duplicate' if n > 1 else 'empty')
            statuses[side] = status
        base['sides'] = statuses
        if record['reserved_role'] == 'development_candidate':
            base['eligible'] = all(s == 'finite_unique' for s in statuses.values())
            if base['eligible']: base['GDI_LR'] = labels
            dev_rows.append(base)
        else:
            base['linkage_present'] = all(s == 'unique_nonempty_unvalidated' for s in statuses.values())
            final_rows.append(base)
    assert not set(r['patient_id'] for r in dev_rows) & set(r['patient_id'] for r in final_rows)
    result = {
        'verified': True, 'question': 'Can frozen ready inputs link to their GDI labels?',
        'development': {'ready_people':520,'ready_records':861,
            'eligible_people':len({r['patient_id'] for r in dev_rows if r['eligible']}),
            'eligible_records':sum(r['eligible'] for r in dev_rows),
            'side_status_counts':dict(Counter(s for r in dev_rows for s in r['sides'].values()))},
        'final_reserved': {'ready_people':223,'ready_records':378,
            'linkage_present_people':len({r['patient_id'] for r in final_rows if r['linkage_present']}),
            'linkage_present_records':sum(r['linkage_present'] for r in final_rows),
            'side_status_counts':dict(Counter(s for r in final_rows for s in r['sides'].values())),
            'numeric_labels_parsed_or_saved':False,'confirmed_valid_GDI_people':None,
            'predictions_or_scores_computed':False},
        'original250_GDI_fields_accessed':False,'skipped_old_rows':skipped_old,
        'skipped_other_rows':skipped_other,'person_split_changed':False,'metric_or_input_rules_changed':False,
        'training_or_evaluation_run':False,
        'archive_member':member,'archive_member_sha256':hashlib.sha256(payload).hexdigest(),
        'archive_stat':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns},
        'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in (
            HERE/'plan.md',Path(__file__),LEDGER,EXCLUDE,OLD/'new_person_split.json',
            OLD/'new_record_split.json',OLD/'study_receipt.json',
            ROOT/'tools/stanford-sample-check/inspect_archive.py')},
        'caution':'Whole CSV decompressed in memory; final fields checked for presence only; clinical validity not assessed.'}
    save('development_linkage.json',{'records':dev_rows})
    save('final_linkage_presence_only.json',{'records':final_rows})
    save('linkage_summary.json',result)
    print(json.dumps({k:result[k] for k in ('verified','development','final_reserved','training_or_evaluation_run')},indent=2))

if __name__ == '__main__': main()
