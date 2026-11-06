"""Audit existing development visits; no model fitting or protected outcome parsing."""
import ast
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import math
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = ROOT / 'gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/data.npz'
INDEX = ROOT / 'outputs/stanford-sample-check-20261016/archive-index.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(archive, name):
    """Read only small, non-object C-order NPY metadata with the standard library."""
    b = archive.read(name + '.npy')
    assert b[:6] == b'\x93NUMPY'
    offset = 10 if b[6] == 1 else 12
    length = struct.unpack('<H' if offset == 10 else '<I', b[8:offset])[0]
    h = ast.literal_eval(b[offset:offset + length].decode('latin1'))
    assert not h['fortran_order']
    raw = b[offset + length:]
    dtype = h['descr']
    count = math.prod(h['shape'])
    if dtype.startswith('<U'):
        size = int(dtype[2:]) * 4
        assert len(raw) == count * size
        return [raw[i:i + size].decode('utf-32-le').rstrip('\0') for i in range(0, len(raw), size)]
    assert dtype in ('<i8', '<f8')
    assert len(raw) == count * 8
    return list(struct.unpack('<' + str(count) + ('q' if dtype == '<i8' else 'd'), raw))


def main():
    spec = importlib.util.spec_from_file_location('bounded_archive', ROOT / 'tools/stanford-sample-check/inspect_archive.py')
    arc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arc)
    before = (arc.ARCHIVE.stat().st_size, arc.ARCHIVE.stat().st_mtime_ns)
    with zipfile.ZipFile(BUNDLE) as z:
        people, videos = array(z, 'person_id'), array(z, 'video_id')
        labels, folds = array(z, 'y'), array(z, 'fold')
    assert len(people) == 1061 and len(set(people)) == 692
    assert len(set(videos)) == len(videos) and len(labels) == 2 * len(videos)
    dev = set(people)
    index = json.loads(INDEX.read_text(encoding='utf-8'))
    tables, schemas, sources = {}, {}, {}
    for member in index['non_json_members']:
        if not member['name'].endswith('.csv'):
            continue
        payload = arc.read_member(member)
        reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
        name = Path(member['name']).name
        schemas[name] = reader.fieldnames
        # GDI values for protected/non-development people are never converted or saved.
        tables[name] = list(reader)
        sources[name] = {'sha256': hashlib.sha256(payload).hexdigest(), 'rows': len(tables[name])}
    date_columns = {name: [c for c in cols if any(w in c.lower() for w in ('date', 'time', 'visit'))]
                    for name, cols in schemas.items()}
    exam_people, ages, gdi, video_exams = defaultdict(set), defaultdict(set), defaultdict(list), defaultdict(set)
    for r in tables['alldata.csv']:
        exam_people[r['examid']].add(r['Patient_ID'])
        if r['Patient_ID'] in dev:
            value = float(r['age']) if r['age'].strip() else float('nan')
            if math.isfinite(value):
                ages[(r['Patient_ID'], r['examid'])].add(value)
    for r in tables['gdi.csv']:
        if r['Patient_ID'] in dev:
            gdi[(r['Patient_ID'], r['examid'], r['side'])].append(float(r['GDI']))
    for r in tables['video_list.csv']:
        video_exams[Path(r['Video_File']).stem].add(r['Exam_ID'])
    records, visits = [], {}
    for i, (person, video, fold) in enumerate(zip(people, videos, folds)):
        exams = video_exams[video]
        assert len(exams) == 1, (video, 'ambiguous exam')
        exam = next(iter(exams))
        assert exam_people[exam] == {person}, 'ambiguous person'
        truth = []
        for side in ('L', 'R'):
            values = gdi[(person, exam, side)]
            assert len(values) == 1 and math.isfinite(values[0])
            truth.append(values[0])
        assert max(abs(a - b) for a, b in zip(truth, labels[2*i:2*i+2])) < 1e-10
        age_values = sorted(ages[(person, exam)])
        row = dict(person_id=person, exam_id=exam, video_id=video, fold=fold,
                   age_values=age_values, gdi_lr=truth, measurement_date=None)
        records.append(row)
        key = (person, exam)
        if key not in visits:
            visits[key] = dict(row, video_ids=[])
            del visits[key]['video_id']
        assert visits[key]['gdi_lr'] == truth and visits[key]['fold'] == fold
        visits[key]['video_ids'].append(video)
    by_person = defaultdict(list)
    for row in visits.values():
        by_person[row['person_id']].append(row)
    pairs = []
    for person, rows in sorted(by_person.items()):
        for a, b in itertools.combinations(sorted(rows, key=lambda r: r['exam_id']), 2):
            # Lexical ID order is a reproducible enumeration, NOT time order.
            aa, bb = a['age_values'], b['age_values']
            pairs.append(dict(person_id=person, exam_a=a['exam_id'], exam_b=b['exam_id'],
                              age_difference=bb[0]-aa[0] if len(aa)==len(bb)==1 else None,
                              chronological_order_confirmed=False,
                              clinical_improvement_label_available=False))
    summary = dict(
        schema='gdi_longitudinal_metadata_audit_v1',
        development_records=len(records), development_people=len(dev),
        unique_person_exams=len(visits),
        people_multiple_exams=sum(len(r)>1 for r in by_person.values()),
        exams_per_person_histogram=dict(sorted(Counter(len(r) for r in by_person.values()).items())),
        unordered_exam_pairs=len(pairs),
        pairs_with_distinct_reported_age=sum(p['age_difference'] not in (None, 0) for p in pairs),
        people_with_distinct_reported_age=len({p['person_id'] for p in pairs if p['age_difference'] not in (None,0)}),
        same_exam_multiple_videos=sum(len(r['video_ids'])>1 for r in visits.values()),
        exams_with_missing_age=sum(not r['age_values'] for r in visits.values()),
        exams_with_conflicting_age=sum(len(r['age_values'])>1 for r in visits.values()),
        explicit_date_columns=date_columns, explicit_dated_pairs=0,
        gdi_bundle_vs_source_max_tolerance=1e-10, all_development_labels_match_source=True,
        source_schemas=schemas, source_members=sources,
        source_bundle_sha256=digest(BUNDLE), index_sha256=digest(INDEX),
        audit_script_sha256=digest(Path(__file__)),
        protected_numeric_outcomes_parsed_or_saved=False,
        whole_annotation_csv_decompressed_in_memory=True,
        training_or_model_evaluation_performed=False,
        caveat='Different exam IDs and reported ages support repeated examinations; exact dates, elapsed days, intervention and clinical change are unconfirmed. Unordered pairs are not independent participants. ZIP timestamps are not measurement dates.',
    )
    assert before == (arc.ARCHIVE.stat().st_size, arc.ARCHIVE.stat().st_mtime_ns)
    for name, value in [('summary.json', summary), ('development-visits.json', list(visits.values())), ('candidate-pairs.json', pairs)]:
        (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_schemas','source_members')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
