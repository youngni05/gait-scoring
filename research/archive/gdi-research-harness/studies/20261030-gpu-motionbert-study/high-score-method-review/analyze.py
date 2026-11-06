"""Bounded development-cohort linkage and batch diagnostics; no model training."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,io,json,hashlib,importlib.util,itertools
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent;ROOT=S.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
source=S/'cnn-multiwindow695/bundle/data.npz'
with np.load(source,allow_pickle=False) as z:
    d={k:z[k].copy() for k in ['person_id','video_id','y','fold','window_record']}
assert len(set(d['person_id']))==692 and len(d['y'])==1061
spec=importlib.util.spec_from_file_location('archive',ROOT/'tools/stanford-sample-check/inspect_archive.py')
arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)
index=load(ROOT/'outputs/stanford-sample-check-20261016/archive-index.json')
members={x['name']:x for x in index['non_json_members']};member_hashes={}
def reader(name):
    payload=arc.read_member(members[name]);member_hashes[name]=hashlib.sha256(payload).hexdigest()
    return csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
videos=set(d['video_id']);mapping=defaultdict(set)
for row in reader('annotations/video_list.csv'):
    if row['Video_File'] in videos:mapping[row['Video_File']].add(row['Exam_ID'])
assert set(mapping)==videos and all(len(v)==1 for v in mapping.values())
exams=np.array([next(iter(mapping[v])) for v in d['video_id']])
keys=set(zip(d['person_id'],exams));ages=defaultdict(set);surgery=defaultdict(set)
clinical=reader('annotations/alldata.csv');clinical_fields=clinical.fieldnames
for row in clinical:
    key=(row['Patient_ID'],row['examid'])
    if key not in keys:continue
    # Select only age after checking approved development identities.
    if row['age'].strip():ages[key].add(float(row['age']))
surgical=reader('annotations/surgInfo.csv');surgical_fields=surgical.fieldnames
for row in surgical:
    key=(row['Patient_ID'],row['examid'])
    if key not in keys:continue
    values=[float(row[k]) for k in surgical_fields if k.startswith('upcoming_') and row[k].strip()]
    assert all(np.isfinite(v) for v in values)
    surgery[key].add(any(v>0 for v in values))
age_row_max_difference=max(max(v)-min(v) for v in ages.values())
# Left/right metadata can differ only in decimal serialization. This check is
# for descriptive age spans; GDI identity below still requires exact equality.
assert all(all(np.isfinite(x) for x in v) and max(v)-min(v)<1e-10 for v in ages.values())
age_precision_duplicates=sum(len(v)>1 for v in ages.values())
ages={k:float(np.mean(list(v))) for k,v in ages.items()}
people={p:np.flatnonzero(d['person_id']==p) for p in np.unique(d['person_id'])}
groups=defaultdict(list)
for i,key in enumerate(zip(d['person_id'],exams)):groups[key].append(i)
pair_differences=[];person_pair_mean=[];age_spans=[]
for p,ix in people.items():
    assert len(set(d['fold'][ix]))==1
    if len(ix)<2:continue
    differences=[np.abs(d['y'][a]-d['y'][b]).mean() for a,b in itertools.combinations(ix,2)]
    pair_differences.extend(differences);person_pair_mean.append(float(np.mean(differences)))
    aa=[ages[(p,exams[i])] for i in ix if (p,exams[i]) in ages]
    if len(aa)==len(ix):age_spans.append(float(np.ptp(aa)))
# Independent existing development linkage joins (new518 subset only).
dev={r['video_id']:r for r in load(ROOT/'gdi-research-harness/studies/20261027-gdi-linkage/development_linkage.json')['records']}
verified=0
for i,v in enumerate(d['video_id']):
    if v not in dev:continue
    r=dev[v];assert r['reserved_role']=='development_candidate'
    assert r['patient_id']==d['person_id'][i] and r['exam_id']==exams[i]
    np.testing.assert_array_equal(d['y'][i],r['GDI_LR']);verified+=1
assert verified==859
batch_reports=[]
for f in range(5):
    for j in range(5):
        if j==f:continue
        tr=np.flatnonzero((d['fold']!=f)&(d['fold']!=j))
        owners=d['window_record'][np.isin(d['window_record'],tr)]
        sample_rows=np.repeat(owners,2);side=np.tile([0,1],len(owners))
        yy=d['y'][sample_rows,side];pp=d['person_id'][sample_rows]
        # Descriptive shuffled batches; no optimizer, not a reproduction of run seed.
        order=np.random.default_rng(20261028+100*f+j).permutation(len(yy))
        bs=[order[k:k+32] for k in range(0,len(order),32) if len(order[k:k+32])==32]
        batch_reports.append(dict(outer=f,inner=j,batches=len(bs),
            mean_unique_people=float(np.mean([len(set(pp[b])) for b in bs])),
            mean_unique_scores=float(np.mean([len(set(yy[b])) for b in bs])),
            fraction_repeated_person=float(np.mean([len(set(pp[b]))<32 for b in bs])),
            fraction_tied_scores=float(np.mean([len(set(yy[b]))<32 for b in bs])),
            fraction_no_target90=float(np.mean([not (yy[b]>=90).any() for b in bs]))))
def stats(x):
    a=np.asarray(x);return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()))
report=dict(passed=True,people=692,records=1061,windows=len(d['window_record']),
    independent_person_exam_keys=len(keys),record_count_histogram=dict(Counter(len(x) for x in people.values())),
    people_multiple_records=sum(len(x)>1 for x in people.values()),
    people_multiple_exams=sum(len(set(exams[x]))>1 for x in people.values()),
    same_exam_multiple_videos=sum(len(x)>1 for x in groups.values()),
    independently_checked_linkage_records=verified,all_people_single_fold=True,
    same_person_cross_exam_pairs=len(pair_differences),
    pair_mean_absolute_GDI_difference=stats(pair_differences),
    equal_person_mean_absolute_cross_exam_difference=stats(person_pair_mean),
    age_available_exam_keys=len(ages),age_row_max_difference=age_row_max_difference,
    age_precision_duplicates=age_precision_duplicates,repeated_person_age_span_years=stats(age_spans),
    repeated_people_age_span_ge_half_year=sum(x>=.5 for x in age_spans),
    exams_with_any_upcoming_surgery_flag=sum(True in v for v in surgery.values()),
    clinical_metadata_fields=clinical_fields,surgery_metadata_fields=surgical_fields,
    limitation='Cross-exam descriptive change, NOT test-retest noise, MAE floor, or attribution of prediction bias. No dates or confirmed intervening-treatment history in inspected tables. Upcoming flags do not establish completed surgery.',
    batch_diagnostics=batch_reports,data_sha256=sha(source),archive_member_sha256=member_hashes,
    source_sha256=sha(__file__),actual_optimizer_steps=0,protected_numeric_outcomes_selected=False,
    diagnostic_correction='Initial exact age uniqueness assertion failed for one exam: 5.9945205479452 versus 5.99452054794521. Accepted only finite age spreads below 1e-10 years for descriptive metadata; exact GDI linkage assertions unchanged.')
report['batch_mean_over_20_inner_splits']={k:float(np.mean([r[k] for r in batch_reports])) for k in batch_reports[0] if k not in ['outer','inner','batches']}
(H/'linkage-and-batch.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['batch_diagnostics','clinical_metadata_fields','surgery_metadata_fields','archive_member_sha256']},ensure_ascii=False))
print('BATCH_MEANS',json.dumps({k:float(np.mean([r[k] for r in batch_reports])) for k in batch_reports[0] if k not in ['outer','inner','batches']}))
