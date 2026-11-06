"""Audit development-only clinical targets; do not optimize or score holdout data."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HARNESS = ROOT / 'gdi-research-harness'
BASE = HARNESS / 'studies/20261030-gpu-motionbert-study/cnn-multiwindow695/bundle'
TABLE = ROOT / 'outputs/stanford-sample-check-20261016/alldata.csv'
TARGETS = ['KneeFlex_meanStance', 'KneeFlex_maxExtension']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def audited_data():
    assert sha(BASE / 'data.npz') == read(BASE / 'manifest.json')['data.npz']
    with np.load(BASE / 'data.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ['person_id', 'video_id', 'y', 'fold']}
    videos = set(data['video_id'])
    exam_by_video = defaultdict(set)
    with (TABLE.parent / 'video_list.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            video = Path(row['Video_File']).stem
            if video in videos:
                exam_by_video[video].add(row['Exam_ID'])
    old = read(HARNESS / 'control/desktop-sessions/20261026T100042Z-9d41cd/native124_tasks.json')['tasks']
    new = read(HARNESS / 'studies/20261027-gdi-linkage/development_linkage.json')['records']
    lookup = {}
    for row in old + new:
        if row['video_id'] in videos:
            exams = exam_by_video[row['video_id']]
            assert len(exams) == 1
            exam = next(iter(exams))
            assert row.get('exam_id', exam) == exam
            value = (row['patient_id'], exam)
            if row['video_id'] in lookup:
                assert lookup[row['video_id']] == value
            lookup[row['video_id']] = value
    keys = set()
    for person, video in zip(data['person_id'], data['video_id']):
        assert video in lookup and lookup[video][0] == person
        keys.update((person, lookup[video][1], side) for side in ['L', 'R'])
    rows = defaultdict(list)
    with TABLE.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        assert set(TARGETS).issubset(reader.fieldnames)
        for row in reader:
            key = (row['Patient_ID'], row['examid'], row['side'])
            if key in keys:
                rows[key].append(row)
    aux = np.full((len(data['y']), 2, len(TARGETS)), np.nan)
    statuses = Counter()
    for i, (person, video) in enumerate(zip(data['person_id'], data['video_id'])):
        for side_index, side in enumerate(['L', 'R']):
            matches = rows[(person, lookup[video][1], side)]
            if len(matches) != 1:
                statuses['missing_row' if not matches else 'duplicate_rows'] += 1
                continue
            statuses['unique_row'] += 1
            for j, name in enumerate(TARGETS):
                try:
                    value = float(matches[0][name])
                except (ValueError, TypeError):
                    continue
                if math.isfinite(value):
                    aux[i, side_index, j] = value
    assert len(data['y']) == 1061 and len(set(data['person_id'])) == 692
    for person in set(data['person_id']):
        assert len(set(data['fold'][data['person_id'] == person])) == 1
    return data, aux, statuses


def main():
    data, aux, statuses = audited_data()
    targets = {}
    for j, name in enumerate(TARGETS):
        mask = np.isfinite(aux[:, :, j])
        values = aux[:, :, j][mask]
        targets[name] = dict(finite_record_sides=int(mask.sum()), missing_record_sides=int((~mask).sum()),
                             finite_records_both_sides=int(mask.all(1).sum()),
                             observed_min=float(values.min()) if len(values) else None,
                             observed_max=float(values.max()) if len(values) else None,
                             fold_finite_sides=[int(mask[data['fold'] == f].sum()) for f in range(5)])
    report = dict(stage='development_metadata_audit_only', people=692, records=1061,
                  record_sides=2122, target_columns=TARGETS, target_audit=targets,
                  clinical_row_status=dict(statuses), all_four_values_present_records=int(np.isfinite(aux).all((1, 2)).sum()),
                  data_sha256=sha(BASE / 'data.npz'), clinical_table_sha256=sha(TABLE),
                  optimizer_steps=0, reserved_holdout_values_accessed=False,
                  notes=['Only development person/exam/side keys are retained and converted to numeric targets.',
                         'Raw clinical column names are preserved. Units and phase definitions require source verification.',
                         'Clinical targets are not screen-space angles calculated from input video.',
                         'No clinical range clipping or label-based record exclusion is applied.'])
    (HERE / 'input-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
