"""Read-only feasibility audit; emit aggregates, never person-level records."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import io
import json
import pickle
import re
import sys

import numpy as np
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'outputs/gait-public-access-20261015'

def rows(path, start=2):
    workbook = load_workbook(path, read_only=True, data_only=True)
    result = list(workbook.active.iter_rows(min_row=start, values_only=True))
    workbook.close()
    return result

def numeric(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

def hist(values):
    return dict(sorted(Counter(str(float(x)) for x in values if numeric(x)).items(), key=lambda item: float(item[0])))

def subject(x):
    return isinstance(x, str) and re.fullmatch(r'Subject\s+\d+', x.strip()) is not None

class ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported global: {module}.{name}')
        return allowed[module, name]

def care():
    path = DATA / 'CARE-PD/3DGait.pkl'
    blob = path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == 'b5349f34ace0c3bd50d3dcf2ceed243ce6b2b7987b4e5caa8a3ccad7d527e550'
    data = ArrayOnlyUnpickler(io.BytesIO(blob)).load()
    assert isinstance(data, dict)
    walks, fields, scores, person_classes, shapes, fps = [], Counter(), Counter(), defaultdict(set), Counter(), Counter()
    missing, invalid, nonfinite = 0, 0, 0
    durations = []
    for person, recordings in data.items():
        assert isinstance(recordings, dict)
        for walk in recordings.values():
            assert isinstance(walk, dict)
            fields.update(walk.keys())
            label = walk.get('UPDRS_GAIT')
            if label is None:
                missing += 1
            elif not numeric(label) or int(label) != label or not 0 <= label <= 3:
                invalid += 1
            else:
                scores[str(int(label))] += 1
                person_classes[person].add(int(label))
            for field in ('pose', 'trans'):
                arr = np.asarray(walk.get(field))
                shapes[f'{field}:{arr.shape[1:]}'] += 1
                if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
                    nonfinite += 1
            rate = walk.get('fps')
            fps[str(rate)] += 1
            if numeric(rate) and rate > 0:
                durations.append(len(walk['pose']) / float(rate))
            walks.append(walk)
    person_counts = {str(c): sum(c in cs for cs in person_classes.values()) for c in range(4)}
    return {'people': len(data), 'walks': len(walks), 'fields': dict(fields), 'label_walk_counts': dict(scores),
            'label_person_counts_membership': person_counts, 'people_with_multiple_scores': sum(len(x)>1 for x in person_classes.values()),
            'people_with_labels': len(person_classes), 'missing_labels': missing, 'invalid_labels': invalid,
            'array_shapes_without_time': dict(shapes), 'nonfinite_arrays': nonfinite, 'fps': dict(fps),
            'duration_seconds_range': [min(durations),max(durations)] if durations else None,
            'input_note': 'SMPL pose parameters and translations; not directly existing OpenPose/MediaPipe joint inputs.'}

def older():
    base = DATA / 'OlderAdults44'
    demo = {r[0].strip(): r for r in rows(base/'01_demography_sppb_mmse.xlsx') if subject(r[0])}
    parameters = defaultdict(list)
    current = None
    invalid_input = 0
    for row in rows(base/'05_gait_parameters.xlsx'):
        if subject(row[0]):
            current = row[0].strip()
        if current in demo and numeric(row[1]):
            parameters[current].append(row)
            invalid_input += not all(numeric(x) for x in row[2:])
    groups = defaultdict(dict)
    ratings = rows(base/'03_poma_test.xlsx',3)
    duplicates = 0
    for row in ratings:
        if subject(row[0]) and row[1] is not None and numeric(row[2]):
            key=(str(row[1]),int(row[2]))
            pid=row[0].strip()
            duplicates += pid in groups[key]
            groups[key][pid]=row
    group_stats={}
    for key, entries in groups.items():
        valid={p:r for p,r in entries.items() if numeric(r[13])}
        full={p:r for p,r in valid.items() if all(numeric(x) for x in r[3:13])}
        group_stats[f'{key[0]} week {key[1]}']={
            'people_rows':len(entries),'valid_totals':len(valid),'missing_totals':len(entries)-len(valid),
            'complete_item_rows':len(full),'total_score_counts':hist([r[13] for r in valid.values()]),
            'sum_mismatch':sum(abs(sum(r[3:13])-r[13])>1e-8 for r in full.values()),
            'linked_to_parameters_people':len(set(valid)&set(parameters)),
            'item_value_counts':[hist([r[j] for r in entries.values()]) for j in range(3,13)]}
    agreements=[]
    keys=list(groups)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            if a[0]==b[0] or (a[1]==1 and b[1]==1):
                common=[p for p in groups[a].keys()&groups[b].keys() if numeric(groups[a][p][13]) and numeric(groups[b][p][13])]
                diffs=[abs(groups[a][p][13]-groups[b][p][13]) for p in common]
                agreements.append({'comparison':f'{a} vs {b}','paired_people':len(common),
                                   'exact_agreement_people':sum(d==0 for d in diffs),'mean_absolute_difference':sum(diffs)/len(diffs) if diffs else None})
    sppb=[r[7] for r in demo.values()]
    return {'demographic_people':len(demo),'sppb_valid':sum(numeric(x) for x in sppb),'sppb_counts':hist(sppb),
            'parameter_people':len(parameters),'parameter_walks':sum(map(len,parameters.values())),
            'walks_per_person':dict(Counter(map(len,parameters.values()))),'parameter_rows_with_nonfinite_or_missing':int(invalid_input),
            'sppb_parameter_people_overlap':len(set(demo)&set(parameters)), 'poma_groups':group_stats,'poma_duplicate_person_rater_week':duplicates,
            'poma_descriptive_agreement':agreements,'local_c3d_files':len(list(base.glob('*.c3d'))),
            'local_coordinate_csv_files':len(list(base.glob('*.csv'))), 'local_rgb_files':len(list(base.glob('*.mp4')))}

def toronto():
    data=[r for r in rows(DATA/'Toronto/Table 1.xlsx') if numeric(r[0]) and len(r)>9 and numeric(r[1])]
    return {'participant_rows':len(data),'unique_participant_ids':len(set(r[0] for r in data)),
            'poma_gait_valid':sum(numeric(r[7]) for r in data),'poma_gait_counts':hist([r[7] for r in data]),
            'tug_valid':sum(numeric(r[9]) for r in data), 'note':'Clinical sheet only; input linkage is not established by this count.'}

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    result={'CARE_PD_3DGait':care(),'OlderAdults44':older(),'Toronto':toronto()}
    hashes=json.loads((HERE/'source_hashes.json').read_text(encoding='utf-8'))
    result['source_hashes_unchanged']=all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value for path,value in hashes.items())
    assert result['source_hashes_unchanged']
    encoded=json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
    (HERE/'audit_result.json').write_text(encoded,encoding='utf-8')
    for group in result['OlderAdults44']['poma_groups'].values():
        group.pop('item_value_counts')
    print(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
