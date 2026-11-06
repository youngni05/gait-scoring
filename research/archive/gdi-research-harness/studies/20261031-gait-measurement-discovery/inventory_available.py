"""Inventory existing measurements and coverage; no feature ranking or model fitting."""
import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
SOURCE_HASHES = {}


def tracked(path):
    SOURCE_HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(tracked(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with tracked(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def assigned_list(path, name):
    tree = ast.parse(tracked(path).read_text(encoding='utf-8-sig'))
    node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    return ast.literal_eval(node.value)


def coverage(records, names):
    result = {}
    for name in names:
        values = []
        for row in records:
            value = row[name]
            if value.strip() and math.isfinite(float(value)):
                values.append(float(value))
        result[name] = {'available_people': len(values), 'missing_people': len(records) - len(values), 'distinct_values': len(set(values))}
    return result


def main():
    opencap = read_json(STUDIES / '20261030-opencap415-tug-timeseries/all3s/protocol.json')
    assert opencap['people'] == 298 and len(opencap['channels']) == 19
    patterns = assigned_list(STUDIES / '20261031-opencap-walk-pattern/common.py', 'PATTERN_NAMES')
    quality = read_json(STUDIES / '20261031-opencap-walk-quality/preparation.json')
    mobilise_code = STUDIES / '20261031-mobilised-function-models/common.py'
    speed = assigned_list(mobilise_code, 'SPEED')
    other = assigned_list(mobilise_code, 'PATTERN')
    names = speed + other
    tug = read_csv(Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-mobilised-function-models/TUG-cohort.csv'))
    assert len(tug) == len({row['participantid'] for row in tug}) == 1980
    first = read_csv(STUDIES / '20261031-mobilised-uncertainty-review/first-eligible-per-target.csv')
    counts = Counter(row['audit_target'] for row in first)
    sppb = [row for row in first if row['audit_target'] == 'SPPB']
    assert len(sppb) == len({row['participantid'] for row in sppb}) == 2232
    label_names = ['total_sppb_score', '4mwt_speed_score', 'balance_score', '5crt_score']
    result = {
        'status': 'Existing measurement inventory and missingness only; no usefulness ranking or new learning experiment.',
        'research_question': 'Which measurable gait quantities are useful for predicting SPPB and TUG?',
        'opencap_3s_TUG': {'people': 298, 'channels': opencap['channels'], 'scope': opencap['scope'], 'already_used_summaries': ['mean', 'standard deviation', 'mean absolute adjacent change'], 'limits': 'Angles are outputs of OpenCap inverse kinematics. Summary coverage does not establish measurement accuracy.'},
        'opencap_existing_6s_pattern_candidates': {'names': patterns, 'TUG_people': quality['cohorts']['TUG_common6']['people'], 'TUG_existing_stability_condition_met_people': quality['cohorts']['TUG_common6']['stable'], 'limits': 'Knee-flexion peaks define cycle proxies; not verified foot-contact events. Stability is algorithmic consistency, not ground-truth accuracy. Different people and duration from the 298-person study.'},
        'mobilise_existing_TUG': {'people': len(tug), 'speed_names': speed, 'other_names': other, 'coverage': coverage(tug, names), 'limits': 'Provider weekly wearable summaries, not single short video; existing cleaned first eligible TUG cohort.'},
        'mobilise_existing_SPPB_candidates': {'people': len(sppb), 'label_coverage': coverage(sppb, label_names), 'limits': 'Existing first-eligible linked candidates, not a newly cleaned training cohort; no new exclusions or imputation. Full DMO feature linkage not rechecked in this inventory.'},
        'existing_first_eligible_target_counts': dict(counts),
        'source_sha256': SOURCE_HASHES,
    }
    for path, digest in SOURCE_HASHES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    (HERE / 'measurement-inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': True, 'opencap_channels': len(opencap['channels']), 'existing_pattern_candidates': len(patterns), 'mobilise_measurements': len(names), 'TUG_people': len(tug), 'SPPB_linked_candidates': len(sppb), 'SPPB_label_coverage': result['mobilise_existing_SPPB_candidates']['label_coverage']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
