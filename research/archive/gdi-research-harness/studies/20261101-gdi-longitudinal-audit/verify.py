"""Independent recount and frozen prediction linkage check, without scoring changes."""
import itertools
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from audit import HERE, ROOT, array, digest


def main():
    read = lambda p: Path(p).read_text(encoding='utf-8-sig')
    summary = json.loads(read(HERE/'summary.json'))
    visits = json.loads(read(HERE/'development-visits.json'))
    pairs = json.loads(read(HERE/'candidate-pairs.json'))
    by_person = defaultdict(list)
    for r in visits:
        by_person[r['person_id']].append(r)
    histogram = Counter(len(v) for v in by_person.values())
    assert sum(histogram.values()) == 692
    assert sum(n*c for n,c in histogram.items()) == 1061
    assert sum(c for n,c in histogram.items() if n>1) == 229
    assert sum(math.comb(n,2)*c for n,c in histogram.items()) == 581
    expected = {(p, *sorted((a['exam_id'], b['exam_id']))) for p,rows in by_person.items()
                for a,b in itertools.combinations(rows,2)}
    assert expected == {(p['person_id'],p['exam_a'],p['exam_b']) for p in pairs}
    assert len(expected) == len(pairs) == summary['unordered_exam_pairs']
    assert not any(r['measurement_date'] for r in visits)
    assert len({r['person_id'] for r in visits if len(r['age_values'])>1}) == 1
    assert all(len({r['fold'] for r in rows})==1 for rows in by_person.values())
    report_path = ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study/inner-bagging692/analysis/comparison.json'
    report = json.loads(read(report_path))
    prediction_path = Path(report['predictions_path'])
    assert digest(prediction_path) == report['predictions_sha256']
    with zipfile.ZipFile(prediction_path) as z:
        ids, vids, folds = array(z,'person_id'),array(z,'video_id'),array(z,'fold')
        labels = array(z,'y')
        assert 'bagged.npy' in z.namelist() and 'CNN_bag.npy' in z.namelist()
    lookup = {v:(p,f,labels[2*i:2*i+2]) for i,(p,v,f) in enumerate(zip(ids,vids,folds))}
    assert len(lookup)==1061
    for r in visits:
        for video in r['video_ids']:
            assert lookup[video] == (r['person_id'],r['fold'],r['gdi_lr'])
    result = dict(passed=True, unique_people=692, unique_exams=1061,
                  repeated_people=229, unordered_pairs=581,
                  same_person_fold_consistent=True, existing_predictions_linked=True,
                  prediction_path=str(prediction_path), prediction_sha256=digest(prediction_path),
                  prediction_report_sha256=digest(report_path),
                  source_artifacts={p.name:digest(p) for p in [HERE/'audit.py',HERE/'summary.json',HERE/'development-visits.json',HERE/'candidate-pairs.json']},
                  verification_script_sha256=digest(Path(__file__)),
                  training_provenance_reaudited=False, change_prediction_metrics_computed=False,
                  note='Metadata and existing prediction identity linkage only; no new performance or clinical-change claim.')
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
