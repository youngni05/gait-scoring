"""Prepare age-ordered visit references only. Does not evaluate model performance."""
import json
from collections import Counter
from pathlib import Path
from audit import HERE, digest


def main():
    visits_path, pairs_path = HERE/'development-visits.json', HERE/'candidate-pairs.json'
    visits = json.loads(visits_path.read_text(encoding='utf-8'))
    candidates = json.loads(pairs_path.read_text(encoding='utf-8'))
    lookup = {(r['person_id'],r['exam_id']):r for r in visits}
    counts = Counter(r['person_id'] for r in candidates)
    ready, excluded = [], []
    for p in candidates:
        a = lookup[p['person_id'],p['exam_a']]
        b = lookup[p['person_id'],p['exam_b']]
        reason = ('missing_or_conflicting_age' if len(a['age_values'])!=1 or len(b['age_values'])!=1
                  else 'equal_reported_age' if a['age_values']==b['age_values'] else None)
        if reason:
            excluded.append(dict(p,reason=reason))
            continue
        assert a['fold']==b['fold']
        first, last = sorted((a,b),key=lambda r:r['age_values'][0])
        ready.append(dict(person_id=p['person_id'],fold=a['fold'],
                          lower_age_exam=first['exam_id'],higher_age_exam=last['exam_id'],
                          lower_age_video_ids=first['video_ids'],higher_age_video_ids=last['video_ids'],
                          reported_age_increase=last['age_values'][0]-first['age_values'][0],
                          order_basis='reported age only; no explicit dates',
                          measurement_dates_available=False,
                          candidate_pairs_for_person=counts[p['person_id']]))
    assert len(ready)+len(excluded)==len(candidates)
    obj = dict(schema='age_ordered_gdi_pair_references_v1',
               pairs=ready,excluded=excluded,
               clinical_change_labels_available=False,
               elapsed_days_available=False,
               prediction_metrics_computed=False,
               source_sha256={p.name:digest(p) for p in (visits_path,pairs_path,Path(__file__))})
    out = HERE/'prepared-pairs.json'
    out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(ready_pairs=len(ready),ready_people=len({r['person_id'] for r in ready}),
                          excluded_pairs=len(excluded),output_sha256=digest(out),
                          prediction_metrics_computed=False),indent=2))


if __name__=='__main__':
    main()
