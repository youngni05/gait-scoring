"""User-approved exploratory change analysis using frozen existing predictions."""
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from audit import HERE, array, digest
from change_metrics import summarize


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    approval = read(HERE/'analysis-approval.json')
    assert approval['approved'] and approval['fixed_model']=='bagged'
    proof = read(HERE/'verification.json')
    source = Path(proof['prediction_path'])
    assert digest(source)==proof['prediction_sha256']
    pairs = read(HERE/'prepared-pairs.json')['pairs']
    visits = read(HERE/'development-visits.json')
    visit_lookup = {(r['person_id'],r['exam_id']):r for r in visits}
    with zipfile.ZipFile(source) as z:
        people,videos,folds = array(z,'person_id'),array(z,'video_id'),array(z,'fold')
        truth,prediction = array(z,'y'),array(z,'bagged')
    lookup = {v:dict(person_id=p,fold=f,truth=truth[2*i:2*i+2],prediction=prediction[2*i:2*i+2])
              for i,(p,v,f) in enumerate(zip(people,videos,folds))}
    assert len(lookup)==1061 and len(prediction)==2122
    rows = []
    for pair in pairs:
        a,b = [visit_lookup[pair['person_id'],pair[k]] for k in ('lower_age_exam','higher_age_exam')]
        assert len(a['age_values'])==len(b['age_values'])==1 and a['age_values'][0]<b['age_values'][0]
        assert len(a['video_ids'])==len(b['video_ids'])==1
        first,last = lookup[a['video_ids'][0]],lookup[b['video_ids'][0]]
        assert first['person_id']==last['person_id']==pair['person_id']
        assert first['fold']==last['fold']==pair['fold']
        assert first['truth']==a['gdi_lr'] and last['truth']==b['gdi_lr']
        rows.append(dict(person_id=pair['person_id'],fold=pair['fold'],
                         lower_age_exam=a['exam_id'],higher_age_exam=b['exam_id'],
                         true_delta_lr=[y-x for x,y in zip(first['truth'],last['truth'])],
                         predicted_delta_lr=[y-x for x,y in zip(first['prediction'],last['prediction'])]))
    assert len(rows)==581 and len({r['person_id'] for r in rows})==229
    def metrics(rs):
        return dict(fixed_model=summarize(rs),no_change=summarize(rs,no_change=True))
    by_person = defaultdict(list)
    for r in rows:
        by_person[r['person_id']].append(r)
    person_results=[]
    for person,rs in sorted(by_person.items()):
        m=metrics(rs)
        person_results.append(dict(person_id=person,pairs=len(rs),**m,
                                   model_minus_no_change_mae=m['fixed_model']['person_mean_absolute_delta_error']-m['no_change']['person_mean_absolute_delta_error']))
    overall=metrics(rows)
    sides={}
    for s,name in enumerate(('left','right')):
        # Duplicate one side into both positions to reuse the exact same person weighting.
        rs=[dict(r,true_delta_lr=[r['true_delta_lr'][s]]*2,predicted_delta_lr=[r['predicted_delta_lr'][s]]*2) for r in rows]
        sides[name]=metrics(rs)
        for m in sides[name].values():
            m['changed_side_pair_entries']//=2
    report=dict(analysis='fixed_model_same_person_GDI_change_exploratory',
                people=229,pairs=581,side_pair_entries=1162,
                fixed_model='existing bagged CNN/STT combination; no refit or selection',
                overall=overall,sides=sides,
                person_mae_comparison=dict(better=sum(r['model_minus_no_change_mae']<0 for r in person_results),
                                           worse=sum(r['model_minus_no_change_mae']>0 for r in person_results),
                                           tied=sum(r['model_minus_no_change_mae']==0 for r in person_results)),
                model_minus_no_change_mae=mean(r['model_minus_no_change_mae'] for r in person_results),
                order_basis='reported age; exact dates and elapsed days unavailable',
                direction_rule='per-person fraction among nonzero true side changes; predicted ties count 0.5',
                actual_baseline_GDI_used_as_model_input=False,
                training_or_model_inference_performed=False,clinical_improvement_validated=False,
                independent_final_validation=False,
                limitations=['Reused development cohort; no new independent validation',
                             'Reference GDI includes visit measurement variability',
                             'No stable-state retest, MDC or clinical change threshold estimated',
                             'All within-person pairs retained but person weights equal; pairs are dependent',
                             'No causal treatment effect or change per day inferred'],
                source_sha256={p.name:digest(p) for p in (HERE/'prepared-pairs.json',HERE/'development-visits.json',HERE/'analysis-approval.json',HERE/'change_metrics.py',Path(__file__))},
                frozen_prediction_path=str(source),frozen_prediction_sha256=digest(source))
    for filename,value in [('change-results.json',report),('change-pairs.json',rows),('change-people.json',person_results)]:
        (HERE/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
