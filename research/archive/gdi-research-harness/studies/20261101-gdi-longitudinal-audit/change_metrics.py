"""Person-balanced GDI score-change metrics.

User-approved real-data evaluation is in analyze_changes.py; this entry point
checks synthetic examples only. GDI changes are not clinical improvement labels.
"""
import math
from collections import defaultdict
from statistics import mean


def summarize(rows, no_change=False):
    """Average pairs within person first, then average people with equal weight.

    Each row contains person_id, true_delta_lr, predicted_delta_lr. Direction
    scores exclude exactly zero reference changes; prediction ties count as 0.5.
    Direction is averaged within each person over their changed side/pair entries.
    No clinical change threshold, chronological ordering or label is inferred.
    """
    if not rows:
        raise ValueError('No pairs')
    grouped = defaultdict(lambda: dict(errors=[], biases=[], directions=[]))
    for row in rows:
        actual = row['true_delta_lr']
        predicted = (0.0, 0.0) if no_change else row['predicted_delta_lr']
        if len(actual) != 2 or len(predicted) != 2:
            raise ValueError('Exactly two sides are required')
        if not all(math.isfinite(x) for x in (*actual, *predicted)):
            raise ValueError('Non-finite score change')
        g = grouped[row['person_id']]
        errors = [p-t for t,p in zip(actual,predicted)]
        g['errors'].append(mean(abs(e) for e in errors))
        g['biases'].append(mean(errors))
        for t,p in zip(actual,predicted):
            if t != 0:
                g['directions'].append(0.5 if p == 0 else float((t>0)==(p>0)))
    direction_people = [mean(g['directions']) for g in grouped.values() if g['directions']]
    return dict(people=len(grouped), pairs=len(rows),
                person_mean_absolute_delta_error=mean(mean(g['errors']) for g in grouped.values()),
                person_mean_signed_delta_error=mean(mean(g['biases']) for g in grouped.values()),
                person_mean_direction_agreement=mean(direction_people) if direction_people else None,
                direction_people=len(direction_people),
                changed_side_pair_entries=sum(len(g['directions']) for g in grouped.values()),
                independent_final_validation=False, clinical_improvement_measured=False)


def synthetic_check():
    # P1 has two pairs with errors 1 and 3; P2 has one with error 4.
    # Equal person weight gives (2+4)/2=3, rather than pooled (1+3+4)/3.
    rows = [dict(person_id='P1',true_delta_lr=(2,-2),predicted_delta_lr=(1,-1)),
            dict(person_id='P1',true_delta_lr=(4,-4),predicted_delta_lr=(1,-1)),
            dict(person_id='P2',true_delta_lr=(6,-6),predicted_delta_lr=(2,-2))]
    model, baseline = summarize(rows), summarize(rows,no_change=True)
    assert model['person_mean_absolute_delta_error'] == 3
    assert baseline['person_mean_absolute_delta_error'] == 4.5
    assert model['person_mean_direction_agreement'] == 1
    assert baseline['person_mean_direction_agreement'] == 0.5
    assert model['person_mean_signed_delta_error'] == 0
    zero = [dict(person_id='P0',true_delta_lr=(0,0),predicted_delta_lr=(1,-1))]
    assert summarize(zero)['person_mean_direction_agreement'] is None
    assert summarize(zero)['direction_people'] == 0
    reverse = [dict(person_id='R',true_delta_lr=(1,-1),predicted_delta_lr=(-1,1))]
    assert summarize(reverse)['person_mean_direction_agreement'] == 0
    tied = [dict(person_id='T',true_delta_lr=(1,0),predicted_delta_lr=(0,3))]
    assert summarize(tied)['person_mean_direction_agreement'] == 0.5
    for invalid in [[],[dict(person_id='X',true_delta_lr=(math.nan,1),predicted_delta_lr=(0,0))],
                    [dict(person_id='X',true_delta_lr=(1,),predicted_delta_lr=(0,0))]]:
        try:
            summarize(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    return dict(passed=True, data='synthetic only', people=2,pairs=3,
                hand_calculated_model_delta_mae=3,hand_calculated_no_change_delta_mae=4.5,
                real_development_metrics_computed=False)


if __name__ == '__main__':
    import json
    print(json.dumps(synthetic_check(),indent=2))
