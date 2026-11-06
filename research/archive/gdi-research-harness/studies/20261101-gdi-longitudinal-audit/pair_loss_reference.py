"""Unadopted full-batch loss reference and metadata-only split preparation.

No model, optimizer, training loop, loss coefficient or success threshold.
The state term preserves record weighting; the change term weights people equally.
"""
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from audit import HERE,digest


def make_pairs(records,allowed_people):
    if any(r['person_id'] not in allowed_people for r in records):
        raise ValueError('Records include a person outside the allowed training set')
    keys=[(r['person_id'],r['exam_id']) for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate person/exam')
    grouped=defaultdict(list)
    for i,r in enumerate(records):
        if len(r['age_values'])==1 and math.isfinite(r['age_values'][0]):
            grouped[r['person_id']].append(i)
    result=[]
    for person,ix in sorted(grouped.items()):
        for a,b in itertools.combinations(sorted(ix,key=lambda i:records[i]['age_values'][0]),2):
            if records[a]['age_values']==records[b]['age_values']:continue
            if records[a]['fold']!=records[b]['fold']:raise ValueError('Person crosses folds')
            result.append((a,b))
    return result


def loss_terms(predicted,actual,records,pairs,allowed_people):
    if not records or len(predicted)!=len(actual) or len(actual)!=len(records):
        raise ValueError('Missing or mismatched records')
    if any(r['person_id'] not in allowed_people for r in records):
        raise ValueError('Non-training record in loss')
    if any(len(v)!=2 or not all(math.isfinite(x) for x in v) for v in list(predicted)+list(actual)):
        raise ValueError('Invalid left/right predictions or labels')
    state=mean(mean((p-t)**2 for p,t in zip(pr,tr)) for pr,tr in zip(predicted,actual))
    grouped=defaultdict(list)
    for a,b in pairs:
        if a==b or not 0<=a<len(records) or not 0<=b<len(records):raise ValueError('Invalid pair index')
        if records[a]['person_id']!=records[b]['person_id']:raise ValueError('Cross-person pair')
        if records[a]['fold']!=records[b]['fold']:raise ValueError('Cross-fold pair')
        grouped[records[a]['person_id']].append(mean(((predicted[b][s]-predicted[a][s])-(actual[b][s]-actual[a][s]))**2 for s in (0,1)))
    return dict(record_mean_state_mse=state,
                person_mean_change_mse=mean(mean(v) for v in grouped.values()) if grouped else None,
                paired_people=len(grouped),pairs=len(pairs),
                note='No paired people => unavailable change term; caller must decide handling before training')


def synthetic_check():
    records=[dict(person_id=p,exam_id=str(i),fold=0,age_values=[age])
             for i,(p,age) in enumerate([('A',1),('A',2),('A',3),('B',1),('B',2),('C',1)])]
    allowed={'A','B','C'};pairs=make_pairs(records,allowed)
    actual=[[0,0]]*6;predicted=[[e,e] for e in [0,1,3,2,0,4]]
    terms=loss_terms(predicted,actual,records,pairs,allowed)
    assert len(pairs)==4 and terms['paired_people']==2
    assert terms['record_mean_state_mse']==5
    assert abs(terms['person_mean_change_mse']-13/3)<1e-12
    reverse=loss_terms(predicted,actual,records,[(b,a) for a,b in pairs],allowed)
    assert reverse==terms
    assert loss_terms(predicted,actual,records,[],allowed)['person_mean_change_mse'] is None
    for action in [lambda:make_pairs(records,{'A'}),
                   lambda:make_pairs(records+[records[0]],allowed),
                   lambda:loss_terms(predicted,actual,records,[(0,3)],allowed),
                   lambda:loss_terms(predicted,actual,records,[(0,99)],allowed)]:
        try:action()
        except ValueError:pass
        else:raise AssertionError('Invalid split/pair was accepted')
    return dict(passed=True,data='synthetic only',state_mse=5,change_mse=13/3,optimizer_steps=0)


def main():
    synthetic=synthetic_check();visits=json.loads((HERE/'development-visits.json').read_text(encoding='utf-8'))
    people={r['person_id'] for r in visits};counts=[]
    for outer in range(5):
        for inner in [None]+[i for i in range(5) if i!=outer]:
            excluded={r['person_id'] for r in visits if r['fold']==outer or r['fold']==inner}
            allowed=people-excluded;train=[r for r in visits if r['person_id'] in allowed]
            pairs=make_pairs(train,allowed)
            assert not {train[a]['person_id'] for a,b in pairs}&excluded
            counts.append(dict(outer=outer,inner=inner,train_people=len(allowed),train_records=len(train),
                               train_paired_people=len({train[a]['person_id'] for a,b in pairs}),train_pairs=len(pairs)))
    result=dict(proposal_only=True,synthetic_check=synthetic,partitions=counts,
                metadata_only=True,actual_labels_used_in_loss=False,
                optimizer_steps=0,loss_weight_selected=False,
                script_sha256=digest(Path(__file__)),visits_sha256=digest(HERE/'development-visits.json'))
    (HERE/'pair-loss-preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(synthetic_check=synthetic,partitions=len(counts),
                         train_paired_people_range=[min(r['train_paired_people'] for r in counts),max(r['train_paired_people'] for r in counts)],
                         train_pairs_range=[min(r['train_pairs'] for r in counts),max(r['train_pairs'] for r in counts)],
                         actual_training_steps=0,proposal_only=True),indent=2))


if __name__=='__main__':main()
