"""Rebuild changes and verify metrics with explicit person/side/pair weights."""
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from audit import HERE, array, digest


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    report=read(HERE/'change-results.json')
    saved=read(HERE/'change-pairs.json')
    prepared=read(HERE/'prepared-pairs.json')['pairs']
    path=Path(report['frozen_prediction_path'])
    assert digest(path)==report['frozen_prediction_sha256']
    with zipfile.ZipFile(path) as z:
        ids,vids,folds=array(z,'person_id'),array(z,'video_id'),array(z,'fold')
        y,p=array(z,'y'),array(z,'bagged')
    index={v:i for i,v in enumerate(vids)}
    counts=Counter(r['person_id'] for r in prepared)
    assert len(counts)==229 and len(prepared)==581
    rebuilt=[]
    differences=[]
    for spec,row in zip(prepared,saved):
        assert len(spec['lower_age_video_ids'])==len(spec['higher_age_video_ids'])==1
        a,b=index[spec['lower_age_video_ids'][0]],index[spec['higher_age_video_ids'][0]]
        assert ids[a]==ids[b]==row['person_id']==spec['person_id']
        assert folds[a]==folds[b]==row['fold']
        for side in (0,1):
            actual=y[2*b+side]-y[2*a+side]
            # Independent residual identity: prediction change - true change = error_b-error_a.
            residual=(p[2*b+side]-y[2*b+side])-(p[2*a+side]-y[2*a+side])
            predicted=actual+residual
            differences.extend((abs(actual-row['true_delta_lr'][side]),abs(predicted-row['predicted_delta_lr'][side])))
            rebuilt.append((ids[a],side,actual,predicted,residual))
    assert len(saved)==len(prepared) and max(differences)<1e-10
    checked=[]
    for name,sides in [('overall',(0,1)),('left',(0,)),('right',(1,))]:
        entries=[r for r in rebuilt if r[1] in sides]
        changed=Counter(r[0] for r in entries if r[2]!=0)
        for model in ('fixed_model','no_change'):
            expected=(report['overall'] if name=='overall' else report['sides'][name])[model]
            mae=math.fsum(abs(r[4] if model=='fixed_model' else -r[2])/(len(counts)*len(sides)*counts[r[0]]) for r in entries)
            bias=math.fsum((r[4] if model=='fixed_model' else -r[2])/(len(counts)*len(sides)*counts[r[0]]) for r in entries)
            direction=[]
            for person,side,actual,predicted,residual in entries:
                if actual==0:
                    continue
                prediction=predicted if model=='fixed_model' else 0
                score=.5 if prediction==0 else float((actual>0)==(prediction>0))
                direction.append(score/(len(changed)*changed[person]))
            values=dict(person_mean_absolute_delta_error=mae,person_mean_signed_delta_error=bias,
                        person_mean_direction_agreement=math.fsum(direction))
            for k,value in values.items():
                checked.append(abs(value-expected[k]))
            assert expected['changed_side_pair_entries']==sum(changed.values())
    assert max(checked)<1e-10
    person_rows=read(HERE/'change-people.json')
    assert len(person_rows)==229 and len({r['person_id'] for r in person_rows})==229
    assert sum(r['model_minus_no_change_mae']<0 for r in person_rows)==report['person_mae_comparison']['better']==132
    assert sum(r['model_minus_no_change_mae']>0 for r in person_rows)==report['person_mae_comparison']['worse']==97
    result=dict(passed=True,people=229,pairs=581,side_pair_entries=len(rebuilt),
                independent_metric_max_difference=max(checked),pair_reconstruction_max_difference=max(differences),
                verification='Residual identity and direct weights, independent of change_metrics.summarize',
                source_sha256={p.name:digest(p) for p in [HERE/'change-results.json',HERE/'change-pairs.json',HERE/'change-people.json',Path(__file__)]},
                clinical_change_validation=False)
    (HERE/'change-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
