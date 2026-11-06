"""Independent standard-library verification of category and case arithmetic."""
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from audit import HERE,digest

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    rows=read(HERE/'cause-pairs.json');report=read(HERE/'cause-results.json')
    n=Counter(r['person_id'] for r in rows);weights=defaultdict(list);errors=defaultdict(list);counts=Counter()
    maxdiff=0.0
    for r in rows:
        for s,(t,p) in enumerate(zip(r['true_delta_lr'],r['predicted_delta_lr'])):
            assert t!=0
            if t*p<0:k='direction_reversed'
            elif p==0:k='predicted_no_change'
            elif p/t<1:k='correct_direction_smaller_magnitude'
            else:k='correct_direction_equal_or_larger_magnitude'
            assert k==r['category_lr'][s]
            counts[k]+=1;w=1/(len(n)*2*n[r['person_id']]);weights[k].append(w);errors[k].append(abs(p-t)*w)
    assert sum(counts.values())==1162 and len(rows)==581 and len(n)==229
    for k in counts:
        c=report['categories'][k];assert counts[k]==c['raw_count']
        maxdiff=max(maxdiff,abs(math.fsum(weights[k])-c['weight']),abs(math.fsum(errors[k])-c['error_contribution']))
    total=math.fsum(math.fsum(e) for e in errors.values())
    assert abs(total-read(HERE/'change-results.json')['overall']['fixed_model']['person_mean_absolute_delta_error'])<1e-10
    cases=read(HERE/'cause-cases.json');perperson=defaultdict(list)
    for r in rows:perperson[r['person_id']].append(r)
    ranking=sorted(perperson,key=lambda p:sum(r['pair_mae'] for r in perperson[p])/len(perperson[p]),reverse=True)[:10]
    assert [r['person_id'] for r in cases]==ranking
    for r in cases:
        assert r['pair_mae']==max(t['pair_mae'] for t in perperson[r['person_id']])
        for s in (0,1):
            assert abs(r['actual_later_lr'][s]-r['actual_initial_lr'][s]-r['true_delta_lr'][s])<1e-10
            assert abs(r['predicted_later_lr'][s]-r['predicted_initial_lr'][s]-r['predicted_delta_lr'][s])<1e-10
    assert maxdiff<1e-10
    result=dict(passed=True,people=229,pairs=581,cases=10,category_arithmetic_max_difference=maxdiff,
                error_contributions_sum=total,clinical_or_causal_validation=False,
                verification_scope='Categories, weights, selected-case ranking and endpoint arithmetic; input quality comes from hash-checked saved source artifacts',
                sources={p.name:digest(p) for p in [HERE/'analyze_causes.py',HERE/'cause-results.json',HERE/'cause-cases.json',HERE/'cause-pairs.json',Path(__file__)]})
    (HERE/'cause-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
